import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import torch
from torch.utils.data import DataLoader


from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logger import setup_logger
from src.data.manifest import DatasetManifest
from src.data.splitter import create_leakage_safe_splits, save_split_manifests
from src.data.transforms import get_transforms
from src.data.dataset import CropDiseaseDataset
from src.data.episodes import EpisodeBuilder
from src.models.classifier import DiseaseClassifier
from src.models.trainer import ClassifierTrainer
from src.evaluation.benchmark import PolicyBenchmarkRunner
from src.evaluation.ablations import AblationStudyRunner

def run_all_experiments(config_path: str = "configs/config.yaml", force_retrain: bool = True):
    logger = setup_logger("EvidenceGainX_ExperimentEngine")
    logger.info("Starting EvidenceGain-X End-to-End Experimental Suite...")

    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))

    manifest_path = "data/manifests/dataset_manifest.json"
    if not Path(manifest_path).exists():
        logger.info("Manifest not found. Generating benchmark dataset first...")
        from scripts.download_data import create_synthetic_benchmark_dataset
        manifest = create_synthetic_benchmark_dataset()
    else:
        manifest = DatasetManifest.load_json(manifest_path)

    classes = cfg["dataset"]["classes"]
    cand_types = cfg["evidence"]["candidate_types"]

    # 1. Leakage-safe plant-grouped splits
    train_cases, val_cases, test_cases = create_leakage_safe_splits(
        manifest=manifest,
        train_ratio=cfg["dataset"]["split_ratios"]["train"],
        val_ratio=cfg["dataset"]["split_ratios"]["val"],
        test_ratio=cfg["dataset"]["split_ratios"]["test"],
        seed=cfg.get("seed", 42)
    )
    save_split_manifests(train_cases, val_cases, test_cases, manifest)
    logger.info(f"Dataset split complete: Train={len(train_cases)}, Val={len(val_cases)}, Test={len(test_cases)} cases.")

    # 2. Create PyTorch datasets and loaders
    train_tf, eval_tf = get_transforms(img_size=cfg["dataset"]["img_size"])
    train_ds = CropDiseaseDataset(train_cases, classes, transform=train_tf)
    val_ds = CropDiseaseDataset(val_cases, classes, transform=eval_tf)
    test_ds = CropDiseaseDataset(test_cases, classes, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=cfg["dataset"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["dataset"]["batch_size"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg["dataset"]["batch_size"], shuffle=False)

    # 3. Baseline Classifier Training & Calibration (Phase 1)
    device = "cuda" if torch.cuda.is_available() and cfg.get("device") == "cuda" else "cpu"
    logger.info(f"Execution device: {device}")

    pretrained_backbone = cfg["classifier"].get("pretrained", True)
    classifier = DiseaseClassifier(
        num_classes=len(classes),
        backbone_name=cfg["classifier"]["backbone"],
        pretrained=pretrained_backbone
    )
    ckpt_path = "checkpoints/classifier_best.pt"
    if Path(ckpt_path).exists() and not force_retrain:
        logger.info(f"Loading existing trained classifier checkpoint from {ckpt_path}...")
        classifier.load_checkpoint(ckpt_path, device=device)
    else:
        logger.info(f"Training ImageNet-pretrained {cfg['classifier']['backbone']} on authentic PlantVillage dataset...")
        trainer = ClassifierTrainer(
            model=classifier,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            lr=cfg["classifier"]["lr"],
            weight_decay=cfg["classifier"]["weight_decay"]
        )
        train_results = trainer.fit(epochs=cfg["classifier"]["epochs"], checkpoint_path=ckpt_path)


    # 4. Phase 2: Evidence Episodes & Empirical Utility Target Construction
    logger.info("Executing Phase 2: Constructing empirical utility targets across diagnostic training episodes...")
    from src.diagnosis.pipeline import EvidenceGainXPipeline
    from src.evidence.policies import InitialOnlyPolicy
    from src.evidence.selector import EvidenceGainSelector

    probe_pipeline = EvidenceGainXPipeline(
        classifier=classifier,
        policy=InitialOnlyPolicy(),
        classes=classes,
        device=device
    )
    ep_builder = EpisodeBuilder(classes=classes, max_budget=cfg["evidence"]["max_budget"])
    train_episodes = ep_builder.build_episodes_from_cases(train_cases)

    utility_training_records = []
    for ep in train_episodes:
        init_state = probe_pipeline.diagnose_initial(ep.initial_image_path)
        init_entropy = init_state["entropy"]
        init_leading = init_state["leading_disease"]
        
        probs_vec = np.array([init_state["full_distribution"].get(c, 0.25) for c in classes], dtype=np.float32)
        state_repr = np.concatenate([probs_vec, [init_entropy, init_state["margin"]]])

        for cand_type, ev_item in ep.available_candidates.items():
            temp_state = {
                "initial_probs": init_state["initial_probs"].copy(),
                "current_probs": init_state["current_probs"].copy(),
                "entropy": init_state["entropy"],
                "margin": init_state["margin"],
                "leading_disease": init_state["leading_disease"],
                "leading_prob": init_state["leading_prob"],
                "rival_disease": init_state["rival_disease"],
                "rival_prob": init_state["rival_prob"],
                "top_hypotheses": list(init_state["top_hypotheses"]),
                "full_distribution": dict(init_state["full_distribution"]),
                "is_ambiguous": init_state["is_ambiguous"],
                "conflict": dict(init_state["conflict"]),
                "evidence_trail": []
            }
            updated_state = probe_pipeline.process_single_evidence(temp_state, cand_type, ev_item)
            
            ent_reduction = max(0.0, init_entropy - updated_state["entropy"])
            was_correct = (init_leading == ep.true_disease)
            is_correct = (updated_state["leading_disease"] == ep.true_disease)
            diag_bonus = 1.0 if (not was_correct and is_correct) else (0.2 if is_correct else -0.3)
            
            target_utility = ent_reduction * 1.5 + diag_bonus
            utility_training_records.append({
                "state": state_repr,
                "candidate": cand_type,
                "target_utility": target_utility
            })

    # 5. Phase 3: Train Neural Utility Selector
    logger.info(f"Executing Phase 3: Training EvidenceGainSelector on {len(utility_training_records)} empirical targets...")
    selector = EvidenceGainSelector(
        candidate_types=cand_types,
        state_dim=len(classes) + 2,
        device=device
    )
    loss = selector.fit(utility_training_records, epochs=cfg["selector"].get("epochs", 20))
    logger.info(f"Phase 3 complete: Neural Utility Selector trained (Final MSE Loss: {loss:.4f})")

    # 6. Build Test Episodes & Phase 4: Closed-Loop Benchmarking across all 7 Baselines (B0 to B6)
    test_episodes = ep_builder.build_episodes_from_cases(test_cases)
    bench_runner = PolicyBenchmarkRunner(
        classifier=classifier,
        classes=classes,
        candidate_types=cand_types,
        learned_selector=selector,
        device=device
    )
    bench_report = bench_runner.run_benchmark(test_episodes)
    bench_report_path = "results/benchmark_results.json"
    bench_report.save_json(bench_report_path)
    logger.info(f"Benchmark results saved to {bench_report_path}")

    # 7. Ablation Studies
    ablation_runner = AblationStudyRunner(
        classifier=classifier,
        classes=classes,
        candidate_types=cand_types,
        device=device
    )
    ablation_results = ablation_runner.run_ablations(test_episodes)
    ablation_path = Path("results/ablation_results.json")
    with open(ablation_path, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)
    logger.info(f"Ablation results saved to {ablation_path}")

    # 8. Publication Figure Generation (Section 23)
    try:
        from scripts.generate_figures import generate_all_figures
        generate_all_figures(bench_report_path, str(ablation_path), "results/figures")
        logger.info("Paper figures generated successfully in results/figures/")
    except Exception as e:
        logger.warning(f"Figure generation notice: {e}")

    logger.info("=== EVIDENCEGAIN-X EXPERIMENTAL SUMMARY ===")
    for p_name, res in bench_report.results.items():
        logger.info(
            f"[{p_name:20s}] Acc: {res.accuracy:.4f} | F1: {res.macro_f1:.4f} | "
            f"Steps: {res.mean_steps:.2f} | Delta_Entropy: {res.mean_entropy_reduction:.4f}"
        )

if __name__ == "__main__":
    run_all_experiments()

