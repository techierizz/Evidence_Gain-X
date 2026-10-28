import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.manifest import DatasetManifest
from src.data.episodes import EpisodeBuilder
from src.models.classifier import DiseaseClassifier
from src.diagnosis.pipeline import EvidenceGainXPipeline
from src.evidence.policies import InitialOnlyPolicy, RandomPolicy, FixedOrderPolicy
from src.evidence.selector import EvidenceGainXPolicy
from src.verification.reliability import EvidenceReliabilityChecker
from src.verification.contradiction import ContradictionVerifier

def test_full_pipeline_flow():
    classes = ["early_blight", "late_blight", "septoria_leaf_spot", "healthy"]
    cand_types = ["lesion_closeup", "leaf_underside", "stem_view", "symptom_text", "weather_context"]
    
    manifest_path = "data/manifests/dataset_manifest.json"
    assert Path(manifest_path).exists()
    
    manifest = DatasetManifest.load_json(manifest_path)
    sample_case = manifest.cases[0]

    classifier = DiseaseClassifier(num_classes=len(classes), pretrained=False)
    policy = EvidenceGainXPolicy(classes=classes, candidate_types=cand_types)
    pipeline = EvidenceGainXPipeline(
        classifier=classifier,
        policy=policy,
        classes=classes,
        max_budget=3
    )

    # 1. Provisional diagnosis
    init_state = pipeline.diagnose_initial(sample_case.initial_image_path)
    assert "leading_disease" in init_state
    assert "entropy" in init_state
    assert "margin" in init_state

    # 2. Run full episode
    final_state = pipeline.run_episode(sample_case.initial_image_path, sample_case.available_evidence)
    assert "final_diagnosis" in final_state
    assert "total_steps_taken" in final_state
    assert len(final_state["evidence_trail"]) > 0

def test_all_baselines():
    classes = ["early_blight", "late_blight", "septoria_leaf_spot", "healthy"]
    cand_types = ["lesion_closeup", "leaf_underside", "stem_view", "symptom_text", "weather_context"]
    
    from src.evaluation.benchmark import PolicyBenchmarkRunner
    from src.evidence.selector import EvidenceGainSelector
    
    classifier = DiseaseClassifier(num_classes=len(classes), pretrained=False)
    selector = EvidenceGainSelector(candidate_types=cand_types, state_dim=len(classes)+2)
    
    runner = PolicyBenchmarkRunner(
        classifier=classifier,
        classes=classes,
        candidate_types=cand_types,
        learned_selector=selector
    )
    # Check that all 7 baselines (B0 to B6) are properly registered
    assert "B0_InitialOnly" in runner.policies
    assert "B1_Random" in runner.policies
    assert "B2_FixedOrder" in runner.policies
    assert "B3_ConfidenceDriven" in runner.policies
    assert "B4_UncertaintyOnly" in runner.policies
    assert "B5_EvidenceGain" in runner.policies
    assert "B6_EvidenceGainX" in runner.policies

def test_reliability_checker():
    checker = EvidenceReliabilityChecker()
    manifest = DatasetManifest.load_json("data/manifests/dataset_manifest.json")
    sample_ev = manifest.cases[0].available_evidence["lesion_closeup"]
    
    report = checker.assess_evidence(sample_ev)
    assert report.quality_score >= 0.0
    assert report.action in ["ACCEPT", "DOWNWEIGHT", "REJECT"]

def test_contradiction_verifier():
    verifier = ContradictionVerifier()
    prior = np.array([0.45, 0.40, 0.10, 0.05])
    expected = np.array([0.70, 0.20, 0.05, 0.05])
    observed_consistent = np.array([0.75, 0.15, 0.05, 0.05])
    observed_contradictory = np.array([0.15, 0.75, 0.05, 0.05])

    classes = ["early_blight", "late_blight", "septoria_leaf_spot", "healthy"]
    rep_ok = verifier.verify(prior, expected, observed_consistent, 0, 1, classes)
    assert not rep_ok.is_contradiction

    rep_contra = verifier.verify(prior, expected, observed_contradictory, 0, 1, classes)
    assert rep_contra.is_contradiction
    assert rep_contra.penalty_factor < 1.0

