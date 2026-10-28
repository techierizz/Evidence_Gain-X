import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
import numpy as np

from src.data.manifest import DatasetManifest, EvidenceItem
from src.models.classifier import DiseaseClassifier
from src.evidence.selector import EvidenceGainXPolicy
from src.diagnosis.pipeline import EvidenceGainXPipeline
from src.verification.reliability import EvidenceReliabilityChecker

def run_scenario_tests():
    print("=" * 70)
    print("EVIDENCEGAIN-X: VERIFICATION OF SCENARIO B & SCENARIO C")
    print("=" * 70)

    # -------------------------------------------------------------
    # TEST SCENARIO C: IMAGE QUALITY & BLUR REJECTION (MODULE 7)
    # -------------------------------------------------------------
    print("\n[TEST SCENARIO C] Image Quality, Blur & Lighting Gating (Module 7)")
    print("-" * 70)
    checker = EvidenceReliabilityChecker(blur_threshold=80.0, min_brightness=30.0)

    # Test 1: Blurry Image
    blurry_item = EvidenceItem(
        evidence_type="lesion_closeup",
        modality="image",
        data_path_or_content="data/test_samples/sample_blurry.jpg"
    )
    rep_blur = checker.assess_evidence(blurry_item)
    print(f"1. Blurry Image Assessment:")
    print(f"   - File: {blurry_item.data_path_or_content}")
    print(f"   - Action: [{rep_blur.action}] (Quality Score: {rep_blur.quality_score:.2f})")
    print(f"   - Blur Score (Laplacian variance): {rep_blur.blur_score:.1f} (Threshold: > 80.0)")
    print(f"   - Diagnostic Reasons: {rep_blur.reasons}")
    assert rep_blur.action in ["REJECT", "DOWNWEIGHT"]
    print("   >>> PASS: Blurry image successfully gated!")

    # Test 2: Dark / Underexposed Image
    dark_item = EvidenceItem(
        evidence_type="lesion_closeup",
        modality="image",
        data_path_or_content="data/test_samples/sample_dark_underexposed.jpg"
    )
    rep_dark = checker.assess_evidence(dark_item)
    print(f"\n2. Dark/Underexposed Image Assessment:")
    print(f"   - File: {dark_item.data_path_or_content}")
    print(f"   - Action: [{rep_dark.action}] (Quality Score: {rep_dark.quality_score:.2f})")
    print(f"   - Brightness: {rep_dark.brightness_score:.1f} (Min Required: > 30.0)")
    print(f"   - Diagnostic Reasons: {rep_dark.reasons}")
    assert rep_dark.action == "REJECT"
    print("   >>> PASS: Underexposed image successfully rejected!")

    # -------------------------------------------------------------
    # TEST SCENARIO B: CONTRADICTION & WRONG LEAF UPLOAD (MODULE 8)
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[TEST SCENARIO B] Contradiction Verification & Wrong Leaf Upload (Module 8)")
    print("-" * 70)

    manifest = DatasetManifest.load_json("data/manifests/dataset_manifest.json")
    classes = manifest.classes
    cand_types = ["lesion_closeup", "leaf_underside", "stem_view", "symptom_text", "weather_context"]

    classifier = DiseaseClassifier(num_classes=len(classes), pretrained=False)
    classifier.load_checkpoint("checkpoints/classifier_best.pt")
    
    policy = EvidenceGainXPolicy(classes=classes, candidate_types=cand_types)
    pipeline = EvidenceGainXPipeline(
        classifier=classifier,
        policy=policy,
        classes=classes,
        max_budget=3
    )

    # Pick an ambiguous blight case
    sample_case = manifest.cases[0]
    print(f"Initial Plant Case: {sample_case.case_id} (True Disease: {sample_case.disease})")
    
    # 1. Provisional Diagnosis
    init_state = pipeline.diagnose_initial(sample_case.initial_image_path)
    print(f"1. Provisional Diagnosis from Initial Image:")
    print(f"   - Leading Hypothesis: {init_state['leading_disease']} ({init_state['leading_prob']*100:.1f}%)")
    print(f"   - Critical Rival:     {init_state['rival_disease']} ({init_state['rival_prob']*100:.1f}%)")
    print(f"   - Shannon Entropy:    {init_state['entropy']:.3f} bits (Ambiguous: {init_state['is_ambiguous']})")

    # 2. Simulate User uploading a CONTRADICTORY Healthy Leaf instead of requested blight close-up
    print(f"\n2. User Uploads Contradictory Evidence (Healthy Leaf from another plant):")
    contradictory_ev = EvidenceItem(
        evidence_type="lesion_closeup",
        modality="image",
        data_path_or_content="data/test_samples/sample_healthy_contradiction.jpg"
    )

    updated_state = pipeline.process_single_evidence(init_state, "lesion_closeup", contradictory_ev)
    trail_item = updated_state["evidence_trail"][-1]

    print(f"   - Quality Action:       [{trail_item['action']}] (Quality: {trail_item['quality_score']:.2f})")
    print(f"   - Contradiction Flag:   {trail_item['contradiction']}")
    print(f"   - Contradiction Score:  {trail_item['contradiction_score']:.3f}")
    print(f"   - System Explanation:   {trail_item['contradiction_explanation']}")
    print(f"   - Final Belief After:   {trail_item['leading_after']} ({trail_item['leading_prob_after']*100:.1f}%)")
    
    assert trail_item["contradiction"] == True
    print("\n>>> PASS: Contradiction successfully detected and penalized! Model prevented false confidence.")
    print("=" * 70)

if __name__ == "__main__":
    run_scenario_tests()
