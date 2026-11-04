import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.data.manifest import DatasetManifest, EvidenceItem
from src.models.classifier import DiseaseClassifier
from src.evidence.selector import EvidenceGainXPolicy
from src.diagnosis.pipeline import EvidenceGainXPipeline
import copy

def test_full_website_workflow():
    cfg = load_config('configs/config.yaml')
    classes = cfg['dataset']['classes']
    cand_types = cfg['evidence']['candidate_types']

    classifier = DiseaseClassifier(num_classes=len(classes), backbone_name=cfg['classifier']['backbone'], pretrained=False)
    classifier.load_checkpoint('checkpoints/classifier_best.pt', device='cpu')
    policy = EvidenceGainXPolicy(classes=classes, candidate_types=cand_types)
    pipeline = EvidenceGainXPipeline(classifier=classifier, policy=policy, classes=classes, device='cpu')

    manifest = DatasetManifest.load_json('data/manifests/dataset_manifest.json')
    case_dict = {c.case_id: c for c in manifest.cases}

    # Test Preset 1: Early Blight
    c1 = case_dict['tomato_early_blight_0001']
    s1 = pipeline.diagnose_initial(c1.initial_image_path)
    assert s1['leading_disease'] == 'early_blight'
    print("Test 1 Early Blight: OK")

    # Test Preset 3: Septoria Spot (Look-alike Ambiguity)
    c3 = case_dict['tomato_septoria_leaf_spot_0001']
    s3 = pipeline.diagnose_initial(c3.initial_image_path)
    assert s3['is_ambiguous'] == True
    print("Test 3 Septoria Ambiguity: OK")

    # Test Preset 5: Blurry Gating (Scenario C)
    b_ev = EvidenceItem(
        evidence_type='lesion_closeup',
        modality='image',
        data_path_or_content='data/test_samples/sample_blurry.jpg'
    )
    s_blur = pipeline.process_single_evidence(copy.deepcopy(s3), 'lesion_closeup', b_ev)
    assert s_blur['evidence_trail'][-1]['action'] == 'REJECTED'
    print("Test 5 Blurry Gating: OK")

    # Test Preset 6: Contradiction Gating (Scenario B)
    c_ev = EvidenceItem(
        evidence_type='lesion_closeup',
        modality='image',
        data_path_or_content='data/test_samples/sample_healthy_contradiction.jpg'
    )
    s_contra = pipeline.process_single_evidence(copy.deepcopy(s1), 'lesion_closeup', c_ev)
    assert s_contra['evidence_trail'][-1]['contradiction'] == True
    print("Test 6 Contradiction Gating: OK")

    # Test Benchmark acquisition
    ev_key = 'lesion_closeup'
    bench_ev = c1.available_evidence[ev_key]
    s_bench = pipeline.process_single_evidence(copy.deepcopy(s1), ev_key, bench_ev)
    assert s_bench['evidence_trail'][-1]['action'] == 'ACCEPT'
    print("Test Benchmark acquisition: OK")

    print("ALL 5 WORKFLOW TESTS PASSED!")

if __name__ == "__main__":
    test_full_website_workflow()
