from typing import Dict, Any, List, Optional
from pathlib import Path
from PIL import Image
import torch
import numpy as np

from src.models.classifier import DiseaseClassifier
from src.diagnosis.hypotheses import CompetingHypothesisTracker
from src.diagnosis.confusion import DiseaseConfusionAnalyzer
from src.diagnosis.updater import CalibratedBeliefUpdater
from src.evidence.policies import BaseAcquisitionPolicy
from src.planning.counterfactual import CounterfactualEvidencePlanner
from src.verification.reliability import EvidenceReliabilityChecker
from src.verification.contradiction import ContradictionVerifier
from src.data.manifest import EvidenceItem

class EvidenceGainXPipeline:
    def __init__(
        self,
        classifier: DiseaseClassifier,
        policy: BaseAcquisitionPolicy,
        classes: List[str],
        device: str = "cpu",
        img_size: int = 224,
        max_budget: int = 3,
        stopping_entropy: float = 0.35,
        stopping_margin: float = 0.70,
        candidate_types: Optional[List[str]] = None
    ):
        self.classifier = classifier.to(device)
        self.classifier.eval()
        self.policy = policy
        self.classes = classes
        self.device = device
        self.img_size = img_size
        self.max_budget = max_budget
        self.stopping_entropy = stopping_entropy
        self.stopping_margin = stopping_margin
        
        self.candidate_types = candidate_types or [
            "lesion_closeup", "leaf_underside", "stem_view", "symptom_text", "weather_context"
        ]

        self.hypothesis_tracker = CompetingHypothesisTracker(classes, margin_threshold=0.25)
        self.confusion_analyzer = DiseaseConfusionAnalyzer(classes)
        self.counterfactual_planner = CounterfactualEvidencePlanner(classes, self.candidate_types)
        self.reliability_checker = EvidenceReliabilityChecker()
        self.contradiction_verifier = ContradictionVerifier()
        self.belief_updater = CalibratedBeliefUpdater(classes)

    def _load_image(self, path_or_pil: Any) -> torch.Tensor:
        from torchvision.transforms.functional import resize, to_tensor, normalize
        if isinstance(path_or_pil, (str, Path)):
            img = Image.open(path_or_pil).convert("RGB")
        elif isinstance(path_or_pil, Image.Image):
            img = path_or_pil.convert("RGB")
        else:
            img = path_or_pil
        
        tensor = to_tensor(resize(img, [self.img_size, self.img_size]))
        tensor = normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return tensor.unsqueeze(0).to(self.device)

    def diagnose_initial(self, initial_image: Any) -> Dict[str, Any]:
        x0 = self._load_image(initial_image)
        with torch.no_grad():
            res = self.classifier.predict_provisional(x0)
        probs = res["probs"].squeeze(0).cpu().numpy()
        hypo_analysis = self.hypothesis_tracker.analyze(torch.tensor(probs))
        conflict_analysis = self.confusion_analyzer.characterize_conflict(hypo_analysis["top_hypotheses"])

        return {
            "initial_probs": probs.copy(),
            "current_probs": probs.copy(),
            "entropy": hypo_analysis["entropy"],
            "margin": hypo_analysis["margin"],
            "leading_disease": hypo_analysis["leading_disease"],
            "leading_prob": hypo_analysis["leading_prob"],
            "rival_disease": hypo_analysis["rival_disease"],
            "rival_prob": hypo_analysis["rival_prob"],
            "top_hypotheses": hypo_analysis["top_hypotheses"],
            "full_distribution": hypo_analysis["full_distribution"],
            "is_ambiguous": hypo_analysis["is_ambiguous"],
            "conflict": conflict_analysis,
            "evidence_trail": []
        }

    def _extract_evidence_likelihood(self, ev_item: EvidenceItem) -> np.ndarray:
        """
        Real multimodal evidence feature extraction & likelihood estimation.
        Processes either visual images or textual/contextual symptom strings.
        """
        with torch.no_grad():
            if ev_item.modality == "image":
                ev_tensor = self._load_image(ev_item.data_path_or_content)
                ev_logits, _ = self.classifier(ev_tensor)
                scaled_logits = ev_logits / self.classifier.temperature
                ev_probs = torch.softmax(scaled_logits, dim=-1).squeeze(0).cpu().numpy()
            else:
                text_content = str(ev_item.data_path_or_content)
                text_emb = self.classifier.encode_text_evidence([text_content], device=self.device)
                text_logits = self.classifier.predict_logits_from_features(text_emb)
                scaled_logits = text_logits / self.classifier.temperature
                ev_probs = torch.softmax(scaled_logits, dim=-1).squeeze(0).cpu().numpy()
        return ev_probs

    def process_single_evidence(
        self,
        state: Dict[str, Any],
        evidence_type: str,
        ev_item: EvidenceItem
    ) -> Dict[str, Any]:
        """
        Executes Module 7 (Reliability) -> Module 5 (Counterfactual Expectation) ->
        Module 8 (Contradiction Verification) -> Module 9 (Bayesian Posterior Update).
        """
        step_num = len(state["evidence_trail"]) + 1

        # 1. Evidence Quality / Reliability Check
        rel_report = self.reliability_checker.assess_evidence(ev_item)
        if rel_report.action == "REJECT":
            state["evidence_trail"].append({
                "step": step_num,
                "evidence_type": evidence_type,
                "modality": ev_item.modality,
                "action": "REJECTED",
                "quality_score": rel_report.quality_score,
                "reasons": rel_report.reasons,
                "contradiction": False,
                "contradiction_score": 0.0,
                "contradiction_explanation": "Blocked by reliability gating before contradiction verification.",
                "entropy_after": state["entropy"],
                "leading_after": state["leading_disease"],
                "leading_prob_after": state["leading_prob"]
            })
            return state

        # 2. Extract Real Evidence Likelihood Profile
        ev_likelihood = self._extract_evidence_likelihood(ev_item)

        # 3. Competing hypothesis indices & Counterfactual Planning expectation
        h1_name = state["leading_disease"]
        h2_name = state["rival_disease"] or state["top_hypotheses"][1].disease if len(state["top_hypotheses"]) > 1 else self.classes[(self.classes.index(h1_name) + 1) % len(self.classes)]
        
        h1_idx = self.classes.index(h1_name)
        h2_idx = self.classes.index(h2_name)

        prior_p = state["current_probs"].copy()
        cf_plan = self.counterfactual_planner.simulate_counterfactual_outcomes(
            current_probs=prior_p,
            h1_idx=h1_idx,
            h2_idx=h2_idx,
            evidence_type=evidence_type
        )
        expected_p = cf_plan["p_outcome_if_h1"]

        # 4. Tentative update to check for contradiction
        tentative_p = self.belief_updater.update_beliefs(
            prior_probs=prior_p,
            evidence_likelihood=ev_likelihood,
            quality_score=rel_report.quality_score,
            contradiction_penalty=1.0
        )

        # 5. Contradiction Verification (Module 8)
        contra_report = self.contradiction_verifier.verify(
            prior_probs=prior_p,
            expected_probs=expected_p,
            observed_probs=tentative_p,
            h1_idx=h1_idx,
            h2_idx=h2_idx,
            classes=self.classes
        )

        # 6. Final Bayesian belief update with contradiction penalty factor
        final_updated_p = self.belief_updater.update_beliefs(
            prior_probs=prior_p,
            evidence_likelihood=ev_likelihood,
            quality_score=rel_report.quality_score,
            contradiction_penalty=contra_report.penalty_factor
        )

        # 7. Update diagnostic state
        hypo_analysis = self.hypothesis_tracker.analyze(torch.tensor(final_updated_p))
        state["current_probs"] = final_updated_p
        state["entropy"] = hypo_analysis["entropy"]
        state["margin"] = hypo_analysis["margin"]
        state["leading_disease"] = hypo_analysis["leading_disease"]
        state["leading_prob"] = hypo_analysis["leading_prob"]
        state["rival_disease"] = hypo_analysis["rival_disease"]
        state["rival_prob"] = hypo_analysis["rival_prob"]
        state["top_hypotheses"] = hypo_analysis["top_hypotheses"]
        state["full_distribution"] = hypo_analysis["full_distribution"]
        state["is_ambiguous"] = hypo_analysis["is_ambiguous"]

        state["evidence_trail"].append({
            "step": step_num,
            "evidence_type": evidence_type,
            "modality": ev_item.modality,
            "action": rel_report.action,
            "quality_score": rel_report.quality_score,
            "reasons": rel_report.reasons,
            "contradiction": contra_report.is_contradiction,
            "contradiction_score": contra_report.contradiction_score,
            "contradiction_explanation": contra_report.explanation,
            "entropy_after": hypo_analysis["entropy"],
            "leading_after": hypo_analysis["leading_disease"],
            "leading_prob_after": hypo_analysis["leading_prob"]
        })
        return state

    def run_episode(
        self,
        initial_image: Any,
        available_candidates: Dict[str, EvidenceItem]
    ) -> Dict[str, Any]:
        state = self.diagnose_initial(initial_image)
        remaining_candidates = list(available_candidates.keys())
        step = 0

        while step < self.max_budget:
            # Check stopping criteria
            if state["entropy"] <= self.stopping_entropy and state["margin"] >= self.stopping_margin:
                break
            if not remaining_candidates:
                break

            # Policy selects next-best candidate observation
            selected_cand = self.policy.select_next(state, remaining_candidates)
            if selected_cand is None:
                break

            remaining_candidates.remove(selected_cand)
            ev_item = available_candidates[selected_cand]

            state = self.process_single_evidence(state, selected_cand, ev_item)
            step += 1

        state["total_steps_taken"] = step
        state["final_diagnosis"] = state["leading_disease"]
        state["final_confidence"] = state["leading_prob"]
        state["is_sufficient"] = (state["entropy"] <= self.stopping_entropy and state["margin"] >= self.stopping_margin)
        return state
