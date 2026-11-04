from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np
import torch

@dataclass
class ContradictionReport:
    is_contradiction: bool
    contradiction_score: float
    expected_shift_direction: str
    observed_shift_direction: str
    penalty_factor: float
    explanation: str
    is_specimen_mismatch: bool = False
    mismatch_disease: Optional[str] = None

class ContradictionVerifier:
    def __init__(self, contradiction_threshold: float = 0.40, penalty_scale: float = 0.50):
        self.contradiction_threshold = contradiction_threshold
        self.penalty_scale = penalty_scale

    def verify(
        self,
        prior_probs: np.ndarray,
        expected_probs: np.ndarray,
        observed_probs: np.ndarray,
        h1_idx: int,
        h2_idx: int,
        classes: List[str],
        expected_probs_h2: Optional[np.ndarray] = None,
        evidence_likelihood: Optional[np.ndarray] = None
    ) -> ContradictionReport:
        delta_exp_h1 = expected_probs - prior_probs
        delta_obs = observed_probs - prior_probs

        norm_obs = float(np.linalg.norm(delta_obs))
        norm_exp1 = float(np.linalg.norm(delta_exp_h1))

        if norm_obs < 1e-6 or norm_exp1 < 1e-6:
            cos1 = 1.0
        else:
            cos1 = float(np.dot(delta_exp_h1, delta_obs) / (norm_exp1 * norm_obs))

        if expected_probs_h2 is not None:
            delta_exp_h2 = expected_probs_h2 - prior_probs
            norm_exp2 = float(np.linalg.norm(delta_exp_h2))
            if norm_exp2 < 1e-6 or norm_obs < 1e-6:
                cos2 = 1.0
            else:
                cos2 = float(np.dot(delta_exp_h2, delta_obs) / (norm_exp2 * norm_obs))
            best_cos = max(cos1, cos2)
        else:
            best_cos = cos1

        trajectory_score = float(np.clip((1.0 - best_cos) / 2.0, 0.0, 1.0))
        is_trajectory_contra = trajectory_score >= self.contradiction_threshold

        is_specimen_mismatch = False
        mismatch_disease = None
        mismatch_score = 0.0
        if evidence_likelihood is not None:
            ev_top_idx = int(np.argmax(evidence_likelihood))
            ev_top_prob = float(evidence_likelihood[ev_top_idx])
            prior_top_prob = float(prior_probs[ev_top_idx])
            if ev_top_idx != h1_idx and ev_top_idx != h2_idx and prior_top_prob < 0.05 and ev_top_prob >= 0.35:
                is_specimen_mismatch = True
                mismatch_disease = classes[ev_top_idx]
                mismatch_score = float(np.clip(ev_top_prob - prior_top_prob, 0.45, 1.0))

        is_contradiction = is_trajectory_contra or is_specimen_mismatch
        contradiction_score = max(trajectory_score, mismatch_score)

        h1_name = classes[h1_idx]
        h2_name = classes[h2_idx]

        exp_dir = f"Support {h1_name} or {h2_name}"
        obs_dir = f"Support {classes[int(np.argmax(observed_probs))]}"

        if is_specimen_mismatch and mismatch_disease:
            m_title = mismatch_disease.replace('_', ' ').title()
            penalty_factor = 0.0
            explanation = (
                f"Specimen Mismatch detected (Module 8): Uploaded evidence exhibits primary morphology of "
                f"'{m_title}' ({evidence_likelihood[ev_top_idx] * 100:.1f}%), which was ruled out on initial specimen "
                f"observation (prior: {prior_probs[ev_top_idx] * 100:.1f}%). Evidence quarantined to protect diagnostic integrity."
            )
        elif is_contradiction:
            penalty_factor = float(1.0 - self.penalty_scale * contradiction_score)
            explanation = (
                f"Contradiction detected: Acquired evidence shifts belief toward '{obs_dir}' "
                f"which contradicts both planned look-alike hypotheses '{h1_name}' and '{h2_name}' "
                f"(divergence: {contradiction_score:.2f}). Confidence is dampened to protect diagnosis."
            )
        else:
            penalty_factor = 1.0
            explanation = f"Observation consistent with expected diagnostic shift (divergence: {contradiction_score:.2f})."

        return ContradictionReport(
            is_contradiction=is_contradiction,
            contradiction_score=contradiction_score,
            expected_shift_direction=exp_dir,
            observed_shift_direction=obs_dir,
            penalty_factor=penalty_factor,
            explanation=explanation,
            is_specimen_mismatch=is_specimen_mismatch,
            mismatch_disease=mismatch_disease
        )
