from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np
import torch

@dataclass
class ContradictionReport:
    is_contradiction: bool
    contradiction_score: float # 0.0 to 1.0 (higher = more contradictory)
    expected_shift_direction: str
    observed_shift_direction: str
    penalty_factor: float
    explanation: str

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
        expected_probs_h2: Optional[np.ndarray] = None
    ) -> ContradictionReport:
        # Directional divergence between expected update and observed update
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
            # Observation aligns if it supports EITHER of the competing hypotheses (h1 or h2)
            best_cos = max(cos1, cos2)
        else:
            best_cos = cos1

        # Contradiction score: 0 (aligned with h1 or h2) to 1 (opposes both look-alike hypotheses)
        contradiction_score = float(np.clip((1.0 - best_cos) / 2.0, 0.0, 1.0))
        is_contradiction = contradiction_score >= self.contradiction_threshold

        h1_name = classes[h1_idx]
        h2_name = classes[h2_idx]

        exp_dir = f"Support {h1_name} or {h2_name}"
        obs_dir = f"Support {classes[int(np.argmax(observed_probs))]}"

        if is_contradiction:
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
            explanation=explanation
        )
