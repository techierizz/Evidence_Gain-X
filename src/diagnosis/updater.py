from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
import numpy as np

class MultimodalDiagnosisUpdater(nn.Module):
    """
    Module 9: Multimodal Diagnosis Updater.
    Fuses the base initial observation features with the newly acquired
    evidence features (visual, textual, or contextual), gated by
    evidence reliability quality and contradiction verification factor.
    """
    def __init__(self, embed_dim: int = 512, num_classes: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # Adaptive multimodal fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dim * 2 + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(
        self,
        base_features: torch.Tensor,
        evidence_features: torch.Tensor,
        quality_score: float,
        contradiction_penalty: float
    ) -> torch.Tensor:
        b_size = base_features.size(0)
        q_tensor = torch.full((b_size, 1), quality_score, dtype=torch.float32, device=base_features.device)
        p_tensor = torch.full((b_size, 1), contradiction_penalty, dtype=torch.float32, device=base_features.device)

        # Modulate evidence by reliability quality
        weighted_ev = evidence_features * float(quality_score)
        
        fused_input = torch.cat([base_features, weighted_ev, q_tensor, p_tensor], dim=-1)
        delta_logits = self.fusion_gate(fused_input)
        return delta_logits

class CalibratedBeliefUpdater:
    """
    Principled Bayesian belief updater combining prior disease probability,
    real evidence likelihood, evidence quality score (Module 7), and
    contradiction penalty (Module 8).
    
    P(D | X_0, e) ∝ P(D | X_0) * [ P(e | D) ] ^ (quality * contradiction_penalty)
    """
    def __init__(self, classes: List[str]):
        self.classes = classes
        self.num_classes = len(classes)

    def update_beliefs(
        self,
        prior_probs: np.ndarray,
        evidence_likelihood: np.ndarray,
        quality_score: float = 1.0,
        contradiction_penalty: float = 1.0
    ) -> np.ndarray:
        # Effective exponent based on reliability & contradiction verification
        eff_weight = float(np.clip(quality_score * contradiction_penalty, 0.05, 1.5))
        
        # Guard against zeros
        safe_ev = np.maximum(1e-5, evidence_likelihood)
        safe_prior = np.maximum(1e-5, prior_probs)
        
        # Likelihood weighting
        weighted_likelihood = np.power(safe_ev, eff_weight)
        posterior = safe_prior * weighted_likelihood
        
        norm_sum = float(posterior.sum())
        if norm_sum > 0:
            posterior = posterior / norm_sum
        else:
            posterior = prior_probs.copy()
            
        return posterior.astype(np.float32)

# Backward-compatibility alias
RuleBasedBeliefUpdater = CalibratedBeliefUpdater

