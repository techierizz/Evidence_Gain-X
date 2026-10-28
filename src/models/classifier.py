import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path

class TextEvidenceEncoder(nn.Module):
    """
    Lightweight agricultural text and context encoder.
    Maps diagnostic symptom keywords and microclimate weather data
    into the shared visual-semantic embedding space.
    """
    def __init__(self, embed_dim: int = 512, vocab_size: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        # Keyword vocabulary hash embedding + linear projection
        self.embedding = nn.EmbeddingBag(vocab_size, 64, mode="mean")
        self.proj = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        # Built-in agricultural vocabulary mapping for key symptomatic terms
        self.vocab = {
            "concentric": 1, "rings": 2, "target": 3, "dark": 4, "brown": 5,
            "chlorotic": 6, "halo": 7, "lower": 8, "water-soaked": 9, "irregular": 10,
            "pale": 11, "white": 12, "mold": 13, "downy": 14, "margin": 15,
            "circular": 16, "spots": 17, "grey": 18, "border": 19, "pycnidia": 20,
            "uniform": 21, "green": 22, "healthy": 23, "canker": 24, "stem": 25,
            "warm": 26, "dew": 27, "rain": 28, "cool": 29, "humid": 30,
            "high": 31, "wet": 32, "dry": 33, "lesion": 34, "blight": 35
        }

    def _text_to_indices(self, text: str) -> torch.Tensor:
        words = text.lower().replace("-", " ").replace(",", " ").replace(".", " ").split()
        indices = [self.vocab.get(w, hash(w) % 200 + 36) for w in words]
        if not indices:
            indices = [0]
        return torch.tensor(indices, dtype=torch.long)

    def forward(self, texts: List[str], device: torch.device) -> torch.Tensor:
        batch_indices = [self._text_to_indices(t).to(device) for t in texts]
        offsets = torch.tensor([0] + [len(idx) for idx in batch_indices[:-1]], dtype=torch.long, device=device).cumsum(0)
        flat_indices = torch.cat(batch_indices)
        emb = self.embedding(flat_indices, offsets)
        return self.proj(emb)

class DiseaseClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        backbone_name: str = "resnet18",
        pretrained: bool = True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone_name

        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base = models.resnet18(weights=weights)
            in_features = base.fc.in_features
            base.fc = nn.Identity()
            self.backbone = base
            self.embed_dim = in_features
        elif backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base = models.resnet50(weights=weights)
            in_features = base.fc.in_features
            base.fc = nn.Identity()
            self.backbone = base
            self.embed_dim = in_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.classifier_head = nn.Linear(self.embed_dim, num_classes)
        self.text_encoder = TextEvidenceEncoder(embed_dim=self.embed_dim)
        self.temperature = nn.Parameter(torch.ones(1) * 1.0, requires_grad=False)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract visual representation embedding."""
        return self.backbone(x)

    def encode_text_evidence(self, texts: List[str], device: Optional[torch.device] = None) -> torch.Tensor:
        """Encode textual symptoms or weather data into evidence embedding space."""
        if device is None:
            device = next(self.parameters()).device
        return self.text_encoder(texts, device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.extract_features(x) # (B, embed_dim)
        logits = self.classifier_head(features) # (B, num_classes)
        return logits, features

    def predict_logits_from_features(self, features: torch.Tensor) -> torch.Tensor:
        """Compute logits from pre-extracted features."""
        return self.classifier_head(features)

    def predict_provisional(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits, features = self.forward(x)
        scaled_logits = logits / self.temperature
        probs = torch.softmax(scaled_logits, dim=-1)
        
        # Calculate entropy H(p) = -sum(p * log2(p + eps))
        eps = 1e-8
        entropy = -torch.sum(probs * torch.log2(probs + eps), dim=-1)
        
        # Sort probabilities for margin computation
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        margin = sorted_probs[:, 0] - sorted_probs[:, 1]

        return {
            "logits": logits,
            "scaled_logits": scaled_logits,
            "probs": probs,
            "features": features,
            "entropy": entropy,
            "margin": margin,
            "top1_pred": sorted_indices[:, 0],
            "top1_prob": sorted_probs[:, 0],
            "sorted_indices": sorted_indices,
            "sorted_probs": sorted_probs
        }

    def load_checkpoint(self, checkpoint_path: str, device: str = "cpu") -> bool:
        p = Path(checkpoint_path)
        if not p.exists():
            return False
        state = torch.load(p, map_location=device)
        if "model_state_dict" in state:
            self.load_state_dict(state["model_state_dict"], strict=False)
            if "temperature" in state:
                self.temperature.data.fill_(state["temperature"])
        else:
            self.load_state_dict(state, strict=False)
        return True
