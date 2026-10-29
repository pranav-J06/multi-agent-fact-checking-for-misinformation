"""
Verdict Aggregator Model module (The Novelty Head).
Implements a lightweight 2-layer MLP: Dense(10, 16) -> ReLU -> Dense(16, 3) -> Softmax
Replaces the expensive LLM Judge used in prior debate frameworks (e.g., MADR, MAD-Fact).
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any
import numpy as np

from src.config import FEATURE_DIM, HIDDEN_DIM, NUM_CLASSES, CLASS_NAMES

class VerdictAggregatorMLP(nn.Module):
    def __init__(self, in_features: int = FEATURE_DIM, hidden_dim: int = HIDDEN_DIM, num_classes: int = NUM_CLASSES):
        super(VerdictAggregatorMLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out = self.relu(out)
        return self.fc2(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape [batch_size, 10]
        Returns:
            probabilities: Tensor of shape [batch_size, 3]
        """
        logits = self.get_logits(x)
        return self.softmax(logits)

    def predict_single(self, feature_vector: np.ndarray) -> Dict[str, Any]:
        """
        Inference on a single feature vector of shape [1, 10].
        Returns prediction dictionary with label, confidence, and class probabilities.
        """
        self.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(feature_vector, dtype=torch.float32)
            probs_tensor = self.forward(x_tensor).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs_tensor))
            confidence = float(probs_tensor[pred_idx])

            return {
                "predicted_label": CLASS_NAMES[pred_idx],
                "confidence": confidence,
                "probabilities": {
                    CLASS_NAMES[i]: float(probs_tensor[i]) for i in range(len(CLASS_NAMES))
                },
                "predicted_class_id": pred_idx
            }

    def count_parameters(self) -> int:
        """Counts total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
