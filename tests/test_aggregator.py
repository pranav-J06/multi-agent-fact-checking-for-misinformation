"""Unit tests for the Verdict Aggregator and Feature Extractor."""
import numpy as np
import torch
from src.config import FEATURE_DIM, NUM_CLASSES
from src.aggregator.feature_extractor import FeatureExtractor
from src.aggregator.model import VerdictAggregatorMLP

def test_feature_extractor_shape():
    # 3 sub-claims
    stance_tensor = np.array([
        [0.9, 0.1],
        [0.85, 0.15],
        [0.95, 0.05]
    ], dtype=np.float32)

    retrieval_results = [
        {"top_score": 0.85},
        {"top_score": 0.78},
        {"top_score": 0.90}
    ]

    features = FeatureExtractor.extract_features(stance_tensor, retrieval_results)
    assert features.shape == (1, FEATURE_DIM)
    assert features[0, 0] > 0.8  # support_mean
    assert features[0, 3] < 0.2  # refute_mean
    assert features[0, 6] > 0.5  # agreement_ratio
    assert abs(features[0, 9] - 0.5) < 1e-5  # num_subclaims_norm

def test_verdict_aggregator_mlp_architecture():
    model = VerdictAggregatorMLP()
    # Check layer parameters: 10*16 + 16 + 16*3 + 3 = 227
    total_params = model.count_parameters()
    assert total_params == 227

    dummy_input = torch.randn(2, FEATURE_DIM)
    probs = model(dummy_input)
    assert probs.shape == (2, NUM_CLASSES)
    # Check softmax sum to 1
    row_sums = probs.sum(dim=-1).detach().numpy()
    assert np.allclose(row_sums, [1.0, 1.0])

if __name__ == "__main__":
    test_feature_extractor_shape()
    test_verdict_aggregator_mlp_architecture()
    print("Aggregator tests passed!")
