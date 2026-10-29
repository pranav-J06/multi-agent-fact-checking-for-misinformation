"""
Feature Extractor / Pooling module.
Converts sub-claim stance tensors [N, 2] and retrieval confidence scores into
the exact 10-dimensional engineered feature vector [1, 10] defined in Figure 2.
"""

from typing import List, Dict, Any
import numpy as np

from src.config import FEATURE_DIM, FEATURE_NAMES

class FeatureExtractor:
    @staticmethod
    def extract_features(
        stance_tensor: np.ndarray,
        retrieval_results: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Pools stance scores and retrieval confidence into a 10-D feature vector.

        Args:
            stance_tensor: np.ndarray of shape [N, 2], column 0 = support, column 1 = refute
            retrieval_results: List of retrieval dicts containing 'top_score'

        Returns:
            feature_vector: np.ndarray of shape [1, 10]
        """
        N = stance_tensor.shape[0]
        if N == 0:
            return np.zeros((1, FEATURE_DIM), dtype=np.float32)

        support_scores = stance_tensor[:, 0]
        refute_scores = stance_tensor[:, 1]
        retrieval_scores = np.array([r.get("top_score", 0.0) for r in retrieval_results], dtype=np.float32)

        # 1. Support statistics
        sup_mean = float(np.mean(support_scores))
        sup_max = float(np.max(support_scores))
        sup_std = float(np.std(support_scores)) if N > 1 else 0.0

        # 2. Refute statistics
        ref_mean = float(np.mean(refute_scores))
        ref_max = float(np.max(refute_scores))
        ref_std = float(np.std(refute_scores)) if N > 1 else 0.0

        # 3. Agreement ratio: difference normalized by sum, bounded in [-1.0, 1.0]
        denom = sup_mean + ref_mean + 1e-6
        agreement_ratio = float((sup_mean - ref_mean) / denom)
        agreement_ratio = max(-1.0, min(1.0, agreement_ratio))

        # 4. Retrieval confidence statistics
        ret_mean = float(np.mean(retrieval_scores)) if len(retrieval_scores) > 0 else 0.0
        ret_min = float(np.min(retrieval_scores)) if len(retrieval_scores) > 0 else 0.0

        # 5. Normalized sub-claim count (N bounded in [2, 6])
        n_norm = float(N / 6.0)

        features = np.array([
            sup_mean,
            sup_max,
            sup_std,
            ref_mean,
            ref_max,
            ref_std,
            agreement_ratio,
            ret_mean,
            ret_min,
            n_norm
        ], dtype=np.float32).reshape(1, FEATURE_DIM)

        return features

    @staticmethod
    def get_feature_dict(feature_vector: np.ndarray) -> Dict[str, float]:
        """Maps 10-D feature vector to named dictionary for visualization."""
        vec = feature_vector.flatten()
        return {name: float(vec[i]) for i, name in enumerate(FEATURE_NAMES)}
