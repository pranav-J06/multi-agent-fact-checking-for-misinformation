"""
Evaluation metrics module for Multi-Agent Fact-Checking.
Calculates Accuracy, Precision, Recall, Macro F1, Confusion Matrix, and latency profiles.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from src.config import CLASS_NAMES, LABEL_MAP

class Evaluator:
    @staticmethod
    def evaluate_predictions(
        y_true: List[str],
        y_pred: List[str]
    ) -> Dict[str, Any]:
        """
        Computes standard multi-class classification metrics.
        """
        # Map labels to integer IDs
        y_true_ids = [LABEL_MAP.get(lbl, 2) for lbl in y_true]
        y_pred_ids = [LABEL_MAP.get(lbl, 2) for lbl in y_pred]

        acc = float(accuracy_score(y_true_ids, y_pred_ids))
        correct = int(np.sum(np.array(y_true_ids) == np.array(y_pred_ids)))
        ci_low, ci_high = Evaluator.compute_wilson_ci(correct, len(y_true_ids))

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true_ids, y_pred_ids, labels=[0, 1, 2], average=None, zero_division=0
        )
        macro_f1 = float(np.mean(f1))
        macro_prec = float(np.mean(precision))
        macro_rec = float(np.mean(recall))

        cm = confusion_matrix(y_true_ids, y_pred_ids, labels=[0, 1, 2]).tolist()

        per_class = {}
        for i, name in enumerate(CLASS_NAMES):
            per_class[name] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i])
            }

        return {
            "accuracy": acc,
            "accuracy_ci_95": [round(ci_low, 4), round(ci_high, 4)],
            "macro_f1": macro_f1,
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
            "per_class": per_class,
            "confusion_matrix": cm,
            "class_names": CLASS_NAMES,
            "total_samples": len(y_true)
        }

    @staticmethod
    def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculates 95% Wilson score interval for binomial proportions."""
        if n == 0:
            return 0.0, 0.0
        z = 1.95996  # 95% confidence
        p = k / n
        denom = 1 + (z**2) / n
        center = (p + (z**2) / (2 * n)) / denom
        delta = (z * np.sqrt((p * (1 - p) + (z**2) / (4 * n)) / n)) / denom
        return float(np.clip(center - delta, 0.0, 1.0)), float(np.clip(center + delta, 0.0, 1.0))

    @staticmethod
    def compute_latency_stats(latencies_list: List[Dict[str, float]]) -> Dict[str, Any]:
        """Calculates latency statistics across benchmark runs."""
        if not latencies_list:
            return {}

        agg_ms = [l.get("verdict_aggregator_ms", 0.0) for l in latencies_list]
        total_s = [l.get("total_pipeline_s", 0.0) for l in latencies_list]
        ext_s = [l.get("claim_extractor_s", 0.0) for l in latencies_list]
        ret_s = [l.get("evidence_retriever_s", 0.0) for l in latencies_list]
        ver_s = [l.get("cross_verifier_s", 0.0) for l in latencies_list]

        return {
            "verdict_aggregator_ms": {
                "mean": float(np.mean(agg_ms)),
                "p50": float(np.percentile(agg_ms, 50)),
                "p95": float(np.percentile(agg_ms, 95)),
                "min": float(np.min(agg_ms)),
                "max": float(np.max(agg_ms))
            },
            "total_pipeline_s": {
                "mean": float(np.mean(total_s)),
                "p50": float(np.percentile(total_s, 50)),
                "p95": float(np.percentile(total_s, 95))
            },
            "stage_means_s": {
                "claim_extractor": float(np.mean(ext_s)),
                "evidence_retriever": float(np.mean(ret_s)),
                "cross_verifier": float(np.mean(ver_s)),
                "verdict_aggregator": float(np.mean(agg_ms) / 1000.0)
            }
        }
