"""
Empirical Feature Extraction & Caching Module.

Runs the real multi-agent pipeline over authentic benchmark claims,
extracting genuine 10-D feature vectors produced by the Claim Extractor,
Evidence Retriever, and Cross-Verifier without synthetic simulation.
Saves empirical feature representations into data/empirical_features.npz.
"""

from pathlib import Path
import json
import time
from typing import Tuple
import numpy as np

from src.config import DATA_DIR, LABEL_MAP
from data.data_loader import load_train_claims
from src.pipeline import FactCheckingPipeline
from src.aggregator.feature_extractor import FeatureExtractor

def extract_and_cache_empirical_features(output_path: Path = DATA_DIR / "empirical_features.npz") -> Tuple[np.ndarray, np.ndarray]:
    print("=" * 65)
    print("EXTRACTING EMPIRICAL FEATURE REPRESENTATIONS FROM TRAINING POOL")
    print("=" * 65)

    pipeline = FactCheckingPipeline(enable_gemini_fallback=False)
    all_claims = load_train_claims()

    X_list = []
    y_list = []
    metadata = []

    print(f"Extracting empirical features for {len(all_claims)} real claims across FEVER and Political datasets...")
    for idx, item in enumerate(all_claims, 1):
        claim_text = item["claim"]
        true_label = item["label"]
        label_idx = LABEL_MAP.get(true_label, 2)
        print(f"  [{idx}/{len(all_claims)}] ({true_label}) {claim_text[:50]}...", flush=True)

        t0 = time.perf_counter()
        # 1. Atomic decomposition
        sub_claims, _ = pipeline.claim_extractor.extract_subclaims(claim_text)
        # 2. Hybrid dense + sparse retrieval
        retrieval_results = pipeline.retriever.retrieve(sub_claims)
        # 3. Stateless dual-pass cross-verification
        stance_tensor, details, _ = pipeline.cross_verifier.verify_subclaims(retrieval_results)
        # 4. Feature pooling
        feat_vec = FeatureExtractor.extract_features(stance_tensor, retrieval_results) # shape [1, 10]
        lat = time.perf_counter() - t0

        X_list.append(feat_vec[0])
        y_list.append(label_idx)
        metadata.append({
            "id": item.get("id", f"claim_{idx}"),
            "claim": claim_text,
            "true_label": true_label,
            "label_idx": label_idx,
            "sub_claims": sub_claims,
            "feature_vector": feat_vec[0].tolist(),
            "latency_s": round(lat, 3)
        })

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, X=X, y=y)

    meta_path = output_path.with_suffix(".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_samples": len(X),
            "class_distribution": {
                "SUPPORTED": int(np.sum(y == 0)),
                "REFUTED": int(np.sum(y == 1)),
                "NOT_ENOUGH_INFO": int(np.sum(y == 2))
            },
            "samples": metadata
        }, f, indent=2)

    print("\n" + "=" * 50)
    print(f"[Done] Saved {len(X)} empirical feature vectors to {output_path}")
    print(f"  Class breakdown: SUPPORTED={np.sum(y == 0)}, REFUTED={np.sum(y == 1)}, NEI={np.sum(y == 2)}")
    print(f"  Metadata saved to {meta_path}")
    print("=" * 50)
    return X, y

if __name__ == "__main__":
    extract_and_cache_empirical_features()

