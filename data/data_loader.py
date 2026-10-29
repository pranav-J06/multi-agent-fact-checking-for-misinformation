"""
Data loader module for the fact-checking pipeline.
Handles loading Wikipedia corpus, FEVER claims, LIAR claims, and feature tensors.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

from src.config import CORPUS_FILE_PATH, FEVER_DIR, LIAR_DIR, DATA_DIR

def load_wiki_corpus(corpus_path: Path = CORPUS_FILE_PATH) -> List[Dict[str, Any]]:
    """Loads Wikipedia evidence sentences corpus."""
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found at {corpus_path}")
    with open(corpus_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_fever_claims() -> List[Dict[str, Any]]:
    """Loads curated FEVER held-out benchmark claims (30 claims)."""
    fever_path = FEVER_DIR / "test_claims.json"
    if not fever_path.exists():
        fever_path = FEVER_DIR / "fever_claims.json"
    if not fever_path.exists():
        raise FileNotFoundError(f"FEVER claims file not found at {fever_path}")
    with open(fever_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_train_claims() -> List[Dict[str, Any]]:
    """Loads empirical training pool claims (60 claims)."""
    train_path = FEVER_DIR / "train_claims.json"
    if not train_path.exists():
        from data.setup_clean_eval import setup_evaluation_files
        setup_evaluation_files()
    with open(train_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_test_claims() -> List[Dict[str, Any]]:
    """Loads held-out evaluation benchmark claims (30 claims)."""
    return load_fever_claims()

def load_liar_claims() -> List[Dict[str, Any]]:
    """Loads curated LIAR stress test claims."""
    liar_path = LIAR_DIR / "liar_claims.json"
    if not liar_path.exists():
        raise FileNotFoundError(f"LIAR claims file not found at {liar_path}")
    with open(liar_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_training_features(feature_file: Path = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads 10-D feature vectors and class labels for training the Verdict Aggregator head.
    Defaults to simulated_features_6k.npz, or loads from empirical_features.npz when specified.
    Returns:
        X: shape [N, 10]
        y: shape [N] (classes 0, 1, 2)
    """
    target_file = feature_file or (DATA_DIR / "simulated_features_6k.npz")
    if not target_file.exists():
        if feature_file and "empirical" in str(feature_file):
            from data.cache_empirical_features import extract_and_cache_empirical_features
            return extract_and_cache_empirical_features(output_path=feature_file)
        legacy_file = DATA_DIR / "fever_features_5k.npz"
        if legacy_file.exists():
            target_file = legacy_file
        else:
            from data.generate_training_data import generate_feature_dataset
            generate_feature_dataset(num_samples=6000, save_path=target_file)
    data = np.load(target_file)
    return data["X"], data["y"]

def load_empirical_features(feature_file: Path = DATA_DIR / "empirical_features.npz") -> Tuple[np.ndarray, np.ndarray]:
    """Loads genuine empirical 10-D feature vectors extracted from real claim executions."""
    if not feature_file.exists():
        from data.cache_empirical_features import extract_and_cache_empirical_features
        return extract_and_cache_empirical_features(output_path=feature_file)
    data = np.load(feature_file)
    return data["X"], data["y"]
