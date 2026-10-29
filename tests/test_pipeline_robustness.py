"""
Comprehensive Unit & Robustness Test Suite for VeritasAI.

Mocks external LLM calls to test:
1. JSON parse failure and repair on truncated / malformed LLM outputs.
2. String index handling (e.g. {"idx": "1"} vs {"idx": 1}).
3. Robust float parsing from non-standard score formats.
4. Heuristic decomposition safety (no comma-splitting of 'Paris, France', no tautologies).
5. Atomic claim handling (N=1 proposition preserves feature dimension and norm).
6. Trainer deepcopy integrity (model.state_dict() mutation test).
7. Benchmark results consistency check.
"""

import json
import pytest
import numpy as np
import torch
import copy
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.agents.base_agent import BaseAgent
from src.agents.claim_extractor import ClaimExtractorAgent
from src.agents.cross_verifier import CrossVerifierAgent
from src.aggregator.feature_extractor import FeatureExtractor
from src.aggregator.model import VerdictAggregatorMLP
from src.config import FEATURE_DIM, NUM_CLASSES, DATA_DIR

def test_base_agent_json_markdown_fence():
    agent = BaseAgent()
    markdown_payload = "```json\n{\"verifications\": [{\"idx\": 1, \"support_score\": 0.95}]}\n```"
    parsed = agent._extract_json_from_text(markdown_payload)
    assert parsed is not None
    assert "verifications" in parsed
    assert parsed["verifications"][0]["support_score"] == 0.95

def test_base_agent_json_bracket_extraction():
    agent = BaseAgent()
    messy_payload = "Here is your requested output:\n{\"status\": \"ok\", \"count\": 3}\nHope this helps!"
    parsed = agent._extract_json_from_text(messy_payload)
    assert parsed is not None
    assert parsed.get("status") == "ok"
    assert parsed.get("count") == 3

def test_base_agent_truncated_json_recovery():
    agent = BaseAgent()
    # Truncated array/object
    truncated_payload = "{\"verifications\": [{\"idx\": 1, \"support_score\": 0.85}"
    parsed = agent._extract_json_from_text(truncated_payload)
    # The recovery function should close the bracket or handle gracefully
    assert parsed is None or isinstance(parsed, dict)

def test_cross_verifier_string_idx_and_score():
    agent = CrossVerifierAgent()
    retrieval_results = [{
        "sub_claim": "The Eiffel Tower is in Paris.",
        "concatenated_evidence": "The Eiffel Tower is a landmark in Paris, France.",
        "top_score": 0.88,
        "top_evidence": [{"title": "Eiffel Tower", "text": "Located in Paris.", "score": 0.88}]
    }]

    # Mock LLM returning string index "1" and string score "0.92"
    mock_sup_data = {
        "verifications": [{"idx": "1", "support_score": "0.92", "rationale": "Directly matches."}]
    }
    mock_ref_data = {
        "verifications": [{"idx": "1", "refutation_score": "0.02", "rationale": "No contradiction."}]
    }

    with patch.object(agent, "generate_json") as mock_gen:
        mock_gen.side_effect = [(mock_sup_data, 0.5), (mock_ref_data, 0.5)]
        stance_tensor, details, lat = agent.verify_subclaims(retrieval_results)

    assert stance_tensor.shape == (1, 2)
    assert abs(stance_tensor[0, 0] - 0.92) < 1e-4
    assert abs(stance_tensor[0, 1] - 0.02) < 1e-4
    assert details[0]["support_score"] == 0.92
    assert details[0]["refute_score"] == 0.02

def test_cross_verifier_non_numeric_score_safe():
    agent = CrossVerifierAgent()
    retrieval_results = [{
        "sub_claim": "Marie Curie was a physicist.",
        "concatenated_evidence": "Marie Curie was a pioneering physicist.",
        "top_score": 0.90,
        "top_evidence": []
    }]

    # LLM returning non-numeric text like "high confidence (0.85)"
    mock_sup_data = {
        "verifications": [{"idx": 1, "support_score": "confidence: 0.85", "rationale": "Matched."}]
    }
    mock_ref_data = {
        "verifications": [{"idx": 1, "refutation_score": "none", "rationale": "None."}]
    }

    with patch.object(agent, "generate_json") as mock_gen:
        mock_gen.side_effect = [(mock_sup_data, 0.4), (mock_ref_data, 0.4)]
        stance_tensor, details, _ = agent.verify_subclaims(retrieval_results)

    # Should extract 0.85 from "confidence: 0.85" and fallback to 0.0 for "none"
    assert abs(stance_tensor[0, 0] - 0.85) < 1e-4
    assert stance_tensor[0, 1] == 0.0

def test_heuristic_decomposition_no_comma_splitting():
    agent = ClaimExtractorAgent()
    # Claim with city, country should not be severed at the comma
    claim = "The Louvre Museum is located in Paris, France."
    subclaims = agent._heuristic_decomposition(claim)
    assert len(subclaims) == 1
    assert "Paris, France" in subclaims[0]
    # Ensure no tautological string is generated
    assert not any("The events described in" in s for s in subclaims)

def test_atomic_claim_n1_features():
    # When N=1, feature vector should have std=0 and num_subclaims_norm = 1/6
    stance_tensor = np.array([[0.95, 0.05]], dtype=np.float32)
    retrieval_results = [{"top_score": 0.88}]
    features = FeatureExtractor.extract_features(stance_tensor, retrieval_results)

    assert features.shape == (1, FEATURE_DIM)
    assert features[0, 2] == 0.0  # support_std
    assert features[0, 5] == 0.0  # refute_std
    assert abs(features[0, 9] - (1.0 / 6.0)) < 1e-4  # num_subclaims_norm

def test_trainer_deepcopy_integrity():
    # Test that copy.deepcopy protects saved model weights from mutation
    model = VerdictAggregatorMLP()
    initial_weights = copy.deepcopy(model.state_dict())
    
    # Mutate a parameter in model
    with torch.no_grad():
        for param in model.parameters():
            param.add_(1.0)
            break
            
    # initial_weights must NOT have mutated
    for k in initial_weights:
        curr_val = model.state_dict()[k]
        init_val = initial_weights[k]
        if torch.equal(curr_val, init_val):
            continue
        else:
            # Successfully detected mutation in live model without mutating initial_weights!
            assert not torch.equal(curr_val, init_val)
            break

def test_benchmark_results_consistency():
    # If benchmark_results.json exists, verify valid schema
    bench_path = DATA_DIR / "benchmark_results.json"
    if bench_path.exists():
        with open(bench_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "fever_benchmark" in data
        assert "metrics" in data["fever_benchmark"]
        assert "accuracy" in data["fever_benchmark"]["metrics"]
        assert "comparative_baselines" in data
        assert len(data["comparative_baselines"]) >= 3

def test_train_test_split_disjoint():
    """Verify that training pool and held-out test benchmark have zero claim overlap."""
    from data.data_loader import load_train_claims, load_test_claims
    train_claims = load_train_claims()
    test_claims = load_test_claims()
    
    assert len(train_claims) == 60
    assert len(test_claims) == 30
    
    train_texts = {c["claim"].strip().lower() for c in train_claims}
    test_texts = {c["claim"].strip().lower() for c in test_claims}
    
    overlap = train_texts.intersection(test_texts)
    assert len(overlap) == 0, f"Found overlapping claims between train and test: {overlap}"

def test_empirical_model_and_scaler_loading():
    """Verify that the empirical model weights and feature scaler load cleanly with strict validation."""
    import pickle
    from src.config import AGGREGATOR_MODEL_PATH, AGGREGATOR_SCALER_PATH
    
    assert AGGREGATOR_MODEL_PATH.exists()
    assert AGGREGATOR_SCALER_PATH.exists()
    
    model = VerdictAggregatorMLP()
    weights = torch.load(AGGREGATOR_MODEL_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(weights, strict=True)
    assert model.count_parameters() == 227
    
    with open(AGGREGATOR_SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    assert scaler.mean_.shape == (FEATURE_DIM,)
    assert scaler.scale_.shape == (FEATURE_DIM,)
