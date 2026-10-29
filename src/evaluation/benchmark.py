"""
Benchmark Runner and Comparative Evaluation Module.

Evaluates the fact-checking system with empirical rigor and honesty:
1. In-Domain Benchmark: 30 author-curated held-out claims across SUPPORTED, REFUTED, and NOT ENOUGH INFO.
2. Out-of-Domain Political Stress Test: 6 author-curated claims across political rhetoric and discourse.
3. Real Executed Baselines & Ablations:
   - VeritasAI (Proposed: Extractor + Isolated Retrieval + Dual-Pass + MLP Head)
   - Baseline 1: Monolithic Single-Prompt LLM (Direct raw claim prompt to Llama-3.1-8B with retrieval)
   - Baseline 2: Simple Threshold Hand Rule (Ablation: refute_max >= 0.5 -> REFUTED, else support_max >= 0.5 -> SUPPORTED, else NEI)
   - Baseline 3: Trained Logistic Regression Head (Linear classifier over 10-D empirical features)

Outputs all metrics, latency profiles, and Wilson 95% confidence intervals to data/benchmark_results.json.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.config import DATA_DIR, CLASS_NAMES, LABEL_MAP, FEATURE_NAMES
from data.data_loader import load_fever_claims, load_liar_claims, load_empirical_features
from src.pipeline import FactCheckingPipeline
from src.evaluation.evaluator import Evaluator
from src.agents.base_agent import BaseAgent

BENCHMARK_OUTPUT_PATH = DATA_DIR / "benchmark_results.json"

MONOLITHIC_PROMPT = """You are a rigorous factual verification system.
Given the claim and retrieved evidence passages, determine if the claim is SUPPORTED, REFUTED, or NOT ENOUGH INFO.

Claim: "{claim}"
Evidence:
{evidence}

Respond strictly in JSON format:
{{
  "verdict": "<SUPPORTED, REFUTED, or NOT ENOUGH INFO>",
  "confidence": <float 0.0 to 1.0>,
  "rationale": "<brief explanation>"
}}"""

def run_monolithic_baseline(agent: BaseAgent, claim: str, evidence: str) -> Dict[str, Any]:
    """Evaluates a claim directly with a single LLM prompt without decomposition."""
    prompt = MONOLITHIC_PROMPT.format(claim=claim, evidence=evidence)
    t0 = time.perf_counter()
    data, lat = agent.generate_json(prompt)
    verdict = "NOT ENOUGH INFO"
    conf = 0.5
    if isinstance(data, dict):
        raw_v = str(data.get("verdict", "")).strip().upper()
        if "SUPPORT" in raw_v:
            verdict = "SUPPORTED"
        elif "REFUT" in raw_v:
            verdict = "REFUTED"
        elif "NOT ENOUGH" in raw_v or "NEI" in raw_v:
            verdict = "NOT ENOUGH INFO"
        try:
            conf = float(data.get("confidence", 0.5))
        except (ValueError, TypeError):
            conf = 0.5
    return {"verdict": verdict, "confidence": conf, "latency_s": lat}

def run_benchmarks(num_fever_samples: int = None, num_liar_samples: int = None):
    print("=" * 65)
    print("STARTING MULTI-AGENT FACT-CHECKING SYSTEM BENCHMARKS")
    print("=" * 65)

    pipeline = FactCheckingPipeline(enable_gemini_fallback=False)
    mono_agent = BaseAgent()

    # Train a real Logistic Regression baseline on the 10-D empirical training feature set
    print("[Baselines] Fitting Logistic Regression head on empirical training features...")
    X_train_raw, y_train_raw = load_empirical_features()
    lr_scaler = pipeline.scaler
    X_train_scaled = lr_scaler.transform(X_train_raw)
    lr_model = LogisticRegression(max_iter=500, random_state=42)
    lr_model.fit(X_train_scaled, y_train_raw)

    fever_claims = load_fever_claims()[:num_fever_samples] if num_fever_samples else load_fever_claims()
    liar_claims = load_liar_claims()[:num_liar_samples] if num_liar_samples else load_liar_claims()

    print(f"\n[1/4] Running Proposed Pipeline on FEVER Benchmark ({len(fever_claims)} claims)...")
    fever_y_true = []
    fever_y_pred = []
    fever_latencies = []
    fever_details = []

    # Lists for real baseline predictions on the exact same claims
    mono_preds = []
    mono_latencies = []
    rule_preds = []
    rule_latencies = []
    lr_preds = []
    lr_latencies = []

    for idx, item in enumerate(fever_claims, 1):
        claim_text = item["claim"]
        true_label = item["label"]
        print(f"  [{idx}/{len(fever_claims)}] {claim_text[:55]}... [GT: {true_label}]")

        # 1. Proposed VeritasAI
        res = pipeline.verify_claim(claim_text)
        fever_y_true.append(true_label)
        fever_y_pred.append(res["verdict"])
        fever_latencies.append(res["latencies"])
        fever_details.append({
            "id": item["id"],
            "claim": claim_text,
            "true_label": true_label,
            "predicted_label": res["verdict"],
            "confidence": res["confidence"],
            "latencies": res["latencies"],
            "sub_claims": res["sub_claims"]
        })

        # 2. Baseline 1: Monolithic Single-Prompt LLM (No decomposition)
        claim_ret = pipeline.retriever.retrieve([claim_text])
        ev_concat = claim_ret[0]["concatenated_evidence"] if claim_ret else "No evidence."
        mono_res = run_monolithic_baseline(mono_agent, claim_text, ev_concat)
        mono_preds.append(mono_res["verdict"])
        mono_latencies.append(mono_res["latency_s"])

        # 3. Baseline 2: Simple Threshold Hand Rule on the dual-pass stances
        # refute_max >= 0.5 -> REFUTED; support_max >= 0.5 -> SUPPORTED; else NEI
        feat_dict = res["feature_vector"]
        r_max = feat_dict.get("refute_max", 0.0)
        s_max = feat_dict.get("support_max", 0.0)
        t_rule_0 = time.perf_counter()
        if r_max >= 0.50:
            rule_pred = "REFUTED"
        elif s_max >= 0.50:
            rule_pred = "SUPPORTED"
        else:
            rule_pred = "NOT ENOUGH INFO"
        rule_lat_ms = (time.perf_counter() - t_rule_0) * 1000
        rule_preds.append(rule_pred)
        rule_latencies.append(rule_lat_ms)

        # 4. Baseline 3: Logistic Regression Head on the extracted 10-D feature vector
        raw_feat_vec = np.array([[feat_dict.get(k, 0.0) for k in FEATURE_NAMES]])
        t_lr_0 = time.perf_counter()
        if raw_feat_vec.shape[1] == 10:
            scaled_feat_vec = lr_scaler.transform(raw_feat_vec)
            lr_class_idx = int(lr_model.predict(scaled_feat_vec)[0])
            lr_pred = CLASS_NAMES[lr_class_idx]
        else:
            lr_pred = "NOT ENOUGH INFO"
        lr_lat_ms = (time.perf_counter() - t_lr_0) * 1000
        lr_preds.append(lr_pred)
        lr_latencies.append(lr_lat_ms)

    fever_metrics = Evaluator.evaluate_predictions(fever_y_true, fever_y_pred)
    fever_latency_stats = Evaluator.compute_latency_stats(fever_latencies)

    # Evaluate all baselines
    mono_metrics = Evaluator.evaluate_predictions(fever_y_true, mono_preds)
    rule_metrics = Evaluator.evaluate_predictions(fever_y_true, rule_preds)
    lr_metrics = Evaluator.evaluate_predictions(fever_y_true, lr_preds)

    print("\n" + "="*50)
    print(f"Proposed Pipeline Accuracy: {fever_metrics['accuracy']:.4f} (95% CI: {fever_metrics['accuracy_ci_95']}) | Macro F1: {fever_metrics['macro_f1']:.4f}")
    print(f"Monolithic LLM Baseline Accuracy: {mono_metrics['accuracy']:.4f} | Macro F1: {mono_metrics['macro_f1']:.4f}")
    print(f"Simple Threshold Rule Accuracy:   {rule_metrics['accuracy']:.4f} | Macro F1: {rule_metrics['macro_f1']:.4f}")
    print(f"Logistic Regression Head Accuracy: {lr_metrics['accuracy']:.4f} | Macro F1: {lr_metrics['macro_f1']:.4f}")
    print(f"Average Aggregator Latency: {fever_latency_stats['verdict_aggregator_ms']['mean']:.2f} ms")
    print("="*50)

    print(f"\n[2/4] Evaluating Cross-Domain Robustness on Out-of-Domain Political Test ({len(liar_claims)} claims)...")
    liar_y_true = []
    liar_y_pred = []
    liar_latencies = []
    liar_details = []

    for idx, item in enumerate(liar_claims, 1):
        claim_text = item["claim"]
        true_label = item["label"]
        print(f"  [{idx}/{len(liar_claims)}] Political: {claim_text[:50]}... [GT: {true_label}]")

        res = pipeline.verify_claim(claim_text)
        liar_y_true.append(true_label)
        liar_y_pred.append(res["verdict"])
        liar_latencies.append(res["latencies"])
        liar_details.append({
            "id": item["id"],
            "claim": claim_text,
            "true_label": true_label,
            "predicted_label": res["verdict"],
            "confidence": res["confidence"],
            "speaker": item.get("speaker", "unknown")
        })

    liar_metrics = Evaluator.evaluate_predictions(liar_y_true, liar_y_pred)
    print(f"  Cross-Domain Accuracy: {liar_metrics['accuracy']:.4f} | Macro F1: {liar_metrics['macro_f1']:.4f}")

    print("\n[3/4] Compiling Comparative Empirical Benchmark Table...")
    baselines = [
        {
            "model_architecture": "VeritasAI (Proposed Pipeline)",
            "novelty": "Stateless Dual-Pass + 227-parameter Trained MLP Head",
            "fever_accuracy": round(fever_metrics["accuracy"], 3),
            "fever_accuracy_ci_95": fever_metrics["accuracy_ci_95"],
            "fever_macro_f1": round(fever_metrics["macro_f1"], 3),
            "avg_aggregator_latency_ms": round(fever_latency_stats["verdict_aggregator_ms"]["mean"], 2),
            "total_pipeline_latency_s": round(fever_latency_stats["total_pipeline_s"]["mean"], 2),
            "vram_allocation": "~4.7 GB (4-bit GGUF backbone via Ollama)",
            "hallucination_cascade_risk": "Low (Stateless independent passes)",
            "trainable_parameters": 227,
            "evaluation_mode": f"Empirically measured (n={len(fever_claims)})"
        },
        {
            "model_architecture": "Ablation: Simple Threshold Rule (No MLP)",
            "novelty": "Dual-Pass stances with hard threshold (refute>=0.5 -> REFUTED, else support>=0.5 -> SUPPORTED)",
            "fever_accuracy": round(rule_metrics["accuracy"], 3),
            "fever_accuracy_ci_95": rule_metrics["accuracy_ci_95"],
            "fever_macro_f1": round(rule_metrics["macro_f1"], 3),
            "avg_aggregator_latency_ms": round(float(np.mean(rule_latencies)), 3),
            "total_pipeline_latency_s": round(fever_latency_stats["total_pipeline_s"]["mean"], 2),
            "vram_allocation": "~4.7 GB (Ollama)",
            "hallucination_cascade_risk": "Low (Stateless passes)",
            "trainable_parameters": 0,
            "evaluation_mode": f"Empirically measured (n={len(fever_claims)})"
        },
        {
            "model_architecture": "Baseline: Multi-Agent + Logistic Regression Head",
            "novelty": "Linear classification head over standardized 10-D features",
            "fever_accuracy": round(lr_metrics["accuracy"], 3),
            "fever_accuracy_ci_95": lr_metrics["accuracy_ci_95"],
            "fever_macro_f1": round(lr_metrics["macro_f1"], 3),
            "avg_aggregator_latency_ms": round(float(np.mean(lr_latencies)), 2),
            "total_pipeline_latency_s": round(fever_latency_stats["total_pipeline_s"]["mean"], 2),
            "vram_allocation": "~4.7 GB (Ollama)",
            "hallucination_cascade_risk": "Low (Stateless passes)",
            "trainable_parameters": 33,
            "evaluation_mode": f"Empirically measured (n={len(fever_claims)})"
        },
        {
            "model_architecture": "Baseline: Monolithic Single-Prompt LLM (with Retrieval)",
            "novelty": "Single-shot prompt to Llama-3.1-8B with top retrieved passage; no sub-claim decomposition",
            "fever_accuracy": round(mono_metrics["accuracy"], 3),
            "fever_accuracy_ci_95": mono_metrics["accuracy_ci_95"],
            "fever_macro_f1": round(mono_metrics["macro_f1"], 3),
            "avg_aggregator_latency_ms": 0.0,
            "total_pipeline_latency_s": round(float(np.mean(mono_latencies)), 2),
            "vram_allocation": "~4.7 GB (Ollama)",
            "hallucination_cascade_risk": "Low on retrieved facts, but operates as an opaque black box with zero auditable sub-claim attribution",
            "trainable_parameters": 0,
            "evaluation_mode": f"Empirically measured (n={len(fever_claims)})"
        }
    ]

    benchmark_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware_environment": {
            "gpu": "NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM)",
            "llm_backbone": "Llama-3.1-8B-Instruct (4-bit GGUF via Ollama, ~4.7GB VRAM)",
            "embedder": "all-MiniLM-L6-v2 (CPU-side)",
            "retriever": "FAISS Flat Inner Product (CPU-side)",
            "verdict_aggregator": "PyTorch 2-layer MLP (10->16->3, 227 trainable parameters)"
        },
        "fever_benchmark": {
            "dataset_description": f"Curated FEVER benchmark subset ({len(fever_claims)} claims: balanced across Supported, Refuted, and NEI)",
            "metrics": fever_metrics,
            "latency_stats": fever_latency_stats,
            "sample_details": fever_details
        },
        "liar_benchmark": {
            "dataset_description": f"Curated out-of-domain political statements stress test ({len(liar_claims)} claims)",
            "metrics": liar_metrics,
            "sample_details": liar_details
        },
        "comparative_baselines": baselines
    }

    BENCHMARK_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BENCHMARK_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)

    print(f"\n[Done] Benchmark results successfully saved to {BENCHMARK_OUTPUT_PATH}")
    return benchmark_summary

if __name__ == "__main__":
    run_benchmarks(num_fever_samples=None, num_liar_samples=None)
