# VeritasAI: Multi-Agent LLM Framework for Fact-Checking & Misinformation Detection

**Authors:** Pranav Jahagirdar [Reg No. 24BCE1503] &bull; Umang Raj [Reg No. 24BCE1470]  
**Institution:** Department of Computer Science and Engineering, VIT Chennai  
**Course:** BCSE409L &mdash; Natural Language Processing | Digital Assignment 1 (DA1)  
**Hardware Budget:** Single Consumer GPU (NVIDIA GeForce RTX 4060, 8GB VRAM)  

---

## 1. Project Overview

**VeritasAI** is a multi-agent fact-checking and misinformation detection system engineered to resolve the fundamental limitations of single-model classifiers and conversational debate frameworks. 

Based on our comprehensive literature review (*"Multi-Agent LLM Frameworks for Fact-Checking and Misinformation Detection"*, DA1), prior architectures suffer from four major bottlenecks: high inference latency (10–30s per claim due to LLM judges), fragile single-pass search queries, communicative hallucination cascades in multi-turn dialogues, and heavy multi-GPU resource demands.

VeritasAI addresses all four challenges on a **single consumer GPU (RTX 4060, 8GB VRAM)** by introducing:
1. **Stateless Dual-Pass Verification:** Independent *Support* and *Refute* passes over a shared, quantized **Llama-3.1-8B-Instruct** backbone with zero shared conversation history, eliminating cognitive shortcuts and hallucination propagation.
2. **Sub-Claim Level Dense Retrieval:** Decomposes complex compound statements into 1–6 atomic sub-claims, executing CPU-side **FAISS** vector search (`all-MiniLM-L6-v2`) and preserving retrieval confidence scores.
3. **Novel Trained Verdict Aggregator Head:** Replaces the traditional GPU-bound LLM "Judge" (as used in MAD and FACT-AUDIT) with a lightweight, trained **2-layer MLP classification head (227 trainable parameters: 208 weights + 19 biases)** operating on a 10-dimensional pooled feature vector. This slashes final decision latency from **10–30 seconds down to <0.5 milliseconds** (<50ms hard constraint).
4. **Honest Architecture Disclosures:** The shipped MLP head is trained directly on genuine 10-D empirical feature representations produced by the multi-agent pipeline across real training claims, completely replacing simulated weights. An optional Tier-2 Gemini fallback is available for online out-of-corpus queries, but is kept strictly disabled during baseline benchmark evaluations to report authentic, self-contained offline performance. Evaluation is conducted on an author-curated controlled test benchmark (n=30 held-out claims, strictly partitioned from the 60-claim training pool), alongside an exploratory 6-claim political stress test.

---

## 2. System Architecture & Tensor Flow

The pipeline maps directly to the six canonical NLP stages and tensor transformations:

```
+---------------------------------------------------------------------------------------------------+
|                                       VERITAS-AI PIPELINE                                         |
+---------------------------------------------------------------------------------------------------+
  [Raw Claim] -> [1, L] (L <= 128)
       |
       v
+-------------------------------+
|  1. Claim Extractor Agent     |  Llama-3.1-8B-Instruct (4-bit GGUF via Ollama)
|     (Atomic Decomposition)    |  Output: N atomic sub-claims -> [N, L_sub] (N in [1, 6])
+-------------------------------+
       |
       v
+-------------------------------+
|  2. Evidence Retriever Agent  |  all-MiniLM-L6-v2 Embedder (384-D) + FAISS IndexFlatIP (CPU-side)
|     (Dense Vector Search)     |  Output: Top-k evidence sentences + similarity scores -> [N, k=5, L_evid]
+-------------------------------+
       |
       v
+-------------------------------+
|  3. Cross-Verifier Agent      |  Llama-3.1-8B-Instruct (Reused sequentially on GPU)
|     (Stateless Dual-Pass)     |  Pass A: 'Support' Stance -> s_i in [0, 1]
|                               |  Pass B: 'Refute' Stance  -> r_i in [0, 1]
|                               |  Output: Stance Tensor -> [N, 2]
+-------------------------------+
       |
       v
+-------------------------------+
|  4. Feature Pooling Engine    |  Aggregates across N sub-claims:
|     (10-D Feature Vector)     |  [sup_mean, sup_max, sup_std, ref_mean, ref_max, ref_std,
|                               |   agreement_ratio, ret_mean, ret_min, num_subclaims_norm]
|                               |  Output: Feature Vector -> [1, F=10]
+-------------------------------+
       |
       v
+-------------------------------+  ★ PROPOSED NOVELTY HEAD
|  5. Verdict Aggregator Head   |  Trained PyTorch MLP: Dense(10, 16, ReLU) -> Dense(16, 3, Softmax)
|     (Trained Classification)  |  Trainable Parameters: 227 (160+48=208 weights, 16+3=19 biases)
|                               |  Output: Class Probabilities -> [1, 3] {SUPPORTED, REFUTED, NEI}
|                               |  Latency: <50ms (measured ~0.34 ms)
+-------------------------------+
       |
       v
+-------------------------------+
|  6. Output / Explanation      |  Verdict Label + Confidence Score + Cited Evidence Sentences +
|     (Structured Explanations) |  Dual-Pass Rationales + Stage Latency Trace
+-------------------------------+
```

---

## 3. How the 4 Critical Research Gaps are Resolved

| Research Gap Identified in Review | Industry / Literature Failure Mode | VeritasAI Architectural Resolution |
| :--- | :--- | :--- |
| **1. Inference Latency & Overhead** | Debate frameworks (MAD, FACT-AUDIT) use an extra LLM call for the Judge, adding 10–30s latency per claim. | Substituted with a small trained **2-layer MLP head** running on CPU/GPU in **~0.34 ms**, cutting final aggregation time by >99.9%. |
| **2. Fragile Query Generation** | FacTool and HiSS issue single-pass queries for entire compound claims, missing fine-grained evidence. | Decomposes claims into **1–6 atomic sub-claims**, queries FAISS per sub-claim, and carries retrieval confidence forward as an engineered feature. |
| **3. Communicative Hallucinations** | Multi-turn debate agents observe each other's history, causing premature consensus and hallucination cascades. | The Cross-Verifier runs **two independent, stateless passes** (Pass A Support, Pass B Refute) with zero shared history, combined statistically. |
| **4. GPU Hardware Constraints** | Prior frameworks require multiple concurrent 70B models requiring multi-GPU server clusters (>40GB VRAM). | **Sequential backbone reuse** of quantized Llama-3.1-8B (4-bit GGUF) and CPU-side FAISS indexing, fitting comfortably in **<5.0GB VRAM on an RTX 4060**. |

---

## 4. Benchmark Evaluation & Baseline Comparison

All metrics below are generated directly from real executions and recorded in `data/benchmark_results.json`:

### Comparative Performance Table
| Framework / Model | Paradigm | FEVER Acc (95% CI) | Macro F1 | Aggregator Latency | Total Pipeline Latency | Hallucination Cascade Risk | Parameters | Evaluation Mode |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **VeritasAI (Proposed)** | **Dual-Pass + 227-param MLP Head** | **86.7%** [70.3%, 94.7%] | **0.864** | **0.34 ms** | **7.48 s** | **Low (Stateless passes)** | **227** | **Author-Curated Held-Out (n=30)** |
| Ablation: Simple Threshold Rule | Dual-Pass Stances + Hand Rule (No MLP) | 80.0% [62.7%, 90.5%] | 0.800 | <0.01 ms | 7.48 s | Low (Stateless passes) | 0 | Author-Curated Held-Out (n=30) |
| Baseline: Logistic Regression Head | Multi-Agent + Linear 10-D Classifier | 83.3% [66.4%, 92.7%] | 0.833 | 0.62 ms | 7.48 s | Low (Stateless passes) | 33 | Author-Curated Held-Out (n=30) |
| Baseline: Monolithic Single-Prompt LLM | Direct Prompt with Retrieval (No Decomposition) | 96.7% [83.3%, 99.4%] | 0.967 | 0.0 ms | 1.60 s | Low on retrieved text; opaque black box without attribution | 0 | Author-Curated Held-Out (n=30) |

### Transparent Benchmark & Scope Disclosures
* **Author-Curated Evaluation Subsets (Not Full Benchmark Datasets):** The evaluation sets (`data/fever/test_claims.json`, n=30, and `data/liar/political_test_claims.json`, n=6) are author-curated controlled evaluation subsets designed around the FEVER 3-class schema (SUPPORTED, REFUTED, NOT ENOUGH INFO) and political rhetoric style. They are NOT the official full FEVER (185k claims) or LIAR (12.8k claims) benchmarks, which would require multi-day compute budgets on local 8B models.
* **Reference Corpus & Topic Overlap Limitation:** The local reference corpus contains 91 curated Wikipedia evidence passages covering 78 distinct article topics (`data/corpus/wiki_corpus.json`). 28 of the 30 test claims share topics with articles in this reference corpus (e.g. Apollo program, Nobel laureates, World Wars, European Union). Consequently, the benchmark measures atomic decomposition, cross-stance reasoning, and adjudication accuracy over known topical domains rather than open-web retrieval recall.
* **Accurate Nature of NOT ENOUGH INFO (NEI):** The 10 NEI test claims represent plausible assertions with non-entailed biographical or historical details (e.g., uncorroborated musical hobbies, alternative draft project names, comparative superlatives) where reference evidence provides insufficient grounding to either verify or refute. They are not necessarily false in the real world, but lack substantiation in the available corpus.
* **Statistical Interpretation of MLP Head vs. Baselines:** On the n=30 test set, the 227-parameter MLP head scores 86.7% (26/30 correct) vs. 80.0% (24/30 correct) for the simple threshold rule and 83.3% (25/30 correct) for logistic regression. With n=30, the 95% Wilson confidence intervals overlap (MLP: `[70.3%, 94.7%]`, Rule: `[62.7%, 90.5%]`, LR: `[66.4%, 92.7%]`), meaning this performance difference does not establish definitive statistical superiority. The primary value of the MLP head is providing continuous, calibrated class posteriors and sub-millisecond execution (0.34 ms) without hand-crafted threshold rules or conversational LLM judge overhead.
* **Explainability vs. Monolithic Accuracy:** While the monolithic single-prompt baseline achieves higher raw classification accuracy (96.7%), it functions as an opaque black box with zero auditable sub-claim attribution, intermediate provenance, or falsifiable evidence linking. VeritasAI trades monolithic accuracy for granular explainability and sub-millisecond adjudication.
* **Exploratory Political Stress Test:** On the 6-claim author-curated political stress test, VeritasAI scores 83.3% (5/6 correct, 0.822 Macro-F1). High stance variance (σ > 0.35) and low agreement ratio (< 0.1) act as an intrinsic uncertainty detector on complex, out-of-domain political rhetoric.

---

## 5. Directory Structure

```
NLP_Proj/
├── data/
│   ├── corpus/                # 91 curated Wikipedia evidence documents (FAISS + Okapi BM25 indexed)
│   │   ├── wiki_corpus.json
│   │   └── wiki_article_cache.json # Persistent Wikipedia extract cache
│   ├── fever/                 # Curated claim benchmarks (strictly partitioned train & held-out test)
│   │   ├── train_claims.json  # 60 training pool claims (20 SUP, 20 REF, 20 realistic NEI)
│   │   ├── test_claims.json   # 30 held-out evaluation claims (10 SUP, 10 REF, 10 NEI)
│   │   └── fever_claims.json  # Benchmark entrypoint
│   ├── liar/                  # Political statements stress test
│   │   ├── political_test_claims.json # 6 hand-written political stress-test claims
│   │   └── liar_claims.json
│   ├── benchmark_results.json # Single source of truth for all measured metrics & CIs
│   ├── simulated_features_6k.npz # 6,000 simulated 10-D feature vectors (synthetic pre-training)
│   ├── empirical_features.npz # 60 genuine 10-D feature vectors from training pool
│   ├── data_loader.py         # Unified data loader
│   ├── generate_training_data.py # Synthetic feature distribution generator
│   ├── cache_empirical_features.py # Live pipeline feature extraction script
│   └── setup_clean_eval.py    # Clean train/test setup script
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration constants, tensor dimensions, model names
│   ├── agents/
│   │   ├── base_agent.py      # Resilient Ollama client wrapper
│   │   ├── claim_extractor.py # Claim Extractor Agent (atomic decomposition)
│   │   └── cross_verifier.py  # Cross-Verifier Agent (dual-pass Support/Refute)
│   ├── retriever/
│   │   ├── embedder.py        # SentenceTransformer all-MiniLM-L6-v2 singleton
│   │   └── faiss_retriever.py # CPU-side FAISS Flat IP index retriever
│   ├── aggregator/
│   │   ├── feature_extractor.py # 10-D feature pooling engine
│   │   ├── model.py           # VerdictAggregatorMLP (Dense(10, 16) -> Dense(16, 3))
│   │   └── trainer.py         # MLP training script with Adam & CrossEntropyLoss
│   ├── evaluation/
│   │   ├── evaluator.py       # Metrics calculator (Acc, Prec, Rec, Macro F1, CM)
│   │   └── benchmark.py       # Benchmark evaluation script
│   ├── pipeline.py            # End-to-end FactCheckingPipeline orchestrator
│   └── cli.py                 # Command-line interface
├── web/
│   ├── app.py                 # FastAPI backend server
│   └── static/
│       ├── index.html         # Rich modern glassmorphic Web UI
│       ├── styles.css         # Dark theme styling, glowing badges, gauge meters
│       ├── app.js             # Interactive frontend controller & live stepper
│       └── images/            # Architecture diagrams (Fig. 1 & Fig. 2)
├── models/
│   ├── verdict_aggregator.pt  # Trained PyTorch MLP weights
│   ├── feature_scaler.pkl     # Feature standardizer
│   └── faiss_wiki_index.bin   # Cached FAISS Flat vector index
├── tests/
│   ├── test_aggregator.py     # Unit tests for 10-D feature extraction & MLP head
│   ├── test_retriever.py      # Unit tests for FAISS retrieval & cosine similarity
│   └── test_extractor.py      # Unit tests for claim decomposition & heuristic fallbacks
├── run_demo.py                # Quick 3-claim end-to-end demonstration
├── requirements.txt           # Python dependencies
└── README.md                  # System documentation
```

---

## 6. Installation & Quick Start

### 1. Prerequisites
- Python 3.10+ (Tested on Python 3.11.9)
- NVIDIA GPU with CUDA support (e.g., RTX 4060, 8GB VRAM)
- [Ollama](https://ollama.ai/) installed with the Llama 3.1 8B model:
  ```bash
  ollama pull llama3.1:latest
  ```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Demonstration
Execute the end-to-end pipeline on sample claims across all three veracity classes:
```bash
python run_demo.py
```

### 4. Interactive Command-Line Interface (CLI)
Verify any arbitrary claim from the terminal:
```bash
python -m src.cli --claim "Marie Curie was the first woman to win a Nobel Prize."
```
Or start an interactive verification session:
```bash
python -m src.cli --interactive
```

### 5. Launch the Web Studio
Start the FastAPI server:
```bash
python -m uvicorn web.app:app --host 127.0.0.1 --port 8000
```
Then open your browser at **`http://127.0.0.1:8000`** to access:
- **Live Fact-Checking Studio:** Interactive claim input, quick preset chips, real-time stage execution stepper, probability gauges, 10-D feature vector cards, and cited evidence.
- **Architecture & Tensors Tab:** High-resolution pipeline diagrams (Fig. 1 & Fig. 2) and layer-level tensor transformation specifications.
- **Evaluation & Baselines Tab:** Detailed benchmark metrics, confusion matrices, and comparative baseline analysis.
- **Research Gaps & Design Tab:** Theoretical synthesis mapping the four literature gaps to our implementation decisions.

### 6. Run Unit Tests
```bash
python -m tests.test_aggregator
python -m tests.test_retriever
python -m tests.test_extractor
```

---

## 7. Citation & Course Information
This codebase represents the primary implementation artifact for:
- **Title:** Multi-Agent LLM Frameworks for Fact-Checking and Misinformation Detection: A Comprehensive Literature Review
- **Course:** BCSE409L &mdash; Natural Language Processing | Digital Assignment 1 (DA1)
- **Institution:** School of Computer Science and Engineering, Vellore Institute of Technology (VIT), Chennai Campus.
