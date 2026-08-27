\# Multi-Agent LLM Framework for Fact-Checking and Misinformation Detection



\## Overview

This project introduces a multi-agent system designed to evaluate whether a given claim is supported by evidence, refuted by evidence, or if there is not enough information to decide\[cite: 1]. Developed as a Computer Science Engineering project at Vellore Institute of Technology, the system is optimized to run entirely on a single consumer GPU (RTX 4060, 8GB VRAM)\[cite: 1]. 



Unlike traditional multi-agent debate architectures that rely on expensive LLM-based judges to resolve disagreements, this framework replaces the final LLM call with a lightweight trained classifier (Verdict Aggregator), significantly reducing latency and computational overhead\[cite: 1].



\## System Architecture

The pipeline consists of four cooperating stages:

\* \*\*Claim Extractor Agent:\*\* Decomposes complex input claims into 2 to 6 atomic, independently checkable sub-claims\[cite: 1].

\* \*\*Evidence Retriever Agent:\*\* Embeds sub-claims into vectors and searches a FAISS index of Wikipedia-derived evidence sentences\[cite: 1]. This process runs on the CPU to preserve GPU memory\[cite: 1].

\* \*\*Cross-Verifier Agent:\*\* Uses a 4-bit quantized Llama-3.1-8B-Instruct model (via Ollama/llama.cpp) to evaluate each sub-claim twice: once arguing for support, and once arguing for refutation\[cite: 1]. 

\* \*\*Verdict Aggregator:\*\* A trained classifier (logistic regression or 2-layer neural network) that pools numeric support/refute scores into a final verdict (SUPPORTED, REFUTED, or NOT ENOUGH INFO)\[cite: 1].



\## Datasets

The experimental plan utilizes two primary datasets:

\* \*\*FEVER (Fact Extraction and VERification):\*\* The primary benchmark dataset, consisting of claims generated from Wikipedia sentences\[cite: 1]. A subsample is used for training and primary evaluation\[cite: 1].

\* \*\*LIAR:\*\* A dataset of short political statements used as a zero-shot, out-of-domain stress test to evaluate the system's cross-domain robustness\[cite: 1].



\## Setup and Installation

1\. Clone this repository:

&#x20;  ```bash

&#x20;  git clone \[https://github.com/YOUR\_USERNAME/multi-agent-fact-checking-for-misinformation.git](https://github.com/YOUR\_USERNAME/multi-agent-fact-checking-for-misinformation.git)

&#x20;  cd multi-agent-fact-checking-for-misinformation

