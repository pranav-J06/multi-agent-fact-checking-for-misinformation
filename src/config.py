"""
Configuration module for Multi-Agent Fact-Checking System.
Defines paths, model settings, hyperparameters, and feature vector constants.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
FEVER_DIR = DATA_DIR / "fever"
LIAR_DIR = DATA_DIR / "liar"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

# LLM Agent Configuration (Backbone: Llama-3.1-8B-Instruct via Ollama)
LLM_MODEL_NAME = "llama3.1:latest"
LLM_TEMPERATURE = 0.1
LLM_TIMEOUT_SECONDS = 45

# Evidence Retriever Configuration
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
RETRIEVAL_TOP_K = 5
FAISS_INDEX_PATH = MODELS_DIR / "faiss_wiki_index.bin"
CORPUS_FILE_PATH = CORPUS_DIR / "wiki_corpus.json"

# Feature Extractor Configuration (10 Engineered Features)
FEATURE_DIM = 10
FEATURE_NAMES = [
    "support_mean",
    "support_max",
    "support_std",
    "refute_mean",
    "refute_max",
    "refute_std",
    "agreement_ratio",
    "retrieval_conf_mean",
    "retrieval_conf_min",
    "num_subclaims_norm",
]

# Verdict Aggregator Configuration (Novel Trained Head)
AGGREGATOR_MODEL_PATH = MODELS_DIR / "verdict_aggregator.pt"
AGGREGATOR_SCALER_PATH = MODELS_DIR / "feature_scaler.pkl"
HIDDEN_DIM = 16
NUM_CLASSES = 3
CLASS_NAMES = ["SUPPORTED", "REFUTED", "NOT ENOUGH INFO"]
LABEL_MAP = {
    "SUPPORTED": 0,
    "REFUTED": 1,
    "NOT ENOUGH INFO": 2,
    "SUPPORTS": 0,
    "REFUTES": 1,
    "NOT_ENOUGH_INFO": 2,
}
INV_LABEL_MAP = {0: "SUPPORTED", 1: "REFUTED", 2: "NOT ENOUGH INFO"}

# Training Hyperparameters
TRAIN_EPOCHS = 40
TRAIN_BATCH_SIZE = 64
TRAIN_LR = 0.003
TRAIN_WEIGHT_DECAY = 1e-4

# Web App Configuration
WEB_HOST = "127.0.0.1"
WEB_PORT = 8000
