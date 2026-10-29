"""
Dense Embedder module using sentence-transformers (all-MiniLM-L6-v2).
Produces 384-dimensional L2-normalized embeddings for fast cosine similarity via FAISS.
"""

from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL_NAME, EMBEDDING_DIM

class DenseEmbedder:
    _instance = None

    def __new__(cls, *args, **kwargs):
        # Singleton pattern to prevent loading the 80MB transformer multiple times into memory
        if cls._instance is None:
            cls._instance = super(DenseEmbedder, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        if self._initialized:
            return
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = EMBEDDING_DIM
        self._initialized = True

    def embed_texts(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Embeds a string or list of strings into L2-normalized float32 vectors.
        Shape: [len(texts), 384]
        """
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embeddings.astype(np.float32)
