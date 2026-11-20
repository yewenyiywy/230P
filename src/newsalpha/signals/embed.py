"""Text embedders. All return L2-normalised float32 rows, so a dot product is a cosine.

hashing                offline, no model download; word + bigram hashing (tests, CI, quick runs)
sentence_transformers  any sentence-transformers model; MiniLM runs fine on CPU
endpoint               OpenAI-compatible /embeddings (e.g. vLLM serving an embedding model)
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from newsalpha.config import SignalsConfig


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _normalise(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.where(n == 0, 1, n)).astype(np.float32)


class HashingEmbedder:
    name = "hashing"

    def __init__(self, n_features: int = 2**14):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.vec = HashingVectorizer(
            n_features=n_features, ngram_range=(1, 2), alternate_sign=False, norm=None
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        return _normalise(self.vec.transform(texts).toarray())


class SentenceTransformerEmbedder:
    def __init__(self, model: str):
        from sentence_transformers import SentenceTransformer

        self.name = model.replace("/", "_")
        self.model = SentenceTransformer(model)

    def embed(self, texts: list[str]) -> np.ndarray:
        return _normalise(self.model.encode(texts, batch_size=64, show_progress_bar=False))


class EndpointEmbedder:
    def __init__(self, model: str, base_url: str):
        from newsalpha.extraction.llm import OpenAICompatClient

        self.name = model.replace("/", "_")
        self.model = model
        self.client = OpenAICompatClient(base_url)

    def embed(self, texts: list[str]) -> np.ndarray:
        rows = []
        for i in range(0, len(texts), 256):
            resp = self.client.post(
                "embeddings", {"model": self.model, "input": texts[i : i + 256]}
            )
            rows += [d["embedding"] for d in sorted(resp["data"], key=lambda d: d["index"])]
        return _normalise(np.asarray(rows, dtype=np.float32))


def get_embedder(cfg: SignalsConfig) -> Embedder:
    if cfg.embedder == "hashing":
        return HashingEmbedder()
    if cfg.embedder == "sentence_transformers":
        return SentenceTransformerEmbedder(cfg.embed_model)
    return EndpointEmbedder(cfg.embed_model, cfg.embed_base_url)
