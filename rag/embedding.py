"""Embedding utilities for RAG.

This module provides a configurable embedding abstraction that uses a
deterministic local fallback when no external provider key is present.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import time
from typing import Iterable, List

EMBED_DIM = 128


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def normalize_vector(vector: List[float]) -> List[float]:
    magnitude = math.sqrt(sum(x * x for x in vector))
    if magnitude == 0:
        return vector
    return [x / magnitude for x in vector]


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    return sum(a * b for a, b in zip(vec_a, vec_b))


def _local_embedding(text: str, dimension: int = EMBED_DIM) -> List[float]:
    counts: dict[int, float] = {}
    for token in tokenize(text):
        if not token:
            continue
        index = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % dimension
        counts[index] = counts.get(index, 0.0) + 1.0

    vector = [0.0] * dimension
    for index, value in counts.items():
        vector[index] = value
    return normalize_vector(vector)


class TextEmbedder:
    def __init__(self, model_name: str = "local-text-embed-001", batch_size: int = 16):
        self.model_name = model_name
        self.batch_size = batch_size
        self.api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self.provider = "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "local"

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        texts = list(texts)

        if self.provider == "anthropic":
            try:
                from anthropic import Anthropic

                client = Anthropic(api_key=self.api_key)
                embeddings = []
                for i in range(0, len(texts), self.batch_size):
                    batch = texts[i:i + self.batch_size]
                    response = client.embeddings.create(
                        model=self.model_name,
                        input=batch,
                    )
                    embeddings.extend([item.embedding for item in response.data])
                return [normalize_vector(vec) for vec in embeddings]
            except Exception:
                pass

        return [self._safe_local_embed(text) for text in texts]

    def _safe_local_embed(self, text: str) -> List[float]:
        return _local_embedding(text)
