"""Vector store for RAG chunk retrieval.

A lightweight in-memory vector database with cosine similarity search and
metadata lookup. The implementation is intentionally simple so it can be used
without heavy external dependencies.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .chunking import Chunk
from .embedding import TextEmbedder, cosine_similarity


class VectorDB:
    def __init__(self, embedder: TextEmbedder):
        self.embedder = embedder
        self.vectors: List[List[float]] = []
        self.metadata: List[Chunk] = []

    def build(self, chunks: List[Chunk]):
        self.metadata = chunks
        self.vectors = self.embedder.embed_texts([chunk.text for chunk in chunks])

    def search(self, query: str, k: int = 4) -> List[tuple[Chunk, float]]:
        if not self.vectors:
            return []
        query_vector = self.embedder.embed_texts([query])[0]
        scored: List[tuple[Chunk, float]] = []
        for metadata, vector in zip(self.metadata, self.vectors):
            score = cosine_similarity(query_vector, vector)
            scored.append((metadata, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]

    def all_chunks(self) -> List[Chunk]:
        return list(self.metadata)
