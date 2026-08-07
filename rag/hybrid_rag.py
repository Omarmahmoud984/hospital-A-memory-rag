"""Hybrid RAG architecture.

This pipeline combines vector similarity retrieval with keyword-based ranking
and then generates an answer from the merged top results.
"""

from __future__ import annotations

import math
import re
from typing import List

from .chunking import Chunk
from .vectordb import VectorDB
from .generation import AnswerGenerator
from .self_rag import SelfRAGVerifier


def bm25_score(query: str, text: str) -> float:
    query_terms = re.findall(r"\w+", query.lower())
    if not query_terms:
        return 0.0
    text_terms = re.findall(r"\w+", text.lower())
    if not text_terms:
        return 0.0
    avgdl = len(text_terms)
    k1 = 1.5
    b = 0.75
    doc_freq = sum(1 for term in set(query_terms) if term in text_terms)
    score = 0.0
    for term in query_terms:
        freq = text_terms.count(term)
        if freq == 0:
            continue
        score += ((freq * (k1 + 1)) / (freq + k1 * (1 - b + b * len(text_terms) / avgdl)))
    return score


def _chunk_key(chunk: Chunk) -> str:
    return f"{chunk.source}:{chunk.section_title}:{chunk.chunk_index}"


def merge_rankings(vector_results: List[tuple[Chunk, float]], keyword_results: List[tuple[Chunk, float]], k: int) -> List[Chunk]:
    merged: dict[str, tuple[Chunk, float]] = {}
    for rank, (chunk, score) in enumerate(vector_results, start=1):
        key = _chunk_key(chunk)
        merged[key] = (chunk, merged.get(key, (chunk, 0.0))[1] + 1.0 / rank)
    for rank, (chunk, score) in enumerate(keyword_results, start=1):
        key = _chunk_key(chunk)
        merged[key] = (chunk, merged.get(key, (chunk, 0.0))[1] + 1.0 / rank)
    selected = sorted(merged.values(), key=lambda item: item[1], reverse=True)
    return [item[0] for item in selected][:k]


class HybridRAG:
    def __init__(
        self,
        db: VectorDB,
        generator: AnswerGenerator,
        verifier: SelfRAGVerifier | None = None,
        top_k: int = 6,
    ):
        self.db = db
        self.generator = generator
        self.verifier = verifier
        self.top_k = top_k

    def answer(self, question: str) -> dict:
        import time
        start_t = time.perf_counter()
        vector_results = self.db.search(question, k=self.top_k)
        keyword_results = [
            (chunk, bm25_score(question, chunk.text))
            for chunk in self.db.all_chunks()
        ]
        keyword_results.sort(key=lambda item: item[1], reverse=True)
        selected_chunks = merge_rankings(vector_results, keyword_results[: self.top_k], self.top_k)
        prompt = self._build_prompt(question, selected_chunks)
        answer = self.generator.generate(prompt)
        verification = None
        if self.verifier:
            verification = self.verifier.verify(answer, selected_chunks)
        latency = time.perf_counter() - start_t
        return {
            "architecture": "hybrid",
            "question": question,
            "answer": answer,
            "chunks": selected_chunks,
            "verification": verification,
            "latency_seconds": latency,
            "token_usage": self.generator.last_token_usage,
        }

    def _build_prompt(self, question: str, chunks: List[Chunk]) -> str:
        context = "\n\n".join(
            f"[{chunk.source} | {chunk.section_title} | chunk {chunk.chunk_index}] {chunk.text}"
            for chunk in chunks
        )
        return (
            "You are an assistant using hospital policy documents. Use the most relevant "
            "policy excerpts to answer the question.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}\n\n"
            f"ANSWER:"
        )
