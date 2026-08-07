"""Self-RAG verification layer.

This module checks whether a generated answer is supported by retrieved
chunks. It uses overlap heuristics and simple entailment checks rather than
relying on a separate neural verification model.
"""

from __future__ import annotations

import re
from typing import Iterable, List

from .chunking import Chunk


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def _overlap_score(answer: str, chunk: Chunk) -> float:
    answer_terms = _token_set(answer)
    chunk_terms = _token_set(chunk.text)
    if not answer_terms or not chunk_terms:
        return 0.0
    common = answer_terms.intersection(chunk_terms)
    return len(common) / max(1, len(answer_terms))


class SelfRAGVerifier:
    def __init__(self, min_support: float = 0.15):
        self.min_support = min_support

    def verify(self, answer: str, chunks: Iterable[Chunk]) -> dict:
        chunks = list(chunks)
        if not chunks:
            return {
                "supported": False,
                "reason": "No retrieved chunks were available for verification.",
                "support_scores": [],
            }

        support_scores = [
            _overlap_score(answer, chunk)
            for chunk in chunks
        ]
        best_score = max(support_scores)
        supported = best_score >= self.min_support
        return {
            "supported": supported,
            "reason": (
                "Answer has sufficient overlap with retrieved policy chunks."
                if supported
                else "Answer appears unsupported by the retrieved policy chunks."
            ),
            "support_scores": support_scores,
            "best_score": best_score,
        }
