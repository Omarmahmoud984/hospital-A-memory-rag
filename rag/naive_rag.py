"""Naive RAG architecture.

The naive pipeline embeds the query, retrieves the top vector-similar chunks,
stuffs them into a prompt, and generates an answer without any additional
re-ranking or iterative retrieval.
"""

from __future__ import annotations

from typing import List

from .generation import AnswerGenerator
from .vectordb import VectorDB
from .self_rag import SelfRAGVerifier
from .chunking import Chunk


class NaiveRAG:
    def __init__(
        self,
        db: VectorDB,
        generator: AnswerGenerator,
        verifier: SelfRAGVerifier | None = None,
        top_k: int = 4,
    ):
        self.db = db
        self.generator = generator
        self.verifier = verifier
        self.top_k = top_k

    def build_prompt(self, question: str, chunks: List[Chunk]) -> str:
        context_lines = []
        for chunk in chunks:
            citation = f"[{chunk.source} | {chunk.section_title} | chunk {chunk.chunk_index}]"
            context_lines.append(f"{citation} {chunk.text}")
        context = "\n\n".join(context_lines)
        return (
            f"Use the following hospital policy excerpts to answer the question.\n"
            f"Cite the source if relevant.\n\n"
            f"POLICY CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}\n\n"
            f"ANSWER:"
        )

    def answer(self, question: str) -> dict:
        import time
        start_t = time.perf_counter()
        chunks = [item[0] for item in self.db.search(question, k=self.top_k)]
        prompt = self.build_prompt(question, chunks)
        answer = self.generator.generate(prompt)
        verification = None
        if self.verifier:
            verification = self.verifier.verify(answer, chunks)
        latency = time.perf_counter() - start_t
        return {
            "architecture": "naive",
            "question": question,
            "answer": answer,
            "chunks": chunks,
            "verification": verification,
            "latency_seconds": latency,
            "token_usage": self.generator.last_token_usage,
        }
