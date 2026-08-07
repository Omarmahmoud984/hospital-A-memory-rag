"""Agentic RAG architecture.

This module implements an iterative retrieval strategy where the model may
refine the query and request additional chunks before generating the final
answer.
"""

from __future__ import annotations

from typing import List

from .chunking import Chunk
from .vectordb import VectorDB
from .generation import AnswerGenerator
from .self_rag import SelfRAGVerifier


class AgenticRAG:
    def __init__(
        self,
        db: VectorDB,
        generator: AnswerGenerator,
        verifier: SelfRAGVerifier | None = None,
        max_rounds: int = 2,
        top_k: int = 3,
    ):
        self.db = db
        self.generator = generator
        self.verifier = verifier
        self.max_rounds = max_rounds
        self.top_k = top_k

    def answer(self, question: str) -> dict:
        import time
        start_t = time.perf_counter()
        query = question
        all_chunks: List[Chunk] = []
        for round_number in range(1, self.max_rounds + 1):
            round_chunks = [item[0] for item in self.db.search(query, k=self.top_k)]
            for chunk in round_chunks:
                if chunk not in all_chunks:
                    all_chunks.append(chunk)

            if self._has_enough_context(query, all_chunks):
                break

            query = self._refine_query(question, all_chunks, round_number)

        prompt = self._build_prompt(question, all_chunks)
        answer = self.generator.generate(prompt)
        verification = None
        if self.verifier:
            verification = self.verifier.verify(answer, all_chunks)
        latency = time.perf_counter() - start_t
        return {
            "architecture": "agentic",
            "question": question,
            "answer": answer,
            "chunks": all_chunks,
            "verification": verification,
            "latency_seconds": latency,
            "token_usage": self.generator.last_token_usage,
        }

    def _has_enough_context(self, question: str, chunks: List[Chunk]) -> bool:
        if not chunks:
            return False
        required_terms = [term for term in question.lower().split() if len(term) > 3]
        found = any(term in chunk.text.lower() for chunk in chunks for term in required_terms)
        return found and len(chunks) >= self.top_k

    def _refine_query(self, original_question: str, chunks: List[Chunk], round_number: int) -> str:
        overview = " ".join(chunk.text for chunk in chunks[: self.top_k])
        return f"{original_question} Additional relevant context: {overview[:256]}"

    def _build_prompt(self, question: str, chunks: List[Chunk]) -> str:
        context = "\n\n".join(
            f"[{chunk.source} | {chunk.section_title} | chunk {chunk.chunk_index}] {chunk.text}"
            for chunk in chunks
        )
        return (
            f"You are an assistant retrieving hospital policy context iteratively.\n\n"
            f"RETRIEVED CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}\n\n"
            f"FINAL ANSWER:"
        )
