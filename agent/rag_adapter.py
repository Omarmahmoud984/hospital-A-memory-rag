"""RAG integration adapter for the MediCore agent.

This adapter loads MCP policy resources, creates a retrieval pipeline, and
provides a simple API for the agent to ask hospital-policy questions when
structured tools and memory are not sufficient.
"""

from __future__ import annotations

import os
import time
from typing import List

from rag.chunking import Chunk, DocumentSource, chunk_documents, load_documents_from_folder
from rag.embedding import TextEmbedder
from rag.vectordb import VectorDB
from rag.naive_rag import NaiveRAG
from rag.hybrid_rag import HybridRAG
from rag.agentic_rag import AgenticRAG
from rag.self_rag import SelfRAGVerifier
from rag.generation import AnswerGenerator


class RAGService:
    DEFAULT_RESOURCE_URIS = [
        "triage://protocols/guidelines",
        "hospital://operating-rooms/rules",
    ]

    def __init__(self, agent: MediCoreAgent, architecture: str = "hybrid"):
        self.agent = agent
        self.architecture = architecture
        self.embedder = TextEmbedder()
        self.generator = AnswerGenerator()
        self.verifier = SelfRAGVerifier()
        self.db = VectorDB(self.embedder)
        self.pipeline = None

    async def initialize(self):
        documents = await self._load_policy_documents()
        chunks = chunk_documents(documents)
        self.db.build(chunks)
        self.pipeline = self._choose_pipeline()

    async def _load_policy_documents(self) -> List[DocumentSource]:
        sources: List[DocumentSource] = []
        for uri in self.DEFAULT_RESOURCE_URIS:
            try:
                response = await self.agent.read_resource(uri)
                contents = response.get("contents", [])
                if contents:
                    text = "\n".join(item.get("text", "") for item in contents)
                    sources.append(DocumentSource(uri=uri, name=uri, text=text))
            except Exception:
                continue

        if sources:
            return sources

        fallback_folder = os.path.join(
            os.path.dirname(__file__), "..", "rag", "documents"
        )
        return load_documents_from_folder(fallback_folder)

    def _choose_pipeline(self):
        if self.architecture == "agentic":
            return AgenticRAG(self.db, self.generator, self.verifier)
        if self.architecture == "naive":
            return NaiveRAG(self.db, self.generator, self.verifier)
        return HybridRAG(self.db, self.generator, self.verifier)

    def answer_question(self, question: str) -> dict:
        if self.pipeline is None:
            raise RuntimeError("RAGService.initialize() must be called before answer_question()")
        start = time.perf_counter()
        result = self.pipeline.answer(question)
        end = time.perf_counter()
        result["latency_seconds"] = end - start
        result["token_usage"] = self._estimate_tokens(question, result["answer"])
        return result

    def _estimate_tokens(self, question: str, answer: str) -> int:
        return len(question.split()) + len(answer.split())
