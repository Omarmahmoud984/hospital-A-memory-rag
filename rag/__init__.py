"""RAG package for document ingestion, retrieval, and answer generation.

This package provides a small retrieval-augmented generation stack for hospital
policy resources, including chunking, embedding, vector retrieval, and answer
prediction. It is intentionally lightweight and works with a deterministic
fallback when no external LLM or embedding service is configured.
"""

from .chunking import DocumentSource, Chunk, chunk_documents, load_documents_from_folder
from .embedding import TextEmbedder
from .vectordb import VectorDB
from .generation import AnswerGenerator
from .naive_rag import NaiveRAG
from .hybrid_rag import HybridRAG
from .agentic_rag import AgenticRAG
from .graph_rag import GraphRAG
from .self_rag import SelfRAGVerifier

__all__ = [
    "DocumentSource",
    "Chunk",
    "chunk_documents",
    "load_documents_from_folder",
    "TextEmbedder",
    "VectorDB",
    "AnswerGenerator",
    "NaiveRAG",
    "HybridRAG",
    "AgenticRAG",
    "GraphRAG",
    "SelfRAGVerifier",
]
