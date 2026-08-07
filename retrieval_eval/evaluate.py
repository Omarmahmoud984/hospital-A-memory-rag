"""Evaluation framework for RAG architectures.

Compares Naive, Hybrid, and Agentic pipelines on accuracy, token usage,
and latency using the hospital policy question set.
"""

from __future__ import annotations

import csv
import os
from typing import List

from rag.chunking import DocumentSource, chunk_documents
from rag.embedding import TextEmbedder
from rag.vectordb import VectorDB
from rag.generation import AnswerGenerator
from rag.naive_rag import NaiveRAG
from rag.hybrid_rag import HybridRAG
from rag.agentic_rag import AgenticRAG
from rag.graph_rag import GraphRAG
from rag.self_rag import SelfRAGVerifier
from retrieval_eval.questions import get_questions


def load_documents() -> List[DocumentSource]:
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rag", "documents"))
    return [
        DocumentSource(uri="triage://protocols/guidelines", name="triage_guidelines", text=open(os.path.join(base, "triage_guidelines.txt"), encoding="utf-8").read()),
        DocumentSource(uri="hospital://operating-rooms/rules", name="or_rules", text=open(os.path.join(base, "or_rules.txt"), encoding="utf-8").read()),
    ]


def compare_answer(answer: str, expected: str, unanswerable: bool) -> bool:
    normalized = answer.strip().lower()
    expected_norm = expected.strip().lower()
    if unanswerable:
        return "could not find" in normalized or "not supported" in normalized or expected_norm == ""
    if not expected_norm:
        return False
    return expected_norm in normalized or normalized in expected_norm


def run_evaluation():
    questions = get_questions()
    docs = load_documents()
    chunks = chunk_documents(docs)
    embedder = TextEmbedder()
    db = VectorDB(embedder)
    db.build(chunks)
    generator = AnswerGenerator()
    verifier = SelfRAGVerifier()

    pipelines = {
        "naive": NaiveRAG(db, generator, verifier),
        "hybrid": HybridRAG(db, generator, verifier),
        "agentic": AgenticRAG(db, generator, verifier),
        "graph_rag": GraphRAG(db, generator, verifier),
    }

    results = []
    for name, pipeline in pipelines.items():
        for question in questions:
            outcome = pipeline.answer(question.question)
            correct = compare_answer(outcome["answer"], question.expected_answer, question.unanswerable)
            results.append({
                "architecture": name,
                "question": question.question,
                "expected": question.expected_answer,
                "answer": outcome["answer"],
                "supported": outcome["verification"]["supported"],
                "best_score": outcome["verification"]["best_score"],
                "correct": correct,
                "latency_seconds": outcome.get("latency_seconds", 0.0),
                "token_usage": outcome.get("token_usage", 0),
            })

    summary = []
    for name in pipelines:
        arch_results = [r for r in results if r["architecture"] == name]
        accuracy = sum(1 for r in arch_results if r["correct"]) / max(1, len(arch_results))
        avg_latency = sum(r["latency_seconds"] for r in arch_results) / max(1, len(arch_results))
        avg_tokens = sum(r["token_usage"] for r in arch_results) / max(1, len(arch_results))
        supported_rate = sum(1 for r in arch_results if r["supported"]) / max(1, len(arch_results))
        summary.append({
            "architecture": name,
            "accuracy": accuracy,
            "avg_latency": avg_latency,
            "avg_tokens": avg_tokens,
            "support_rate": supported_rate,
        })

    output_path = os.path.join(os.path.dirname(__file__), "evaluation_summary.csv")
    with open(output_path, "w", encoding="utf-8", newline="") as csvfile:
        fieldnames = ["architecture", "accuracy", "support_rate", "avg_latency_ms", "avg_tokens"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow({
                "architecture": row["architecture"],
                "accuracy": row["accuracy"],
                "support_rate": row["support_rate"],
                "avg_latency_ms": round(row["avg_latency"] * 1000.0, 2),
                "avg_tokens": row["avg_tokens"],
            })

    print("RAG Evaluation Summary")
    print("architecture | accuracy | support_rate | avg_latency_ms | avg_tokens")
    for row in summary:
        latency_ms = row['avg_latency'] * 1000.0
        print(
            f"{row['architecture']} | {row['accuracy']:.2f} | {row['support_rate']:.2f} | {latency_ms:.2f} ms | {row['avg_tokens']:.0f}"
        )
    print(f"Saved summary to: {output_path}")


if __name__ == "__main__":
    run_evaluation()
