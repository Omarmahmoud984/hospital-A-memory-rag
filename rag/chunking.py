"""Document chunking utilities for RAG.

This module is responsible for splitting hospital policy documents into
retrievable chunks while preserving source and section metadata.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

CHUNK_SIZE = 120
CHUNK_OVERLAP = 20


@dataclass
class DocumentSource:
    uri: str
    name: str
    text: str


@dataclass
class Chunk:
    text: str
    source: str
    section_title: str
    chunk_index: int
    token_count: int


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def tokenize(text: str) -> List[str]:
    return [tok.lower() for tok in re.findall(r"\w+", text)]


def _detect_section_titles(lines: List[str]) -> List[str]:
    titles = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.isupper() or stripped.endswith(":"):
            titles.append(stripped.rstrip(":"))
    return titles


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_title = "Document"
    current_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.isupper() or stripped.endswith(":"):
            if current_lines:
                sections.append((current_title, "\n".join(current_lines)))
            current_title = stripped.rstrip(":")
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_title, "\n".join(current_lines)))

    if not sections:
        sections.append((current_title, text))

    return sections


def chunk_text(text: str, source: str = "unknown", chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
    text = normalize_whitespace(text)
    sections = _split_into_sections(text)
    chunks: List[Chunk] = []
    index = 0

    for section_title, section_text in sections:
        tokens = tokenize(section_text)
        if not tokens:
            continue

        stride = max(1, chunk_size - overlap)
        for i in range(0, len(tokens), stride):
            window = tokens[i:i + chunk_size]
            if not window:
                break
            chunk_text = " ".join(window)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    source=source,
                    section_title=section_title,
                    chunk_index=index,
                    token_count=len(window),
                )
            )
            index += 1
            if i + chunk_size >= len(tokens):
                break

    return chunks


def chunk_documents(documents: Iterable[DocumentSource], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
    chunks: List[Chunk] = []
    for document in documents:
        chunks.extend(
            chunk_text(
                text=document.text,
                source=document.uri,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )
    return chunks


def load_documents_from_folder(folder_path: str) -> List[DocumentSource]:
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Document folder not found: {folder_path}")

    documents: List[DocumentSource] = []
    for file_path in sorted(folder.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8")
        documents.append(
            DocumentSource(
                uri=f"file://{file_path.name}",
                name=file_path.stem,
                text=text,
            )
        )
    return documents
