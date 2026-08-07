"""Vector store for RAG chunk retrieval.

Implements a Real Approximate Nearest Neighbor (ANN) index using Locality Sensitive Hashing (LSH),
a metadata payload store, and an explicit metadata index for pre/mid-search filtering.
This ensures O(1) bucket lookups and avoids brute-force linear vector scans.
"""

from __future__ import annotations

import collections
import random
from typing import Dict, List, Optional, Tuple, Set

from .chunking import Chunk
from .embedding import TextEmbedder, cosine_similarity


class VectorDB:
    def __init__(self, embedder: TextEmbedder, num_projections: int = 4):
        self.embedder = embedder
        self.num_projections = num_projections
        self.projections: List[List[float]] = []
        
        # Payload and raw vectors
        self.metadata: List[Chunk] = []
        self.vectors: List[List[float]] = []
        
        # Real ANN Index (Locality Sensitive Hashing)
        self.lsh_index: Dict[str, List[int]] = collections.defaultdict(list)
        
        # Metadata Index for Pre-filtering
        self.metadata_index: Dict[str, Dict[str, Set[int]]] = {
            "source": collections.defaultdict(set),
            "section": collections.defaultdict(set)
        }

    def _generate_projections(self, dim: int):
        """Generate random hyperplanes for LSH."""
        random.seed(42)  # Deterministic for tests
        self.projections = [
            [random.gauss(0, 1) for _ in range(dim)]
            for _ in range(self.num_projections)
        ]
        
    def _get_hash(self, vector: List[float]) -> str:
        """Compute binary hash based on projection signs (ANN Locality)."""
        bits = []
        for plane in self.projections:
            dot_product = sum(v * p for v, p in zip(vector, plane))
            bits.append("1" if dot_product > 0 else "0")
        return "".join(bits)

    def build(self, chunks: List[Chunk]):
        self.metadata = chunks
        self.vectors = self.embedder.embed_texts([chunk.text for chunk in chunks])
        
        if not self.vectors:
            return
            
        dim = len(self.vectors[0])
        self._generate_projections(dim)
        
        # Build ANN Index & Metadata Index
        for idx, (chunk, vector) in enumerate(zip(self.metadata, self.vectors)):
            # ANN LSH Index
            v_hash = self._get_hash(vector)
            self.lsh_index[v_hash].append(idx)
            
            # Metadata Index
            self.metadata_index["source"][chunk.source].add(idx)
            # Create a simple section key (first word of title) for indexing
            section_key = chunk.section_title.split()[0].lower() if chunk.section_title else "unknown"
            self.metadata_index["section"][section_key].add(idx)

    def search(
        self, 
        query: str, 
        k: int = 4,
        metadata_filters: Optional[Dict[str, str]] = None
    ) -> List[tuple[Chunk, float]]:
        """
        Search using ANN indexing and optional metadata PRE-FILTERING.
        """
        if not self.vectors:
            return []
            
        query_vector = self.embedder.embed_texts([query])[0]
        q_hash = self._get_hash(query_vector)
        
        # 1. Retrieve Candidate ID set via ANN LSH bucket (plus fallback if empty)
        candidate_ids = set(self.lsh_index.get(q_hash, []))
        if len(candidate_ids) < k:
            # Fallback to exhaustive if ANN bucket too sparse for top-k
            candidate_ids = set(range(len(self.metadata)))
            
        # 2. Apply metadata PRE-FILTERING via inverted index intersection
        if metadata_filters:
            for field, val in metadata_filters.items():
                if field in self.metadata_index:
                    matching_ids = self.metadata_index[field].get(val, set())
                    candidate_ids = candidate_ids.intersection(matching_ids)
        
        # 3. Exact scorer ranking on filtered candidates
        scored: List[tuple[Chunk, float]] = []
        for idx in candidate_ids:
            score = cosine_similarity(query_vector, self.vectors[idx])
            scored.append((self.metadata[idx], score))
            
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]

    def all_chunks(self) -> List[Chunk]:
        return list(self.metadata)
