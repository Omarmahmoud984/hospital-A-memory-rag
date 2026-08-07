"""
memory/semantic.py
------------------
Semantic knowledge storage for consolidated facts, entities, and relationships.

Why this file exists:
    Episodic memory captures isolated events over time, but agents require a unified, structured
    knowledge base representing factual truths (e.g. user warehouse preferences, hospital protocols,
    patient allergy history) decoupled from specific temporal occurrences.

What problem it solves:
    Provides a version-controlled, queryable semantic store (`SemanticMemory`). Crucially, to prevent
    uncontrolled corruption or conflicting factual overrides, direct external writes to Semantic Memory
    are strictly forbidden. Modifying or adding facts can ONLY be performed by the authorized
    Consolidation Layer (`memory.consolidation.SemanticConsolidationEngine`).

How it connects to other memory modules:
    - Read API is publicly accessible to agents and context window management tools.
    - Write/Update API enforces access control, accepting modifications ONLY from `SemanticConsolidationEngine`.
    - Receives synthesized facts extracted periodically from `memory.episodic.EpisodicMemory`.

How it fits into the overall memory architecture:
    Semantic Memory represents Layer 4 (Top Tier) of the 4-layer memory hierarchy. It holds the most
    distilled, stable, and verified domain knowledge.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
import json
import logging
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class FactState(str, Enum):
    """Lifecycle status of a semantic fact."""
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    EXPIRED = "expired"
    CONTRADICTED = "contradicted"


@dataclass
class SemanticFact:
    """
    Represents a single structured fact or entity assertion.

    Attributes:
        fact_id: Unique string key for the fact.
        subject: The subject entity (e.g., "warehouse", "patient_101").
        predicate: The relationship or property (e.g., "preferred_location", "allergic_to").
        object_value: The value assertion (e.g., "Warehouse Alpha", "Penicillin").
        confidence: Float from 0.0 to 1.0 reflecting confidence.
        version: Integer version counter incremented on updates.
        state: Current status (`FactState`).
        created_at: ISO 8601 creation timestamp.
        updated_at: ISO 8601 last update timestamp.
        history: Historical log of previous values and supersessions.
        source_episode_ids: List of episodic event IDs supporting this fact.
        metadata: Flexible storage for auxiliary annotations.
    """
    fact_id: str
    subject: str
    predicate: str
    object_value: Any
    confidence: float = 1.0
    version: int = 1
    state: FactState = FactState.ACTIVE
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    history: List[Dict[str, Any]] = field(default_factory=list)
    source_episode_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.state, str) and not isinstance(self.state, FactState):
            self.state = FactState(self.state)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize fact to dictionary format."""
        data = asdict(self)
        data["state"] = self.state.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticFact":
        """Reconstruct fact instance from dictionary."""
        data_copy = data.copy()
        if "state" in data_copy:
            data_copy["state"] = FactState(data_copy["state"])
        return cls(**data_copy)


class ConsolidationToken:
    """
    Private authorization token type. Only instances created by
    `SemanticConsolidationEngine` allow write operations on `SemanticMemory`.
    """
    pass


class SemanticMemory:
    """
    Structured long-term knowledge repository with strict write-access protection.

    Enforces the core rule: Direct writes from external callers are rejected with
    PermissionError. Only calls presenting a valid `ConsolidationToken` can modify facts.
    """

    def __init__(self, initial_facts: Optional[List[SemanticFact]] = None) -> None:
        """Initialize SemanticMemory store."""
        self._facts: Dict[str, SemanticFact] = {}
        if initial_facts:
            for fact in initial_facts:
                self._facts[fact.fact_id] = fact

    @property
    def count(self) -> int:
        """Total number of facts in memory (including active and historical)."""
        return len(self._facts)

    @property
    def active_count(self) -> int:
        """Total number of active facts."""
        return sum(1 for f in self._facts.values() if f.state == FactState.ACTIVE)

    # --- Public Read Interface ---

    def get_fact(self, fact_id: str) -> Optional[SemanticFact]:
        """Retrieve a fact by its unique ID."""
        return self._facts.get(fact_id)

    def get_active_facts(self) -> List[SemanticFact]:
        """Return all facts currently marked ACTIVE."""
        return [f for f in self._facts.values() if f.state == FactState.ACTIVE]

    def query_by_subject(self, subject: str, active_only: bool = True) -> List[SemanticFact]:
        """Query facts matching a specific subject entity."""
        subj_lower = subject.lower()
        return [
            f for f in self._facts.values()
            if f.subject.lower() == subj_lower and (not active_only or f.state == FactState.ACTIVE)
        ]

    def query_by_triple(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        active_only: bool = True,
    ) -> List[SemanticFact]:
        """Query facts matching subject and/or predicate patterns."""
        results = []
        for fact in self._facts.values():
            if active_only and fact.state != FactState.ACTIVE:
                continue
            if subject and fact.subject.lower() != subject.lower():
                continue
            if predicate and fact.predicate.lower() != predicate.lower():
                continue
            results.append(fact)
        return results

    def search_knowledge(self, query: str) -> List[SemanticFact]:
        """Search active facts by keyword across subject, predicate, and value."""
        q = query.lower()
        results = []
        for fact in self.get_active_facts():
            val_str = str(fact.object_value).lower()
            if q in fact.subject.lower() or q in fact.predicate.lower() or q in val_str:
                results.append(fact)
        return results

    # --- Restricted Consolidation Write Interface ---

    def _verify_write_permission(self, token: Any) -> None:
        """Internal guard ensuring only ConsolidationToken holders can write."""
        if not isinstance(token, ConsolidationToken):
            raise PermissionError(
                "Direct write to SemanticMemory is forbidden. "
                "Only the SemanticConsolidationEngine with a valid ConsolidationToken may write."
            )

    def consolidate_fact(self, token: ConsolidationToken, fact: SemanticFact) -> None:
        """
        Add or update a semantic fact. Requires a valid ConsolidationToken.
        """
        self._verify_write_permission(token)
        self._facts[fact.fact_id] = fact
        logger.info("Consolidated SemanticFact [%s]: %s %s = %s (v%d, State: %s)",
                    fact.fact_id, fact.subject, fact.predicate, fact.object_value, fact.version, fact.state.value)

    def mark_fact_superseded(
        self,
        token: ConsolidationToken,
        fact_id: str,
        superseded_by_id: str,
    ) -> bool:
        """Mark an existing fact as SUPERSEDED. Requires ConsolidationToken."""
        self._verify_write_permission(token)
        fact = self._facts.get(fact_id)
        if not fact:
            return False
        fact.state = FactState.SUPERSEDED
        fact.updated_at = datetime.now(timezone.utc).isoformat()
        fact.history.append({
            "action": "superseded",
            "superseded_by": superseded_by_id,
            "timestamp": fact.updated_at,
        })
        return True

    def mark_fact_contradicted(
        self,
        token: ConsolidationToken,
        fact_id: str,
        resolution_notes: str,
    ) -> bool:
        """Mark an existing fact as CONTRADICTED. Requires ConsolidationToken."""
        self._verify_write_permission(token)
        fact = self._facts.get(fact_id)
        if not fact:
            return False
        fact.state = FactState.CONTRADICTED
        fact.updated_at = datetime.now(timezone.utc).isoformat()
        fact.history.append({
            "action": "contradicted",
            "notes": resolution_notes,
            "timestamp": fact.updated_at,
        })
        return True

    # --- Serialization ---

    def to_dict(self) -> Dict[str, Any]:
        """Serialize SemanticMemory state."""
        return {
            "count": self.count,
            "active_count": self.active_count,
            "facts": [f.to_dict() for f in self._facts.values()],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticMemory":
        """Reconstruct SemanticMemory instance."""
        raw_facts = data.get("facts", [])
        facts = [SemanticFact.from_dict(item) for item in raw_facts]
        return cls(initial_facts=facts)

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "SemanticMemory":
        """Reconstruct from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
