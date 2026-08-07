"""
memory/consolidation.py
-----------------------
Semantic Consolidation Engine for periodic factual extraction and contradiction resolution.

Why this file exists:
    Episodic Memory accumulates time-bound experiences. Over time, recurring patterns or updated
    user preferences (e.g. changing preferred warehouse from Warehouse Alpha to Warehouse Bravo)
    emerge in episodic events. A dedicated background process is required to extract stable truths,
    resolve contradictions, and version semantic knowledge.

What problem it solves:
    Prevents silent factual overwrites or knowledge corruption. When contradictory episodes occur
    (e.g., Episode A says "Preferred warehouse: Alpha", Episode B says "Preferred warehouse: Bravo"),
    the consolidation engine detects the conflict, marks the outdated fact as SUPERSEDED, creates
    a new versioned fact, records the resolution history with timestamps, and updates state explicitly.

How it connects to other memory modules:
    - Queries persistent events from `memory.episodic.EpisodicMemory`.
    - Holds the private `ConsolidationToken` required to write/update `memory.semantic.SemanticMemory`.
    - Completely decoupled from `memory.router.PromoteOrDropRouter` (the router evaluates short-term
      evictions; consolidation synthesizes long-term episodic events into semantic facts).

How it fits into the overall memory architecture:
    Acts as the periodic background processing bridge between Layer 3 (Episodic Experience) and
    Layer 4 (Semantic Knowledge Store).
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from memory.episodic import EpisodicMemory, EpisodicEvent, EventCategory
from memory.semantic import (
    SemanticMemory,
    SemanticFact,
    FactState,
    ConsolidationToken,
)

logger = logging.getLogger(__name__)


class ConflictResolutionStrategy(str, Enum):
    """Strategies for resolving conflicting episodic assertions."""
    LATEST_WINS = "latest_wins"
    HIGHEST_CONFIDENCE = "highest_confidence"
    FLAG_CONTRADICTION = "flag_contradiction"


@dataclass
class ConsolidationResult:
    """
    Summary report produced by a consolidation run.

    Attributes:
        run_id: Unique identifier for the consolidation pass.
        timestamp: ISO 8601 string of execution time.
        episodes_scanned: Total count of episodic events processed.
        facts_created: Count of new facts inserted into SemanticMemory.
        facts_updated: Count of facts updated or incremented in version.
        facts_superseded: Count of facts marked SUPERSEDED due to conflicts.
        contradictions_flagged: Count of unresolved conflicts marked CONTRADICTED.
        details: List of descriptive action strings for audit reporting.
    """
    run_id: str
    timestamp: str
    episodes_scanned: int
    facts_created: int
    facts_updated: int
    facts_superseded: int
    contradictions_flagged: int
    details: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConsolidationResult":
        """Reconstruct result instance from dictionary."""
        return cls(**data)


class SemanticConsolidationEngine:
    """
    Periodic consolidation engine extracting facts from Episodic Memory
    and updating Semantic Memory using an authorized ConsolidationToken.
    """

    def __init__(
        self,
        episodic_memory: EpisodicMemory,
        semantic_memory: SemanticMemory,
        default_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.LATEST_WINS,
    ) -> None:
        """
        Initialize the Consolidation Engine.

        Args:
            episodic_memory: Source episodic memory store.
            semantic_memory: Target semantic memory store.
            default_strategy: Default conflict resolution strategy.
        """
        self.episodic_memory: EpisodicMemory = episodic_memory
        self.semantic_memory: SemanticMemory = semantic_memory
        self.default_strategy: ConflictResolutionStrategy = default_strategy
        
        # Private token instantiation granting exclusive write rights to SemanticMemory
        self._token: ConsolidationToken = ConsolidationToken()
        self._run_history: List[ConsolidationResult] = []

    def run_consolidation(
        self,
        min_importance: float = 0.5,
        strategy: Optional[ConflictResolutionStrategy] = None,
    ) -> ConsolidationResult:
        """
        Execute a complete consolidation pass across Episodic Memory.

        Scans episodic events with importance >= min_importance, extracts triple assertions,
        detects conflicts against active semantic facts, applies versioning and state changes,
        and logs full audit details.
        """
        strat = strategy or self.default_strategy
        run_id = f"cons_{len(self._run_history) + 1}_{int(datetime.now(timezone.utc).timestamp())}"
        now_iso = datetime.now(timezone.utc).isoformat()

        episodes = self.episodic_memory.query_by_min_importance(min_importance)
        
        facts_created = 0
        facts_updated = 0
        facts_superseded = 0
        contradictions_flagged = 0
        details: List[str] = []

        logger.info("Starting Semantic Consolidation Run [%s] processing %d episodes (Strategy: %s)",
                    run_id, len(episodes), strat.value)

        # Step 1: Extract candidate assertions from episodic events
        candidate_facts = self._extract_facts_from_episodes(episodes)

        # Step 2: Process each candidate assertion against Semantic Memory
        for subj, pred, obj_val, confidence, ep_ids, ep_timestamp in candidate_facts:
            # Check for existing active facts matching (subject, predicate)
            existing_facts = self.semantic_memory.query_by_triple(
                subject=subj, predicate=pred, active_only=True
            )

            if not existing_facts:
                # Case A: Brand new fact -> Create v1
                fact_id = f"fact_{subj}_{pred}".lower().replace(" ", "_")
                new_fact = SemanticFact(
                    fact_id=fact_id,
                    subject=subj,
                    predicate=pred,
                    object_value=obj_val,
                    confidence=confidence,
                    version=1,
                    state=FactState.ACTIVE,
                    created_at=ep_timestamp,
                    updated_at=now_iso,
                    history=[{
                        "version": 1,
                        "action": "created",
                        "object_value": obj_val,
                        "timestamp": ep_timestamp,
                    }],
                    source_episode_ids=ep_ids,
                )
                self.semantic_memory.consolidate_fact(self._token, new_fact)
                facts_created += 1
                details.append(f"Created new fact [{fact_id}]: {subj} {pred} = '{obj_val}'")

            else:
                for existing in existing_facts:
                    if existing.object_value == obj_val:
                        # Case B: Fact confirmed -> Update confidence/source episodes if needed
                        existing.confidence = max(existing.confidence, confidence)
                        for ep_id in ep_ids:
                            if ep_id not in existing.source_episode_ids:
                                existing.source_episode_ids.append(ep_id)
                        existing.updated_at = now_iso
                        facts_updated += 1
                        details.append(f"Reinforced existing fact [{existing.fact_id}]: {subj} {pred} = '{obj_val}'")

                    else:
                        # Case C: REAL CONTRADICTION DETECTED!
                        logger.warning(
                            "Contradiction detected for (%s, %s): Existing='%s' vs New='%s'",
                            subj, pred, existing.object_value, obj_val
                        )

                        if strat == ConflictResolutionStrategy.LATEST_WINS:
                            # Supersede existing fact, create new versioned fact
                            self.semantic_memory.mark_fact_superseded(
                                self._token, existing.fact_id, f"fact_{subj}_{pred}_v{existing.version + 1}"
                            )
                            facts_superseded += 1

                            new_fact_id = f"fact_{subj}_{pred}".lower().replace(" ", "_")
                            updated_fact = SemanticFact(
                                fact_id=new_fact_id,
                                subject=subj,
                                predicate=pred,
                                object_value=obj_val,
                                confidence=confidence,
                                version=existing.version + 1,
                                state=FactState.ACTIVE,
                                created_at=existing.created_at,
                                updated_at=now_iso,
                                history=existing.history + [{
                                    "version": existing.version + 1,
                                    "action": "superseded_previous",
                                    "previous_value": existing.object_value,
                                    "new_value": obj_val,
                                    "timestamp": now_iso,
                                }],
                                source_episode_ids=existing.source_episode_ids + ep_ids,
                            )
                            self.semantic_memory.consolidate_fact(self._token, updated_fact)
                            facts_updated += 1
                            details.append(
                                f"Resolved contradiction via LATEST_WINS: Superseded '{existing.object_value}' "
                                f"with '{obj_val}' for [{subj} {pred}] (New Version: v{updated_fact.version})"
                            )

                        elif strat == ConflictResolutionStrategy.FLAG_CONTRADICTION:
                            self.semantic_memory.mark_fact_contradicted(
                                self._token,
                                existing.fact_id,
                                f"Contradictory candidate value '{obj_val}' observed in episode(s) {ep_ids}",
                            )
                            contradictions_flagged += 1
                            details.append(
                                f"Flagged contradiction for [{existing.fact_id}]: "
                                f"Existing='{existing.object_value}' vs New='{obj_val}'"
                            )

        result = ConsolidationResult(
            run_id=run_id,
            timestamp=now_iso,
            episodes_scanned=len(episodes),
            facts_created=facts_created,
            facts_updated=facts_updated,
            facts_superseded=facts_superseded,
            contradictions_flagged=contradictions_flagged,
            details=details,
        )

        self._run_history.append(result)
        logger.info("Consolidation Run [%s] Complete. Created: %d, Updated: %d, Superseded: %d, Contradictions: %d",
                    run_id, facts_created, facts_updated, facts_superseded, contradictions_flagged)

        return result

    def _extract_facts_from_episodes(
        self,
        episodes: List[EpisodicEvent],
    ) -> List[Tuple[str, str, Any, float, List[str], str]]:
        """
        Helper method parsing episodic event text/metadata into triple assertions.

        Returns list of tuples: (subject, predicate, object_value, confidence, [episode_ids], timestamp)
        """
        extracted = []
        for ep in episodes:
            # Check metadata for explicit triples first
            if "subject" in ep.metadata and "predicate" in ep.metadata and "object_value" in ep.metadata:
                extracted.append((
                    ep.metadata["subject"],
                    ep.metadata["predicate"],
                    ep.metadata["object_value"],
                    ep.importance_score,
                    [ep.event_id],
                    ep.timestamp,
                ))
                continue

            # Textual pattern heuristics for demo domain (e.g. warehouse preference, allergy, status)
            detail_lower = ep.detail.lower()
            if "preferred warehouse" in detail_lower or "warehouse preference" in detail_lower:
                val = "Warehouse Bravo" if "bravo" in detail_lower else "Warehouse Alpha"
                extracted.append((
                    "user",
                    "preferred_warehouse",
                    val,
                    ep.importance_score,
                    [ep.event_id],
                    ep.timestamp,
                ))

            elif "allergic to" in detail_lower or "allergy" in detail_lower:
                words = detail_lower.split("allergic to")
                allergen = words[1].strip().split()[0] if len(words) > 1 else "unknown"
                extracted.append((
                    "patient",
                    "allergy",
                    allergen.title(),
                    ep.importance_score,
                    [ep.event_id],
                    ep.timestamp,
                ))

        return extracted

    def get_run_history(self) -> List[ConsolidationResult]:
        """Return history of consolidation run reports."""
        return list(self._run_history)
