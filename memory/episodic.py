"""
memory/episodic.py
------------------
Persistent episodic storage for high-value agent experiences and routing decisions.

Why this file exists:
    Agents encounter critical decisions, user preferences, resolved incidents, business milestones,
    and workflow outcomes during operation. Raw dialogue is noisy, but distilled experiences
    represent valuable historical context that must persist long after a conversation ends.

What problem it solves:
    Provides a structured, persistent episodic memory store (`EpisodicMemory`) that records
    only meaningful, outcome-bearing events. Prevents storage bloat by excluding routine,
    ephemeral dialogue, ensuring fast retrieval of relevant historical experiences when making
    future decisions.

How it connects to other memory modules:
    - Receives promoted messages and structured evaluation audit records from `memory.router.PromoteOrDropRouter`.
    - Serves as the sole input data source for `memory.consolidation.SemanticConsolidationEngine`
      to synthesize stable, versioned knowledge into `memory.semantic.SemanticMemory`.

How it fits into the overall memory architecture:
    Episodic Memory represents Layer 3 of the 4-layer memory hierarchy. It forms the permanent
    experience log of the agent, bridging short-term interaction events to consolidated long-term facts.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
import json
import logging
from typing import Any, Dict, List, Optional, Predicate

logger = logging.getLogger(__name__)


class EventCategory(str, Enum):
    """Categories for persistent episodic memory records."""
    USER_DECISION = "user_decision"
    USER_PREFERENCE = "user_preference"
    RESOLVED_INCIDENT = "resolved_incident"
    BUSINESS_EVENT = "business_event"
    RECURRING_PATTERN = "recurring_pattern"
    COMPLETED_WORKFLOW = "completed_workflow"
    ROUTER_PROMOTION = "router_promotion"


@dataclass
class EpisodicEvent:
    """
    Represents a single meaningful experience or event in the agent's lifetime.

    Attributes:
        event_id: Unique string identifier for the event.
        category: The classification category (`EventCategory`).
        summary: High-level textual summary of the experience.
        detail: In-depth information, payload, or context.
        timestamp: ISO 8601 string recording when the event occurred.
        importance_score: Float between 0.0 and 1.0 indicating significance.
        source_message_id: Optional ID of the short-term message that triggered this record.
        tags: List of descriptive tags for indexing and filtering.
        metadata: Flexible dictionary for category-specific auxiliary details.
    """
    event_id: str
    category: EventCategory
    summary: str
    detail: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    importance_score: float = 1.0
    source_message_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.category, str) and not isinstance(self.category, EventCategory):
            self.category = EventCategory(self.category)
        # Enforce importance score bounds
        self.importance_score = max(0.0, min(1.0, float(self.importance_score)))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize event to dictionary format."""
        data = asdict(self)
        data["category"] = self.category.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodicEvent":
        """Reconstruct event instance from dictionary."""
        data_copy = data.copy()
        if "category" in data_copy:
            data_copy["category"] = EventCategory(data_copy["category"])
        return cls(**data_copy)


class EpisodicMemory:
    """
    Persistent storage and query engine for EpisodicEvents.

    Supports structured event logging, category-based querying, tag filtering,
    time-window retrieval, and JSON serialization.
    """

    def __init__(self, initial_events: Optional[List[EpisodicEvent]] = None) -> None:
        """Initialize EpisodicMemory with an optional list of events."""
        self._events: Dict[str, EpisodicEvent] = {}
        if initial_events:
            for event in initial_events:
                self._events[event.event_id] = event

    @property
    def count(self) -> int:
        """Total number of stored episodic events."""
        return len(self._events)

    def record_event(self, event: EpisodicEvent) -> None:
        """Store an episodic event. Overwrites if an event with the same ID exists."""
        self._events[event.event_id] = event
        logger.info("Recorded episodic event [%s]: %s (Category: %s)", event.event_id, event.summary, event.category.value)

    def add_event(
        self,
        event_id: str,
        category: EventCategory,
        summary: str,
        detail: str,
        importance_score: float = 1.0,
        source_message_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EpisodicEvent:
        """Convenience method to construct and record an EpisodicEvent."""
        event = EpisodicEvent(
            event_id=event_id,
            category=category,
            summary=summary,
            detail=detail,
            importance_score=importance_score,
            source_message_id=source_message_id,
            tags=tags or [],
            metadata=metadata or {},
        )
        self.record_event(event)
        return event

    def get_event(self, event_id: str) -> Optional[EpisodicEvent]:
        """Retrieve a specific event by its ID."""
        return self._events.get(event_id)

    def get_all_events(() -> List[EpisodicEvent]:
        """Return all recorded events sorted chronologically."""
        return sorted(self._events.values(), key=lambda e: e.timestamp)

    def get_all_events(self) -> List[EpisodicEvent]:
        """Return all recorded events sorted chronologically."""
        return sorted(self._events.values(), key=lambda e: e.timestamp)

    def query_by_category(self, category: EventCategory) -> List[EpisodicEvent]:
        """Filter events matching a specific category."""
        target_cat = category if isinstance(category, EventCategory) else EventCategory(category)
        return [e for e in self.get_all_events() if e.category == target_cat]

    def query_by_tag(self, tag: str) -> List[EpisodicEvent]:
        """Filter events containing a specific tag."""
        return [e for e in self.get_all_events() if tag in e.tags]

    def query_by_min_importance(self, min_importance: float) -> List[EpisodicEvent]:
        """Filter events with importance score >= min_importance."""
        return [e for e in self.get_all_events() if e.importance_score >= min_importance]

    def query_by_search_term(self, query: str) -> List[EpisodicEvent]:
        """Basic keyword search across summary, detail, and tags."""
        term = query.lower()
        results = []
        for e in self.get_all_events():
            if (
                term in e.summary.lower()
                or term in e.detail.lower()
                or any(term in t.lower() for t in e.tags)
            ):
                results.append(e)
        return results

    def clear(self) -> None:
        """Clear all stored episodic events."""
        self._events.clear()
        logger.info("Episodic memory cleared.")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize EpisodicMemory to dictionary format."""
        return {
            "count": self.count,
            "events": [e.to_dict() for e in self.get_all_events()],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodicMemory":
        """Reconstruct EpisodicMemory instance from dictionary data."""
        raw_events = data.get("events", [])
        events = [EpisodicEvent.from_dict(item) for item in raw_events]
        return cls(initial_events=events)

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Serialize state to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "EpisodicMemory":
        """Reconstruct instance from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
