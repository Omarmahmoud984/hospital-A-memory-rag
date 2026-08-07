"""
memory/scratchpad.py
--------------------
Agent working memory storage for active task state, reasoning, and plans.

Why this file exists:
    Agents need an isolated, scratch space to hold operational state—such as current goals,
    step-by-step execution plans, intermediate reasoning steps, sub-task statuses, temporary
    variables, and partial tool results. Conflating working state with raw dialogue history leads
    to state loss when conversation history is trimmed or summarized.

What problem it solves:
    Provides an explicit, structured working memory container (`Scratchpad`) that persists
    independently of raw conversation transcripts. Transcript pruning or sliding window truncation
    in Short-Term Memory will NEVER wipe or corrupt the agent's current task context, goals,
    assumptions, or execution progress.

How it connects to other memory modules:
    - Complements `memory.short_term.ShortTermMemory` by maintaining semantic state across turns.
    - Provides working context consumed by context selection strategies in `context_eval/`.
    - Captured snapshots can be logged to `memory.episodic.EpisodicMemory` upon task completion.

How it fits into the overall memory architecture:
    Scratchpad represents Layer 2 of the 4-layer memory hierarchy. It acts as the agent's "mental desk"
    or scratchpad—active, mutable, and protected from conversational garbage collection.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Execution status for agent goals and sub-goals."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class PlanStep:
    """
    Represents a single step in the agent's current execution plan.

    Attributes:
        step_number: Sequential index of the step.
        description: Textual explanation of the planned action.
        status: Current execution status of the step.
        result: Optional summary of the result or tool output for this step.
    """
    step_number: int
    description: str
    status: TaskStatus = TaskStatus.NOT_STARTED
    result: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize PlanStep to dictionary."""
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanStep":
        """Reconstruct PlanStep from dictionary."""
        data_copy = data.copy()
        if "status" in data_copy:
            data_copy["status"] = TaskStatus(data_copy["status"])
        return cls(**data_copy)


@dataclass
class ScratchpadState:
    """
    Complete state representation of the agent's working memory.

    Attributes:
        current_goal: The top-level goal or query the agent is attempting to resolve.
        current_subgoal: Active sub-task currently being executed.
        execution_plan: Ordered list of `PlanStep` items.
        reasoning_trace: Chain-of-thought or decision history for the current session.
        assumptions: Working assumptions established by the agent during problem solving.
        temporary_variables: Key-value store for arbitrary transient runtime variables.
        partial_tool_results: Cache of intermediate tool outputs keyed by tool name or ID.
        last_updated: Timestamp of the latest state modification.
    """
    current_goal: Optional[str] = None
    current_subgoal: Optional[str] = None
    execution_plan: List[PlanStep] = field(default_factory=list)
    reasoning_trace: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    temporary_variables: Dict[str, Any] = field(default_factory=dict)
    partial_tool_results: Dict[str, Any] = field(default_factory=dict)
    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Serialize complete state to dictionary."""
        return {
            "current_goal": self.current_goal,
            "current_subgoal": self.current_subgoal,
            "execution_plan": [step.to_dict() for step in self.execution_plan],
            "reasoning_trace": list(self.reasoning_trace),
            "assumptions": list(self.assumptions),
            "temporary_variables": dict(self.temporary_variables),
            "partial_tool_results": dict(self.partial_tool_results),
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScratchpadState":
        """Reconstruct state from dictionary."""
        plan_raw = data.get("execution_plan", [])
        plan_steps = [PlanStep.from_dict(step) for step in plan_raw]
        return cls(
            current_goal=data.get("current_goal"),
            current_subgoal=data.get("current_subgoal"),
            execution_plan=plan_steps,
            reasoning_trace=data.get("reasoning_trace", []),
            assumptions=data.get("assumptions", []),
            temporary_variables=data.get("temporary_variables", {}),
            partial_tool_results=data.get("partial_tool_results", {}),
            last_updated=data.get("last_updated", datetime.now(timezone.utc).isoformat()),
        )


class Scratchpad:
    """
    Encapsulates reading, updating, resetting, and serializing agent working memory.

    Guarantees that state changes update `last_updated` timestamps and remain immune
    to conversation transcript clearance or context truncation.
    """

    def __init__(self, initial_state: Optional[ScratchpadState] = None) -> None:
        """Initialize Scratchpad with an optional existing state."""
        self._state: ScratchpadState = initial_state or ScratchpadState()

    @property
    def state(self) -> ScratchpadState:
        """Access the current scratchpad state."""
        return self._state

    def _touch(self) -> None:
        """Update last_updated timestamp on state modification."""
        self._state.last_updated = datetime.now(timezone.utc).isoformat()

    # --- Goal & Plan Management ---

    def set_goal(self, goal: str) -> None:
        """Set or update the primary active goal."""
        self._state.current_goal = goal
        self._touch()
        logger.debug("Scratchpad goal set to: %s", goal)

    def set_subgoal(self, subgoal: Optional[str]) -> None:
        """Set or clear the active sub-goal."""
        self._state.current_subgoal = subgoal
        self._touch()
        logger.debug("Scratchpad subgoal set to: %s", subgoal)

    def set_execution_plan(self, step_descriptions: List[str]) -> None:
        """Initialize or overwrite the execution plan from a list of step descriptions."""
        self._state.execution_plan = [
            PlanStep(step_number=idx + 1, description=desc)
            for idx, desc in enumerate(step_descriptions)
        ]
        self._touch()

    def update_plan_step(
        self,
        step_number: int,
        status: TaskStatus,
        result: Optional[str] = None,
    ) -> bool:
        """
        Update status and result for a specific step in the execution plan.

        Returns True if the step was found and updated, False otherwise.
        """
        for step in self._state.execution_plan:
            if step.step_number == step_number:
                step.status = status
                if result is not None:
                    step.result = result
                self._touch()
                return True
        return False

    # --- Reasoning & Assumptions ---

    def add_reasoning_step(self, reasoning: str) -> None:
        """Append a reasoning entry or thought step."""
        self._state.reasoning_trace.append(reasoning)
        self._touch()

    def add_assumption(self, assumption: str) -> None:
        """Record a working assumption."""
        if assumption not in self._state.assumptions:
            self._state.assumptions.append(assumption)
            self._touch()

    def remove_assumption(self, assumption: str) -> bool:
        """Remove a working assumption if invalidated."""
        if assumption in self._state.assumptions:
            self._state.assumptions.remove(assumption)
            self._touch()
            return True
        return False

    # --- Variables & Tool Results ---

    def set_variable(self, key: str, value: Any) -> None:
        """Store a temporary key-value pair."""
        self._state.temporary_variables[key] = value
        self._touch()

    def get_variable(self, key: str, default: Any = None) -> Any:
        """Retrieve a stored temporary variable."""
        return self._state.temporary_variables.get(key, default)

    def store_partial_tool_result(self, tool_identifier: str, result: Any) -> None:
        """Cache an intermediate or partial result from a tool execution."""
        self._state.partial_tool_results[tool_identifier] = result
        self._touch()

    def get_partial_tool_result(self, tool_identifier: str, default: Any = None) -> Any:
        """Retrieve a cached partial tool result."""
        return self._state.partial_tool_results.get(tool_identifier, default)

    # --- Reset & Lifecycle ---

    def reset() -> None:
        """Completely reset the scratchpad working memory to empty state."""
        self._state = ScratchpadState()
        logger.info("Scratchpad state reset.")

    def reset(self) -> None:
        """Completely reset the scratchpad working memory to empty state."""
        self._state = ScratchpadState()
        logger.info("Scratchpad state reset.")

    # --- Serialization ---

    def to_dict(self) -> Dict[str, Any]:
        """Serialize Scratchpad state to dictionary format."""
        return self._state.to_dict()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Scratchpad":
        """Reconstruct Scratchpad from dictionary data."""
        state = ScratchpadState.from_dict(data)
        return cls(initial_state=state)

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Serialize Scratchpad state to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "Scratchpad":
        """Reconstruct Scratchpad from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
