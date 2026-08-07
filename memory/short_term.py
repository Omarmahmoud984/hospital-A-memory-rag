"""
memory/short_term.py
--------------------
Rolling conversation buffer for the AI Agent Memory Subsystem.

Why this file exists:
    In complex multi-turn interactions, an AI agent receives user requests, generates replies,
    issues tool calls, and observes tool results. The raw stream of interactions forms the
    immediate conversation history. Short-Term Memory provides a bounded, temporary storage
    layer for this ongoing dialogue stream.

What problem it solves:
    Context windows are finite and costly. Unbounded conversation history leads to context
    overflows and increased latency/token costs. This module implements a rolling FIFO buffer
    with configurable capacity limits (by message count and/or total tokens). When capacity is
    exceeded, oldest items are evicted and dispatched to an overflow handler (the Promote-or-Drop
    Router) to decide if they should be discarded or preserved in Episodic Memory.

How it connects to other memory modules:
    - Evicted messages are sent directly to `memory.router.PromoteOrDropRouter` via an overflow callback.
    - Operates alongside `memory.scratchpad.Scratchpad` (which stores working agent state independently
      of raw turn history).
    - Serves as the primary input feed for context window management strategies in `context_eval/`.

How it fits into the overall memory architecture:
    Short-Term Memory is Tier 1 of the 4-layer memory hierarchy (Short-Term, Scratchpad, Episodic,
    Semantic). It is completely ephemeral: no state here is permanent until processed by the router.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
import json
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class MessageRole(str, Enum):
    """Supported roles for items in short-term conversation memory."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_CALL = "tool_call"
    TOOL_OBSERVATION = "tool_observation"


@dataclass
class Message:
    """
    Represents a single conversational turn or interaction event.

    Attributes:
        role: The role of the entity producing the message.
        content: Primary textual content or serialized representation.
        timestamp: ISO 8601 string of when the message was recorded.
        message_id: Unique identifier for tracking and correlation.
        tool_name: Optional name of the tool (for TOOL_CALL / TOOL_OBSERVATION).
        tool_call_id: Optional ID linking a tool observation to its call.
        metadata: Flexible dictionary for auxiliary attributes (e.g. latency, status).
        estimated_tokens: Estimated token count for context budget management.
    """
    role: MessageRole
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    message_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_call_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    estimated_tokens: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.role, str) and not isinstance(self.role, MessageRole):
            self.role = MessageRole(self.role)
        if self.estimated_tokens <= 0 and self.content:
            # Simple fallback estimation: ~4 chars per token if not explicitly provided
            self.estimated_tokens = max(1, len(self.content) // 4)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize Message to dictionary format."""
        data = asdict(self)
        data["role"] = self.role.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Reconstruct Message instance from dictionary."""
        data_copy = data.copy()
        if "role" in data_copy:
            data_copy["role"] = MessageRole(data_copy["role"])
        return cls(**data_copy)


# Type hint for the overflow callback function (e.g., PromoteOrDropRouter.evaluate_overflow)
OverflowCallback = Callable[[Message], None]


class ShortTermMemory:
    """
    Rolling conversation buffer enforcing capacity limits via FIFO eviction.

    Supports dual capacity bounds:
        - `capacity`: Maximum number of messages allowed in the buffer.
        - `max_tokens`: Optional ceiling on total aggregated tokens across all messages.
    """

    def __init__(
        self,
        capacity: int = 20,
        max_tokens: Optional[int] = None,
        overflow_callback: Optional[OverflowCallback] = None,
    ) -> None:
        """
        Initialize Short-Term Memory buffer.

        Args:
            capacity: Maximum number of messages the buffer can hold.
            max_tokens: Maximum cumulative tokens allowed. None for unlimited.
            overflow_callback: Callback function executed when a message is evicted due to overflow.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        
        self.capacity: int = capacity
        self.max_tokens: Optional[int] = max_tokens
        self.overflow_callback: Optional[OverflowCallback] = overflow_callback
        
        self._buffer: List[Message] = []
        self._total_tokens: int = 0

    @property
    def size(self) -> int:
        """Current number of messages in the buffer."""
        return len(self._buffer)

    @property
    def total_tokens(self) -> int:
        """Current cumulative estimated tokens in the buffer."""
        return self._total_tokens

    def set_overflow_callback(self, callback: OverflowCallback) -> None:
        """Register or update the overflow notification callback."""
        self.overflow_callback = callback

    def add_message(self, message: Message) -> None:
        """
        Append a new message to the buffer and enforce capacity limits.

        If capacity or max_tokens thresholds are breached, oldest messages are evicted
        FIFO-style and forwarded to `overflow_callback`.
        """
        self._buffer.append(message)
        self._total_tokens += message.estimated_tokens
        logger.debug("Added message [%s] (%d tokens). Current size: %d", message.role.value, message.estimated_tokens, self.size)

        self._enforce_capacity()

    def add_user_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Message:
        """Convenience method to record a user message."""
        msg = Message(role=MessageRole.USER, content=content, metadata=metadata or {})
        self.add_message(msg)
        return msg

    def add_assistant_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Message:
        """Convenience method to record an assistant reply."""
        msg = Message(role=MessageRole.ASSISTANT, content=content, metadata=metadata or {})
        self.add_message(msg)
        return msg

    def add_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        tool_call_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """Convenience method to record an outgoing tool call."""
        content = json.dumps({"tool": tool_name, "arguments": arguments})
        msg = Message(
            role=MessageRole.TOOL_CALL,
            content=content,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            metadata=metadata or {},
        )
        self.add_message(msg)
        return msg

    def add_tool_observation(
        self,
        tool_name: str,
        observation: Any,
        tool_call_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """Convenience method to record a tool execution observation/result."""
        content = observation if isinstance(observation, str) else json.dumps(observation)
        msg = Message(
            role=MessageRole.TOOL_OBSERVATION,
            content=content,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            metadata=metadata or {},
        )
        self.add_message(msg)
        return msg

    def _enforce_capacity(self) -> None:
        """Evict oldest messages until size and token constraints are satisfied."""
        while self._buffer and (
            len(self._buffer) > self.capacity
            or (self.max_tokens is not None and self._total_tokens > self.max_tokens)
        ):
            evicted = self._buffer.pop(0)
            self._total_tokens -= evicted.estimated_tokens
            logger.info("Evicted message [%s] (ID: %s) from Short-Term Memory", evicted.role.value, evicted.message_id)

            if self.overflow_callback is not None:
                try:
                    self.overflow_callback(evicted)
                except Exception as e:
                    logger.error("Error executing overflow_callback for evicted message: %s", e)

    def get_messages(self) -> List[Message]:
        """Return a shallow copy of all active messages in the buffer."""
        return list(self._buffer)

    def peek_latest(() -> Optional[Message]:
        """Return the most recent message without modifying the buffer."""
        return self._buffer[-1] if self._buffer else None

    def peek_latest(self) -> Optional[Message]:
        """Return the most recent message without modifying the buffer."""
        return self._buffer[-1] if self._buffer else None

    def clear(() -> List[Message]:
        """Clear all messages from buffer and return evicted items."""
        evicted_items = list(self._buffer)
        self._buffer.clear()
        self._total_tokens = 0
        return evicted_items

    def clear(self) -> List[Message]:
        """Clear all messages from buffer and return cleared items."""
        evicted_items = list(self._buffer)
        self._buffer.clear()
        self._total_tokens = 0
        return evicted_items

    def to_dict(self) -> Dict[str, Any]:
        """Serialize ShortTermMemory state to dictionary."""
        return {
            "capacity": self.capacity,
            "max_tokens": self.max_tokens,
            "total_tokens": self._total_tokens,
            "messages": [msg.to_dict() for msg in self._buffer],
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        overflow_callback: Optional[OverflowCallback] = None,
    ) -> "ShortTermMemory":
        """Reconstruct ShortTermMemory from dictionary state."""
        instance = cls(
            capacity=data.get("capacity", 20),
            max_tokens=data.get("max_tokens"),
            overflow_callback=overflow_callback,
        )
        messages_data = data.get("messages", [])
        for msg_dict in messages_data:
            msg = Message.from_dict(msg_dict)
            instance._buffer.append(msg)
            instance._total_tokens += msg.estimated_tokens
        return instance

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Serialize state to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(
        cls,
        json_str: str,
        overflow_callback: Optional[OverflowCallback] = None,
    ) -> "ShortTermMemory":
        """Reconstruct instance from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data, overflow_callback=overflow_callback)
