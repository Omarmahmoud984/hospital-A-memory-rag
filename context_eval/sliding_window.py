"""
context_eval/sliding_window.py
------------------------------
Sliding Window strategy for context window management.

Why this file exists:
    When an LLM agent engages in extended multi-turn interactions, total context can quickly
    exceed model context limits. A baseline method for managing context volume is the Sliding
    Window strategy.

What problem it solves:
    Implements a bounded window context strategy (`SlidingWindowStrategy`) that retains only
    the most recent $N$ messages or $K$ tokens of conversation history while preserving critical
    system prompts and working state (`Scratchpad`).

How it connects to other memory modules:
    - Implements the shared `BaseContextStrategy` interface defined in this module.
    - Consumes raw turn history from `memory.short_term.ShortTermMemory`.
    - Protects working memory in `memory.scratchpad.Scratchpad` from truncation.
    - Evaluated against other strategies in `context_eval.evaluate`.

How it fits into the overall memory architecture:
    Context Strategy 1 of 4 in the Context Window Management & Evaluation framework.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import json
import logging
from typing import Any, Dict, List, Optional

from memory.short_term import Message, MessageRole, ShortTermMemory
from memory.scratchpad import Scratchpad

logger = logging.getLogger(__name__)


@dataclass
class FormattedContext:
    """
    Standardized payload representing the context prepared for LLM submission.

    Attributes:
        system_instruction: Preserved system prompt or scratchpad summary.
        messages: List of active formatted messages included in the prompt window.
        total_tokens: Total estimated tokens in the formatted context.
        pruned_count: Number of historical messages excluded by the strategy.
        strategy_name: Name identifier of the strategy used.
    """
    system_instruction: str
    messages: List[Dict[str, Any]]
    total_tokens: int
    pruned_count: int
    strategy_name: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize payload to dictionary."""
        return asdict(self)


class BaseContextStrategy(ABC):
    """
    Abstract base class establishing a uniform interface for all context management strategies.
    """

    def __init__(self, name: str) -> None:
        self.name: str = name

    @abstractmethod
    def prepare_context(
        self,
        short_term_memory: ShortTermMemory,
        scratchpad: Optional[Scratchpad] = None,
        system_prompt: str = "",
        max_context_tokens: int = 4000,
    ) -> FormattedContext:
        """
        Transform memory stores into a bounded FormattedContext payload.
        """
        pass


class SlidingWindowStrategy(BaseContextStrategy):
    """
    Sliding Window context management strategy.

    Retains the system instruction and Scratchpad working state, while taking the $N$ most
    recent messages from ShortTermMemory that fit within `max_context_tokens`.
    """

    def __init__(self, window_size: int = 10) -> None:
        """
        Initialize Sliding Window strategy.

        Args:
            window_size: Maximum number of recent conversation messages to keep.
        """
        super().__init__(name="SlidingWindow")
        self.window_size: int = window_size

    def prepare_context(
        self,
        short_term_memory: ShortTermMemory,
        scratchpad: Optional[Scratchpad] = None,
        system_prompt: str = "",
        max_context_tokens: int = 4000,
    ) -> FormattedContext:
        """
        Prepare context by applying sliding window truncation.
        """
        # 1. Build composite system instruction incorporating scratchpad state if available
        system_components = []
        if system_prompt:
            system_components.append(system_prompt)
        
        if scratchpad and scratchpad.state.current_goal:
            scratch_dict = scratchpad.to_dict()
            system_components.append(
                f"[ACTIVE SCRATCHPAD WORKING STATE]\n{json.dumps(scratch_dict, indent=2)}"
            )

        full_system_text = "\n\n".join(system_components)
        system_tokens = max(1, len(full_system_text) // 4) if full_system_text else 0

        budget_remaining = max_context_tokens - system_tokens
        if budget_remaining < 0:
            logger.warning("System prompt and scratchpad exceed total max_context_tokens budget!")
            budget_remaining = 100

        # 2. Get all raw short-term messages and take the most recent `window_size` items
        all_messages = short_term_memory.get_messages()
        windowed_messages = all_messages[-self.window_size:] if len(all_messages) > self.window_size else list(all_messages)

        # 3. Trim from oldest to newest if cumulative tokens exceed remaining budget
        selected_messages: List[Message] = []
        accumulated_tokens = 0

        for msg in reversed(windowed_messages):
            msg_tokens = msg.estimated_tokens
            if accumulated_tokens + msg_tokens <= budget_remaining:
                selected_messages.insert(0, msg)
                accumulated_tokens += msg_tokens
            else:
                break

        pruned_count = len(all_messages) - len(selected_messages)
        formatted_messages = [msg.to_dict() for msg in selected_messages]

        return FormattedContext(
            system_instruction=full_system_text,
            messages=formatted_messages,
            total_tokens=system_tokens + accumulated_tokens,
            pruned_count=pruned_count,
            strategy_name=self.name,
        )
