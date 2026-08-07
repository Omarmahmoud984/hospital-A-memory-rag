"""
context_eval/summarization.py
-----------------------------
Recursive Summarization strategy for context window management.

Why this file exists:
    In long, multi-turn conversations, dropping older turns entirely (as in Sliding Window) risks
    losing important historical context. Recursive Summarization compresses evicted dialogue blocks
    into a growing, structured summary while preserving recent raw turns.

What problem it solves:
    Implements `RecursiveSummarizationStrategy`. When conversation history exceeds specified thresholds,
    older messages are summarized and prepended to the context stream as a running summary block,
    ensuring historical continuity without context budget explosion.

How it connects to other memory modules:
    - Inherits from `BaseContextStrategy` in `context_eval.sliding_window`.
    - Processes raw messages from `memory.short_term.ShortTermMemory`.
    - Preserves working state in `memory.scratchpad.Scratchpad`.
    - Evaluated in `context_eval.evaluate`.

How it fits into the overall memory architecture:
    Context Strategy 3 of 4 in the Context Window Management & Evaluation framework.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from memory.short_term import Message, MessageRole, ShortTermMemory
from memory.scratchpad import Scratchpad
from context_eval.sliding_window import BaseContextStrategy, FormattedContext

logger = logging.getLogger(__name__)


class RecursiveSummarizationStrategy(BaseContextStrategy):
    """
    Recursive Summarization context management strategy.

    Recursively compresses older conversation turns into a cumulative summary narrative while
    retaining the $M$ most recent turns in raw message format.
    """

    def __init__(
        self,
        recent_raw_messages: int = 4,
        max_summary_tokens: int = 300,
    ) -> None:
        """
        Initialize Recursive Summarization strategy.

        Args:
            recent_raw_messages: Number of recent messages to keep as raw turns.
            max_summary_tokens: Maximum token budget allocated for the running summary block.
        """
        super().__init__(name="RecursiveSummarization")
        self.recent_raw_messages: int = recent_raw_messages
        self.max_summary_tokens: int = max_summary_tokens
        self._running_summary: str = ""

    @property
    def current_summary(self) -> str:
        """Access the current cumulative summary string."""
        return self._running_summary

    def prepare_context(
        self,
        short_term_memory: ShortTermMemory,
        scratchpad: Optional[Scratchpad] = None,
        system_prompt: str = "",
        max_context_tokens: int = 4000,
    ) -> FormattedContext:
        """
        Transform short-term memory into a formatted context payload using recursive summarization.
        """
        all_messages = short_term_memory.get_messages()

        # Split messages into historical (to summarize) and recent (kept raw)
        if len(all_messages) > self.recent_raw_messages:
            historical_messages = all_messages[:-self.recent_raw_messages]
            raw_recent_messages = all_messages[-self.recent_raw_messages:]
        else:
            historical_messages = []
            raw_recent_messages = list(all_messages)

        # Update running summary with historical messages if any are newly evicted
        if historical_messages:
            self._update_running_summary(historical_messages)

        # 1. Build composite system instruction incorporating running summary and scratchpad
        system_components = []
        if system_prompt:
            system_components.append(system_prompt)

        if self._running_summary:
            system_components.append(
                f"[RECURSIVE DIALOGUE SUMMARY]\n{self._running_summary}"
            )

        if scratchpad and scratchpad.state.current_goal:
            scratch_dict = scratchpad.to_dict()
            system_components.append(
                f"[ACTIVE SCRATCHPAD WORKING STATE]\n{json.dumps(scratch_dict, indent=2)}"
            )

        full_system_text = "\n\n".join(system_components)
        system_tokens = max(1, len(full_system_text) // 4) if full_system_text else 0

        budget_remaining = max_context_tokens - system_tokens

        # 2. Format raw recent messages within remaining token budget
        selected_messages: List[Dict[str, Any]] = []
        accumulated_tokens = 0
        pruned_count = len(historical_messages)

        for msg in reversed(raw_recent_messages):
            msg_dict = msg.to_dict()
            msg_tokens = msg.estimated_tokens

            if accumulated_tokens + msg_tokens <= budget_remaining:
                selected_messages.insert(0, msg_dict)
                accumulated_tokens += msg_tokens
            else:
                pruned_count += 1

        return FormattedContext(
            system_instruction=full_system_text,
            messages=selected_messages,
            total_tokens=system_tokens + accumulated_tokens,
            pruned_count=pruned_count,
            strategy_name=self.name,
        )

    def _update_running_summary(self, historical_messages: List[Message]) -> None:
        """
        Synthesize historical messages into the running summary string.
        """
        summary_lines = []
        if self._running_summary:
            summary_lines.append(self._running_summary)

        for msg in historical_messages:
            role_label = msg.role.value.upper()
            snippet = msg.content[:100].replace("\n", " ")
            summary_lines.append(f"- {role_label}: {snippet}")

        raw_summary = "\n".join(summary_lines)

        # Truncate summary to max_summary_tokens bound if needed
        max_chars = self.max_summary_tokens * 4
        if len(raw_summary) > max_chars:
            self._running_summary = raw_summary[-max_chars:] + "..."
        else:
            self._running_summary = raw_summary
