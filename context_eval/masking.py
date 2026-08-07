"""
context_eval/masking.py
-----------------------
Observation / Tool Output Masking strategy for context window management.

Why this file exists:
    Agent tool executions (e.g. database lookups, hospital capacity reports, external API calls)
    frequently produce large JSON payloads or multi-line observations. Keeping raw, verbose tool
    outputs in history quickly consumes available context budgets without adding long-term value.

What problem it solves:
    Implements a selective masking strategy (`ObservationMaskingStrategy`) that compresses or
    replaces bulky historical tool observations with lightweight structural summaries (e.g.
    `[Tool Output Masked: 1,200 chars | Status: Success]`). Older or oversized tool observations
    are masked while recent user/assistant turns and critical system state (`Scratchpad`) remain intact.

How it connects to other memory modules:
    - Inherits from `BaseContextStrategy` in `context_eval.sliding_window`.
    - Scans `MessageRole.TOOL_OBSERVATION` items from `memory.short_term.ShortTermMemory`.
    - Preserves `Scratchpad` working state.
    - Benchmarked against other context management strategies in `context_eval.evaluate`.

How it fits into the overall memory architecture:
    Context Strategy 2 of 4 in the Context Window Management & Evaluation framework.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from memory.short_term import Message, MessageRole, ShortTermMemory
from memory.scratchpad import Scratchpad
from context_eval.sliding_window import BaseContextStrategy, FormattedContext

logger = logging.getLogger(__name__)


class ObservationMaskingStrategy(BaseContextStrategy):
    """
    Observation / Tool Output Masking context management strategy.

    Selectively masks or condenses historical `TOOL_OBSERVATION` messages that exceed a token/length
    threshold or fall outside the immediate execution window, drastically reducing token bloat.
    """

    def __init__(
        self,
        max_observation_chars: int = 200,
        keep_unmasked_recent: int = 2,
    ) -> None:
        """
        Initialize Observation Masking strategy.

        Args:
            max_observation_chars: Threshold character length above which older tool outputs are masked.
            keep_unmasked_recent: Number of most recent tool observations to keep unmasked regardless of size.
        """
        super().__init__(name="ObservationMasking")
        self.max_observation_chars: int = max_observation_chars
        self.keep_unmasked_recent: int = keep_unmasked_recent

    def prepare_context(
        self,
        short_term_memory: ShortTermMemory,
        scratchpad: Optional[Scratchpad] = None,
        system_prompt: str = "",
        max_context_tokens: int = 4000,
    ) -> FormattedContext:
        """
        Transform short-term memory into a formatted context payload with tool output masking.
        """
        # 1. Build composite system instruction incorporating scratchpad
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

        all_messages = short_term_memory.get_messages()

        # Identify indices of all TOOL_OBSERVATION messages to exempt the most recent `keep_unmasked_recent`
        tool_obs_indices = [
            i for i, msg in enumerate(all_messages)
            if msg.role == MessageRole.TOOL_OBSERVATION
        ]
        recent_unmasked_set = set(tool_obs_indices[-self.keep_unmasked_recent:]) if tool_obs_indices else set()

        processed_messages: List[Dict[str, Any]] = []
        accumulated_tokens = 0
        pruned_count = 0

        # 2. Process messages from newest to oldest fitting within budget
        for idx in range(len(all_messages) - 1, -1, -1):
            msg = all_messages[idx]
            msg_dict = msg.to_dict()

            # Check if this tool observation should be masked
            if (
                msg.role == MessageRole.TOOL_OBSERVATION
                and idx not in recent_unmasked_set
                and len(msg.content) > self.max_observation_chars
            ):
                tool_name = msg.tool_name or "tool"
                char_len = len(msg.content)
                summary_preview = msg.content[:60].replace("\n", " ")
                
                masked_text = (
                    f"[TOOL OBSERVATION MASKED | Tool: '{tool_name}' | "
                    f"Original Length: {char_len} chars | Preview: '{summary_preview}...']"
                )
                msg_dict["content"] = masked_text
                estimated_msg_tokens = max(1, len(masked_text) // 4)
                pruned_count += 1
            else:
                estimated_msg_tokens = msg.estimated_tokens

            if accumulated_tokens + estimated_msg_tokens <= budget_remaining:
                processed_messages.insert(0, msg_dict)
                accumulated_tokens += estimated_msg_tokens
            else:
                # Truncate older messages if context budget is exhausted
                pruned_count += 1

        return FormattedContext(
            system_instruction=full_system_text,
            messages=processed_messages,
            total_tokens=system_tokens + accumulated_tokens,
            pruned_count=pruned_count,
            strategy_name=self.name,
        )
