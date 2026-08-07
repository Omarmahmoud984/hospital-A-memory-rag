"""
context_eval/zone_pruning.py
----------------------------
Zone-Based Pruning strategy for context window management.

Why this file exists:
    Homogeneous context truncation treats all historical elements equally. However, different
    parts of a prompt have vastly different functional importance: system instructions and
    scratchpad state must never be lost, current user instructions must stay intact, while
    older intermediate tool outputs can be aggressively pruned.

What problem it solves:
    Implements `ZonePruningStrategy`, which partitions prompt context into 4 priority zones:
        - Zone A (System & Scratchpad): Protected / Never Pruned.
        - Zone B (Latest Turn): Protected / High Priority.
        - Zone C (Recent Dialogue): Medium Priority (Lightly trimmed if needed).
        - Zone D (Historical Dialogue & Tool Observations): Low Priority (First to be pruned).

How it connects to other memory modules:
    - Inherits from `BaseContextStrategy` in `context_eval.sliding_window`.
    - Segregates messages from `memory.short_term.ShortTermMemory`.
    - Protects `memory.scratchpad.Scratchpad` in Zone A.
    - Benchmarked against all other strategies in `context_eval.evaluate`.

How it fits into the overall memory architecture:
    Context Strategy 4 of 4 in the Context Window Management & Evaluation framework.
"""

from enum import Enum
import json
import logging
from typing import Any, Dict, List, Optional

from memory.short_term import Message, MessageRole, ShortTermMemory
from memory.scratchpad import Scratchpad
from context_eval.sliding_window import BaseContextStrategy, FormattedContext

logger = logging.getLogger(__name__)


class PriorityZone(str, Enum):
    """Priority zones for context allocation."""
    ZONE_A_SYSTEM_SCRATCHPAD = "zone_a"  # System Prompt & Scratchpad (Never pruned)
    ZONE_B_LATEST_TURN = "zone_b"         # Latest User query / Assistant prompt (Protected)
    ZONE_C_RECENT_DIALOGUE = "zone_c"     # Recent turns & active tool headers (Medium priority)
    ZONE_D_HISTORICAL_TOOL_OBS = "zone_d" # Older dialogue & raw tool outputs (Low priority / First to prune)


class ZonePruningStrategy(BaseContextStrategy):
    """
    Zone-Based Pruning context management strategy.

    Allocates token budget by priority zones, ensuring critical system instructions and active
    user requests are never evicted while low-priority historical tool observations are shed first.
    """

    def __init__(
        self,
        recent_turns_count: int = 3,
        historical_obs_max_chars: int = 150,
    ) -> None:
        """
        Initialize Zone-Based Pruning strategy.

        Args:
            recent_turns_count: Number of turns designated for Zone C (Recent Dialogue).
            historical_obs_max_chars: Character limit for Zone D tool outputs before pruning.
        """
        super().__init__(name="ZoneBasedPruning")
        self.recent_turns_count: int = recent_turns_count
        self.historical_obs_max_chars: int = historical_obs_max_chars

    def prepare_context(
        self,
        short_term_memory: ShortTermMemory,
        scratchpad: Optional[Scratchpad] = None,
        system_prompt: str = "",
        max_context_tokens: int = 4000,
    ) -> FormattedContext:
        """
        Transform short-term memory into a formatted context payload using zone-based priority allocation.
        """
        # --- ZONE A: System Prompt & Scratchpad (Highest Priority) ---
        system_components = []
        if system_prompt:
            system_components.append(system_prompt)

        if scratchpad and scratchpad.state.current_goal:
            scratch_dict = scratchpad.to_dict()
            system_components.append(
                f"[ACTIVE SCRATCHPAD WORKING STATE]\n{json.dumps(scratch_dict, indent=2)}"
            )

        full_system_text = "\n\n".join(system_components)
        zone_a_tokens = max(1, len(full_system_text) // 4) if full_system_text else 0

        budget_remaining = max_context_tokens - zone_a_tokens
        if budget_remaining < 0:
            logger.warning("Zone A exceeds total max_context_tokens budget!")
            budget_remaining = 100

        all_messages = short_term_memory.get_messages()
        if not all_messages:
            return FormattedContext(
                system_instruction=full_system_text,
                messages=[],
                total_tokens=zone_a_tokens,
                pruned_count=0,
                strategy_name=self.name,
            )

        # --- ZONE B: Latest Turn (High Priority) ---
        latest_message = all_messages[-1]
        zone_b_messages: List[Message] = [latest_message]
        remaining_messages = all_messages[:-1]

        # --- ZONE C: Recent Dialogue & ZONE D: Historical Tool Observations ---
        if len(remaining_messages) > self.recent_turns_count:
            zone_c_messages = remaining_messages[-self.recent_turns_count:]
            zone_d_messages = remaining_messages[:-self.recent_turns_count]
        else:
            zone_c_messages = list(remaining_messages)
            zone_d_messages = []

        formatted_output_messages: List[Dict[str, Any]] = []
        accumulated_tokens = 0
        pruned_count = 0

        # Step 1: Always include Zone B (Latest Turn)
        zone_b_tokens = latest_message.estimated_tokens
        formatted_output_messages.append(latest_message.to_dict())
        accumulated_tokens += zone_b_tokens
        budget_remaining -= zone_b_tokens

        # Step 2: Include Zone C (Recent Dialogue)
        for msg in reversed(zone_c_messages):
            msg_tokens = msg.estimated_tokens
            if accumulated_tokens + msg_tokens <= (max_context_tokens - zone_a_tokens):
                formatted_output_messages.insert(0, msg.to_dict())
                accumulated_tokens += msg_tokens
            else:
                pruned_count += 1

        # Step 3: Include Zone D (Historical Dialogue & Tool Observations) with aggressive pruning
        for msg in reversed(zone_d_messages):
            msg_dict = msg.to_dict()
            if msg.role == MessageRole.TOOL_OBSERVATION and len(msg.content) > self.historical_obs_max_chars:
                # Zone D Pruning: Mask oversized historical observations
                msg_dict["content"] = f"[ZONE D PRUNED | Tool: {msg.tool_name or 'tool'}]"
                msg_tokens = max(1, len(msg_dict["content"]) // 4)
                pruned_count += 1
            else:
                msg_tokens = msg.estimated_tokens

            if accumulated_tokens + msg_tokens <= (max_context_tokens - zone_a_tokens):
                formatted_output_messages.insert(0, msg_dict)
                accumulated_tokens += msg_tokens
            else:
                pruned_count += 1

        return FormattedContext(
            system_instruction=full_system_text,
            messages=formatted_output_messages,
            total_tokens=zone_a_tokens + accumulated_tokens,
            pruned_count=pruned_count,
            strategy_name=self.name,
        )
