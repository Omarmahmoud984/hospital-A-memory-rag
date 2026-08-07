"""
context_eval/test_cases.py
--------------------------
Dataset generator for long-context evaluation test scenarios.

Why this file exists:
    Evaluating context window management strategies requires realistic, multi-turn conversation
    datasets containing heavy tool output payloads, deep turn counts, and buried needle-in-a-haystack
    facts.

What problem it solves:
    Provides programmatic dataset generators (`DatasetGenerator`) producing standardized `TestCase`
    suite objects with dozens of dialogue turns, complex JSON tool observations, and buried facts
    (e.g., patient drug allergies, preferred ICU beds, assigned doctor IDs) used to test retrieval accuracy.

How it connects to other memory modules:
    - Generates populating inputs for `memory.short_term.ShortTermMemory` and `memory.scratchpad.Scratchpad`.
    - Executed directly by the benchmark harness in `context_eval.evaluate`.

How it fits into the overall memory architecture:
    Data Generation Engine in the Context Window Management & Evaluation framework.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import json
import logging
import random
from typing import Any, Dict, List, Optional

from memory.short_term import Message, MessageRole, ShortTermMemory
from memory.scratchpad import Scratchpad, TaskStatus

logger = logging.getLogger(__name__)


@dataclass
class BuriedFact:
    """
    Represents a specific target fact hidden within a long conversation history.

    Attributes:
        fact_key: Unique key identifier for the fact (e.g., "patient_allergy").
        fact_value: Ground truth string or structured value (e.g., "Penicillin").
        turn_index: Zero-indexed turn number where the fact was introduced.
        context_hint: Clue snippet surrounding the fact.
    """
    fact_key: str
    fact_value: str
    turn_index: int
    context_hint: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)


@dataclass
class TestCase:
    """
    Complete evaluation test case containing conversation memory and target evaluation parameters.

    Attributes:
        case_id: Unique identifier string.
        name: Human-readable benchmark title.
        description: Summary of scenario goals.
        total_turns: Number of dialogue turns generated.
        short_term_memory: Pre-populated ShortTermMemory instance.
        scratchpad: Pre-populated Scratchpad instance.
        buried_facts: List of `BuriedFact` needles inserted into the conversation.
        query: Final evaluation question posed to test context recall.
        expected_answer_contains: List of substring tokens required in correct response.
    """
    case_id: str
    name: str
    description: str
    total_turns: int
    short_term_memory: ShortTermMemory
    scratchpad: Scratchpad
    buried_facts: List[BuriedFact]
    query: str
    expected_answer_contains: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize test case state."""
        return {
            "case_id": self.case_id,
            "name": self.name,
            "description": self.description,
            "total_turns": self.total_turns,
            "short_term_memory": self.short_term_memory.to_dict(),
            "scratchpad": self.scratchpad.to_dict(),
            "buried_facts": [f.to_dict() for f in self.buried_facts],
            "query": self.query,
            "expected_answer_contains": self.expected_answer_contains,
        }


class DatasetGenerator:
    """
    Generates realistic, long-context evaluation scenarios for Meridian General Hospital triage.
    """

    @staticmethod
    def generate_hospital_triage_benchmark(
        turn_count: int = 50,
        large_json_payloads: bool = True,
    ) -> TestCase:
        """
        Generate a multi-turn (50+ turns) hospital triage dataset with buried clinical facts
        and heavy JSON tool outputs.
        """
        stm = ShortTermMemory(capacity=turn_count + 10)
        scratchpad = Scratchpad()
        scratchpad.set_goal("Triage patient emergency admissions and optimize ICU bed utilization")
        scratchpad.set_execution_plan([
            "Register patient intake details",
            "Check ICU bed availability",
            "Verify allergy history and assigned doctor",
            "Create admission record",
        ])

        buried_facts: List[BuriedFact] = []

        # Turn 1-5: Initial greetings and routine hospital capacity inquiries
        stm.add_user_message("Good morning, what is the overall hospital capacity today?")
        stm.add_assistant_message("Let me fetch current hospital capacity metrics.")
        
        cap_obs = (
            {"hospital_id": 1, "name": "Meridian General", "available_icu_beds": 8, "occupancy_rate": 0.82}
            if not large_json_payloads else
            {
                "hospital_id": 1,
                "name": "Meridian General Hospital",
                "city": "Central Metro",
                "total_beds": 500,
                "occupied_beds": 410,
                "available_icu_beds": 8,
                "departments": [
                    {"name": "Emergency", "head": "Dr. Sarah Jenkins", "active_cases": 24},
                    {"name": "Cardiology", "head": "Dr. Robert Chen", "active_cases": 15},
                    {"name": "ICU", "head": "Dr. Marcus Vance", "active_cases": 12},
                ],
            }
        )
        stm.add_tool_call("get_hospital_capacity", {"hospital_id": 1})
        stm.add_tool_observation("get_hospital_capacity", cap_obs)

        # Turn 6 (BURIED FACT 1): Critical Patient Allergy
        msg_allergy = stm.add_user_message(
            "Patient John Doe (ID #101) arrived. Note: Patient has a severe, life-threatening allergy to Ceftriaxone."
        )
        buried_facts.append(
            BuriedFact(
                fact_key="patient_allergy",
                fact_value="Ceftriaxone",
                turn_index=6,
                context_hint="Patient John Doe has severe allergy to Ceftriaxone",
            )
        )
        stm.add_assistant_message("Recorded patient John Doe intake. Checking allergy flags in record.")

        # Turn 7 (BURIED FACT 2): Assigned Doctor Preference
        stm.add_user_message("Assign Dr. Emily Carter (Doctor ID #405) as primary attending physician.")
        buried_facts.append(
            BuriedFact(
                fact_key="assigned_doctor",
                fact_value="405",
                turn_index=8,
                context_hint="Assign Dr. Emily Carter Doctor ID #405",
            )
        )

        # Generate intermediate multi-turn noise & large tool JSON payloads (Turns 10 to turn_count-5)
        for i in range(10, turn_count - 5):
            stm.add_user_message(f"Status check turn #{i}: Please check operating room {i % 4 + 1} status.")
            stm.add_tool_call("update_operating_room_status", {"room_id": i % 4 + 1, "status": "Maintenance"})
            
            # Heavy multi-line JSON payload to stress token budgets
            heavy_obs = {
                "room_id": i % 4 + 1,
                "status": "Maintenance",
                "sanitation_log": [
                    {"timestamp": f"2026-08-07T0{i%9}:00:00Z", "inspector": f"Tech_{i}", "passed": True}
                    for _ in range(5)
                ],
                "equipment_status": {"ventilator": "OK", "monitor": "OK", "anesthesia_machine": "Calibrated"},
            }
            stm.add_tool_observation("update_operating_room_status", heavy_obs)
            stm.add_assistant_message(f"Operating room {i % 4 + 1} status verified as Maintenance.")

        # Turn N-4 (BURIED FACT 3): Preferred ICU Bed Allocation
        stm.add_user_message("Patient John Doe requires ICU bed placement. Preferred bed location: ICU Bed #103.")
        buried_facts.append(
            BuriedFact(
                fact_key="preferred_icu_bed",
                fact_value="ICU Bed #103",
                turn_index=turn_count - 4,
                context_hint="Preferred bed location ICU Bed #103",
            )
        )

        # Final query target
        query = (
            "What is patient John Doe's severe drug allergy, assigned doctor ID, "
            "and preferred ICU bed allocation?"
        )
        expected = ["Ceftriaxone", "405", "103"]

        return TestCase(
            case_id="triage_long_context_01",
            name="Meridian Emergency Triage 50-Turn Needle Benchmark",
            description="Stress tests context retrieval across 50 turns with heavy JSON observations and 3 buried facts.",
            total_turns=stm.size,
            short_term_memory=stm,
            scratchpad=scratchpad,
            buried_facts=buried_facts,
            query=query,
            expected_answer_contains=expected,
        )

    @classmethod
    def get_eval_suite(cls) -> List[TestCase]:
        """Return a suite of evaluation test cases."""
        return [
            cls.generate_hospital_triage_benchmark(turn_count=30, large_json_payloads=False),
            cls.generate_hospital_triage_benchmark(turn_count=50, large_json_payloads=True),
        ]
