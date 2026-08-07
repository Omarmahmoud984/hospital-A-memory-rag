"""Domain-specific RAG evaluation questions.

These questions are designed so the evaluation framework can test factual
retrieval, multi-hop reasoning, and unanswerable query handling using the
hospital policy documents.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class EvaluationQuestion:
    question: str
    expected_answer: str
    unanswerable: bool = False


def get_questions() -> List[EvaluationQuestion]:
    return [
        EvaluationQuestion(
            question="According to the triage guidelines, what status should be assigned to a patient with severe asthma?",
            expected_answer="Admitted",
        ),
        EvaluationQuestion(
            question="What action must be taken immediately after surgical procedures in the operating room?",
            expected_answer="Mark the room as Maintenance immediately after surgical procedures.",
        ),
        EvaluationQuestion(
            question="Can an operating room status be changed to Available before sanitation verification?",
            expected_answer="No, it can only be changed to Available after full sanitation verification.",
        ),
        EvaluationQuestion(
            question="When only one ICU bed remains network-wide, what is required before reservation?",
            expected_answer="Explicit attending physician sign-off is required.",
        ),
        EvaluationQuestion(
            question="What triage level includes cardiac arrest and requires immediate ICU or surgery assignment?",
            expected_answer="Red level.",
        ),
        EvaluationQuestion(
            question="Does the hospital policy describe how to assign insurance codes for surgery?",
            expected_answer="",
            unanswerable=True,
        ),
        EvaluationQuestion(
            question="If a patient has a minor laceration, what status should they receive?",
            expected_answer="Waiting",
        ),
    ]
