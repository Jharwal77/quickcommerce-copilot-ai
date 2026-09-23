"""Run the assistant over the golden set and capture what it produced.

Collection is separated from scoring so that a single set of answers can be scored
more than once without paying for generation again.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field

from qc_copilot.evaluation.golden import GoldenRow
from qc_copilot.llm.ratelimit import BudgetExhaustedError
from qc_copilot.models import AskRequest, ToolCall, Usage

logger = logging.getLogger(__name__)


class EvalSample(BaseModel):
    """One golden row paired with the answer the system actually gave."""

    model_config = ConfigDict(frozen=True)

    row_id: str
    question: str
    ground_truth: str
    category: str
    requires_tool: bool
    expected_tool: str | None = None
    answer: str
    contexts: list[str] = Field(default_factory=list)
    retrieved_doc_ids: list[str] = Field(default_factory=list)
    cited_doc_ids: list[str] = Field(default_factory=list)
    citation_count: int = 0
    tool_calls: list[ToolCall] = Field(default_factory=list)
    refused: bool = False
    mode: str = "retrieval"
    usage: Usage = Field(default_factory=Usage)
    error: str | None = None


SampleCallback = Callable[[EvalSample], None]


def _keep(sample: EvalSample) -> None:
    return None


def collect_samples(
    pipeline,
    rows: list[GoldenRow],
    existing: dict[str, EvalSample] | None = None,
    on_sample: SampleCallback = _keep,
) -> list[EvalSample]:
    """Answer every golden row not already in `existing`, checkpointing each answer.

    Ordinary failures are recorded on the sample so the run continues. A daily budget
    failure propagates, because nothing further can be collected today.
    """
    existing = existing or {}
    samples: list[EvalSample] = []

    for index, row in enumerate(rows, start=1):
        if row.id in existing:
            samples.append(existing[row.id])
            continue
        try:
            response = pipeline.answer(AskRequest(question=row.question))
        except BudgetExhaustedError:
            raise
        except Exception as exc:
            logger.warning("row %s failed: %s", row.id, exc)
            sample = EvalSample(
                row_id=row.id,
                question=row.question,
                ground_truth=row.ground_truth,
                category=row.category.value,
                requires_tool=row.requires_tool,
                expected_tool=row.expected_tool.value if row.expected_tool else None,
                answer="",
                error=f"{type(exc).__name__}: {exc}",
            )
            samples.append(sample)
            on_sample(sample)
            continue

        sample = EvalSample(
            row_id=row.id,
            question=row.question,
            ground_truth=row.ground_truth,
            category=row.category.value,
            requires_tool=row.requires_tool,
            expected_tool=row.expected_tool.value if row.expected_tool else None,
            answer=response.answer,
            contexts=list(response.contexts),
            retrieved_doc_ids=list(response.retrieved_doc_ids),
            cited_doc_ids=sorted({c.doc_id for c in response.citations}),
            citation_count=len(response.citations),
            tool_calls=list(response.tool_calls),
            refused=response.refused,
            mode=response.mode,
            usage=response.usage,
        )
        samples.append(sample)
        on_sample(sample)
        logger.info("collected %d/%d (%s)", index, len(rows), row.id)

    return samples
