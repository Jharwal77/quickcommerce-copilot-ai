"""Loader and schema for the golden question and answer set.

The golden set is a curated artifact, not a generated one. It is validated on load so
that a typo in a document id or a duplicated question fails the evaluation run rather
than quietly skewing a score.
"""

from __future__ import annotations

import json
from collections import Counter
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from qc_copilot.config import PROJECT_ROOT

GOLDEN_SET_PATH = PROJECT_ROOT / "eval" / "golden" / "golden_set.jsonl"

EXPECTED_ROW_COUNT = 60


class GoldenCategory(StrEnum):
    RETURNS = "returns"
    DELIVERY = "delivery"
    SUBSTITUTIONS = "substitutions"
    STORE_OPS = "store_ops"
    PRODUCT = "product"
    DARK_STORE = "dark_store"
    LIVE_OPS = "live_ops"


class ToolName(StrEnum):
    GET_INVENTORY = "get_inventory"
    SEARCH_CATALOG = "search_catalog"


class GoldenRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    question: str = Field(min_length=8)
    ground_truth: str = Field(min_length=8)
    category: GoldenCategory
    requires_tool: bool = False
    expected_tool: ToolName | None = None
    expected_doc_ids: list[str] = Field(default_factory=list)

    @field_validator("question", "ground_truth")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()


class GoldenSetError(ValueError):
    """Raised when the golden set on disk is not internally consistent."""


def _validate(rows: list[GoldenRow]) -> None:
    duplicate_ids = [rid for rid, n in Counter(r.id for r in rows).items() if n > 1]
    if duplicate_ids:
        raise GoldenSetError(f"duplicate golden row ids: {sorted(duplicate_ids)}")

    duplicate_questions = [q for q, n in Counter(r.question.lower() for r in rows).items() if n > 1]
    if duplicate_questions:
        raise GoldenSetError(f"duplicate golden questions: {duplicate_questions}")

    for row in rows:
        # A row that needs a live lookup must name the tool, and must not claim a
        # knowledge-base document as its source, because live state is never indexed.
        if row.requires_tool:
            if row.expected_tool is None:
                raise GoldenSetError(f"{row.id} requires a tool but names none")
            if row.expected_doc_ids:
                raise GoldenSetError(f"{row.id} requires a tool but also expects documents")
        else:
            if row.expected_tool is not None:
                raise GoldenSetError(f"{row.id} names a tool but is not marked requires_tool")
            if not row.expected_doc_ids:
                raise GoldenSetError(f"{row.id} is retrieval-only but expects no documents")


def load_golden_set(path: Path | None = None, strict_count: bool = False) -> list[GoldenRow]:
    path = path or GOLDEN_SET_PATH
    if not path.exists():
        raise GoldenSetError(f"golden set not found at {path}")

    rows: list[GoldenRow] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(GoldenRow.model_validate(json.loads(line)))
        except Exception as exc:
            raise GoldenSetError(f"{path.name} line {line_number}: {exc}") from exc

    if not rows:
        raise GoldenSetError(f"golden set at {path} is empty")

    _validate(rows)

    if strict_count and len(rows) != EXPECTED_ROW_COUNT:
        raise GoldenSetError(f"expected {EXPECTED_ROW_COUNT} golden rows, found {len(rows)}")

    return rows


def category_counts(rows: list[GoldenRow]) -> dict[str, int]:
    return dict(sorted(Counter(row.category.value for row in rows).items()))
