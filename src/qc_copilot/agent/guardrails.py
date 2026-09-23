"""Deterministic output guardrails applied after the model has finished.

The reflection step asks a model whether the draft is supported. This module does not
ask anything: a sentence either carries a marker that points at real evidence or it is
removed. The two checks are complementary. Reflection catches a wrong number next to a
valid marker; citation enforcement catches a confident sentence with no marker at all.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

from qc_copilot.rag.prompts import REFUSAL_TEXT

_MARKER = re.compile(r"\[(\d{1,2})\]")
# Models write markers in several shapes: [1], 【1】, [1, 2], [1][2]. Everything is
# normalised to the plain [n] form before any check runs.
_FULLWIDTH_MARKER = re.compile(r"\u3010\s*(\d{1,2})\s*\u3011")
_GROUPED_MARKER = re.compile(r"\[\s*(\d{1,2}(?:\s*,\s*\d{1,2})+)\s*\]")
_SPACED_MARKER = re.compile(r"\[\s+(\d{1,2})\s+\]|\[\s+(\d{1,2})\]|\[(\d{1,2})\s+\]")


def normalize_markers(text: str) -> str:
    """Rewrite every citation marker variant to the canonical [n] form."""
    text = _FULLWIDTH_MARKER.sub(r"[\1]", text)
    text = _GROUPED_MARKER.sub(
        lambda m: "".join(f"[{n.strip()}]" for n in m.group(1).split(",")), text
    )
    return _SPACED_MARKER.sub(lambda m: f"[{next(g for g in m.groups() if g)}]", text)


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[\"'*(])")
# Lines that only introduce a list, such as "The store has:", carry no claim of their own.
_INTRO_LINE = re.compile(r"^[^.!?]*:\s*$")


class CitationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    answer: str
    refused: bool
    kept: int = Field(ge=0)
    dropped_sentences: list[str] = Field(default_factory=list)
    removed_markers: list[int] = Field(default_factory=list)


_MARKERS_ONLY = re.compile(r"^(\s*\[\d{1,2}\]\s*[.,;]?)+\s*$")


def split_units(text: str) -> list[str]:
    """Split into citable units: each line is split further into sentences.

    Models often place the marker after the full stop or on its own line. A unit made
    only of markers belongs to the sentence before it, so it is merged back rather
    than counted as a cited unit in its own right.
    """
    units: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        for part in _SENTENCE_BOUNDARY.split(stripped):
            part = part.strip()
            if not part:
                continue
            if units and _MARKERS_ONLY.match(part):
                units[-1] = f"{units[-1]} {part}"
            else:
                units.append(part)
    return units


def _markers(text: str) -> list[int]:
    return [int(m) for m in _MARKER.findall(text)]


def enforce_citations(answer: str, valid_markers: set[int]) -> CitationReport:
    """Keep only sentences backed by a valid marker; refuse if nothing survives.

    Markers that point at evidence which does not exist are stripped rather than
    trusted. A sentence whose only markers were invalid is treated as uncited.
    """
    text = normalize_markers(answer).strip()
    if not text or text.startswith(REFUSAL_TEXT):
        return CitationReport(answer=REFUSAL_TEXT, refused=True, kept=0)

    kept_units: list[str] = []
    dropped: list[str] = []
    removed: list[int] = []
    pending_intro: str | None = None

    for unit in split_units(text):
        markers = _markers(unit)
        invalid = [m for m in markers if m not in valid_markers]
        cleaned = unit
        for marker in invalid:
            removed.append(marker)
            cleaned = cleaned.replace(f"[{marker}]", "").strip()
        cleaned = re.sub(r"\s{2,}", " ", cleaned)

        if _INTRO_LINE.match(cleaned) and not markers:
            # Hold the intro line; it is only kept if a cited unit follows it.
            pending_intro = cleaned
            continue

        if any(m in valid_markers for m in markers):
            if pending_intro is not None:
                kept_units.append(pending_intro)
                pending_intro = None
            kept_units.append(cleaned)
        else:
            dropped.append(unit)
            pending_intro = None

    # Markers with no words around them are not an answer.
    if not any(re.search(r"[A-Za-z]", unit) for unit in kept_units):
        kept_units = []

    if not kept_units:
        return CitationReport(
            answer=REFUSAL_TEXT,
            refused=True,
            kept=0,
            dropped_sentences=dropped,
            removed_markers=sorted(set(removed)),
        )

    return CitationReport(
        answer="\n".join(kept_units) if "\n" in text else " ".join(kept_units),
        refused=False,
        kept=len(kept_units),
        dropped_sentences=dropped,
        removed_markers=sorted(set(removed)),
    )
