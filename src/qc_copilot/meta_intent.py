"""Deterministic detection of questions about the assistant itself.

"What can I ask?" has no answer in the knowledge base and no tool can fetch one, so
without this check the citation guardrail would refuse it as out of scope. Matching is
anchored and capped at a few words so that ordinary questions which happen to start the
same way ("what do you know about the returns window?") still go through the agent loop.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence

from qc_copilot.models import AskResponse, TraceStep, Usage

MAX_WORDS = 10

_CONTRACTIONS = re.compile(r"\b(what|how|who|where)'s\b")
_NON_WORD = re.compile(r"[^a-z0-9 ]+")
_PREFIXES = ("hi ", "hello ", "hey ", "please ", "ok ", "okay ", "so ", "um ", "hmm ")
_SUFFIXES = (" please", " here", " today", " exactly", " then", " thanks", " thank you")

_EXACT = {
    "help",
    "help me",
    "i need help",
    "menu",
    "capabilities",
    "examples",
    "example questions",
    "instructions",
    "usage",
    "start",
    "what now",
}

_SUBJECT = (
    r"(this|it|you|this tool|this app|this assistant|this thing"
    r"|the (tool|app|assistant|copilot|agent))"
)

_PATTERNS = [
    re.compile(p)
    for p in (
        rf"what (can|could|should|do|may) i ask( you| here| about| {_SUBJECT})?",
        r"what (kinds? of |sorts? of |types? of )?questions? (can|could|should|do) (i|you) "
        r"(ask|answer|handle|take)",
        r"what (can|could|do) you (do|help( me)? with|answer|handle|cover|tell me|know)"
        r"( for me| here)?",
        r"what are you (able to do|good at|for|capable of)",
        rf"what is {_SUBJECT}",
        r"what is your (scope|purpose|job|role|knowledge base|knowledge)",
        rf"what (does|do) {_SUBJECT} (do|know|cover)",
        r"what (do|did) you know( about)?",
        r"what (topics|areas|policies|data|information|stores|products) (do|can) you "
        r"(cover|know|know about|have|answer)",
        r"what are your (capabilities|limits|limitations|sources)",
        rf"how (does|do) {_SUBJECT} work",
        rf"how (do|can|should) i use {_SUBJECT}",
        rf"how (does|do) {_SUBJECT} (answer|decide|retrieve|find answers)",
        r"who are you",
        r"tell me (about yourself|what you (can do|know))",
        r"show me (some )?(examples?|what you can do|sample questions)",
        r"(can|could|will) you help( me)?",
        r"what is in (your|the) knowledge base",
        r"where (does|do) (your|the) (answers|data|information) come from",
    )
]

SYSTEM_TRACE_LABEL = "System response: capability summary"


def normalize(question: str) -> str:
    text = question.strip().lower().replace("’", "'")
    text = _CONTRACTIONS.sub(r"\1 is", text)
    text = _NON_WORD.sub(" ", text)
    text = " ".join(text.split())
    changed = True
    while changed:
        changed = False
        for prefix in _PREFIXES:
            if text.startswith(prefix):
                text = text[len(prefix) :]
                changed = True
        for suffix in _SUFFIXES:
            if text.endswith(suffix):
                text = text[: -len(suffix)]
                changed = True
    return text.strip()


def is_meta_intent(question: str) -> bool:
    text = normalize(question)
    if not text or len(text.split()) > MAX_WORDS:
        return False
    if text in _EXACT:
        return True
    return any(pattern.fullmatch(text) for pattern in _PATTERNS)


def capability_answer(tool_names: Sequence[str]) -> str:
    tools = ", ".join(f"`{name}`" for name in tool_names)
    live = (
        f"Answered live through typed tools ({tools}) over the operations database: "
        "current availability, reserved units, and reorder points at a named dark store."
        if tool_names
        else "Live-stock tools are switched off in this mode, so stock questions are refused."
    )
    return "\n".join(
        [
            "This is a system note about what the assistant can do, not a grounded answer, "
            "so nothing here is cited.",
            "",
            "**1. Policy and product questions.** Answered from the knowledge base of operator "
            "policies (returns, delivery, substitutions, dark-store operations) and product "
            "documents; every sentence cites the passage it came from.",
            "",
            f"**2. Live stock at a dark store.** {live}",
            "",
            "**3. Refusing when neither applies.** If nothing retrieved or returned by a tool "
            "supports an answer, the assistant says so instead of guessing, and the trace shows "
            "what it checked first.",
            "",
            "The **What I know** panel lists the policies, dark stores, brands, and categories "
            "loaded from the data, and the example chips under the composer show one question "
            "of each kind. Everything here is synthetic and describes a fictional operator.",
        ]
    )


def capability_response(question: str, tool_names: Sequence[str]) -> AskResponse:
    started = time.perf_counter()
    answer = capability_answer(tool_names)
    return AskResponse(
        question=question,
        answer=answer,
        trace=[
            TraceStep(
                kind="answer",
                label=SYSTEM_TRACE_LABEL,
                detail=(
                    "Matched a question about the assistant itself before the agent loop ran: "
                    "no retrieval, no tool calls, no model call."
                ),
            )
        ],
        mode="system",
        refused=False,
        usage=Usage(total_ms=(time.perf_counter() - started) * 1000),
    )
