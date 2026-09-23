"""Metrics computed without an LLM judge: latency, cost, refusals, and citations."""

from __future__ import annotations

from statistics import median

from qc_copilot.evaluation.collect import EvalSample

# A row is counted as hallucinated when the judge's faithfulness score for it falls
# below this level while the system still committed to an answer. Refusals are never
# hallucinations, and rows the judge could not score are excluded.
HALLUCINATION_FAITHFULNESS_THRESHOLD = 0.5


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile. Deterministic and adequate for a 60-row sample."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = max(0, min(len(ordered) - 1, round(fraction * (len(ordered) - 1))))
    return ordered[rank]


def latency_summary(samples: list[EvalSample]) -> dict[str, float]:
    durations = [s.usage.total_ms for s in samples if s.usage.total_ms > 0]
    if not durations:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0, "mean_ms": 0.0}
    return {
        "p50_ms": round(median(durations), 1),
        "p95_ms": round(percentile(durations, 0.95), 1),
        "max_ms": round(max(durations), 1),
        "mean_ms": round(sum(durations) / len(durations), 1),
    }


def cost_summary(samples: list[EvalSample]) -> dict[str, float]:
    answered = [s for s in samples if s.usage.total_tokens > 0]
    total_cost = sum(s.usage.estimated_cost_usd for s in samples)
    total_tokens = sum(s.usage.total_tokens for s in samples)
    queries = len(answered) or 1
    return {
        "queries_measured": len(answered),
        "total_tokens": total_tokens,
        "mean_prompt_tokens": round(sum(s.usage.prompt_tokens for s in samples) / queries, 1),
        "mean_completion_tokens": round(
            sum(s.usage.completion_tokens for s in samples) / queries, 1
        ),
        "total_cost_usd": round(total_cost, 6),
        "cost_per_1k_queries_usd": round(total_cost / queries * 1000, 4),
    }


def citation_summary(samples: list[EvalSample]) -> dict[str, float]:
    """How often a committed answer carries at least one resolvable citation."""
    committed = [s for s in samples if not s.refused and not s.error]
    if not committed:
        return {"committed_answers": 0, "citation_coverage": 0.0, "uncited_answer_rate": 0.0}
    cited = sum(1 for s in committed if s.citation_count > 0)
    return {
        "committed_answers": len(committed),
        "citation_coverage": round(cited / len(committed), 4),
        "uncited_answer_rate": round(1 - cited / len(committed), 4),
    }


def refusal_summary(samples: list[EvalSample]) -> dict[str, float]:
    """Refusal behaviour split by whether a refusal was the right move.

    A row that needs a live tool call cannot be answered from the knowledge base, so
    refusing it is correct. A row backed by an indexed document should be answered, so
    refusing it is a miss.
    """
    total = len(samples) or 1
    tool_rows = [s for s in samples if s.requires_tool]
    kb_rows = [s for s in samples if not s.requires_tool]
    return {
        "refusal_rate": round(sum(1 for s in samples if s.refused) / total, 4),
        "correct_refusal_rate_on_tool_rows": round(
            sum(1 for s in tool_rows if s.refused) / (len(tool_rows) or 1), 4
        ),
        "wrong_refusal_rate_on_kb_rows": round(
            sum(1 for s in kb_rows if s.refused) / (len(kb_rows) or 1), 4
        ),
    }


def retrieval_hit_rate(samples: list[EvalSample], expected: dict[str, list[str]]) -> float:
    """Share of knowledge-base rows where at least one expected document was retrieved."""
    kb_rows = [s for s in samples if not s.requires_tool and expected.get(s.row_id)]
    if not kb_rows:
        return 0.0
    hits = sum(1 for s in kb_rows if set(expected[s.row_id]) & set(s.retrieved_doc_ids))
    return round(hits / len(kb_rows), 4)


def hallucination_rate(samples: list[EvalSample], faithfulness: dict[str, float | None]) -> dict:
    """Share of committed answers the judge scored as unfaithful to their context."""
    scored = [
        s
        for s in samples
        if not s.refused and not s.error and faithfulness.get(s.row_id) is not None
    ]
    if not scored:
        return {"scored_answers": 0, "hallucination_rate": 0.0, "hallucinated_row_ids": []}
    offenders = [
        s.row_id for s in scored if faithfulness[s.row_id] < HALLUCINATION_FAITHFULNESS_THRESHOLD
    ]
    return {
        "scored_answers": len(scored),
        "hallucination_rate": round(len(offenders) / len(scored), 4),
        "hallucinated_row_ids": sorted(offenders),
    }


def tool_call_summary(samples: list[EvalSample]) -> dict:
    """Deterministic scoring for rows that need a live lookup.

    A row succeeds when the expected tool was invoked, returned without error, and the
    system committed to an answer. A row answered with no tool call at all is the worst
    outcome: the number in the answer can only have been invented.

    Catalog rows are the exception. The catalog is static and is also indexed in the
    knowledge base, so a committed, cited answer from retrieval is as authoritative as
    one from search_catalog. Either path counts; a refusal or an uncited answer does not.
    """
    tool_rows = [s for s in samples if s.requires_tool]
    if not tool_rows:
        return {
            "rows": 0,
            "succeeded": 0,
            "tool_call_success_rate": 0.0,
            "refused": 0,
            "answered_without_tool": 0,
            "failed_row_ids": [],
        }
    succeeded: list[str] = []
    refused = 0
    answered_without_tool = 0
    for s in tool_rows:
        called_ok = any(tc.name == s.expected_tool and tc.ok for tc in s.tool_calls)
        cited_from_index = s.expected_tool == "search_catalog" and s.citation_count > 0
        if (called_ok or cited_from_index) and not s.refused and not s.error:
            succeeded.append(s.row_id)
        elif s.refused:
            refused += 1
        elif not s.tool_calls and not s.error:
            answered_without_tool += 1
    by_model: dict[str, dict[str, int]] = {}
    for s in tool_rows:
        answer_model = next((m for m in s.usage.models if not m.startswith("reflect=")), "unknown")
        bucket = by_model.setdefault(answer_model, {"rows": 0, "succeeded": 0})
        bucket["rows"] += 1
        bucket["succeeded"] += int(s.row_id in succeeded)
    return {
        "rows": len(tool_rows),
        "succeeded": len(succeeded),
        "tool_call_success_rate": round(len(succeeded) / len(tool_rows), 4),
        "refused": refused,
        "answered_without_tool": answered_without_tool,
        "failed_row_ids": sorted(s.row_id for s in tool_rows if s.row_id not in succeeded),
        "by_answer_model": by_model,
    }
