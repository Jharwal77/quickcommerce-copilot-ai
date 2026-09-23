import pytest

from qc_copilot.evaluation.collect import EvalSample
from qc_copilot.evaluation.metrics import (
    citation_summary,
    cost_summary,
    hallucination_rate,
    latency_summary,
    percentile,
    refusal_summary,
    retrieval_hit_rate,
)
from qc_copilot.models import Usage


def _sample(
    row_id: str,
    *,
    requires_tool: bool = False,
    refused: bool = False,
    citations: int = 1,
    total_ms: float = 100.0,
    prompt_tokens: int = 100,
    completion_tokens: int = 20,
    cost: float = 0.0001,
    retrieved: list[str] | None = None,
    error: str | None = None,
) -> EvalSample:
    return EvalSample(
        row_id=row_id,
        question="A golden question?",
        ground_truth="A golden answer.",
        category="returns",
        requires_tool=requires_tool,
        answer="" if error else "An answer [1].",
        contexts=[] if refused else ["some context"],
        retrieved_doc_ids=retrieved or ["policy-returns"],
        citation_count=citations,
        refused=refused,
        error=error,
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            total_ms=total_ms,
            estimated_cost_usd=cost,
        ),
    )


def test_percentile_handles_edges():
    assert percentile([], 0.95) == 0.0
    assert percentile([7.0], 0.95) == 7.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.0) == 1.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 1.0) == 4.0


def test_percentile_picks_the_nearest_rank():
    values = [float(v) for v in range(1, 101)]
    assert percentile(values, 0.5) == 51.0
    assert percentile(values, 0.95) == 95.0


def test_latency_summary_reports_p50_and_p95():
    samples = [_sample(f"r{i}", total_ms=float(i * 10)) for i in range(1, 21)]
    summary = latency_summary(samples)
    assert summary["p50_ms"] == 105.0
    assert summary["p95_ms"] == 190.0
    assert summary["max_ms"] == 200.0


def test_latency_summary_ignores_unmeasured_rows():
    assert latency_summary([_sample("r1", total_ms=0.0)])["p95_ms"] == 0.0


def test_cost_summary_scales_to_a_thousand_queries():
    samples = [_sample(f"r{i}", cost=0.001) for i in range(10)]
    summary = cost_summary(samples)
    assert summary["queries_measured"] == 10
    assert summary["total_cost_usd"] == pytest.approx(0.01)
    assert summary["cost_per_1k_queries_usd"] == pytest.approx(1.0)


def test_cost_summary_survives_an_empty_run():
    assert cost_summary([])["cost_per_1k_queries_usd"] == 0.0


def test_citation_summary_excludes_refusals_and_errors():
    samples = [
        _sample("a", citations=2),
        _sample("b", citations=0),
        _sample("c", refused=True, citations=0),
        _sample("d", error="boom", citations=0),
    ]
    summary = citation_summary(samples)
    assert summary["committed_answers"] == 2
    assert summary["citation_coverage"] == 0.5
    assert summary["uncited_answer_rate"] == 0.5


def test_citation_summary_survives_all_refusals():
    assert citation_summary([_sample("a", refused=True)])["citation_coverage"] == 0.0


def test_refusal_summary_separates_correct_from_wrong_refusals():
    samples = [
        _sample("live-1", requires_tool=True, refused=True),
        _sample("live-2", requires_tool=True, refused=False),
        _sample("ret-1", refused=False),
        _sample("ret-2", refused=True),
    ]
    summary = refusal_summary(samples)
    assert summary["refusal_rate"] == 0.5
    assert summary["correct_refusal_rate_on_tool_rows"] == 0.5
    assert summary["wrong_refusal_rate_on_kb_rows"] == 0.5


def test_retrieval_hit_rate_counts_any_expected_document():
    samples = [
        _sample("a", retrieved=["policy-returns", "product-x"]),
        _sample("b", retrieved=["product-y"]),
    ]
    expected = {"a": ["policy-returns"], "b": ["policy-delivery"]}
    assert retrieval_hit_rate(samples, expected) == 0.5


def test_retrieval_hit_rate_ignores_tool_rows():
    samples = [_sample("live-1", requires_tool=True, retrieved=[])]
    assert retrieval_hit_rate(samples, {"live-1": []}) == 0.0


def test_hallucination_rate_flags_low_faithfulness_answers():
    samples = [_sample("a"), _sample("b"), _sample("c")]
    faithfulness = {"a": 0.9, "b": 0.2, "c": 0.45}
    result = hallucination_rate(samples, faithfulness)
    assert result["scored_answers"] == 3
    assert result["hallucination_rate"] == pytest.approx(0.6667, abs=1e-4)
    assert result["hallucinated_row_ids"] == ["b", "c"]


def test_refusals_are_never_counted_as_hallucinations():
    samples = [_sample("a", refused=True), _sample("b")]
    result = hallucination_rate(samples, {"a": 0.0, "b": 1.0})
    assert result["scored_answers"] == 1
    assert result["hallucination_rate"] == 0.0


def test_unscored_rows_are_excluded_from_the_hallucination_rate():
    samples = [_sample("a"), _sample("b")]
    result = hallucination_rate(samples, {"a": None, "b": 0.1})
    assert result["scored_answers"] == 1
    assert result["hallucination_rate"] == 1.0


def test_hallucination_rate_survives_no_scored_rows():
    assert hallucination_rate([], {})["hallucination_rate"] == 0.0
