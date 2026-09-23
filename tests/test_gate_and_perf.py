import json

import pytest

from qc_copilot.evaluation.gate import answer_model, evaluate_gate
from qc_copilot.evaluation.perf import perf_block, update_perf_file


def _report(
    label="x",
    faithfulness=0.9,
    tool_success=1.0,
    errors=0,
    kb_rows=50,
    scored=50,
    wrong_refusal=0.0,
    models=None,
):
    return {
        "label": label,
        "config": {"mode": "agentic", "answer_chain": "m"},
        "dataset": {
            "rows": 60,
            "knowledge_base_rows": kb_rows,
            "collection_errors": errors,
            "answer_models": models
            if models is not None
            else {"groq:a": 60, "reflect=gemini:r": 60},
        },
        "knowledge_base": {
            "ragas": {"faithfulness": faithfulness},
            "hallucination": {"scored_answers": scored},
            "refusals": {"wrong_refusal_rate_on_kb_rows": wrong_refusal},
        },
        "tool_rows": {"tool_call_success_rate": tool_success},
        "latency": {"p50_ms": 800.0, "p95_ms": 2100.0, "mean_ms": 900.0, "max_ms": 3000.0},
        "cost": {"queries_measured": 60, "total_tokens": 60000, "cost_per_1k_queries_usd": 0.12},
    }


def test_gate_passes_within_the_margin():
    result = evaluate_gate(_report(faithfulness=0.90), _report(faithfulness=0.86), margin=0.05)
    assert result.passed
    assert result.threshold == 0.85


def test_gate_fails_below_the_margin():
    result = evaluate_gate(_report(faithfulness=0.90), _report(faithfulness=0.84), margin=0.05)
    assert not result.passed
    assert "below threshold 0.8500" in result.reasons[0]


def test_gate_fails_when_rows_were_lost():
    candidate = _report(faithfulness=0.95, scored=40)
    result = evaluate_gate(_report(), candidate, margin=0.05)
    assert not result.passed
    assert "only 40 of 50" in result.reasons[0]


def test_gate_counts_refused_rows_as_accounted_for():
    candidate = _report(faithfulness=0.95, scored=48, wrong_refusal=0.04)
    assert evaluate_gate(_report(), candidate, margin=0.05).passed


def test_gate_fails_on_collection_errors():
    result = evaluate_gate(_report(), _report(errors=2), margin=0.05)
    assert not result.passed
    assert "failed during answer collection" in result.reasons[0]


def test_gate_optionally_requires_tool_success():
    candidate = _report(tool_success=0.5)
    assert evaluate_gate(_report(), candidate, margin=0.05).passed
    result = evaluate_gate(_report(), candidate, margin=0.05, min_tool_success=0.8)
    assert not result.passed
    assert "tool-call success" in result.reasons[0]


def test_gate_fails_when_candidate_has_no_score():
    candidate = _report()
    candidate["knowledge_base"]["ragas"]["faithfulness"] = None
    assert not evaluate_gate(_report(), candidate, margin=0.05).passed


def test_gate_rejects_a_baseline_without_a_score():
    baseline = _report()
    baseline["knowledge_base"]["ragas"]["faithfulness"] = None
    with pytest.raises(ValueError):
        evaluate_gate(baseline, _report(), margin=0.05)


def test_perf_block_summarises_latency_and_cost():
    block = perf_block(_report(label="baseline"))
    assert block["latency_ms"]["p95"] == 2100.0
    assert block["tokens_per_query"] == 1000.0
    assert block["cost_per_1k_queries_usd"] == 0.12
    assert block["source_report"] == "baseline"


def test_perf_file_merges_labels(tmp_path):
    out = tmp_path / "perf.json"
    update_perf_file(_report(label="baseline"), out)
    merged = update_perf_file(_report(label="after"), out)
    assert set(merged) == {"baseline", "after"}
    assert set(json.loads(out.read_text())) == {"baseline", "after"}


def test_gate_can_require_no_tool_success_regression_against_the_baseline():
    baseline = _report(tool_success=1.0)
    assert not evaluate_gate(
        baseline, _report(tool_success=0.5), margin=0.05, tool_success_from_baseline=True
    ).passed
    assert evaluate_gate(
        baseline, _report(tool_success=1.0), margin=0.05, tool_success_from_baseline=True
    ).passed
    # The explicit floor and the baseline floor combine; the stricter one applies.
    result = evaluate_gate(
        _report(tool_success=0.5),
        _report(tool_success=0.6),
        margin=0.05,
        min_tool_success=0.8,
        tool_success_from_baseline=True,
    )
    assert "below 0.8000" in result.reasons[0]


def test_answer_model_is_the_dominant_non_reflect_entry():
    report = _report(models={"groq:a": 3, "gemini:b": 7, "reflect=gemini:r": 10})
    assert answer_model(report) == "gemini:b"
    assert answer_model(_report(models={})) is None
    assert answer_model({"dataset": {}}) is None


def test_gate_is_not_enforced_when_the_answer_model_differs():
    baseline = _report(models={"groq:a": 60, "reflect=gemini:r": 60})
    fallen_through = _report(faithfulness=0.80, models={"gemini:b": 60, "reflect=gemini:r": 60})
    result = evaluate_gate(
        baseline, fallen_through, margin=0.05, enforce_only_on_baseline_model=True
    )
    assert not result.passed and not result.enforced
    assert result.baseline_model == "groq:a" and result.candidate_model == "gemini:b"
    assert not result.same_answer_model


def test_gate_stays_enforced_when_the_answer_model_matches():
    result = evaluate_gate(
        _report(), _report(faithfulness=0.80), margin=0.05, enforce_only_on_baseline_model=True
    )
    assert not result.passed and result.enforced


def test_gate_passing_on_a_different_model_is_still_a_pass():
    result = evaluate_gate(
        _report(),
        _report(faithfulness=0.9, models={"gemini:b": 60}),
        margin=0.05,
        enforce_only_on_baseline_model=True,
    )
    assert result.passed and result.enforced


def test_gate_enforces_by_default_even_when_models_differ():
    result = evaluate_gate(
        _report(), _report(faithfulness=0.80, models={"gemini:b": 60}), margin=0.05
    )
    assert not result.passed and result.enforced
