import json

from qc_copilot.api.meta import eval_summary


def _report(label, mode, tool_rate, faithfulness):
    return {
        "label": label,
        "config": {"mode": mode},
        "dataset": {"rows": 30, "tool_rows": 4},
        "knowledge_base": {
            "ragas": {"faithfulness": faithfulness, "answer_relevancy": 0.9},
            "citations": {"citation_coverage": 1.0},
            "hallucination": {"hallucination_rate": 0.0},
        },
        "tool_rows": {"tool_call_success_rate": tool_rate},
        "latency": {"p95_ms": 1234.5},
        "created_at": "2026-09-09T00:00:00+00:00",
    }


def test_eval_summary_reports_only_the_pairs_present(tmp_path):
    (tmp_path / "baseline_30.json").write_text(
        json.dumps(_report("baseline", "retrieval", 0.0, 1.0))
    )
    summary = eval_summary(tmp_path)
    assert set(summary) == {"subset_30"}
    assert set(summary["subset_30"]) == {"baseline"}
    assert summary["subset_30"]["baseline"]["tool_call_success_rate"] == 0.0

    (tmp_path / "after_30.json").write_text(json.dumps(_report("after", "agentic", 1.0, 0.98)))
    summary = eval_summary(tmp_path)
    assert summary["subset_30"]["after"]["faithfulness"] == 0.98
    assert summary["subset_30"]["after"]["p95_latency_ms"] == 1234.5


def test_eval_summary_is_empty_without_reports(tmp_path):
    assert eval_summary(tmp_path) == {}
