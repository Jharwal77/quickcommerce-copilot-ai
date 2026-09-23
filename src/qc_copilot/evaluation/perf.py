"""Extract latency and cost figures from an evaluation report into reports/perf.json.

Usage:
    python -m qc_copilot.evaluation.perf eval/results/baseline.json \
        [--out reports/perf.json]

The output file holds one block per report label so the README can quote baseline
and improved figures side by side.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from qc_copilot.config import PROJECT_ROOT

DEFAULT_OUT = PROJECT_ROOT / "reports" / "perf.json"


def perf_block(report: dict) -> dict:
    cost = report["cost"]
    latency = report["latency"]
    queries = cost["queries_measured"] or 1
    return {
        "mode": report["config"]["mode"],
        "answer_chain": report["config"]["answer_chain"],
        "queries_measured": cost["queries_measured"],
        "latency_ms": {
            "p50": latency["p50_ms"],
            "p95": latency["p95_ms"],
            "mean": latency["mean_ms"],
            "max": latency["max_ms"],
        },
        "tokens_per_query": round(cost["total_tokens"] / queries, 1),
        "cost_per_1k_queries_usd": cost["cost_per_1k_queries_usd"],
        "source_report": report["label"],
        "recorded_at": datetime.now(UTC).isoformat(),
    }


def update_perf_file(report: dict, out: Path) -> dict:
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing[report["label"]] = perf_block(report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(existing, indent=2) + "\n")
    return existing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    merged = update_perf_file(json.loads(args.report.read_text()), args.out)
    block = merged[json.loads(args.report.read_text())["label"]]
    print(
        f"{block['source_report']}: p95 {block['latency_ms']['p95']} ms, "
        f"${block['cost_per_1k_queries_usd']} per 1k queries -> {args.out}"
    )


if __name__ == "__main__":
    main()
