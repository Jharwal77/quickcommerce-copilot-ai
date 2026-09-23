"""Run the full evaluation and write a JSON report.

The two halves of the golden set are scored differently and reported separately.
Knowledge-base rows get RAGAS scores from the judge. Tool rows get a deterministic
tool-call success rate, because a live inventory number is either fetched or invented
and no judge is needed to tell which.

Usage:
    python -m qc_copilot.evaluation.run --label baseline --out eval/results/baseline.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from qc_copilot import __version__
from qc_copilot.agent.pipeline import AgentPipeline, build_pipeline
from qc_copilot.config import PROJECT_ROOT, get_settings
from qc_copilot.evaluation.checkpoint import Checkpoint
from qc_copilot.evaluation.collect import EvalSample, collect_samples
from qc_copilot.evaluation.golden import GOLDEN_SET_PATH, load_golden_set
from qc_copilot.evaluation.metrics import (
    HALLUCINATION_FAITHFULNESS_THRESHOLD,
    citation_summary,
    cost_summary,
    hallucination_rate,
    latency_summary,
    refusal_summary,
    retrieval_hit_rate,
    tool_call_summary,
)
from qc_copilot.evaluation.ragas_runner import (
    METRIC_NAMES,
    JudgeError,
    RowScores,
    aggregate,
    faithfulness_by_row,
    judge_usage,
    raise_on_failures,
    score_samples,
)
from qc_copilot.llm.ratelimit import BudgetExhaustedError

logger = logging.getLogger(__name__)


def build_report(
    label: str,
    samples: list[EvalSample],
    scored: list[RowScores],
    expected_docs: dict[str, list[str]],
    elapsed_seconds: float,
) -> dict:
    settings = get_settings()
    kb_samples = [s for s in samples if not s.requires_tool]
    tool_samples = [s for s in samples if s.requires_tool]
    scores_by_row = {r.row_id: r for r in scored}

    return {
        "label": label,
        "created_at": datetime.now(UTC).isoformat(),
        "app_version": __version__,
        "config": {
            "mode": samples[0].mode if samples else "retrieval",
            "answer_chain": settings.answer_chain,
            "reflect_chain": settings.reflect_chain,
            "judge_chain": settings.judge_chain,
            "embedding_model": settings.embedding_model,
            "retrieval_top_k": settings.retrieval_top_k,
            "chunk_tokens": settings.chunk_tokens,
            "chunk_overlap_tokens": settings.chunk_overlap_tokens,
        },
        "dataset": {
            "rows": len(samples),
            "knowledge_base_rows": len(kb_samples),
            "tool_rows": len(tool_samples),
            "collection_errors": sum(1 for s in samples if s.error),
            "answer_models": _model_counts(samples),
        },
        "knowledge_base": {
            "ragas": aggregate(scored),
            "judges": judge_usage(scored),
            "hallucination": {
                **hallucination_rate(kb_samples, faithfulness_by_row(scored)),
                "faithfulness_threshold": HALLUCINATION_FAITHFULNESS_THRESHOLD,
            },
            "citations": citation_summary(kb_samples),
            "refusals": refusal_summary(kb_samples),
            "retrieval": {"expected_doc_hit_rate": retrieval_hit_rate(kb_samples, expected_docs)},
        },
        "tool_rows": tool_call_summary(tool_samples),
        "latency": latency_summary(samples),
        "cost": cost_summary(samples),
        "elapsed_seconds": round(elapsed_seconds, 1),
        "per_row": [
            {
                "row_id": s.row_id,
                "category": s.category,
                "requires_tool": s.requires_tool,
                "expected_tool": s.expected_tool,
                "refused": s.refused,
                "citation_count": s.citation_count,
                "retrieved_doc_ids": s.retrieved_doc_ids,
                "tool_calls": [tc.model_dump() for tc in s.tool_calls],
                "total_ms": round(s.usage.total_ms, 1),
                "total_tokens": s.usage.total_tokens,
                "scores": (
                    {k: scores_by_row[s.row_id].scores.get(k) for k in METRIC_NAMES}
                    if s.row_id in scores_by_row
                    else {}
                ),
                "answer": s.answer,
                "error": s.error,
            }
            for s in samples
        ],
    }


def _model_counts(samples: list[EvalSample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        for model in sample.usage.models:
            counts[model] = counts.get(model, 0) + 1
    return dict(sorted(counts.items()))


def _from_checkpoint(checkpoint: Checkpoint) -> list[RowScores]:
    return [
        RowScores(
            row_id=rid,
            scores={m: v for m, (v, _) in metrics.items()},
            judges={m: j for m, (_, j) in metrics.items()},
        )
        for rid, metrics in checkpoint.load_entries().items()
    ]


def print_summary(report: dict) -> None:
    kb = report["knowledge_base"]
    tools = report["tool_rows"]
    print(f"\n{report['label']}: {report['dataset']['rows']} rows")
    print(f"\nKnowledge-base rows ({report['dataset']['knowledge_base_rows']}), RAGAS via judge")
    print("-" * 52)
    for name, value in kb["ragas"].items():
        print(f"  {name:<26} {value if value is not None else 'n/a'}")
    print(f"  {'hallucination_rate':<26} {kb['hallucination']['hallucination_rate']}")
    print(f"  {'citation_coverage':<26} {kb['citations']['citation_coverage']}")
    print(f"  {'wrong_refusal_rate':<26} {kb['refusals']['wrong_refusal_rate_on_kb_rows']}")
    print(f"  {'expected_doc_hit_rate':<26} {kb['retrieval']['expected_doc_hit_rate']}")
    for metric, judges in kb["judges"].items():
        print(f"  judges/{metric:<19} {judges}")
    print(f"\nTool rows ({tools['rows']}), deterministic")
    print("-" * 52)
    print(f"  {'tool_call_success_rate':<26} {tools['tool_call_success_rate']}")
    print(f"  {'refused':<26} {tools['refused']}")
    print(f"  {'answered_without_tool':<26} {tools['answered_without_tool']}")
    print("\nAll rows")
    print("-" * 52)
    print(f"  {'p95_latency_ms':<26} {report['latency']['p95_ms']}")
    print(f"  {'cost_per_1k_queries_usd':<26} {report['cost']['cost_per_1k_queries_usd']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="baseline", help="Name recorded in the report.")
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "eval" / "results" / "baseline.json",
        help="Where to write the JSON report.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Use only the first N rows, for smoke tests."
    )
    parser.add_argument(
        "--skip-judge", action="store_true", help="Collect answers but do not run the judge."
    )
    parser.add_argument(
        "--mode",
        choices=["retrieval", "agentic"],
        default=None,
        help="Override AGENT_MODE for this run.",
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=GOLDEN_SET_PATH,
        help="Golden set to evaluate (full set, a stratified subset, or the smoke set).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Discard the checkpoint for this label and start over.",
    )
    parser.add_argument(
        "--recollect-category",
        default=None,
        help=(
            "Drop the checkpointed answers and scores for one golden category and collect "
            "that whole stratum again, for example after a provider failover degraded it."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    rows = load_golden_set(
        args.golden, strict_count=args.golden == GOLDEN_SET_PATH and not args.limit
    )
    if args.limit:
        rows = rows[: args.limit]

    settings = get_settings()
    if args.mode:
        settings = settings.model_copy(update={"agent_mode": args.mode})

    # Checkpoints are keyed by label. Resuming is the default because a free-tier run
    # routinely spans more than one daily token budget.
    checkpoint = Checkpoint(args.label)
    if args.fresh:
        checkpoint.reset()
    if args.recollect_category:
        dropped = checkpoint.drop_category(args.recollect_category)
        print(f"recollecting {dropped} rows in category {args.recollect_category}")
    cached_samples = checkpoint.load_samples()
    if cached_samples:
        print(f"resuming: {len(cached_samples)} answers already collected")

    started = time.perf_counter()
    stopped_early: BudgetExhaustedError | None = None
    samples: list[EvalSample] = []
    if all(row.id in cached_samples for row in rows):
        # Keep the golden set's order and ignore checkpointed rows outside this set.
        samples = [cached_samples[row.id] for row in rows]
    else:
        pipeline = build_pipeline(settings)
        try:
            samples = collect_samples(pipeline, rows, cached_samples, checkpoint.save_sample)
            samples = [s for s in samples if s.row_id in {row.id for row in rows}]
        except BudgetExhaustedError as exc:
            stopped_early = exc
        finally:
            if isinstance(pipeline, AgentPipeline):
                pipeline.close()

    scored: list[RowScores] = []
    judge_error: JudgeError | None = None
    if stopped_early is None and not args.skip_judge:
        known = checkpoint.load_scores()
        if known:
            print(f"resuming: {sum(len(v) for v in known.values())} scores already recorded")
        kb_samples = [s for s in samples if not s.requires_tool]
        try:
            # Only knowledge-base rows go to the judge. Tool rows are scored deterministically.
            scored = asyncio.run(score_samples(kb_samples, settings, known, checkpoint.save_score))
        except* BudgetExhaustedError as group:
            stopped_early = group.exceptions[0]
            scored = _from_checkpoint(checkpoint)
        # Rows resumed from the checkpoint carry their judge attribution on disk only.
        entries = checkpoint.load_entries()
        for row in scored:
            row.judges.update({m: j for m, (_, j) in entries.get(row.row_id, {}).items() if j})
        if stopped_early is None:
            try:
                raise_on_failures(scored)
            except JudgeError as exc:
                judge_error = exc

    if stopped_early is not None:
        # The checkpoint may hold rows from a larger golden set; count only this run's rows.
        wanted = {row.id for row in rows}
        collected = sum(1 for rid in checkpoint.load_samples() if rid in wanted)
        scored_rows = sum(
            1
            for rid, v in checkpoint.load_scores().items()
            if rid in wanted and len(v) == len(METRIC_NAMES)
        )
        kb_total = sum(1 for r in rows if not r.requires_tool)
        print(
            f"\nSTOPPED: {stopped_early}\n"
            f"progress saved under {checkpoint.directory}: {collected}/{len(rows)} answers "
            f"collected, {scored_rows}/{kb_total} knowledge-base rows fully scored, "
            f"{len(rows) - collected} answers and {kb_total - scored_rows} judged rows remain. "
            "Re-run the same command to resume.",
            file=sys.stderr,
        )
        sys.exit(2)

    expected_docs = {row.id: row.expected_doc_ids for row in rows}
    report = build_report(args.label, samples, scored, expected_docs, time.perf_counter() - started)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print_summary(report)
    print(f"\nwrote {args.out}")

    if judge_error is not None:
        # The report is still written for inspection, but the run is a failure.
        print(f"\nERROR: {judge_error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
