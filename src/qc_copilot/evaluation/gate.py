"""CI gate: fail when faithfulness drops too far below the committed baseline.

Usage:
    python -m qc_copilot.evaluation.gate --baseline eval/results/baseline.json \
        --candidate eval/results/ci.json --margin 0.05 \
        [--min-tool-success 0.8] [--tool-success-from-baseline]
        [--enforce-only-on-baseline-model]

The margin is a fixed absolute allowance below the baseline mean. A judge is not
perfectly repeatable, so a gate at exactly the baseline would flake; a gate five
points below it catches real regressions without failing on noise.

Exit codes: 0 pass, 1 fail, 3 not enforced. The last one is used when the candidate's
answers came from a different model than the baseline's because a provider was capped:
the comparison is still printed, but a gap between two models is not a regression in
the code under test, so CI reports it as a warning rather than a failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

NOT_ENFORCED_EXIT = 3


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reasons: list[str]
    threshold: float
    candidate_faithfulness: float | None
    baseline_faithfulness: float
    baseline_model: str | None = None
    candidate_model: str | None = None
    enforced: bool = True

    @property
    def same_answer_model(self) -> bool:
        return self.baseline_model == self.candidate_model


def _faithfulness(report: dict) -> float | None:
    return report["knowledge_base"]["ragas"].get("faithfulness")


def answer_model(report: dict) -> str | None:
    """The model that produced most of a report's answers, ignoring reflection entries."""
    counts = report.get("dataset", {}).get("answer_models") or {}
    answers = {k: v for k, v in counts.items() if not k.startswith("reflect=")}
    if not answers:
        return None
    return max(sorted(answers), key=answers.__getitem__)


def evaluate_gate(
    baseline: dict,
    candidate: dict,
    margin: float,
    min_tool_success: float | None = None,
    tool_success_from_baseline: bool = False,
    enforce_only_on_baseline_model: bool = False,
) -> GateResult:
    base = _faithfulness(baseline)
    if base is None:
        raise ValueError("baseline report has no faithfulness score")
    threshold = round(base - margin, 4)
    reasons: list[str] = []

    cand = _faithfulness(candidate)
    if cand is None:
        reasons.append("candidate has no faithfulness score")
    elif cand < threshold:
        reasons.append(f"faithfulness {cand:.4f} is below threshold {threshold:.4f}")

    errors = candidate["dataset"].get("collection_errors", 0)
    if errors:
        reasons.append(f"{errors} row(s) failed during answer collection")

    expected_rows = candidate["dataset"]["knowledge_base_rows"]
    scored_rows = candidate["knowledge_base"]["hallucination"].get("scored_answers", 0)
    refused = round(
        candidate["knowledge_base"]["refusals"]["wrong_refusal_rate_on_kb_rows"] * expected_rows
    )
    if scored_rows + refused < expected_rows:
        reasons.append(
            f"only {scored_rows + refused} of {expected_rows} knowledge-base rows were scored"
        )

    # Faithfulness alone is a weak gate when the baseline already sits at the ceiling, so
    # tool-call success must not regress against the reference report either.
    if tool_success_from_baseline:
        floor = baseline["tool_rows"]["tool_call_success_rate"]
        min_tool_success = max(min_tool_success or 0.0, floor)
    if min_tool_success is not None:
        rate = candidate["tool_rows"]["tool_call_success_rate"]
        if rate < min_tool_success:
            reasons.append(f"tool-call success {rate:.4f} is below {min_tool_success:.4f}")

    base_model = answer_model(baseline)
    cand_model = answer_model(candidate)
    enforced = not (enforce_only_on_baseline_model and reasons and base_model != cand_model)
    return GateResult(
        passed=not reasons,
        reasons=reasons,
        threshold=threshold,
        candidate_faithfulness=cand,
        baseline_faithfulness=base,
        baseline_model=base_model,
        candidate_model=cand_model,
        enforced=enforced,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--min-tool-success", type=float, default=None)
    parser.add_argument(
        "--tool-success-from-baseline",
        action="store_true",
        help="Also require tool-call success at or above the baseline report's rate.",
    )
    parser.add_argument(
        "--enforce-only-on-baseline-model",
        action="store_true",
        help="Report failures as a warning (exit 3) when the candidate's answers came from "
        "a different model than the baseline's, as happens after a provider fallback.",
    )
    args = parser.parse_args()

    result = evaluate_gate(
        json.loads(args.baseline.read_text()),
        json.loads(args.candidate.read_text()),
        args.margin,
        args.min_tool_success,
        args.tool_success_from_baseline,
        args.enforce_only_on_baseline_model,
    )
    print(
        f"baseline faithfulness {result.baseline_faithfulness:.4f}, "
        f"threshold {result.threshold:.4f}, candidate "
        f"{result.candidate_faithfulness if result.candidate_faithfulness is not None else 'n/a'}"
    )
    print(
        f"answer model: baseline {result.baseline_model or 'unknown'}, "
        f"candidate {result.candidate_model or 'unknown'}"
    )
    if result.passed:
        print("gate: PASS")
        return
    if not result.enforced:
        for reason in result.reasons:
            print(f"gate: WARNING - {reason}")
        print(
            "gate: NOT ENFORCED - candidate answers came from "
            f"{result.candidate_model or 'unknown'} but the baseline was answered by "
            f"{result.baseline_model or 'unknown'}; a gap between two models after a provider "
            "fallback is not a regression in the code under test"
        )
        sys.exit(NOT_ENFORCED_EXIT)
    for reason in result.reasons:
        print(f"gate: FAIL - {reason}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
