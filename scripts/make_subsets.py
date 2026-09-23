"""Derive fixed, stratified subsets of the golden set.

Two subsets are written next to the full set and committed, so every run and the CI
gate score exactly the same rows:

- eval/golden/smoke.jsonl: 10 rows, used by CI on every push.
- eval/golden/eval30.jsonl: 30 rows, the fallback when the full set cannot be judged
  inside the free-tier daily budgets.

Rows are taken round-robin across categories in a fixed order, so each subset keeps the
category mix of the full set, and each carries a minimum number of tool-requiring rows so
that tool-call success is measured on more than a coin flip.

Usage:
    python scripts/make_subsets.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "eval" / "golden" / "golden_set.jsonl"

CATEGORY_ORDER = [
    "returns",
    "delivery",
    "substitutions",
    "store_ops",
    "product",
    "dark_store",
    "live_ops",
]

# name -> (size, minimum tool-requiring rows)
SUBSETS = {"smoke.jsonl": (10, 2), "eval30.jsonl": (30, 4)}


def stratified(rows: list[dict], size: int, min_tool_rows: int) -> list[dict]:
    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)

    picked: list[dict] = []
    cursor = 0
    while len(picked) < size:
        progressed = False
        for category in CATEGORY_ORDER:
            bucket = by_category[category]
            if cursor < len(bucket) and len(picked) < size:
                picked.append(bucket[cursor])
                progressed = True
        if not progressed:
            break
        cursor += 1

    spare_tools = [r for r in by_category["live_ops"] if r not in picked]
    while sum(r["requires_tool"] for r in picked) < min_tool_rows and spare_tools:
        # Swap the last knowledge-base row of the largest category for another tool row.
        counts: dict[str, int] = defaultdict(int)
        for r in picked:
            if not r["requires_tool"]:
                counts[r["category"]] += 1
        largest = max(counts, key=lambda c: counts[c])
        victim = [r for r in picked if r["category"] == largest][-1]
        picked.remove(victim)
        picked.append(spare_tools.pop(0))

    order = {row["id"]: index for index, row in enumerate(rows)}
    return sorted(picked, key=lambda row: order[row["id"]])


def main() -> None:
    rows = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line]
    for name, (size, min_tool_rows) in SUBSETS.items():
        subset = stratified(rows, size, min_tool_rows)
        target = GOLDEN.parent / name
        target.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in subset))
        counts: dict[str, int] = defaultdict(int)
        for row in subset:
            counts[row["category"]] += 1
        tools = sum(r["requires_tool"] for r in subset)
        print(f"{name}: {len(subset)} rows, {tools} tool rows, {dict(counts)}")


if __name__ == "__main__":
    main()
