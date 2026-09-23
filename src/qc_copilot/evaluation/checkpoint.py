"""On-disk checkpoints so an evaluation can stop and resume without losing work.

Every collected answer and every judge score is appended to a file the moment it
exists. A run that dies on a daily token budget picks up exactly where it stopped, and
re-scoring never pays for generation again.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from qc_copilot.config import PROJECT_ROOT
from qc_copilot.evaluation.collect import EvalSample

CACHE_ROOT = PROJECT_ROOT / "eval" / "cache"


class Checkpoint:
    def __init__(self, label: str, root: Path = CACHE_ROOT) -> None:
        self.directory = root / label
        self.samples_path = self.directory / "samples.jsonl"
        self.scores_path = self.directory / "scores.jsonl"

    def reset(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def load_samples(self) -> dict[str, EvalSample]:
        if not self.samples_path.exists():
            return {}
        samples = {}
        for line in self.samples_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                sample = EvalSample.model_validate_json(line)
                samples[sample.row_id] = sample
        return samples

    def save_sample(self, sample: EvalSample) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.samples_path.open("a", encoding="utf-8") as handle:
            handle.write(sample.model_dump_json() + "\n")

    def load_scores(self) -> dict[str, dict[str, float]]:
        return {
            rid: {m: v for m, (v, _) in metrics.items()}
            for rid, metrics in self.load_entries().items()
        }

    def load_entries(self) -> dict[str, dict[str, tuple[float, str]]]:
        """row_id -> metric -> (value, judge model key)."""
        if not self.scores_path.exists():
            return {}
        entries: dict[str, dict[str, tuple[float, str]]] = {}
        for line in self.scores_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                entries.setdefault(entry["row_id"], {})[entry["metric"]] = (
                    entry["value"],
                    entry.get("judge", ""),
                )
        return entries

    def save_score(self, row_id: str, metric: str, value: float, judge: str = "") -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.scores_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps({"row_id": row_id, "metric": metric, "value": value, "judge": judge})
                + "\n"
            )
            handle.flush()

    def drop_category(self, category: str) -> int:
        """Remove every answer and score for one category; returns rows removed."""
        samples = self.load_samples()
        keep = {rid: s for rid, s in samples.items() if s.category != category}
        dropped = set(samples) - set(keep)
        if not dropped:
            return 0
        self.samples_path.write_text(
            "".join(s.model_dump_json() + "\n" for s in keep.values()), encoding="utf-8"
        )
        if self.scores_path.exists():
            lines = [
                line
                for line in self.scores_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and json.loads(line)["row_id"] not in dropped
            ]
            self.scores_path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
        return len(dropped)
