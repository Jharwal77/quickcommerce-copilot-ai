import asyncio
from types import SimpleNamespace

import pytest

from qc_copilot.evaluation.checkpoint import Checkpoint
from qc_copilot.evaluation.collect import EvalSample, collect_samples
from qc_copilot.evaluation.golden import GoldenCategory, GoldenRow
from qc_copilot.evaluation.ragas_runner import (
    Throttle,
    _score_one,
    _with_retry,
    slim_prompts,
)
from qc_copilot.llm.ratelimit import BudgetExhaustedError, is_daily_limit, parse_wait_hint
from qc_copilot.models import AskRequest, AskResponse


def _sample(row_id: str, requires_tool: bool = False) -> EvalSample:
    return EvalSample(
        row_id=row_id,
        question="What is the dairy window?",
        ground_truth="Two hours.",
        category="returns",
        requires_tool=requires_tool,
        answer="Two hours [1].",
        contexts=["Dairy: 2 hours."],
    )


def test_checkpoint_round_trips_samples_and_scores(tmp_path):
    cp = Checkpoint("t", root=tmp_path)
    assert cp.load_samples() == {}
    assert cp.load_scores() == {}
    cp.save_sample(_sample("a"))
    cp.save_sample(_sample("b"))
    cp.save_score("a", "faithfulness", 1.0)
    cp.save_score("a", "context_recall", 0.5)
    cp.save_score("b", "faithfulness", 0.0)
    assert set(cp.load_samples()) == {"a", "b"}
    assert cp.load_samples()["a"].answer == "Two hours [1]."
    assert cp.load_scores() == {
        "a": {"faithfulness": 1.0, "context_recall": 0.5},
        "b": {"faithfulness": 0.0},
    }
    cp.reset()
    assert cp.load_samples() == {} and cp.load_scores() == {}


def test_parse_wait_hint_handles_every_groq_format():
    assert parse_wait_hint("Please try again in 8.6175s.") == pytest.approx(8.6175)
    assert parse_wait_hint("try again in 250ms") == pytest.approx(0.25)
    assert parse_wait_hint("Please try again in 7m33.6s.") == pytest.approx(453.6)
    assert parse_wait_hint("try again in 2m") == pytest.approx(120.0)
    assert parse_wait_hint("try again in 1h2m3s") == pytest.approx(3723.0)
    assert parse_wait_hint("no hint here") is None


GEMINI_MINUTE = (
    "You exceeded your current quota, please check your plan and billing details. "
    "Quota exceeded for metric: generate_content_free_tier_requests, limit: 15, "
    "quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier. Please retry in 31.2s."
)
GEMINI_DAY = (
    "You exceeded your current quota. Quota exceeded for metric: "
    "generate_content_free_tier_requests, limit: 20, "
    "quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier. Please retry in 31.2s."
)


def test_daily_limit_detection():
    assert is_daily_limit("Rate limit reached ... on tokens per day (TPD): Limit 200000")
    assert is_daily_limit(GEMINI_DAY)
    assert is_daily_limit("You exceeded your current quota, please check your plan and billing")
    assert not is_daily_limit(GEMINI_MINUTE)
    assert not is_daily_limit("Rate limit reached ... on tokens per minute (TPM): Limit 8000")


def test_gemini_retry_hint_is_parsed():
    assert parse_wait_hint(GEMINI_MINUTE) == pytest.approx(31.2)


async def test_with_retry_stops_at_once_on_a_daily_limit(monkeypatch):
    import qc_copilot.evaluation.ragas_runner as runner

    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(runner.asyncio, "sleep", fake_sleep)

    async def daily():
        raise RuntimeError(
            "Error code: 429 - rate_limit_exceeded on tokens per day (TPD): Limit 200000, "
            "Used 198691. Please try again in 7m33.6s."
        )

    with pytest.raises(BudgetExhaustedError, match="daily budget exhausted"):
        await _with_retry(daily, attempts=5, label="t", throttle=Throttle(), max_wait=90)
    assert sleeps == []


async def test_with_retry_treats_a_very_long_wait_as_exhausted(monkeypatch):
    async def long_wait():
        raise RuntimeError("Error code: 429 - rate_limit_exceeded. Please try again in 5m.")

    with pytest.raises(BudgetExhaustedError):
        await _with_retry(long_wait, attempts=5, label="t", max_wait=90)


class _FakeMetric:
    def __init__(self, value: float):
        self.value = value
        self.calls = 0

    async def ascore(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(value=self.value)


class _FakeChain:
    """Single-judge stand-in for JudgeChain."""

    def __init__(self, metric, judge: str = "gemini:fake-judge"):
        self._metric = metric
        self.judge = SimpleNamespace(key=judge)
        self.throttle = Throttle()
        self.exhausted = False
        self.failovers = 0

    def metric(self):
        return self._metric

    def fail_over(self, reason: str) -> bool:
        self.failovers += 1
        self.exhausted = True
        return False


def _metrics() -> dict:
    return {
        "faithfulness": _FakeChain(_FakeMetric(1.0)),
        "answer_relevancy": _FakeChain(_FakeMetric(0.9)),
        "context_precision": _FakeChain(_FakeMetric(1.0)),
        "context_recall": _FakeChain(_FakeMetric(0.5)),
    }


def _calls(chains: dict, name: str) -> int:
    return chains[name].metric().calls


async def test_score_one_skips_known_metrics_and_reports_new_ones():
    metrics = _metrics()
    recorded: list[tuple[str, str, float]] = []
    known = {"faithfulness": 0.25, "context_precision": 0.75}
    result = await _score_one(_sample("a"), metrics, 2, known, lambda *a: recorded.append(a), 90)
    assert result.scores["faithfulness"] == 0.25
    assert result.scores["context_precision"] == 0.75
    assert result.scores["answer_relevancy"] == 0.9
    assert result.scores["context_recall"] == 0.5
    assert _calls(metrics, "faithfulness") == 0
    assert _calls(metrics, "context_precision") == 0
    assert {m for _, m, _, _ in recorded} == {"answer_relevancy", "context_recall"}
    assert all(judge == "gemini:fake-judge" for _, _, _, judge in recorded)
    assert result.judges["context_recall"] == "gemini:fake-judge"
    assert result.errors == {}


async def test_score_one_scores_a_refusal_without_calling_context_judges():
    metrics = _metrics()
    refusal = _sample("r").model_copy(update={"contexts": [], "refused": True})
    recorded: list[tuple[str, str, float]] = []
    result = await _score_one(refusal, metrics, 2, {}, lambda *a: recorded.append(a), 90)
    assert result.scores["faithfulness"] == 0.0
    assert result.scores["context_recall"] == 0.0
    assert _calls(metrics, "faithfulness") == 0
    assert _calls(metrics, "answer_relevancy") == 1
    assert len(recorded) == 4


async def test_score_one_propagates_budget_exhaustion():
    class Exhausted:
        async def ascore(self, **kwargs):
            raise RuntimeError(
                "Error code: 429 - rate_limit_exceeded on tokens per day (TPD). "
                "Please try again in 9m."
            )

    metrics = _metrics()
    metrics["faithfulness"] = _FakeChain(Exhausted())
    with pytest.raises(BudgetExhaustedError):
        await _score_one(_sample("a"), metrics, 1, {}, lambda *a: None, 90)
    assert metrics["faithfulness"].failovers == 1


def test_slim_prompts_strips_examples_from_every_prompt_attribute():
    metrics = {
        "faithfulness": SimpleNamespace(
            statement_generator_prompt=SimpleNamespace(examples=[1, 2]),
            nli_statement_prompt=SimpleNamespace(examples=[1]),
        ),
        "context_recall": SimpleNamespace(prompt=SimpleNamespace(examples=[1, 2, 3])),
        "other": SimpleNamespace(prompt=SimpleNamespace()),
    }
    assert slim_prompts(metrics) == 3
    assert metrics["faithfulness"].nli_statement_prompt.examples == []
    assert metrics["context_recall"].prompt.examples == []


def _row(rid: str) -> GoldenRow:
    return GoldenRow(
        id=rid,
        question=f"Question number {rid}?",
        ground_truth="An answer.",
        category=GoldenCategory.RETURNS,
        expected_doc_ids=["policy-returns"],
    )


class _Pipeline:
    def __init__(self, fail_on: str | None = None):
        self.asked: list[str] = []
        self.fail_on = fail_on

    def answer(self, request: AskRequest) -> AskResponse:
        self.asked.append(request.question)
        if self.fail_on and self.fail_on in request.question:
            raise BudgetExhaustedError("daily budget exhausted")
        return AskResponse(question=request.question, answer="Fine [1].", contexts=["c"])


def test_collect_reuses_cached_answers_and_checkpoints_new_ones():
    pipeline = _Pipeline()
    cached = {"a": _sample("a")}
    saved: list[str] = []
    samples = collect_samples(
        pipeline, [_row("a"), _row("b")], cached, lambda s: saved.append(s.row_id)
    )
    assert [s.row_id for s in samples] == ["a", "b"]
    assert pipeline.asked == ["Question number b?"]
    assert saved == ["b"]


def test_collect_stops_on_budget_exhaustion_after_saving_earlier_rows():
    pipeline = _Pipeline(fail_on="c")
    saved: list[str] = []
    with pytest.raises(BudgetExhaustedError):
        collect_samples(
            pipeline, [_row("a"), _row("c"), _row("d")], {}, lambda s: saved.append(s.row_id)
        )
    assert saved == ["a"]
    assert pipeline.asked == ["Question number a?", "Question number c?"]


def test_score_samples_task_group_surfaces_budget_error(monkeypatch):
    import qc_copilot.evaluation.ragas_runner as runner

    class Exhausted:
        async def ascore(self, **kwargs):
            raise RuntimeError(
                "Error code: 429 - rate_limit_exceeded on tokens per day (TPD). "
                "Please try again in 9m."
            )

    metrics = _metrics()
    metrics["faithfulness"] = _FakeChain(Exhausted())
    monkeypatch.setattr(runner, "build_metrics", lambda settings: metrics)
    from qc_copilot.config import Settings

    with pytest.raises(ExceptionGroup) as info:
        asyncio.run(
            runner.score_samples([_sample("a"), _sample("b")], Settings(eval_max_retries=0))
        )
    assert info.group_contains(BudgetExhaustedError)


async def test_judge_call_moves_to_the_next_judge_when_one_keeps_failing():
    from qc_copilot.evaluation.ragas_runner import _judge_call

    class Broken:
        async def ascore(self, **kwargs):
            raise RuntimeError("Error code: 429 - rate_limit_exceeded. Please try again in 1s.")

    class Chain:
        def __init__(self):
            self.metrics = [Broken(), _FakeMetric(0.75)]
            self.index = 0
            self.throttle = Throttle()

        @property
        def exhausted(self):
            return self.index >= len(self.metrics)

        @property
        def judge(self):
            return SimpleNamespace(key=f"judge-{self.index}")

        def metric(self):
            return self.metrics[self.index]

        def fail_over(self, reason):
            self.index += 1
            return not self.exhausted

    import qc_copilot.evaluation.ragas_runner as runner

    async def no_sleep(seconds):
        return None

    original = runner.asyncio.sleep
    runner.asyncio.sleep = no_sleep
    try:
        chain = Chain()
        outcome = await _judge_call(chain, lambda m: m.ascore(), attempts=1, label="t", max_wait=90)
    finally:
        runner.asyncio.sleep = original
    assert outcome.value == 0.75
    assert chain.index == 1


def test_drop_category_removes_answers_and_scores_for_that_stratum(tmp_path):
    cp = Checkpoint("t", root=tmp_path)
    cp.save_sample(_sample("ret-1"))
    cp.save_sample(
        _sample("live-1", requires_tool=True).model_copy(update={"category": "live_ops"})
    )
    cp.save_score("ret-1", "faithfulness", 1.0, "j")
    cp.save_score("live-1", "faithfulness", 0.0, "j")
    assert cp.drop_category("live_ops") == 1
    assert set(cp.load_samples()) == {"ret-1"}
    assert set(cp.load_scores()) == {"ret-1"}
    assert cp.drop_category("live_ops") == 0


def test_collect_only_treats_rows_of_the_current_set_as_cached():
    """A checkpoint with more rows than the subset must not mask missing subset rows."""
    pipeline = _Pipeline()
    cached = {f"x{i}": _sample(f"x{i}") for i in range(40)}
    rows = [_row("a"), _row("b")]
    assert not all(r.id in cached for r in rows)
    samples = collect_samples(pipeline, rows, cached, lambda s: None)
    assert [s.row_id for s in samples] == ["a", "b"]
    assert len(pipeline.asked) == 2
