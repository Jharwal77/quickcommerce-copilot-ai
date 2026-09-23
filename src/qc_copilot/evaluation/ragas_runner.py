"""RAGAS scoring with Groq as the judge and local embeddings.

Rows are scored one metric at a time so that a per-row score is available. Aggregate
means alone would hide which questions the system is failing, and the per-row
faithfulness scores are what the hallucination rate is built from.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass, field

from openai import APIConnectionError, APIStatusError, AsyncOpenAI, RateLimitError
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from qc_copilot.config import Settings, get_settings
from qc_copilot.evaluation.collect import EvalSample
from qc_copilot.llm.providers import ModelRef, parse_chain
from qc_copilot.llm.ratelimit import (
    BudgetExhaustedError,
    is_daily_limit,
    is_out_of_credit,
    parse_wait_hint,
)

logger = logging.getLogger(__name__)

METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


@dataclass
class RowScores:
    row_id: str
    scores: dict[str, float | None] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    judges: dict[str, str] = field(default_factory=dict)


def build_judge(settings: Settings, ref: ModelRef):
    """Every provider here speaks the OpenAI wire format, which the RAGAS adapter patches."""
    key = settings.api_keys().get(ref.provider)
    if not key:
        raise RuntimeError(f"no API key configured for judge provider {ref.provider}")
    # SDK retries are disabled so that backoff and failover are handled in one place.
    client = AsyncOpenAI(base_url=ref.base_url, api_key=key, max_retries=0)
    extra = (
        {"reasoning_effort": settings.groq_reasoning_effort}
        if ref.supports_reasoning_effort
        else {}
    )
    # Temperature 0 keeps the judge as repeatable as the API allows, which matters
    # because a CI gate compares its scores against a fixed threshold.
    return llm_factory(ref.model, provider="openai", client=client, temperature=0.0, **extra)


class JudgeChain:
    """A metric bound to an ordered list of judge models, advanced on budget exhaustion.

    Metric instances are created lazily per judge so an unused fallback costs nothing.
    Every recorded score carries the judge that produced it, because a fallback judge is
    a different instrument and the report has to say so.
    """

    def __init__(self, name: str, settings: Settings, refs: list[ModelRef], embeddings) -> None:
        if not refs:
            raise RuntimeError(f"judge chain for {name} has no usable model")
        self.name = name
        self.settings = settings
        self.refs = refs
        self.embeddings = embeddings
        self.index = 0
        self.throttles = [Throttle() for _ in refs]
        self._metrics: dict[int, object] = {}

    @property
    def judge(self) -> ModelRef:
        return self.refs[self.index]

    @property
    def throttle(self) -> Throttle:
        return self.throttles[self.index]

    @property
    def exhausted(self) -> bool:
        return self.index >= len(self.refs)

    def metric(self):
        if self.index not in self._metrics:
            llm = build_judge(self.settings, self.judge)
            factory = {
                "faithfulness": lambda: Faithfulness(llm=llm),
                "answer_relevancy": lambda: AnswerRelevancy(llm=llm, embeddings=self.embeddings),
                "context_precision": lambda: ContextPrecision(llm=llm),
                "context_recall": lambda: ContextRecall(llm=llm),
            }[self.name]
            metric = factory()
            if self.settings.eval_lean_prompts:
                slim_prompts({self.name: metric})
            self._metrics[self.index] = metric
        return self._metrics[self.index]

    def fail_over(self, reason: str) -> bool:
        previous = self.judge.key
        self.index += 1
        if self.exhausted:
            logger.error("judge chain for %s exhausted after %s: %s", self.name, previous, reason)
            return False
        logger.warning(
            "judge chain for %s: %s -> %s (%s)", self.name, previous, self.judge.key, reason
        )
        return True


def build_metrics(settings: Settings) -> dict[str, JudgeChain]:
    refs = parse_chain(settings.judge_chain, settings.api_keys())
    embeddings = HuggingFaceEmbeddings(
        model=settings.embedding_model, use_api=False, device=settings.embedding_device
    )
    return {name: JudgeChain(name, settings, list(refs), embeddings) for name in METRIC_NAMES}


PROMPT_ATTRIBUTES = ("prompt", "statement_generator_prompt", "nli_statement_prompt")


def slim_prompts(metrics: dict) -> int:
    """Remove few-shot demonstrations from every RAGAS prompt; return how many."""
    stripped = 0
    for metric in metrics.values():
        for attribute in PROMPT_ATTRIBUTES:
            prompt = getattr(metric, attribute, None)
            if prompt is not None and hasattr(prompt, "examples"):
                prompt.examples = []
                stripped += 1
    return stripped


class JudgeError(RuntimeError):
    """Raised when a metric could not be scored after every retry."""


class Throttle:
    """Shared cooldown for every task that talks to the same judge model.

    When one call learns the model's token budget is exhausted, every other call to
    that model should wait too. Without this, concurrent tasks each burn a retry
    attempt on the same exhausted budget.
    """

    def __init__(self) -> None:
        self.resume_at = 0.0

    async def wait(self) -> None:
        delay = self.resume_at - asyncio.get_running_loop().time()
        if delay > 0:
            await asyncio.sleep(delay)

    def hold(self, seconds: float) -> None:
        now = asyncio.get_running_loop().time()
        self.resume_at = max(self.resume_at, now + seconds)


def _chain(exc: BaseException):
    """Walk an exception and its causes. Structured-output wrappers re-raise API errors."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _retry_after_seconds(exc: BaseException) -> float | None:
    for item in _chain(exc):
        response = getattr(item, "response", None)
        header = response.headers.get("retry-after") if response is not None else None
        if header:
            try:
                return float(header)
            except ValueError:
                pass
        # Groq puts the precise wait in the message body rather than in a header.
        hint = parse_wait_hint(str(item))
        if hint is not None:
            return hint
    return None


def _daily_limit(exc: BaseException) -> bool:
    return any(is_daily_limit(str(item)) or is_out_of_credit(item) for item in _chain(exc))


def _is_transient(exc: BaseException) -> bool:
    for item in _chain(exc):
        if isinstance(item, RateLimitError | APIConnectionError):
            return True
        if isinstance(item, APIStatusError) and item.status_code >= 500:
            return True
        if "rate_limit_exceeded" in str(item) or "Error code: 429" in str(item):
            return True
    return False


async def _with_retry(
    coro_factory,
    attempts: int,
    label: str,
    throttle: Throttle | None = None,
    max_wait: float = 90.0,
):
    """Backoff on 429s and 5xx; stop at once when the daily budget is gone.

    Groq enforces a per-model tokens-per-minute budget, so a short 429 is expected under
    load and simply means "wait". A wait of minutes means the tokens-per-day budget is
    exhausted, and sleeping through that would stall the run for hours. Progress is
    checkpointed, so stopping and resuming later loses nothing.
    """
    for attempt in range(1, attempts + 2):
        if throttle is not None:
            await throttle.wait()
        try:
            return await coro_factory()
        except Exception as exc:
            if _daily_limit(exc):
                reason = (
                    "provider credit exhausted"
                    if any(is_out_of_credit(item) for item in _chain(exc))
                    else "daily budget exhausted"
                )
                raise BudgetExhaustedError(f"{label}: {reason}") from exc
            if not _is_transient(exc):
                raise
            wait = _retry_after_seconds(exc) or min(2.0**attempt, 60.0)
            wait += random.uniform(0.0, 1.0)
            if _daily_limit(exc) or wait > max_wait:
                raise BudgetExhaustedError(
                    f"{label}: provider asked for a {wait:.0f}s wait; daily budget exhausted"
                ) from exc
            if attempt > attempts:
                raise
            if throttle is not None:
                throttle.hold(wait)
            logger.warning(
                "%s attempt %d hit %s, retrying in %.1fs", label, attempt, type(exc).__name__, wait
            )
            await asyncio.sleep(wait)
    raise RuntimeError("unreachable")


def average_precision(verdicts: list[int]) -> float:
    """RAGAS's context precision formula: precision@k averaged over the useful passages."""
    if not verdicts:
        return 0.0
    numerator = sum((sum(verdicts[: i + 1]) / (i + 1)) * verdicts[i] for i in range(len(verdicts)))
    return numerator / (sum(verdicts) + 1e-10)


async def _context_precision(
    sample: EvalSample, chain: JudgeChain, attempts: int, max_wait: float
) -> float:
    """Score one passage per judge call so a rate limit retries one passage, not five.

    RAGAS scores the passages in sequence inside a single call and gives up on the
    whole row if any one of them is rate limited. Five passages of few-shot prompt do
    not fit in one free-tier token window, so that never converged.
    """
    verdicts: list[int] = []
    for index, context in enumerate(sample.contexts):
        outcome = await _judge_call(
            chain,
            lambda metric, ctx=context: metric.ascore(
                user_input=sample.question, reference=sample.ground_truth, retrieved_contexts=[ctx]
            ),
            attempts,
            f"{sample.row_id}/context_precision[{index}]",
            max_wait,
        )
        verdicts.append(1 if float(outcome.value) >= 0.5 else 0)
    return average_precision(verdicts)


async def _judge_call(chain: JudgeChain, make_call, attempts: int, label: str, max_wait: float):
    """Run one judge call, moving down the chain when the current judge is exhausted."""
    while not chain.exhausted:
        metric = chain.metric()
        try:
            return await _with_retry(
                lambda m=metric: make_call(m), attempts, label, chain.throttle, max_wait
            )
        except BudgetExhaustedError as exc:
            if not chain.fail_over(str(exc)):
                raise
        except Exception as exc:
            # A judge still rate limited after every retry is as good as exhausted; a
            # non-transient error means this judge cannot score at all. Either way the
            # next judge in the chain gets the call rather than the row losing its score.
            reason = (
                f"{type(exc).__name__} after retries"
                if _is_transient(exc)
                else f"{type(exc).__name__}: {exc}"
            )
            if not chain.fail_over(f"{label}: {reason}"):
                raise BudgetExhaustedError(
                    f"{label}: every judge in the chain failed; last error {reason}"
                ) from exc
    raise BudgetExhaustedError(f"{label}: every judge in the chain is exhausted")


ScoreCallback = Callable[[str, str, float, str], None]


async def _score_one(
    sample: EvalSample,
    chains: dict[str, JudgeChain],
    attempts: int,
    known: dict[str, float],
    on_score: ScoreCallback,
    max_wait: float,
) -> RowScores:
    result = RowScores(row_id=sample.row_id, scores=dict(known))

    def record(name: str, value: float, judge: str = "") -> None:
        result.scores[name] = value
        result.judges[name] = judge
        on_score(sample.row_id, name, value, judge)

    # Without retrieved context there is nothing to be faithful to and nothing to
    # measure precision or recall against, so those are zero by definition rather
    # than by judgement. Relevancy is still meaningful: a refusal is a response.
    has_context = bool(sample.contexts) and not sample.error

    async def run(name: str, make_call):
        if name in known:
            return
        chain = chains[name]
        try:
            outcome = await _judge_call(
                chain, make_call, attempts, f"{sample.row_id}/{name}", max_wait
            )
            record(name, float(outcome.value), chain.judge.key)
        except BudgetExhaustedError:
            raise
        except Exception as exc:
            result.scores[name] = None
            result.errors[name] = f"{type(exc).__name__}: {exc}"
            logger.error("%s/%s gave up after retries: %s", sample.row_id, name, exc)

    if has_context:
        await run(
            "faithfulness",
            lambda metric: metric.ascore(
                user_input=sample.question,
                response=sample.answer,
                retrieved_contexts=sample.contexts,
            ),
        )
        if "context_precision" not in known:
            chain = chains["context_precision"]
            try:
                record(
                    "context_precision",
                    await _context_precision(sample, chain, attempts, max_wait),
                    chain.judge.key,
                )
            except BudgetExhaustedError:
                raise
            except Exception as exc:
                result.scores["context_precision"] = None
                result.errors["context_precision"] = f"{type(exc).__name__}: {exc}"
                logger.error("%s/context_precision gave up after retries: %s", sample.row_id, exc)
        await run(
            "context_recall",
            lambda metric: metric.ascore(
                user_input=sample.question,
                retrieved_contexts=sample.contexts,
                reference=sample.ground_truth,
            ),
        )
    else:
        for name in ("faithfulness", "context_precision", "context_recall"):
            if name not in known:
                record(name, 0.0)

    if sample.answer and not sample.error:
        await run(
            "answer_relevancy",
            lambda metric: metric.ascore(
                user_input=sample.question,
                response=sample.answer,
            ),
        )
    elif "answer_relevancy" not in known:
        record("answer_relevancy", 0.0)

    return result


def _ignore(row_id: str, metric: str, value: float, judge: str) -> None:
    return None


async def score_samples(
    samples: list[EvalSample],
    settings: Settings | None = None,
    known: dict[str, dict[str, float]] | None = None,
    on_score: ScoreCallback = _ignore,
) -> list[RowScores]:
    """Score every sample, skipping metrics already present in `known`.

    Raises BudgetExhaustedError as soon as any judge reports its daily budget is gone.
    Scores recorded before that point have already been handed to `on_score`.
    """
    settings = settings or get_settings()
    known = known or {}
    chains = build_metrics(settings)
    semaphore = asyncio.Semaphore(settings.eval_concurrency)
    completed = 0
    results: list[RowScores] = []

    async def worker(sample: EvalSample) -> None:
        nonlocal completed
        async with semaphore:
            scores = await _score_one(
                sample,
                chains,
                settings.eval_max_retries,
                known.get(sample.row_id, {}),
                on_score,
                settings.eval_max_wait_seconds,
            )
            results.append(scores)
            completed += 1
            logger.info("scored %d/%d (%s)", completed, len(samples), sample.row_id)

    # A TaskGroup cancels the remaining rows the moment one of them hits the daily cap.
    async with asyncio.TaskGroup() as group:
        for sample in samples:
            group.create_task(worker(sample))
    return results


def raise_on_failures(rows: list[RowScores]) -> None:
    """A row the judge could not score must never quietly vanish from the mean."""
    failed = {r.row_id: r.errors for r in rows if r.errors}
    if failed:
        detail = "; ".join(
            f"{rid}: " + ", ".join(f"{metric}={err}" for metric, err in errs.items())
            for rid, errs in failed.items()
        )
        raise JudgeError(f"{len(failed)} row(s) could not be scored: {detail}")


def aggregate(rows: list[RowScores]) -> dict[str, float | None]:
    """Mean of each metric across the rows the judge could actually score."""
    summary: dict[str, float | None] = {}
    for name in METRIC_NAMES:
        values = [r.scores.get(name) for r in rows]
        usable = [v for v in values if v is not None]
        summary[name] = round(sum(usable) / len(usable), 4) if usable else None
    return summary


def faithfulness_by_row(rows: list[RowScores]) -> dict[str, float | None]:
    return {r.row_id: r.scores.get("faithfulness") for r in rows}


def judge_usage(rows: list[RowScores]) -> dict[str, dict[str, int]]:
    """metric -> judge model -> number of rows it scored."""
    usage: dict[str, dict[str, int]] = {}
    for row in rows:
        for metric, judge in row.judges.items():
            if judge:
                usage.setdefault(metric, {})[judge] = usage.setdefault(metric, {}).get(judge, 0) + 1
    return usage
