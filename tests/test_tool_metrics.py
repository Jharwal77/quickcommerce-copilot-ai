import httpx
import pytest
from openai import APIConnectionError, APIStatusError, BadRequestError, RateLimitError

from qc_copilot.evaluation.collect import EvalSample
from qc_copilot.evaluation.metrics import tool_call_summary
from qc_copilot.evaluation.ragas_runner import (
    JudgeError,
    RowScores,
    Throttle,
    _is_transient,
    _retry_after_seconds,
    _with_retry,
    average_precision,
    raise_on_failures,
)
from qc_copilot.models import ToolCall


def _tool_sample(
    row_id: str,
    *,
    tool_calls: list[ToolCall] | None = None,
    refused: bool = False,
    error: str | None = None,
    expected: str = "get_inventory",
) -> EvalSample:
    return EvalSample(
        row_id=row_id,
        question="How many units are available at the store?",
        ground_truth="Thirty-five units are available.",
        category="live_ops",
        requires_tool=True,
        expected_tool=expected,
        answer="" if refused else "35 units are available.",
        refused=refused,
        error=error,
        tool_calls=tool_calls or [],
    )


def _call(name: str = "get_inventory", ok: bool = True) -> ToolCall:
    return ToolCall(name=name, arguments={"sku": "X"}, ok=ok, error=None if ok else "boom")


def test_success_requires_the_expected_tool_to_succeed_and_an_answer():
    samples = [
        _tool_sample("a", tool_calls=[_call()]),
        _tool_sample("b", tool_calls=[_call(ok=False)]),
        _tool_sample("c", tool_calls=[_call("search_catalog")]),
        _tool_sample("d", tool_calls=[_call()], refused=True),
    ]
    summary = tool_call_summary(samples)
    assert summary["succeeded"] == 1
    assert summary["tool_call_success_rate"] == 0.25
    assert summary["failed_row_ids"] == ["b", "c", "d"]


def test_answers_with_no_tool_call_are_counted_as_invented():
    samples = [_tool_sample("a"), _tool_sample("b", refused=True)]
    summary = tool_call_summary(samples)
    assert summary["answered_without_tool"] == 1
    assert summary["refused"] == 1
    assert summary["tool_call_success_rate"] == 0.0


def test_tool_summary_ignores_knowledge_base_rows():
    kb = EvalSample(
        row_id="ret-1",
        question="What is the dairy return window?",
        ground_truth="Two hours.",
        category="returns",
        requires_tool=False,
        answer="Two hours [1].",
    )
    assert tool_call_summary([kb])["rows"] == 0


def test_raise_on_failures_names_every_unscored_row():
    rows = [
        RowScores("ok", scores={"faithfulness": 1.0}),
        RowScores("bad", scores={"faithfulness": None}, errors={"faithfulness": "RateLimitError"}),
    ]
    with pytest.raises(JudgeError, match="bad: faithfulness=RateLimitError"):
        raise_on_failures(rows)
    raise_on_failures(rows[:1])


def _http_error(cls, status: int, headers: dict | None = None):
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status, headers=headers or {}, request=request)
    return cls("error", response=response, body=None)


def test_rate_limits_and_server_errors_are_transient_but_bad_requests_are_not():
    assert _is_transient(_http_error(RateLimitError, 429))
    assert _is_transient(_http_error(APIStatusError, 503))
    assert _is_transient(APIConnectionError(request=httpx.Request("POST", "https://x")))
    assert not _is_transient(_http_error(BadRequestError, 400))
    assert not _is_transient(ValueError("schema mismatch"))


def test_retry_after_header_is_honoured_when_present():
    assert _retry_after_seconds(_http_error(RateLimitError, 429, {"retry-after": "7"})) == 7.0
    assert _retry_after_seconds(_http_error(RateLimitError, 429)) is None
    assert _retry_after_seconds(ValueError("no response")) is None


async def test_with_retry_retries_a_rate_limit_then_succeeds(monkeypatch):
    import qc_copilot.evaluation.ragas_runner as runner

    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(runner.asyncio, "sleep", fake_sleep)
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _http_error(RateLimitError, 429, {"retry-after": "2"})
        return "done"

    assert await _with_retry(flaky, attempts=5, label="t") == "done"
    assert calls["n"] == 3
    assert len(sleeps) == 2
    assert all(2.0 <= s < 3.0 for s in sleeps)


async def test_with_retry_gives_up_after_the_configured_attempts(monkeypatch):
    import qc_copilot.evaluation.ragas_runner as runner

    async def fake_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(runner.asyncio, "sleep", fake_sleep)

    async def always_limited():
        raise _http_error(RateLimitError, 429)

    with pytest.raises(RateLimitError):
        await _with_retry(always_limited, attempts=2, label="t")


async def test_with_retry_does_not_retry_a_programming_error():
    calls = {"n": 0}

    async def broken():
        calls["n"] += 1
        raise ValueError("bad schema")

    with pytest.raises(ValueError):
        await _with_retry(broken, attempts=5, label="t")
    assert calls["n"] == 1


class _Wrapped(Exception):
    """Stands in for the structured-output library's own retry exception."""


def test_transient_detection_sees_through_a_wrapping_exception():
    inner = _http_error(RateLimitError, 429)
    try:
        try:
            raise inner
        except RateLimitError as exc:
            raise _Wrapped("gave up") from exc
    except _Wrapped as wrapped:
        assert _is_transient(wrapped)


def test_groq_message_hint_is_used_when_there_is_no_header():
    exc = _Wrapped(
        "Error code: 429 - {'error': {'message': 'Rate limit reached on tokens per minute "
        "(TPM): Limit 8000. Please try again in 8.6175s.', 'code': 'rate_limit_exceeded'}}"
    )
    assert _is_transient(exc)
    assert _retry_after_seconds(exc) == pytest.approx(8.6175)


def test_millisecond_hints_are_converted():
    assert _retry_after_seconds(_Wrapped("Please try again in 350ms")) == pytest.approx(0.35)


async def test_throttle_holds_every_caller_after_one_rate_limit(monkeypatch):
    import qc_copilot.evaluation.ragas_runner as runner

    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(runner.asyncio, "sleep", fake_sleep)
    throttle = Throttle()

    async def limited_once():
        if not sleeps:
            raise _http_error(RateLimitError, 429, {"retry-after": "5"})
        return "ok"

    assert await _with_retry(limited_once, attempts=3, label="a", throttle=throttle) == "ok"
    assert throttle.resume_at > 0

    async def fine():
        return "ok"

    # A second caller on the same judge waits out the shared cooldown before calling.
    await _with_retry(fine, attempts=3, label="b", throttle=throttle)
    assert len(sleeps) >= 2


def test_average_precision_matches_the_ragas_definition():
    assert average_precision([]) == 0.0
    assert average_precision([0, 0, 0]) == pytest.approx(0.0)
    assert average_precision([1, 1, 1]) == pytest.approx(1.0)
    # Useful passage ranked first, then two noise passages: precision@1 = 1.
    assert average_precision([1, 0, 0]) == pytest.approx(1.0)
    # Useful passage ranked last of three: precision@3 = 1/3.
    assert average_precision([0, 0, 1]) == pytest.approx(1 / 3)
    # Two useful passages at ranks 1 and 3: mean of 1/1 and 2/3.
    assert average_precision([1, 0, 1]) == pytest.approx((1 + 2 / 3) / 2)


def test_tool_summary_breaks_success_down_by_answer_model():
    from qc_copilot.models import Usage

    good = _tool_sample("a", tool_calls=[_call()]).model_copy(
        update={"usage": Usage(models=["groq:openai/gpt-oss-20b", "reflect=gemini:x"])}
    )
    bad = _tool_sample("b", refused=True).model_copy(
        update={"usage": Usage(models=["openrouter:nemotron:free"])}
    )
    summary = tool_call_summary([good, bad])
    assert summary["by_answer_model"] == {
        "groq:openai/gpt-oss-20b": {"rows": 1, "succeeded": 1},
        "openrouter:nemotron:free": {"rows": 1, "succeeded": 0},
    }


def test_catalog_rows_accept_a_cited_answer_from_the_index():
    from_index = _tool_sample("c1", expected="search_catalog").model_copy(
        update={"citation_count": 2}
    )
    uncited = _tool_sample("c2", expected="search_catalog")
    stock_from_index = _tool_sample("s1", expected="get_inventory").model_copy(
        update={"citation_count": 2}
    )
    summary = tool_call_summary([from_index, uncited, stock_from_index])
    assert summary["succeeded"] == 1
    assert summary["failed_row_ids"] == ["c2", "s1"]
