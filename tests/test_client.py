"""Failover behaviour of the chat client, driven by a scripted OpenAI-compatible client."""

from types import SimpleNamespace

import httpx
import pytest
from openai import BadRequestError, RateLimitError

from qc_copilot.config import Settings
from qc_copilot.llm.client import ChatClient, LLMNotConfiguredError
from qc_copilot.llm.providers import ModelRef, parse_chain, parse_model_ref
from qc_copilot.llm.ratelimit import BudgetExhaustedError

KEYS = {"groq": "g", "gemini": "m", "openrouter": "o", "hf": "", "github": ""}


def _settings(**overrides) -> Settings:
    base = dict(
        _env_file=None,
        groq_api_key="g",
        gemini_api_key="m",
        openrouter_api_key="o",
        llm_max_retries=1,
        llm_max_wait_seconds=30,
    )
    base.update(overrides)
    return Settings(**base)


def _429(message: str) -> RateLimitError:
    request = httpx.Request("POST", "https://x/chat/completions")
    return RateLimitError(message, response=httpx.Response(429, request=request), body=None)


def _response(text="ok", tool_calls=None):
    message = SimpleNamespace(content=text, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


class ScriptedProvider:
    """Stands in for an OpenAI client; each model has a queue of outcomes."""

    def __init__(self, outcomes: dict[str, list]):
        self.outcomes = outcomes
        self.calls: list[tuple[str, dict]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        model = kwargs["model"]
        self.calls.append((model, kwargs))
        outcome = self.outcomes[model].pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _client(
    outcomes: dict[str, list], chain: str, **overrides
) -> tuple[ChatClient, ScriptedProvider]:
    settings = _settings(**overrides)
    client = ChatClient(settings, chain=parse_chain(chain, settings.api_keys()), role="test")
    provider = ScriptedProvider(outcomes)
    client._client_for = lambda ref: provider
    return client, provider


def test_parse_model_ref_validates_provider():
    assert parse_model_ref("groq:openai/gpt-oss-20b") == ModelRef("groq", "openai/gpt-oss-20b")
    with pytest.raises(ValueError):
        parse_model_ref("nowhere:model")
    with pytest.raises(ValueError):
        parse_model_ref("groq")


def test_parse_chain_skips_providers_without_keys():
    chain = parse_chain("github:openai/gpt-4o-mini,groq:openai/gpt-oss-20b, gemini:g-flash", KEYS)
    assert [r.key for r in chain] == ["groq:openai/gpt-oss-20b", "gemini:g-flash"]
    assert parse_chain("hf:meta-llama/Llama-3.3-70B-Instruct", KEYS) == []


def test_unconfigured_client_is_reported_not_crashed():
    client = ChatClient(
        _settings(groq_api_key="", gemini_api_key="", openrouter_api_key=""), role="answer"
    )
    assert client.is_configured is False
    assert client.model == "unconfigured"
    with pytest.raises(LLMNotConfiguredError):
        client.complete("s", "u")


def test_completion_uses_the_first_model_and_prices_it():
    client, provider = _client({"openai/gpt-oss-20b": [_response("hi")]}, "groq:openai/gpt-oss-20b")
    result = client.complete("system", "user")
    assert result.text == "hi"
    assert result.model == "groq:openai/gpt-oss-20b"
    assert result.cost_usd == pytest.approx((10 * 0.075 + 5 * 0.30) / 1_000_000)
    # reasoning_effort is a Groq gpt-oss parameter and is sent only there.
    assert provider.calls[0][1]["reasoning_effort"] == "low"


def test_reasoning_effort_is_not_sent_to_other_providers():
    client, provider = _client(
        {"gemini-3.5-flash-lite": [_response()]}, "gemini:gemini-3.5-flash-lite"
    )
    client.complete("s", "u")
    assert "reasoning_effort" not in provider.calls[0][1]


def test_daily_cap_fails_over_to_the_next_model(monkeypatch):
    import qc_copilot.llm.client as module

    monkeypatch.setattr(module.time, "sleep", lambda s: None)
    daily = _429("Error code: 429 - on tokens per day (TPD): Limit 200000. Please try again in 7m.")
    client, provider = _client(
        {"openai/gpt-oss-20b": [daily], "openai/gpt-oss-120b": [_response("from 120b")]},
        "groq:openai/gpt-oss-20b,groq:openai/gpt-oss-120b",
    )
    result = client.complete("s", "u")
    assert result.model == "groq:openai/gpt-oss-120b"
    assert client.chain.switches[0][:2] == ("groq:openai/gpt-oss-20b", "groq:openai/gpt-oss-120b")
    # The switch is sticky: the next call goes straight to the fallback.
    provider.outcomes["openai/gpt-oss-120b"].append(_response("again"))
    assert client.complete("s", "u").model == "groq:openai/gpt-oss-120b"
    assert [m for m, _ in provider.calls] == [
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-120b",
    ]


def test_short_rate_limit_is_retried_on_the_same_model(monkeypatch):
    import qc_copilot.llm.client as module

    sleeps: list[float] = []
    monkeypatch.setattr(module.time, "sleep", lambda s: sleeps.append(s))
    client, provider = _client(
        {"openai/gpt-oss-20b": [_429("Please try again in 2s"), _response("ok")]},
        "groq:openai/gpt-oss-20b,groq:openai/gpt-oss-120b",
    )
    assert client.complete("s", "u").model == "groq:openai/gpt-oss-20b"
    assert len(sleeps) == 1 and 2.0 <= sleeps[0] < 3.0
    assert client.chain.switches == []


def test_persistent_rate_limit_fails_over_after_retries(monkeypatch):
    import qc_copilot.llm.client as module

    monkeypatch.setattr(module.time, "sleep", lambda s: None)
    client, provider = _client(
        {
            "openai/gpt-oss-20b": [_429("Please try again in 1s"), _429("Please try again in 1s")],
            "openai/gpt-oss-120b": [_response("fallback")],
        },
        "groq:openai/gpt-oss-20b,groq:openai/gpt-oss-120b",
        llm_max_retries=1,
    )
    assert client.complete("s", "u").text == "fallback"
    assert "still rate limited" in client.chain.switches[0][2]


def test_exhausting_every_model_raises_budget_error(monkeypatch):
    import qc_copilot.llm.client as module

    monkeypatch.setattr(module.time, "sleep", lambda s: None)
    daily = "Error code: 429 - on tokens per day (TPD). Please try again in 9m."
    client, _ = _client(
        {"openai/gpt-oss-20b": [_429(daily)], "openai/gpt-oss-120b": [_429(daily)]},
        "groq:openai/gpt-oss-20b,groq:openai/gpt-oss-120b",
    )
    with pytest.raises(BudgetExhaustedError, match="every model in the test chain"):
        client.complete("s", "u")
    assert client.is_configured is False


def test_non_transient_errors_are_not_retried_or_failed_over():
    request = httpx.Request("POST", "https://x")
    bad = BadRequestError("bad", response=httpx.Response(400, request=request), body=None)
    client, provider = _client(
        {"openai/gpt-oss-20b": [bad], "openai/gpt-oss-120b": [_response()]},
        "groq:openai/gpt-oss-20b,groq:openai/gpt-oss-120b",
    )
    with pytest.raises(BadRequestError):
        client.complete("s", "u")
    assert len(provider.calls) == 1


def test_tool_completion_parses_requested_calls_and_omits_empty_tools():
    call = SimpleNamespace(
        id="c1", function=SimpleNamespace(name="get_inventory", arguments='{"sku": "X"}')
    )
    client, provider = _client(
        {"openai/gpt-oss-20b": [_response("", tool_calls=[call]), _response("done")]},
        "groq:openai/gpt-oss-20b",
    )
    first = client.complete_with_tools(
        [{"role": "user", "content": "q"}], tools=[{"type": "function"}]
    )
    assert first.tool_calls[0].name == "get_inventory"
    assert first.tool_calls[0].arguments == {"sku": "X"}
    assert first.assistant_message["tool_calls"][0]["id"] == "c1"
    assert provider.calls[0][1]["tool_choice"] == "auto"
    second = client.complete_with_tools([{"role": "user", "content": "q"}], tools=[])
    assert second.text == "done"
    assert "tools" not in provider.calls[1][1]


def test_out_of_credit_fails_over_immediately():
    from openai import APIStatusError

    request = httpx.Request("POST", "https://x")
    depleted = APIStatusError(
        "You have depleted your monthly included credits",
        response=httpx.Response(402, request=request),
        body=None,
    )
    client, provider = _client(
        {"openai/gpt-oss-20b": [depleted], "openai/gpt-oss-120b": [_response("fallback")]},
        "groq:openai/gpt-oss-20b,groq:openai/gpt-oss-120b",
    )
    assert client.complete("s", "u").text == "fallback"
    assert "credit exhausted" in client.chain.switches[0][2]


def test_retired_provider_fails_over_immediately():
    from openai import APIStatusError

    request = httpx.Request("POST", "https://x")
    gone = APIStatusError(
        "github_models_retirement_brownout",
        response=httpx.Response(410, request=request),
        body=None,
    )
    client, _ = _client(
        {"openai/gpt-oss-20b": [gone], "openai/gpt-oss-120b": [_response("fallback")]},
        "groq:openai/gpt-oss-20b,groq:openai/gpt-oss-120b",
    )
    assert client.complete("s", "u").text == "fallback"


def test_provider_extras_on_tool_calls_are_echoed_back():
    class FakeCall:
        id = "c9"
        function = SimpleNamespace(name="search_catalog", arguments='{"query": "paneer"}')

        def model_dump(self, exclude_none=False):
            return {
                "id": self.id,
                "type": "function",
                "function": {"name": self.function.name, "arguments": self.function.arguments},
                "extra_content": {"google": {"thought_signature": "sig=="}},
            }

    client, _ = _client(
        {"openai/gpt-oss-20b": [_response("", tool_calls=[FakeCall()])]}, "groq:openai/gpt-oss-20b"
    )
    completion = client.complete_with_tools(
        [{"role": "user", "content": "q"}], tools=[{"type": "function"}]
    )
    echoed = completion.assistant_message["tool_calls"][0]
    assert echoed["extra_content"] == {"google": {"thought_signature": "sig=="}}
    assert echoed["function"] == {"name": "search_catalog", "arguments": '{"query": "paneer"}'}
    assert completion.tool_calls[0].arguments == {"query": "paneer"}


def test_unsigned_tool_calls_get_a_placeholder_signature_only_for_gemini(monkeypatch):
    import qc_copilot.llm.client as module

    monkeypatch.setattr(module.time, "sleep", lambda s: None)
    daily = _429("Error code: 429 - on tokens per day (TPD): Limit 200000. Please try again in 7m.")
    client, provider = _client(
        {"openai/gpt-oss-20b": [daily], "gemini-3.1-flash-lite": [_response("from gemini")]},
        "groq:openai/gpt-oss-20b,gemini:gemini-3.1-flash-lite",
    )
    # A history whose first turn was answered by Groq: the tool call carries no signature.
    history = [
        {"role": "user", "content": "stock?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "get_inventory", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "{}"},
    ]
    result = client.complete_with_tools(history, tools=[{"type": "function"}])
    assert result.model == "gemini:gemini-3.1-flash-lite"
    groq_call, gemini_call = provider.calls
    assert "extra_content" not in groq_call[1]["messages"][1]["tool_calls"][0]
    sent = gemini_call[1]["messages"][1]["tool_calls"][0]
    assert sent["extra_content"] == {
        "google": {"thought_signature": module.GEMINI_PLACEHOLDER_SIGNATURE}
    }
    # The caller's history is left untouched and a real signature is never overwritten.
    assert "extra_content" not in history[1]["tool_calls"][0]
    signed = [
        {
            **history[1],
            "tool_calls": [
                {
                    **history[1]["tool_calls"][0],
                    "extra_content": {"google": {"thought_signature": "real"}},
                }
            ],
        }
    ]
    assert (
        module._messages_for(parse_model_ref("gemini:x"), signed)[0]["tool_calls"][0][
            "extra_content"
        ]["google"]["thought_signature"]
        == "real"
    )
