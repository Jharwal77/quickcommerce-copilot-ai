"""Chat client over a chain of OpenAI-compatible models with automatic failover.

The client holds an ordered chain of provider:model references. Calls go to the first
model. A per-minute rate limit is waited out. A daily budget exhaustion, or a wait too
long to be anything else, moves the chain to the next model for the rest of the process
and logs the switch. Only when every model is exhausted does a call fail, and it fails
loudly with BudgetExhaustedError so a caller can checkpoint and stop.
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

from qc_copilot.config import Settings, get_settings
from qc_copilot.llm.providers import ModelRef, parse_chain
from qc_copilot.llm.ratelimit import (
    BudgetExhaustedError,
    is_daily_limit,
    is_out_of_credit,
    parse_wait_hint,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class LLMNotConfiguredError(RuntimeError):
    """Raised when a generation is attempted with no usable provider key."""


def retry_hint_seconds(exc: BaseException) -> float | None:
    return parse_wait_hint(str(exc))


def is_transient(exc: BaseException) -> bool:
    if isinstance(exc, RateLimitError | APIConnectionError):
        return True
    return isinstance(exc, APIStatusError) and exc.status_code >= 500


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    cost_usd: float

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class RequestedToolCall:
    """A tool invocation the model asked for, with its arguments already parsed."""

    id: str
    name: str
    arguments: dict


@dataclass(frozen=True)
class ToolCompletion:
    text: str
    tool_calls: list[RequestedToolCall]
    assistant_message: dict = field(repr=False)
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    cost_usd: float

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ModelChain:
    """Ordered candidates plus a cursor that only ever moves forward."""

    def __init__(self, refs: list[ModelRef], role: str) -> None:
        if not refs:
            raise LLMNotConfiguredError(f"no models configured for the {role} chain")
        self.refs = list(refs)
        self.role = role
        self.index = 0
        self.switches: list[tuple[str, str, str]] = []

    @property
    def current(self) -> ModelRef:
        return self.refs[self.index]

    @property
    def exhausted(self) -> bool:
        return self.index >= len(self.refs)

    def fail_over(self, reason: str) -> bool:
        """Advance to the next candidate. Returns False when none is left."""
        previous = self.current.key
        self.index += 1
        if self.exhausted:
            logger.error("%s chain exhausted after %s: %s", self.role, previous, reason)
            return False
        self.switches.append((previous, self.current.key, reason))
        logger.warning("%s chain: %s -> %s (%s)", self.role, previous, self.current.key, reason)
        return True


class ChatClient:
    def __init__(
        self,
        settings: Settings | None = None,
        chain: list[ModelRef] | None = None,
        role: str = "answer",
    ) -> None:
        self.settings = settings or get_settings()
        if chain is None:
            chain = parse_chain(self.settings.answer_chain, self.settings.api_keys())
        self.role = role
        self.chain = ModelChain(chain, role) if chain else None
        self._clients: dict[str, OpenAI] = {}

    @property
    def is_configured(self) -> bool:
        return self.chain is not None and not self.chain.exhausted

    @property
    def model(self) -> str:
        return self.chain.current.key if self.is_configured else "unconfigured"

    def _client_for(self, ref: ModelRef) -> OpenAI:
        if ref.provider not in self._clients:
            key = self.settings.api_keys().get(ref.provider)
            if not key:
                raise LLMNotConfiguredError(f"no API key configured for provider {ref.provider}")
            # SDK retries are off; backoff and failover are handled in one place below.
            self._clients[ref.provider] = OpenAI(base_url=ref.base_url, api_key=key, max_retries=0)
        return self._clients[ref.provider]

    def _call(
        self, make_request: Callable[[OpenAI, ModelRef], T], label: str
    ) -> tuple[T, ModelRef]:
        """Run a request against the chain, retrying short waits and failing over long ones."""
        if self.chain is None:
            raise LLMNotConfiguredError(
                "no model provider is configured; add at least one API key to .env"
            )
        while not self.chain.exhausted:
            ref = self.chain.current
            client = self._client_for(ref)
            attempts = self.settings.llm_max_retries
            for attempt in range(1, attempts + 2):
                try:
                    return make_request(client, ref), ref
                except Exception as exc:
                    if is_out_of_credit(exc):
                        self.chain.fail_over(f"{label}: provider credit exhausted")
                        break
                    if not is_transient(exc):
                        raise
                    wait = (retry_hint_seconds(exc) or min(2.0**attempt, 60.0)) + random.uniform(
                        0, 1
                    )
                    daily = is_daily_limit(str(exc))
                    if daily or wait > self.settings.llm_max_wait_seconds or attempt > attempts:
                        reason = (
                            "daily budget exhausted"
                            if daily
                            else f"asked to wait {wait:.0f}s"
                            if wait > self.settings.llm_max_wait_seconds
                            else f"still rate limited after {attempts} retries"
                        )
                        self.chain.fail_over(f"{label}: {reason}")
                        break
                    logger.warning(
                        "%s on %s attempt %d hit %s, retrying in %.1fs",
                        label,
                        ref.key,
                        attempt,
                        type(exc).__name__,
                        wait,
                    )
                    time.sleep(wait)
        raise BudgetExhaustedError(
            f"{label}: every model in the {self.role} chain is exhausted "
            f"({', '.join(r.key for r in self.chain.refs)})"
        )

    def _extra_params(self, ref: ModelRef) -> dict:
        if ref.supports_reasoning_effort:
            return {"reasoning_effort": self.settings.groq_reasoning_effort}
        return {}

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        started = time.perf_counter()
        temp = self.settings.generation_temperature if temperature is None else temperature
        response, ref = self._call(
            lambda client, ref: client.chat.completions.create(
                model=ref.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temp,
                max_tokens=max_tokens or self.settings.generation_max_tokens,
                **self._extra_params(ref),
            ),
            "complete",
        )
        latency_ms = (time.perf_counter() - started) * 1000
        prompt_tokens, completion_tokens = _usage(response)
        return Completion(
            text=(response.choices[0].message.content or "").strip(),
            model=ref.key,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            cost_usd=ref.estimate_cost_usd(prompt_tokens, completion_tokens),
        )

    def complete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ToolCompletion:
        """One turn of a tool-using conversation.

        The caller owns the message history. The returned assistant message is ready
        to be appended to it verbatim, followed by one tool message per requested call.
        """
        started = time.perf_counter()
        temp = self.settings.generation_temperature if temperature is None else temperature
        # An empty tool list means "answer now"; providers reject tools=[] so it is omitted.
        tool_kwargs = {"tools": tools, "tool_choice": "auto"} if tools else {}
        response, ref = self._call(
            lambda client, ref: client.chat.completions.create(
                model=ref.model,
                messages=_messages_for(ref, messages),
                temperature=temp,
                max_tokens=max_tokens or self.settings.generation_max_tokens,
                **tool_kwargs,
                **self._extra_params(ref),
            ),
            "complete_with_tools",
        )
        latency_ms = (time.perf_counter() - started) * 1000
        message = response.choices[0].message
        prompt_tokens, completion_tokens = _usage(response)

        requested: list[RequestedToolCall] = []
        for call in message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {"_raw": call.function.arguments}
            requested.append(RequestedToolCall(call.id, call.function.name, arguments))

        assistant_message: dict = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            assistant_message["tool_calls"] = [_echo_tool_call(call) for call in message.tool_calls]

        return ToolCompletion(
            text=(message.content or "").strip(),
            tool_calls=requested,
            assistant_message=assistant_message,
            model=ref.key,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            cost_usd=ref.estimate_cost_usd(prompt_tokens, completion_tokens),
        )


# Gemini rejects a tool call in the history that has no thought signature, which is the
# case whenever an earlier turn of the same conversation was answered by another provider
# and the chain failed over mid-conversation. Google documents this placeholder for exactly
# that situation: it skips signature validation for calls the model did not produce.
GEMINI_PLACEHOLDER_SIGNATURE = "skip_thought_signature_validator"


def _messages_for(ref: ModelRef, messages: list[dict]) -> list[dict]:
    """Adapt a shared conversation history to the provider about to receive it."""
    if ref.provider != "gemini":
        return messages
    adapted: list[dict] = []
    for message in messages:
        calls = message.get("tool_calls") if message.get("role") == "assistant" else None
        if not calls:
            adapted.append(message)
            continue
        patched = []
        for call in calls:
            signature = (call.get("extra_content") or {}).get("google", {}).get("thought_signature")
            if signature:
                patched.append(call)
            else:
                extra = dict(call.get("extra_content") or {})
                extra["google"] = {
                    **extra.get("google", {}),
                    "thought_signature": GEMINI_PLACEHOLDER_SIGNATURE,
                }
                patched.append({**call, "extra_content": extra})
        adapted.append({**message, "tool_calls": patched})
    return adapted


def _echo_tool_call(call) -> dict:
    """Serialize a tool call so it can be sent back to the provider on the next turn.

    Provider-specific fields are kept: Gemini's OpenAI-compatible endpoint attaches
    extra_content.google.thought_signature to each tool call and rejects the follow-up
    turn with a 400 unless it is echoed back verbatim.
    """
    echoed: dict = call.model_dump(exclude_none=True) if hasattr(call, "model_dump") else {}
    echoed.update(
        {
            "id": call.id,
            "type": "function",
            "function": {"name": call.function.name, "arguments": call.function.arguments},
        }
    )
    return echoed


def _usage(response) -> tuple[int, int]:
    usage = response.usage
    return (
        int(getattr(usage, "prompt_tokens", 0) or 0),
        int(getattr(usage, "completion_tokens", 0) or 0),
    )
