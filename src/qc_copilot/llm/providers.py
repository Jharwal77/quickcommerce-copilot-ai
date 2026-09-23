"""Registry of OpenAI-compatible providers and the model chains built on them.

Every provider here speaks the OpenAI chat-completions wire format, so one client
serves all of them. A chain is an ordered list of "provider:model" references; when a
model's daily budget runs out the chain moves to the next one. Providers without an
API key configured are skipped, so the same chain definition works on any machine.
"""

from __future__ import annotations

from dataclasses import dataclass

PROVIDER_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "openrouter": "https://openrouter.ai/api/v1",
    # Hugging Face Inference Providers, routed through one OpenAI-compatible endpoint.
    "hf": "https://router.huggingface.co/v1",
    # GitHub Models, free and rate limited, authenticated with a GitHub token.
    "github": "https://models.github.ai/inference",
}

# List prices in USD per million tokens (input, output), used only to estimate what the
# measured traffic would cost at published rates. Every run in this project used free
# tiers, so actual spend was zero. Models absent from this table are estimated at zero.
LIST_PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "groq:openai/gpt-oss-20b": (0.075, 0.30),
    "groq:openai/gpt-oss-120b": (0.15, 0.60),
    "gemini:gemini-3.6-flash": (0.30, 2.50),
    "gemini:gemini-3.5-flash-lite": (0.10, 0.40),
    "gemini:gemini-3.1-flash-lite": (0.10, 0.40),
    "gemini:gemini-flash-lite-latest": (0.10, 0.40),
    "gemini:gemini-3.5-flash": (0.30, 2.50),
    "gemini:gemini-3.7-flash": (0.30, 2.50),
    "gemini:gemini-3.8-flash": (0.30, 2.50),
    "hf:meta-llama/Llama-3.3-70B-Instruct": (0.80, 0.80),
    "hf:Qwen/Qwen2.5-72B-Instruct": (0.80, 0.80),
    "hf:meta-llama/Llama-3.1-8B-Instruct": (0.10, 0.10),
    "github:openai/gpt-4o-mini": (0.15, 0.60),
    "github:meta/Llama-3.3-70B-Instruct": (0.80, 0.80),
}

# Only Groq's gpt-oss deployment accepts the reasoning_effort parameter.
REASONING_EFFORT_MODELS = {"groq:openai/gpt-oss-20b", "groq:openai/gpt-oss-120b"}


@dataclass(frozen=True)
class ModelRef:
    provider: str
    model: str

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.model}"

    @property
    def base_url(self) -> str:
        return PROVIDER_BASE_URLS[self.provider]

    @property
    def supports_reasoning_effort(self) -> bool:
        return self.key in REASONING_EFFORT_MODELS

    def price(self) -> tuple[float, float]:
        return LIST_PRICES_USD_PER_MTOK.get(self.key, (0.0, 0.0))

    def estimate_cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        rate_in, rate_out = self.price()
        return (prompt_tokens * rate_in + completion_tokens * rate_out) / 1_000_000

    def __str__(self) -> str:
        return self.key


def parse_model_ref(text: str) -> ModelRef:
    provider, sep, model = text.strip().partition(":")
    if not sep or not model or provider not in PROVIDER_BASE_URLS:
        raise ValueError(
            f"model reference {text!r} must look like provider:model with provider one of "
            f"{sorted(PROVIDER_BASE_URLS)}"
        )
    return ModelRef(provider, model)


def parse_chain(text: str, api_keys: dict[str, str]) -> list[ModelRef]:
    """Parse a comma-separated chain, dropping providers that have no key configured.

    An empty result means nothing is configured; callers decide whether that is fatal.
    """
    refs = [parse_model_ref(part) for part in text.split(",") if part.strip()]
    return [ref for ref in refs if api_keys.get(ref.provider)]
