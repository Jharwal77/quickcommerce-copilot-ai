import httpx
import pytest
from openai import BadRequestError, RateLimitError

from qc_copilot.config import Settings
from qc_copilot.llm.client import (
    ChatClient,
    Completion,
    LLMNotConfiguredError,
    is_transient,
    retry_hint_seconds,
)
from qc_copilot.models import AskRequest, Chunk, RetrievedChunk, SourceKind
from qc_copilot.rag.pipeline import RagPipeline, extract_citations
from qc_copilot.rag.prompts import REFUSAL_TEXT, build_answer_prompt, format_contexts


def _retrieved(n: int = 3) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk=Chunk(
                chunk_id=f"policy-returns::{i:03d}",
                doc_id="policy-returns",
                title="Returns Policy",
                kind=SourceKind.POLICY,
                text=f"Passage {i} about refunds.",
                token_count=6,
                section=f"Section {i}",
                ordinal=i,
            ),
            score=1.0 - i / 10,
        )
        for i in range(n)
    ]


class _FakeEmbedder:
    dimension = 4

    def encode_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]


class _FakeStore:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results
        self.calls: list[int] = []

    def search(self, vector, top_k, kinds=None):
        self.calls.append(top_k)
        return self.results[:top_k]


class _FakeLLM:
    def __init__(self, text: str) -> None:
        self.text = text
        self.prompts: list[str] = []

    def complete(self, system_prompt: str, user_prompt: str, **_):
        self.prompts.append(user_prompt)
        return Completion(
            text=self.text,
            model="fake",
            prompt_tokens=120,
            completion_tokens=30,
            latency_ms=12.5,
            cost_usd=0.0001,
        )


def _pipeline(results: list[RetrievedChunk], answer: str) -> RagPipeline:
    return RagPipeline(
        settings=Settings(retrieval_top_k=3),
        embedder=_FakeEmbedder(),
        store=_FakeStore(results),
        llm=_FakeLLM(answer),
    )


def test_extract_citations_maps_markers_to_passages():
    citations = extract_citations("Dairy is 2 hours [1] and UPI is 24 hours [3].", _retrieved())
    assert [c.marker for c in citations] == [1, 3]
    assert citations[0].chunk_id == "policy-returns::000"
    assert citations[1].chunk_id == "policy-returns::002"


def test_extract_citations_deduplicates_repeated_markers():
    citations = extract_citations("Fact [2]. Another fact [2].", _retrieved())
    assert [c.marker for c in citations] == [2]


def test_extract_citations_ignores_out_of_range_markers():
    assert extract_citations("Claim [9] with no passage.", _retrieved()) == []


def test_extract_citations_returns_markers_in_order():
    citations = extract_citations("Later [3] then earlier [1].", _retrieved())
    assert [c.marker for c in citations] == [1, 3]


def test_extract_citations_handles_an_uncited_answer():
    assert extract_citations("No markers anywhere in this answer.", _retrieved()) == []


def test_format_contexts_numbers_passages_from_one():
    text = format_contexts(_retrieved(2))
    assert "[1] Source: Returns Policy - Section 0" in text
    assert "[2] Source: Returns Policy - Section 1" in text


def test_prompt_carries_the_question_and_every_passage():
    prompt = build_answer_prompt("What is the dairy window?", _retrieved(2))
    assert "What is the dairy window?" in prompt
    assert "Passage 0 about refunds." in prompt
    assert "Passage 1 about refunds." in prompt


def test_answer_returns_citations_and_contexts():
    pipeline = _pipeline(_retrieved(), "Dairy is 2 hours [1].")
    response = pipeline.answer(AskRequest(question="What is the dairy return window?"))
    assert response.mode == "retrieval"
    assert response.refused is False
    assert [c.marker for c in response.citations] == [1]
    assert len(response.contexts) == 3
    assert response.usage.total_tokens == 150
    assert response.usage.total_ms > 0


def test_answer_refuses_when_retrieval_is_empty():
    pipeline = _pipeline([], "unused")
    response = pipeline.answer(AskRequest(question="Something not in the corpus at all?"))
    assert response.refused is True
    assert response.answer == REFUSAL_TEXT
    assert response.citations == []
    assert response.usage.total_tokens == 0


def test_refusal_from_the_model_is_flagged():
    pipeline = _pipeline(_retrieved(), REFUSAL_TEXT)
    response = pipeline.answer(AskRequest(question="What is the CEO's home address?"))
    assert response.refused is True


def test_request_top_k_overrides_the_configured_default():
    pipeline = _pipeline(_retrieved(), "Answer [1].")
    pipeline.answer(AskRequest(question="What is the dairy return window?", top_k=2))
    assert pipeline.store.calls == [2]


def test_default_top_k_comes_from_settings():
    pipeline = _pipeline(_retrieved(), "Answer [1].")
    pipeline.answer(AskRequest(question="What is the dairy return window?"))
    assert pipeline.store.calls == [3]


def test_chat_client_refuses_to_run_without_any_key():
    client = ChatClient(
        Settings(_env_file=None, groq_api_key="", gemini_api_key="", openrouter_api_key="")
    )
    assert client.is_configured is False
    with pytest.raises(LLMNotConfiguredError):
        client.complete("system", "user")


def test_rate_limit_hint_is_parsed_from_the_body():
    assert retry_hint_seconds(RuntimeError("Please try again in 4.2s")) == pytest.approx(4.2)
    assert retry_hint_seconds(ValueError("nothing")) is None


def test_only_rate_limits_and_server_errors_are_transient():
    request = httpx.Request("POST", "https://x")
    limited = RateLimitError("l", response=httpx.Response(429, request=request), body=None)
    bad = BadRequestError("bad", response=httpx.Response(400, request=request), body=None)
    assert is_transient(limited)
    assert not is_transient(bad)
    assert not is_transient(ValueError("x"))
