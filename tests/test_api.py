from contextlib import suppress

import pytest
from httpx import ASGITransport, AsyncClient

from qc_copilot.api.main import app
from qc_copilot.config import get_settings
from qc_copilot.llm.client import LLMNotConfiguredError
from qc_copilot.llm.ratelimit import BudgetExhaustedError
from qc_copilot.models import AskRequest, AskResponse, Citation, Usage


class _FakeLLM:
    def __init__(self, configured: bool = True) -> None:
        self.is_configured = configured


class _FakeStore:
    def __init__(self, count: int = 118, fail: bool = False) -> None:
        self._count = count
        self._fail = fail

    def count(self) -> int:
        if self._fail:
            raise ConnectionError("qdrant is down")
        return self._count


class _FakePipeline:
    def __init__(
        self, vectors: int = 118, configured: bool = True, fail_count: bool = False
    ) -> None:
        self.store = _FakeStore(vectors, fail_count)
        self.llm = _FakeLLM(configured)
        self.raise_llm_error = False
        self.raise_budget_error = False

    def answer(self, request: AskRequest) -> AskResponse:
        if self.raise_llm_error:
            raise LLMNotConfiguredError("GROQ_API_KEY is not set.")
        if self.raise_budget_error:
            raise BudgetExhaustedError("gpt-oss-20b asked for a 450s wait")
        return AskResponse(
            question=request.question,
            answer="Dairy items must be reported within 2 hours [1].",
            citations=[
                Citation(
                    marker=1,
                    chunk_id="policy-returns::001",
                    doc_id="policy-returns",
                    label="Returns and Refunds Policy - Return windows by category",
                    score=0.71,
                )
            ],
            contexts=["Returns and Refunds Policy > Return windows by category"],
            usage=Usage(prompt_tokens=100, completion_tokens=20, total_tokens=120, total_ms=42.0),
        )


async def _client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.fixture(autouse=True)
def _reset_state():
    yield
    # Starlette keeps state in an internal dict, so delattr is the way to clear it.
    with suppress(AttributeError, KeyError):
        delattr(app.state, "pipeline")


async def test_health_reports_ok_when_the_index_and_key_are_present():
    app.state.pipeline = _FakePipeline()
    async with await _client() as client:
        response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["vectors"] == 118
    assert body["llm_configured"] is True


async def test_health_is_degraded_when_the_index_is_empty():
    app.state.pipeline = _FakePipeline(vectors=0)
    async with await _client() as client:
        body = (await client.get("/health")).json()
    assert body["status"] == "degraded"
    assert body["detail"] == "knowledge base is empty"


async def test_health_is_degraded_without_an_llm_key():
    app.state.pipeline = _FakePipeline(configured=False)
    async with await _client() as client:
        body = (await client.get("/health")).json()
    assert body["status"] == "degraded"
    assert body["llm_configured"] is False


async def test_health_is_degraded_when_the_vector_store_is_unreachable():
    app.state.pipeline = _FakePipeline(fail_count=True)
    async with await _client() as client:
        body = (await client.get("/health")).json()
    assert body["status"] == "degraded"
    assert body["detail"] == "vector store unreachable"


async def test_health_is_degraded_before_the_pipeline_is_built():
    async with await _client() as client:
        body = (await client.get("/health")).json()
    assert body["status"] == "degraded"
    assert body["detail"] == "pipeline is not ready"


async def test_ask_returns_an_answer_with_citations():
    app.state.pipeline = _FakePipeline()
    async with await _client() as client:
        response = await client.post("/ask", json={"question": "What is the dairy return window?"})
    assert response.status_code == 200
    body = response.json()
    assert body["citations"][0]["doc_id"] == "policy-returns"
    assert body["usage"]["total_tokens"] == 120
    assert body["mode"] == "retrieval"


async def test_ask_rejects_a_blank_question():
    app.state.pipeline = _FakePipeline()
    async with await _client() as client:
        response = await client.post("/ask", json={"question": "   "})
    assert response.status_code == 422


async def test_ask_rejects_an_unknown_field():
    app.state.pipeline = _FakePipeline()
    async with await _client() as client:
        response = await client.post("/ask", json={"question": "A valid question?", "hack": 1})
    assert response.status_code == 422


def _out_of_range_top_k():
    return {"question": "A valid question?", "top_k": 99}


async def test_ask_rejects_an_out_of_range_top_k():
    app.state.pipeline = _FakePipeline()
    async with await _client() as client:
        response = await client.post("/ask", json=_out_of_range_top_k())
    assert response.status_code == 422


async def test_ask_returns_503_when_the_pipeline_is_missing():
    async with await _client() as client:
        response = await client.post("/ask", json={"question": "A valid question?"})
    assert response.status_code == 503


async def test_ask_returns_503_when_the_llm_key_is_absent():
    pipeline = _FakePipeline()
    pipeline.raise_llm_error = True
    app.state.pipeline = pipeline
    async with await _client() as client:
        response = await client.post("/ask", json={"question": "A valid question?"})
    assert response.status_code == 503
    assert "GROQ_API_KEY" in response.json()["detail"]


async def test_ask_returns_503_when_the_daily_budget_is_exhausted():
    pipeline = _FakePipeline()
    pipeline.raise_budget_error = True
    app.state.pipeline = pipeline
    async with await _client() as client:
        response = await client.post("/ask", json={"question": "A valid question?"})
    assert response.status_code == 503
    assert "budget exhausted" in response.json()["detail"]


async def test_root_falls_back_to_meta_json_without_a_frontend_build(monkeypatch, tmp_path):
    monkeypatch.setenv("FRONTEND_DIST", str(tmp_path / "absent"))
    get_settings.cache_clear()
    try:
        async with await _client() as client:
            response = await client.get("/")
        assert response.status_code == 200
        body = response.json()
        assert body["service"] == "quickcommerce-copilot"
        assert body["links"]["docs"] == "/docs"
        assert body["synthetic_data"] is True
    finally:
        get_settings.cache_clear()


async def test_root_serves_the_built_index_when_present(monkeypatch, tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><title>quickcommerce-copilot</title>")
    monkeypatch.setenv("FRONTEND_DIST", str(dist))
    get_settings.cache_clear()
    try:
        async with await _client() as client:
            response = await client.get("/")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert "quickcommerce-copilot" in response.text
    finally:
        get_settings.cache_clear()


async def test_meta_describes_knowledge_tools_and_evaluation(monkeypatch):
    # Chains only list providers that have a key, so give Gemini one explicitly: CI has none.
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    get_settings.cache_clear()
    app.state.pipeline = _FakePipeline()
    try:
        async with await _client() as client:
            response = await client.get("/meta")
    finally:
        get_settings.cache_clear()
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] in {"agentic", "retrieval"}
    knowledge = body["knowledge"]
    assert {p["doc_id"] for p in knowledge["policies"]} >= {"policy-returns", "policy-delivery"}
    assert any(
        "Return windows" in section for p in knowledge["policies"] for section in p["sections"]
    )
    assert len(knowledge["dark_stores"]) == 6
    assert "Dairy & Eggs" in knowledge["categories"]
    assert "Nandhini Fresh" in knowledge["brands"]
    assert knowledge["product_count"] == 88
    assert knowledge["chunk_count"] == 118
    assert isinstance(body["eval"], dict)
    assert any(ref.startswith("gemini:") for ref in body["answer_chain"])
    assert all(":" in ref for ref in body["answer_chain"] + body["reflect_chain"])
