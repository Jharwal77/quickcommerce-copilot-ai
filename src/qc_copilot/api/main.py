"""FastAPI service exposing the assistant and serving the built frontend."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from qc_copilot import __version__
from qc_copilot.agent.pipeline import AgentPipeline, build_pipeline
from qc_copilot.api.meta import build_meta
from qc_copilot.config import get_settings
from qc_copilot.llm.client import LLMNotConfiguredError
from qc_copilot.llm.ratelimit import BudgetExhaustedError
from qc_copilot.models import AskRequest, AskResponse, HealthResponse, MetaResponse
from qc_copilot.rag.pipeline import RagPipeline

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # The embedding model is loaded and the MCP server started once at startup, not
    # per request, so the first request does not pay for either.
    pipeline = await run_in_threadpool(build_pipeline)
    app.state.pipeline = pipeline
    logger.info("pipeline ready in %s mode", get_settings().agent_mode)
    yield
    if isinstance(pipeline, AgentPipeline):
        pipeline.close()


app = FastAPI(
    title="quickcommerce-copilot",
    description=(
        "Grounded question answering over a synthetic quick-commerce catalog and its "
        "operating policies."
    ),
    version=__version__,
    lifespan=lifespan,
)

_ASSETS = get_settings().frontend_dist / "assets"
if _ASSETS.is_dir():
    app.mount("/assets", StaticFiles(directory=_ASSETS), name="assets")


def _pipeline() -> RagPipeline | AgentPipeline:
    pipeline = getattr(app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="pipeline is not ready")
    return pipeline


def _tool_names() -> list[str]:
    pipeline = getattr(app.state, "pipeline", None)
    if isinstance(pipeline, AgentPipeline):
        return pipeline.graph.mcp.tool_names()
    return []


async def _chunk_count() -> int | None:
    pipeline = getattr(app.state, "pipeline", None)
    if pipeline is None:
        return None
    try:
        return await run_in_threadpool(pipeline.store.count)
    except Exception:
        return None


@app.get("/", include_in_schema=False)
async def root():
    """The built single-page app when present; otherwise the /meta JSON."""
    index = get_settings().frontend_dist / "index.html"
    if index.is_file():
        return FileResponse(index, media_type="text/html")
    return JSONResponse((await meta()).model_dump(mode="json"))


@app.get("/meta", response_model=MetaResponse)
async def meta() -> MetaResponse:
    settings = get_settings()
    return await run_in_threadpool(build_meta, settings, await _chunk_count(), _tool_names())


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = get_settings()
    pipeline = getattr(app.state, "pipeline", None)
    if pipeline is None:
        return HealthResponse(
            status="degraded",
            collection=settings.qdrant_collection,
            llm_configured=any(settings.api_keys().values()),
            detail="pipeline is not ready",
        )
    try:
        vectors = await run_in_threadpool(pipeline.store.count)
    except Exception as exc:
        logger.warning("vector store unreachable: %s", exc)
        return HealthResponse(
            status="degraded",
            collection=settings.qdrant_collection,
            llm_configured=pipeline.llm.is_configured,
            detail="vector store unreachable",
        )

    degraded = vectors == 0 or not pipeline.llm.is_configured
    return HealthResponse(
        status="degraded" if degraded else "ok",
        collection=settings.qdrant_collection,
        vectors=vectors,
        llm_configured=pipeline.llm.is_configured,
        detail="knowledge base is empty" if vectors == 0 else None,
    )


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    pipeline = _pipeline()
    try:
        response = await run_in_threadpool(pipeline.answer, request)
    except LLMNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except BudgetExhaustedError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"model provider budget exhausted; retry later ({exc})",
        ) from exc
    # One structured line per request is the whole observability story here: tokens,
    # latency, cost, and whether the guardrails intervened.
    logger.info(
        "ask %s",
        json.dumps(
            {
                "mode": response.mode,
                "refused": response.refused,
                "citations": len(response.citations),
                "tool_calls": len(response.tool_calls),
                "models": response.usage.models,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_ms": round(response.usage.total_ms, 1),
                "retrieval_ms": round(response.usage.retrieval_ms, 1),
                "generation_ms": round(response.usage.generation_ms, 1),
                "cost_usd": round(response.usage.estimated_cost_usd, 6),
                "dropped_sentences": (
                    response.guardrails.dropped_sentences if response.guardrails else 0
                ),
            }
        ),
    )
    return response
