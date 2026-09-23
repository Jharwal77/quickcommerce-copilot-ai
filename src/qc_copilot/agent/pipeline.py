"""Agentic pipeline with the same interface as the retrieval-only baseline."""

from __future__ import annotations

import time

from qc_copilot.agent.graph import AgentGraph, citations_from_evidence, evidence_items
from qc_copilot.agent.guardrails import enforce_citations
from qc_copilot.config import Settings, get_settings
from qc_copilot.llm.client import ChatClient
from qc_copilot.llm.providers import parse_chain
from qc_copilot.mcp_server.client import MCPToolClient
from qc_copilot.meta_intent import capability_response, is_meta_intent
from qc_copilot.models import AskRequest, AskResponse, GuardrailReport, TraceStep, Usage
from qc_copilot.rag.pipeline import RagPipeline
from qc_copilot.retrieval.embeddings import build_embedder
from qc_copilot.retrieval.store import VectorStore


class AgentPipeline:
    def __init__(self, settings: Settings, graph: AgentGraph) -> None:
        self.settings = settings
        self.graph = graph

    @classmethod
    def build(cls, settings: Settings | None = None) -> AgentPipeline:
        settings = settings or get_settings()
        embedder = build_embedder(settings)
        store = VectorStore.from_settings(settings, embedder.dimension)
        mcp = MCPToolClient.from_settings(settings).start()
        graph = AgentGraph(
            settings,
            embedder,
            store,
            ChatClient(settings),
            mcp,
            reflector=ChatClient(
                settings,
                chain=parse_chain(settings.reflect_chain, settings.api_keys()),
                role="reflect",
            ),
        )
        return cls(settings, graph)

    @property
    def store(self) -> VectorStore:
        return self.graph.store

    @property
    def llm(self) -> ChatClient:
        return self.graph.llm

    def close(self) -> None:
        self.graph.mcp.close()

    def answer(self, request: AskRequest) -> AskResponse:
        if is_meta_intent(request.question):
            return capability_response(request.question, self.graph.mcp.tool_names())
        started = time.perf_counter()
        state = self.graph.run(request.question)
        evidence = state["evidence"]
        report = enforce_citations(state["final"], {e.marker for e in evidence})
        reflection = state["reflection"] or {}
        trace = list(state["trace"])
        if report.dropped_sentences or report.removed_markers:
            trace.append(
                TraceStep(
                    kind="guardrail",
                    label=(
                        f"Citation guardrail dropped {len(report.dropped_sentences)} uncited "
                        f"sentence(s) and {len(report.removed_markers)} invalid marker(s)"
                    ),
                    detail=" | ".join(report.dropped_sentences)[:300],
                    ok=not report.refused,
                )
            )
        else:
            trace.append(
                TraceStep(kind="guardrail", label="Citation guardrail: every sentence is cited")
            )
        return AskResponse(
            question=request.question,
            answer=report.answer,
            citations=citations_from_evidence(report.answer, evidence),
            contexts=[e.text for e in evidence],
            retrieved_doc_ids=[e.doc_id for e in evidence if e.kind == "passage"],
            tool_calls=list(state["tool_calls"]),
            evidence=evidence_items(evidence),
            trace=trace,
            mode="agentic",
            refused=state["refused"] or report.refused,
            guardrails=GuardrailReport(
                reflection_supported=reflection.get("supported"),
                unsupported_claims=list(reflection.get("unsupported_claims", [])),
                dropped_sentences=len(report.dropped_sentences),
                removed_markers=len(report.removed_markers),
                tool_calls_made=state["live_tool_calls"],
                tool_call_cap=self.settings.tool_call_cap,
            ),
            usage=Usage(
                prompt_tokens=state["prompt_tokens"],
                completion_tokens=state["completion_tokens"],
                total_tokens=state["prompt_tokens"] + state["completion_tokens"],
                retrieval_ms=state["retrieval_ms"],
                generation_ms=state["generation_ms"],
                total_ms=(time.perf_counter() - started) * 1000,
                estimated_cost_usd=state["cost_usd"],
                models=list(state["models"]),
            ),
        )


def build_pipeline(settings: Settings | None = None) -> AgentPipeline | RagPipeline:
    settings = settings or get_settings()
    if settings.agent_mode == "retrieval":
        return RagPipeline.build(settings)
    return AgentPipeline.build(settings)
