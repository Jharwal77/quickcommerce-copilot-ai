"""Classic retrieve-then-generate pipeline.

This is the phase-one baseline the agentic loop is measured against. It always
retrieves and never calls a tool, which is exactly why it cannot answer questions
about live stock or a specific order.
"""

from __future__ import annotations

import re
import time

from qc_copilot.agent.guardrails import normalize_markers
from qc_copilot.config import Settings, get_settings
from qc_copilot.llm.client import ChatClient
from qc_copilot.meta_intent import capability_response, is_meta_intent
from qc_copilot.models import (
    AskRequest,
    AskResponse,
    Citation,
    EvidenceItem,
    RetrievedChunk,
    TraceStep,
    Usage,
)
from qc_copilot.rag.prompts import ANSWER_SYSTEM_PROMPT, REFUSAL_TEXT, build_answer_prompt
from qc_copilot.retrieval.embeddings import Embedder, build_embedder
from qc_copilot.retrieval.store import VectorStore

_CITATION_MARKER = re.compile(r"\[(\d{1,2})\]")


def extract_citations(answer: str, retrieved: list[RetrievedChunk]) -> list[Citation]:
    """Map the [n] markers actually present in the answer back to their passages."""
    citations: list[Citation] = []
    seen: set[int] = set()
    for raw_marker in _CITATION_MARKER.findall(answer):
        marker = int(raw_marker)
        if marker in seen or not 1 <= marker <= len(retrieved):
            continue
        seen.add(marker)
        item = retrieved[marker - 1]
        citations.append(
            Citation(
                marker=marker,
                chunk_id=item.chunk.chunk_id,
                doc_id=item.chunk.doc_id,
                label=item.chunk.citation_label,
                score=item.score,
            )
        )
    return sorted(citations, key=lambda c: c.marker)


class RagPipeline:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        store: VectorStore,
        llm: ChatClient,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.store = store
        self.llm = llm

    @classmethod
    def build(cls, settings: Settings | None = None) -> RagPipeline:
        settings = settings or get_settings()
        embedder = build_embedder(settings)
        return cls(
            settings=settings,
            embedder=embedder,
            store=VectorStore.from_settings(settings, embedder.dimension),
            llm=ChatClient(settings),
        )

    def retrieve(self, question: str, top_k: int) -> list[RetrievedChunk]:
        return self.store.search(self.embedder.encode_query(question), top_k)

    def answer(self, request: AskRequest) -> AskResponse:
        if is_meta_intent(request.question):
            return capability_response(request.question, [])
        started = time.perf_counter()
        top_k = request.top_k or self.settings.retrieval_top_k

        retrieval_started = time.perf_counter()
        retrieved = self.retrieve(request.question, top_k)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000

        if not retrieved:
            return AskResponse(
                question=request.question,
                answer=REFUSAL_TEXT,
                refused=True,
                usage=Usage(
                    retrieval_ms=retrieval_ms,
                    total_ms=(time.perf_counter() - started) * 1000,
                ),
            )

        completion = self.llm.complete(
            ANSWER_SYSTEM_PROMPT,
            build_answer_prompt(request.question, retrieved),
        )
        answer = normalize_markers(completion.text)

        return AskResponse(
            question=request.question,
            answer=answer,
            citations=extract_citations(answer, retrieved),
            contexts=[item.chunk.text for item in retrieved],
            retrieved_doc_ids=[item.chunk.doc_id for item in retrieved],
            evidence=[
                EvidenceItem(
                    marker=index,
                    label=item.chunk.citation_label,
                    doc_id=item.chunk.doc_id,
                    kind="passage",
                    text=item.chunk.text,
                    score=item.score,
                )
                for index, item in enumerate(retrieved, start=1)
            ],
            trace=[
                TraceStep(
                    kind="retrieve",
                    label="Searched the knowledge base",
                    detail=request.question,
                    latency_ms=retrieval_ms,
                    markers=list(range(1, len(retrieved) + 1)),
                )
            ],
            mode="retrieval",
            refused=answer.strip().startswith(REFUSAL_TEXT),
            usage=Usage(
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
                total_tokens=completion.total_tokens,
                retrieval_ms=retrieval_ms,
                generation_ms=completion.latency_ms,
                total_ms=(time.perf_counter() - started) * 1000,
                estimated_cost_usd=completion.cost_usd,
                models=[completion.model],
            ),
        )
