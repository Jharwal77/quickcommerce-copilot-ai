"""Typed contracts for the knowledge base, the retrieval layer, and the HTTP API."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SourceKind(StrEnum):
    POLICY = "policy"
    PRODUCT = "product"
    DARK_STORE = "dark_store"


class KBDocument(BaseModel):
    """A document as loaded from disk, before chunking."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    title: str
    kind: SourceKind
    text: str
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A retrievable unit of text derived from exactly one KBDocument."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    title: str
    kind: SourceKind
    text: str
    token_count: int = Field(ge=1)
    section: str | None = None
    ordinal: int = Field(default=0, ge=0)
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)

    @property
    def citation_label(self) -> str:
        return f"{self.title} - {self.section}" if self.section else self.title


class RetrievedChunk(BaseModel):
    """A chunk returned by vector search, with its similarity score."""

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float


class Citation(BaseModel):
    model_config = ConfigDict(frozen=True)

    marker: int = Field(ge=1, description="The [n] marker used in the answer text.")
    chunk_id: str
    doc_id: str
    label: str
    score: float


class ToolCall(BaseModel):
    """One invocation of a live tool during answering, successful or not."""

    model_config = ConfigDict(frozen=True)

    name: str
    arguments: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    ok: bool
    error: str | None = None
    latency_ms: float = Field(default=0.0, ge=0.0)


class Usage(BaseModel):
    """Token, latency, and cost telemetry for a single request."""

    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    retrieval_ms: float = Field(default=0.0, ge=0.0)
    generation_ms: float = Field(default=0.0, ge=0.0)
    total_ms: float = Field(default=0.0, ge=0.0)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    # provider:model references that produced this answer, in order of first use. More
    # than one means a provider failover happened mid-question.
    models: list[str] = Field(default_factory=list)


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=3, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def _strip_question(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question must not be blank")
        return cleaned


class EvidenceItem(BaseModel):
    """One numbered piece of evidence the answer may cite: a passage or a tool result."""

    model_config = ConfigDict(frozen=True)

    marker: int = Field(ge=1)
    label: str
    doc_id: str
    kind: Literal["passage", "tool"]
    text: str
    score: float = 0.0


class TraceStep(BaseModel):
    """One step of the agent loop, in the order it happened."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["retrieve", "tool", "reflect", "guardrail", "answer"]
    label: str
    detail: str = ""
    ok: bool = True
    latency_ms: float = Field(default=0.0, ge=0.0)
    arguments: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    markers: list[int] = Field(default_factory=list)


class GuardrailReport(BaseModel):
    """What the post-generation checks changed before the answer was returned."""

    model_config = ConfigDict(frozen=True)

    reflection_supported: bool | None = None
    unsupported_claims: list[str] = Field(default_factory=list)
    dropped_sentences: int = Field(default=0, ge=0)
    removed_markers: int = Field(default=0, ge=0)
    tool_calls_made: int = Field(default=0, ge=0)
    tool_call_cap: int = Field(default=0, ge=0)


class AskResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    contexts: list[str] = Field(default_factory=list)
    retrieved_doc_ids: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    trace: list[TraceStep] = Field(default_factory=list)
    mode: Literal["retrieval", "agentic", "system"] = "retrieval"
    refused: bool = False
    guardrails: GuardrailReport | None = None
    usage: Usage = Field(default_factory=Usage)


class HealthResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["ok", "degraded"]
    collection: str
    vectors: int | None = None
    llm_configured: bool
    detail: str | None = None


class KnowledgeSummary(BaseModel):
    """What the knowledge base and tools cover, derived from the data on disk."""

    model_config = ConfigDict(frozen=True)

    policies: list[dict[str, str | list[str]]]
    dark_stores: list[dict[str, str]]
    categories: list[str]
    brands: list[str]
    product_count: int
    chunk_count: int | None = None


class MetaResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    service: str
    version: str
    description: str
    links: dict[str, str]
    mode: str
    tools: list[str]
    answer_chain: list[str]
    reflect_chain: list[str]
    tool_call_cap: int
    knowledge: KnowledgeSummary
    eval: dict[str, object]
    synthetic_data: bool = True
