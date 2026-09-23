"""Explicit agent loop: decide, then retrieve or call a tool, then reflect, then answer.

The graph is deliberately small and every transition is visible. The model chooses the
next action by requesting a tool. Knowledge-base search is offered as a tool alongside
the live MCP tools so that one decision step covers both kinds of evidence. Two hard
limits bound the loop: a cap on live tool calls and a cap on model turns. A final
reflection step checks the draft against the gathered evidence before anything is
returned.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from qc_copilot.agent.prompts import (
    AGENT_SYSTEM_PROMPT,
    BUDGET_EXHAUSTED_NOTE,
    KNOWLEDGE_BASE_TOOL,
    REFLECT_SYSTEM_PROMPT,
    build_reflect_prompt,
    format_evidence,
)
from qc_copilot.config import Settings
from qc_copilot.llm.client import ChatClient
from qc_copilot.mcp_server.client import MCPToolClient
from qc_copilot.models import Citation, EvidenceItem, ToolCall, TraceStep
from qc_copilot.rag.prompts import REFUSAL_TEXT
from qc_copilot.retrieval.embeddings import Embedder
from qc_copilot.retrieval.store import VectorStore

logger = logging.getLogger(__name__)

KB_TOOL_NAME = "search_knowledge_base"
_CITATION_MARKER = re.compile(r"\[(\d{1,2})\]")
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class Evidence(BaseModel):
    """One numbered piece of evidence the model may cite."""

    model_config = ConfigDict(frozen=True)

    marker: int
    label: str
    text: str
    kind: Literal["passage", "tool"]
    chunk_id: str
    doc_id: str
    score: float = 0.0


class ReflectionVerdict(BaseModel):
    model_config = ConfigDict(extra="ignore")

    supported: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    revised_answer: str = ""


class AgentState(TypedDict, total=False):
    question: str
    messages: list[dict]
    evidence: list[Evidence]
    tool_calls: list[ToolCall]
    steps: int
    live_tool_calls: int
    draft: str
    final: str
    refused: bool
    reflection: dict | None
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    generation_ms: float
    retrieval_ms: float
    models: list[str]
    trace: list[TraceStep]


class AgentGraph:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        store: VectorStore,
        llm: ChatClient,
        mcp: MCPToolClient,
        reflector: ChatClient | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.mcp = mcp
        self.tools = [KNOWLEDGE_BASE_TOOL] + [t.as_openai_tool() for t in mcp.tools]
        self.live_tool_names = set(mcp.tool_names())
        self.graph = self._build()

    def _build(self):
        graph = StateGraph(AgentState)
        graph.add_node("decide", self.decide)
        graph.add_node("act", self.act)
        graph.add_node("reflect", self.reflect)
        graph.add_edge(START, "decide")
        graph.add_conditional_edges(
            "decide", self.route_after_decide, {"act": "act", "reflect": "reflect"}
        )
        graph.add_edge("act", "decide")
        graph.add_edge("reflect", END)
        return graph.compile()

    def initial_state(self, question: str) -> AgentState:
        return AgentState(
            question=question,
            messages=[
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            evidence=[],
            tool_calls=[],
            steps=0,
            live_tool_calls=0,
            draft="",
            final="",
            refused=False,
            reflection=None,
            prompt_tokens=0,
            completion_tokens=0,
            cost_usd=0.0,
            generation_ms=0.0,
            retrieval_ms=0.0,
            models=[],
            trace=[],
        )

    def run(self, question: str) -> AgentState:
        return self.graph.invoke(self.initial_state(question))

    def _is_live_inventory_question(self, question: str) -> bool:
        """Return True for explicit stock/availability questions about a dark store."""
        q = question.lower()

        inventory_terms = (
            "available",
            "in stock",
            "stock",
            "how many units",
            "units available",
        )

        store_terms = (
            "dark store",
            "store",
        )

        return any(term in q for term in inventory_terms) and any(term in q for term in store_terms)

    # -- nodes -------------------------------------------------------------------

    def _extract_live_inventory_parts(self, question: str) -> tuple[str, str] | None:
        """Extract the exact product and requested store from a live-stock question."""
        patterns = (
            r"^is\s+(.+?)\s+available\s+at\s+(.+?)\s*\??$",
            r"^is\s+(.+?)\s+in\s+stock\s+at\s+(.+?)\s*\??$",
            r"^is\s+(.+?)\s+in\s+stock\s+at\s+the\s+(.+?)\s*\??$",
            r"^how\s+many\s+units\s+of\s+(.+?)\s+are\s+available\s+at\s+(.+?)\s*\??$",
            r"^how\s+many\s+units\s+of\s+(.+?)\s+are\s+in\s+stock\s+at\s+(.+?)\s*\??$",
        )

        normalized = " ".join(question.strip().split())

        for pattern in patterns:
            match = re.match(pattern, normalized, re.IGNORECASE)
            if match:
                product = match.group(1).strip()
                store = match.group(2).strip()
                return product, store

        return None

    def _forced_tool_message(
        self,
        tool_name: str,
        arguments: dict,
        call_id: str,
    ) -> dict:
        """Build an assistant tool-call message without asking the LLM to plan it."""
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(arguments),
                    },
                }
            ],
        }

    def _find_catalog_match(
        self,
        evidence: list[Evidence],
        requested_store: str,
    ) -> tuple[str, str] | None:
        """Return (SKU, store_id) from the successful catalog evidence."""
        for item in reversed(evidence):
            if item.doc_id != "tool:search_catalog":
                continue

            try:
                payload = json.loads(item.text)
            except (TypeError, json.JSONDecodeError):
                continue

            hits = payload.get("hits") or []
            stores = payload.get("stores") or []

            if not hits:
                return None

            sku = hits[0].get("sku")
            requested = requested_store.strip().lower()

            for store in stores:
                name = str(store.get("name") or "").strip().lower()
                store_id = store.get("store_id")

                if name == requested:
                    return sku, store_id

            return None

        return None

    def decide(self, state: AgentState) -> AgentState:
        """Choose deterministic live-tool routing, otherwise ask the LLM."""

        messages = list(state["messages"])
        question = state["question"]

        # Live inventory routing is deterministic for the first two tool calls.
        # This prevents a free LLM from inventing or rewriting the product query.
        if self._is_live_inventory_question(question):
            parts = self._extract_live_inventory_parts(question)

            if parts:
                product, requested_store = parts

                # Step 1: force the exact catalog query from the user's question.
                if state["live_tool_calls"] == 0:
                    call_id = f"forced-search-{state['steps'] + 1}"
                    messages.append(
                        self._forced_tool_message(
                            "search_catalog",
                            {"query": product, "limit": 10},
                            call_id,
                        )
                    )

                    return {
                        "messages": messages,
                        "steps": state["steps"] + 1,
                    }

                # Step 2: use the catalog result to force the exact SKU/store lookup.
                if state["live_tool_calls"] == 1:
                    match = self._find_catalog_match(
                        state["evidence"],
                        requested_store,
                    )

                    if match:
                        sku, store_id = match
                        call_id = f"forced-inventory-{state['steps'] + 1}"

                        messages.append(
                            self._forced_tool_message(
                                "get_inventory",
                                {
                                    "sku": sku,
                                    "dark_store": store_id,
                                },
                                call_id,
                            )
                        )

                        return {
                            "messages": messages,
                            "steps": state["steps"] + 1,
                        }

        # After deterministic live-tool routing, let the LLM write the final answer.
        # No more live tools are needed for this question.
        out_of_budget = (
            state["steps"] >= self.settings.agent_max_steps - 1
            or state["live_tool_calls"] >= self.settings.tool_call_cap
        )

        if out_of_budget:
            messages.append({"role": "user", "content": BUDGET_EXHAUSTED_NOTE})

        if self._is_live_inventory_question(question) and state["live_tool_calls"] >= 2:
            tools = []
        else:
            tools = [] if out_of_budget else self.tools

        completion = self.llm.complete_with_tools(
            messages,
            tools=tools,
        )

        messages.append(completion.assistant_message)

        update: AgentState = {
            "messages": messages,
            "steps": state["steps"] + 1,
            "prompt_tokens": state["prompt_tokens"] + completion.prompt_tokens,
            "completion_tokens": state["completion_tokens"] + completion.completion_tokens,
            "cost_usd": state["cost_usd"] + completion.cost_usd,
            "generation_ms": state["generation_ms"] + completion.latency_ms,
            "models": _note_model(state["models"], completion.model),
        }

        if not completion.tool_calls:
            update["draft"] = completion.text

        return update

    def route_after_decide(self, state: AgentState) -> str:
        last = state["messages"][-1]
        return "act" if last.get("tool_calls") else "reflect"

    def act(self, state: AgentState) -> AgentState:
        """Execute every tool the model requested and feed the results back."""
        messages = list(state["messages"])
        evidence = list(state["evidence"])
        tool_calls = list(state["tool_calls"])
        trace = list(state["trace"])
        live_calls = state["live_tool_calls"]
        retrieval_ms = state["retrieval_ms"]

        for request in messages[-1]["tool_calls"]:
            name = request["function"]["name"]
            try:
                arguments = json.loads(request["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                arguments = {}

            if name == KB_TOOL_NAME:
                started = time.perf_counter()
                before = len(evidence)
                content, evidence, shown = self._search_knowledge_base(arguments, evidence)
                elapsed = (time.perf_counter() - started) * 1000
                retrieval_ms += elapsed
                trace.append(
                    TraceStep(
                        kind="retrieve",
                        label="Searched the knowledge base",
                        detail=str(arguments.get("query") or ""),
                        ok=bool(shown),
                        latency_ms=elapsed,
                        markers=shown,
                        arguments={"new_passages": len(evidence) - before},
                    )
                )
            elif name in self.live_tool_names and live_calls < self.settings.tool_call_cap:
                live_calls += 1
                content, evidence, call, summary = self._call_live_tool(name, arguments, evidence)
                tool_calls.append(call)
                trace.append(
                    TraceStep(
                        kind="tool",
                        label=f"Called {name}",
                        detail=(call.error or summary)[:200],
                        ok=call.ok,
                        latency_ms=call.latency_ms,
                        arguments=call.arguments,
                        markers=[evidence[-1].marker] if call.ok else [],
                    )
                )
            elif name in self.live_tool_names:
                content = BUDGET_EXHAUSTED_NOTE
                trace.append(
                    TraceStep(
                        kind="tool",
                        label=f"Declined {name}: tool budget spent",
                        detail=f"cap of {self.settings.tool_call_cap} live calls reached",
                        ok=False,
                    )
                )
            else:
                content = f"Unknown tool '{name}'. Available: {', '.join(self.live_tool_names)}."

            messages.append({"role": "tool", "tool_call_id": request["id"], "content": content})

        return {
            "messages": messages,
            "evidence": evidence,
            "tool_calls": tool_calls,
            "trace": trace,
            "live_tool_calls": live_calls,
            "retrieval_ms": retrieval_ms,
        }

    def reflect(self, state: AgentState) -> AgentState:
        """Check the draft against the gathered evidence before returning it."""
        trace = list(state.get("trace", []))
        draft = str(state.get("draft") or "").strip()

        if not draft:
            trace.append(
                TraceStep(
                    kind="reflect",
                    label="Refused: empty draft",
                    detail="The answer draft was empty.",
                    ok=False,
                )
            )
            return {
                "final": REFUSAL_TEXT,
                "refused": True,
                "reflection": {
                    "supported": False,
                    "unsupported_claims": ["empty answer"],
                },
                "trace": trace,
            }

        if draft.startswith(REFUSAL_TEXT):
            trace.append(
                TraceStep(
                    kind="reflect",
                    label="Reflection skipped for refusal",
                    detail="The answerer already returned the refusal text.",
                    ok=True,
                    latency_ms=0.0,
                )
            )
            return {
                "final": draft,
                "refused": True,
                "reflection": None,
                "trace": trace,
            }

        if not state["evidence"]:
            trace.append(
                TraceStep(
                    kind="reflect",
                    label="Refused: the evidence does not contain the answer",
                    detail="The draft made claims without any retrieved passage or tool result.",
                    ok=False,
                )
            )
            return {
                "final": REFUSAL_TEXT,
                "refused": True,
                "reflection": {
                    "supported": False,
                    "unsupported_claims": ["no evidence"],
                },
                "trace": trace,
            }

        # A live inventory result is authoritative for stock questions.
        # Do not let the LLM refusal override successful get_inventory evidence.
        if draft.startswith(REFUSAL_TEXT):
            inventory_evidence = next(
                (
                    item
                    for item in reversed(state["evidence"])
                    if getattr(item, "doc_id", "").startswith("tool:get_inventory")
                    and '"found": true' in getattr(item, "text", "")
                ),
                None,
            )

            if inventory_evidence is not None:
                text = inventory_evidence.text
                match = re.search(
                    r'"product_name":\s*"([^"]+)".*?'
                    r'"store_name":\s*"([^"]+)".*?'
                    r'"on_hand_units":\s*(\d+).*?'
                    r'"reserved_units":\s*(\d+).*?'
                    r'"available_units":\s*(\d+).*?'
                    r'"in_stock":\s*(true|false)',
                    text,
                    re.DOTALL,
                )

                if match:
                    product, store, on_hand, reserved, available, in_stock = match.groups()
                    draft = (
                        f"{product} is available at {store}, with {available} "
                        f"available units ({on_hand} on hand, {reserved} reserved). "
                        f"[{inventory_evidence.marker}]"
                    )

        evidence_text = format_evidence(
            [(item.marker, item.label, item.text) for item in state["evidence"]]
        )

        started = time.perf_counter()

        try:
            prompt = build_reflect_prompt(
                question=state["question"],
                evidence=evidence_text,
                draft=draft,
            )

            result = self.llm.complete(
                REFLECT_SYSTEM_PROMPT,
                prompt,
            )

            latency_ms = (time.perf_counter() - started) * 1000
            verdict = parse_verdict(result.text)

            if verdict is None:
                trace.append(
                    TraceStep(
                        kind="reflect",
                        label="Reflection returned an invalid verdict",
                        detail=result.text,
                        ok=False,
                        latency_ms=latency_ms,
                    )
                )
                return {
                    "final": draft,
                    "refused": False,
                    "reflection": None,
                    "trace": trace,
                }

            reflection = verdict.model_dump()

            if verdict.supported:
                final = draft
                refused = False
            elif verdict.revised_answer and verdict.revised_answer.strip() != REFUSAL_TEXT:
                final = verdict.revised_answer.strip()
                refused = False
            else:
                final = REFUSAL_TEXT
                refused = True

            trace.append(
                TraceStep(
                    kind="reflect",
                    label="Reflection completed",
                    detail=result.text,
                    ok=True,
                    latency_ms=latency_ms,
                )
            )

            return {
                "final": final,
                "refused": refused,
                "reflection": reflection,
                "prompt_tokens": state["prompt_tokens"] + result.prompt_tokens,
                "completion_tokens": state["completion_tokens"] + result.completion_tokens,
                "cost_usd": state["cost_usd"] + result.cost_usd,
                "generation_ms": state["generation_ms"] + result.latency_ms,
                "models": _note_model(state["models"], f"reflect={result.model}"),
                "trace": trace,
            }

        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000
            trace.append(
                TraceStep(
                    kind="reflect",
                    label="Reflection failed",
                    detail=str(exc),
                    ok=False,
                    latency_ms=latency_ms,
                )
            )

            return {
                "final": draft,
                "refused": False,
                "reflection": None,
                "trace": trace,
            }

    # -- helpers -----------------------------------------------------------------

    def _search_knowledge_base(
        self, arguments: dict, evidence: list[Evidence]
    ) -> tuple[str, list[Evidence], list[int]]:
        query = str(arguments.get("query") or "").strip()
        if not query:
            return "search_knowledge_base needs a non-empty query.", evidence, []

        retrieved = self.store.search(
            self.embedder.encode_query(query), self.settings.retrieval_top_k
        )
        known = {e.chunk_id for e in evidence}
        shown: list[tuple[int, str, str]] = []
        for item in retrieved:
            existing = next((e for e in evidence if e.chunk_id == item.chunk.chunk_id), None)
            if existing is None:
                existing = Evidence(
                    marker=len(evidence) + 1,
                    label=item.chunk.citation_label,
                    text=item.chunk.text,
                    kind="passage",
                    chunk_id=item.chunk.chunk_id,
                    doc_id=item.chunk.doc_id,
                    score=item.score,
                )
                evidence.append(existing)
                known.add(existing.chunk_id)
            shown.append((existing.marker, existing.label, existing.text))

        if not shown:
            return "No passages matched.", evidence, []
        return format_evidence(shown), evidence, [marker for marker, _, _ in shown]

    def _call_live_tool(
        self, name: str, arguments: dict, evidence: list[Evidence]
    ) -> tuple[str, list[Evidence], ToolCall, str]:
        """Returns the model-facing content, updated evidence, the call record, and a
        one-line human summary of what the tool reported."""
        result = self.mcp.call_tool(name, arguments)
        if not result.call.ok:
            return f"Tool error from {name}: {result.call.error}", evidence, result.call, ""
        structured = result.structured or {}
        summary = str(structured.get("message") or "")
        if not summary and "count" in structured:
            summary = f"{structured['count']} catalog match(es)"
        summary = summary or result.text[:160]

        arg_text = ", ".join(f"{k}={v}" for k, v in result.call.arguments.items())
        marker = len(evidence) + 1
        item = Evidence(
            marker=marker,
            label=f"Live tool {name}({arg_text})",
            text=result.as_model_text(),
            kind="tool",
            chunk_id=f"tool::{name}::{marker:03d}",
            doc_id=f"tool:{name}",
            score=1.0,
        )
        evidence.append(item)
        return (
            format_evidence([(item.marker, item.label, item.text)]),
            evidence,
            result.call,
            summary,
        )


def _note_model(models: list[str], model: str) -> list[str]:
    return models if model in models else [*models, model]


def parse_verdict(text: str) -> ReflectionVerdict | None:
    match = _JSON_BLOCK.search(text)
    if not match:
        return None
    try:
        return ReflectionVerdict.model_validate(json.loads(match.group(0)))
    except (json.JSONDecodeError, ValidationError):
        return None


def evidence_items(evidence: list[Evidence]) -> list[EvidenceItem]:
    return [
        EvidenceItem(
            marker=e.marker,
            label=e.label,
            doc_id=e.doc_id,
            kind=e.kind,
            text=e.text,
            score=e.score,
        )
        for e in evidence
    ]


def citations_from_evidence(answer: str, evidence: list[Evidence]) -> list[Citation]:
    by_marker = {e.marker: e for e in evidence}
    seen: set[int] = set()
    citations: list[Citation] = []
    for raw in _CITATION_MARKER.findall(answer):
        marker = int(raw)
        if marker in seen or marker not in by_marker:
            continue
        seen.add(marker)
        item = by_marker[marker]
        citations.append(
            Citation(
                marker=marker,
                chunk_id=item.chunk_id,
                doc_id=item.doc_id,
                label=item.label,
                score=item.score,
            )
        )
    return sorted(citations, key=lambda c: c.marker)
