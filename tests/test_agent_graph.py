"""The loop is exercised with a scripted model and scripted tools, no network."""

import json
from dataclasses import dataclass

import pytest

from qc_copilot.agent.graph import AgentGraph, citations_from_evidence, parse_verdict
from qc_copilot.agent.pipeline import AgentPipeline
from qc_copilot.agent.prompts import BUDGET_EXHAUSTED_NOTE
from qc_copilot.config import Settings
from qc_copilot.llm.client import Completion, RequestedToolCall, ToolCompletion
from qc_copilot.mcp_server.client import ToolResult, ToolSpec
from qc_copilot.models import AskRequest, Chunk, RetrievedChunk, SourceKind, ToolCall
from qc_copilot.rag.prompts import REFUSAL_TEXT

REFLECT_OK = '{"supported": true, "unsupported_claims": [], "revised_answer": ""}'


def _tool_turn(*calls: tuple[str, dict]) -> ToolCompletion:
    requested = [
        RequestedToolCall(id=f"call_{i}", name=name, arguments=args)
        for i, (name, args) in enumerate(calls)
    ]
    return ToolCompletion(
        text="",
        tool_calls=requested,
        assistant_message={
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": r.id,
                    "type": "function",
                    "function": {"name": r.name, "arguments": json.dumps(r.arguments)},
                }
                for r in requested
            ],
        },
        model="fake",
        prompt_tokens=100,
        completion_tokens=10,
        latency_ms=5.0,
        cost_usd=0.00001,
    )


def _answer_turn(text: str) -> ToolCompletion:
    return ToolCompletion(
        text=text,
        tool_calls=[],
        assistant_message={"role": "assistant", "content": text},
        model="fake",
        prompt_tokens=100,
        completion_tokens=20,
        latency_ms=5.0,
        cost_usd=0.00001,
    )


class ScriptedLLM:
    def __init__(self, turns: list[ToolCompletion], reflections: list[str] | None = None):
        self.turns = list(turns)
        self.reflections = list(reflections or [REFLECT_OK])
        self.tool_turn_log: list[list[dict]] = []

    def complete_with_tools(self, messages, tools, **_):
        self.tool_turn_log.append(tools)
        if not self.turns:
            raise AssertionError("model asked for more turns than scripted")
        return self.turns.pop(0)

    def complete(self, system_prompt, user_prompt, **_):
        text = self.reflections.pop(0) if self.reflections else REFLECT_OK
        return Completion(
            text=text,
            model="fake",
            prompt_tokens=50,
            completion_tokens=15,
            latency_ms=3.0,
            cost_usd=0.000005,
        )


class FakeEmbedder:
    dimension = 4

    def encode_query(self, text):
        return [0.0, 0.0, 0.0, 1.0]


class FakeStore:
    def __init__(self):
        self.queries: list[str] = []

    def search(self, vector, top_k, kinds=None):
        return [
            RetrievedChunk(
                chunk=Chunk(
                    chunk_id="policy-returns::001",
                    doc_id="policy-returns",
                    title="Returns Policy",
                    kind=SourceKind.POLICY,
                    text="Dairy & Eggs: report within 2 hours of delivery.",
                    token_count=12,
                    section="Return windows",
                ),
                score=0.8,
            )
        ][:top_k]


@dataclass
class FakeMCP:
    calls: list = None

    def __post_init__(self):
        self.calls = []
        self.tools = [
            ToolSpec("get_inventory", "stock", {"type": "object", "properties": {}}),
            ToolSpec("search_catalog", "catalog", {"type": "object", "properties": {}}),
        ]

    def tool_names(self):
        return [t.name for t in self.tools]

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if arguments.get("sku") == "BROKEN":
            return ToolResult(call=ToolCall(name=name, arguments=arguments, ok=False, error="boom"))
        return ToolResult(
            call=ToolCall(name=name, arguments=arguments, ok=True, latency_ms=2.0),
            structured={"found": True, "record": {"available_units": 35}},
            text="35 available",
        )

    def close(self):
        pass


def _graph(llm, settings: Settings | None = None) -> AgentGraph:
    settings = settings or Settings(tool_call_cap=2, agent_max_steps=4, retrieval_top_k=3)
    return AgentGraph(settings, FakeEmbedder(), FakeStore(), llm, FakeMCP())


def test_retrieval_then_answer_produces_a_cited_passage():
    llm = ScriptedLLM(
        [
            _tool_turn(("search_knowledge_base", {"query": "dairy return window"})),
            _answer_turn("Dairy must be reported within 2 hours [1]."),
        ]
    )
    state = _graph(llm).run("What is the dairy return window?")
    assert state["final"] == "Dairy must be reported within 2 hours [1]."
    assert state["refused"] is False
    assert [e.kind for e in state["evidence"]] == ["passage"]
    assert state["tool_calls"] == []
    assert state["retrieval_ms"] >= 0


def test_live_tool_result_becomes_numbered_evidence_and_a_tool_call_record():
    llm = ScriptedLLM(
        [
            _tool_turn(("get_inventory", {"sku": "QC-X", "dark_store": "DS-BLR-001"})),
            _answer_turn("There are 35 units available [1]."),
        ]
    )
    graph = _graph(llm)
    state = graph.run("How many units of X at Indiranagar?")
    assert state["tool_calls"][0].name == "get_inventory"
    assert state["tool_calls"][0].ok is True
    assert state["evidence"][0].kind == "tool"
    assert state["evidence"][0].doc_id == "tool:get_inventory"
    assert "35" in state["messages"][-3]["content"] or "35" in state["evidence"][0].text
    citations = citations_from_evidence(state["final"], state["evidence"])
    assert citations[0].doc_id == "tool:get_inventory"


def test_tool_error_is_reported_to_the_model_and_recorded_as_failed():
    llm = ScriptedLLM(
        [
            _tool_turn(("get_inventory", {"sku": "BROKEN", "dark_store": "DS-BLR-001"})),
            _answer_turn(REFUSAL_TEXT),
        ]
    )
    state = _graph(llm).run("How many units of BROKEN?")
    assert state["tool_calls"][0].ok is False
    assert "Tool error" in state["messages"][-2]["content"]
    assert state["refused"] is True
    assert state["evidence"] == []


def test_tool_call_cap_stops_further_live_calls_and_withholds_tools():
    llm = ScriptedLLM(
        [
            _tool_turn(("get_inventory", {"sku": "A", "dark_store": "DS-BLR-001"})),
            _tool_turn(("get_inventory", {"sku": "B", "dark_store": "DS-BLR-001"})),
            _answer_turn("A has 35 [1] and B has 35 [2]."),
        ]
    )
    graph = _graph(llm)
    state = graph.run("Stock of A and B?")
    assert state["live_tool_calls"] == 2
    # After the cap is reached the next decide turn offers no tools at all.
    assert llm.tool_turn_log[-1] == []
    assert any(m.get("content") == BUDGET_EXHAUSTED_NOTE for m in state["messages"])


def test_requests_beyond_the_cap_in_one_turn_are_refused_not_executed():
    llm = ScriptedLLM(
        [
            _tool_turn(
                ("get_inventory", {"sku": "A", "dark_store": "DS-BLR-001"}),
                ("get_inventory", {"sku": "B", "dark_store": "DS-BLR-001"}),
                ("get_inventory", {"sku": "C", "dark_store": "DS-BLR-001"}),
            ),
            _answer_turn("A and B known [1][2]."),
        ]
    )
    graph = _graph(llm)
    state = graph.run("Stock of A, B, C?")
    assert len(graph.mcp.calls) == 2
    assert len(state["tool_calls"]) == 2
    tool_messages = [m for m in state["messages"] if m.get("role") == "tool"]
    assert tool_messages[-1]["content"] == BUDGET_EXHAUSTED_NOTE


def test_step_cap_forces_an_answer_even_if_the_model_keeps_searching():
    settings = Settings(tool_call_cap=5, agent_max_steps=3, retrieval_top_k=3)
    llm = ScriptedLLM(
        [
            _tool_turn(("search_knowledge_base", {"query": "a"})),
            _tool_turn(("search_knowledge_base", {"query": "b"})),
            _answer_turn("Answer [1]."),
        ]
    )
    state = _graph(llm, settings).run("Loop?")
    assert state["steps"] == 3
    assert llm.tool_turn_log[-1] == []
    assert state["final"] == "Answer [1]."


def test_repeated_passages_keep_their_original_marker():
    llm = ScriptedLLM(
        [
            _tool_turn(("search_knowledge_base", {"query": "a"})),
            _tool_turn(("search_knowledge_base", {"query": "a again"})),
            _answer_turn("Two hours [1]."),
        ]
    )
    state = _graph(llm).run("Dairy window?")
    assert len(state["evidence"]) == 1
    assert state["evidence"][0].marker == 1


def test_reflection_revises_an_unsupported_draft():
    verdict = json.dumps(
        {
            "supported": False,
            "unsupported_claims": ["9 days"],
            "revised_answer": "Dairy must be reported within 2 hours [1].",
        }
    )
    llm = ScriptedLLM(
        [
            _tool_turn(("search_knowledge_base", {"query": "dairy"})),
            _answer_turn("Dairy must be reported within 9 days [1]."),
        ],
        reflections=[verdict],
    )
    state = _graph(llm).run("Dairy window?")
    assert state["final"] == "Dairy must be reported within 2 hours [1]."
    assert state["reflection"]["unsupported_claims"] == ["9 days"]
    assert state["refused"] is False


def test_reflection_with_nothing_supportable_refuses():
    verdict = json.dumps(
        {"supported": False, "unsupported_claims": ["all of it"], "revised_answer": REFUSAL_TEXT}
    )
    llm = ScriptedLLM(
        [
            _tool_turn(("search_knowledge_base", {"query": "dairy"})),
            _answer_turn("The CEO lives in Pune [1]."),
        ],
        reflections=[verdict],
    )
    state = _graph(llm).run("Where does the CEO live?")
    assert state["final"] == REFUSAL_TEXT
    assert state["refused"] is True


def test_committed_answer_with_no_evidence_is_never_returned():
    llm = ScriptedLLM([_answer_turn("Paneer costs INR 83.60.")])
    state = _graph(llm).run("Paneer price?")
    assert state["final"] == REFUSAL_TEXT
    assert state["refused"] is True
    assert state["reflection"]["unsupported_claims"] == ["no evidence"]


def test_unparseable_reflection_keeps_the_draft():
    llm = ScriptedLLM(
        [
            _tool_turn(("search_knowledge_base", {"query": "dairy"})),
            _answer_turn("Two hours [1]."),
        ],
        reflections=["I cannot decide."],
    )
    state = _graph(llm).run("Dairy window?")
    assert state["final"] == "Two hours [1]."
    assert state["reflection"] is None


def test_parse_verdict_tolerates_surrounding_prose():
    verdict = parse_verdict('Sure.\n```json\n{"supported": true}\n```')
    assert verdict is not None and verdict.supported is True
    assert parse_verdict("no json here") is None
    assert parse_verdict('{"supported": "maybe"}') is None


def test_pipeline_wraps_the_state_into_a_response_with_usage():
    llm = ScriptedLLM(
        [
            _tool_turn(("search_knowledge_base", {"query": "dairy"})),
            _tool_turn(("get_inventory", {"sku": "QC-X", "dark_store": "DS-BLR-001"})),
            _answer_turn("Two hours [1] and 35 units [2]."),
        ]
    )
    settings = Settings(tool_call_cap=2, agent_max_steps=5, retrieval_top_k=3)
    pipeline = AgentPipeline(settings, _graph(llm, settings))
    response = pipeline.answer(AskRequest(question="Dairy window and stock?"))
    assert response.mode == "agentic"
    assert [c.marker for c in response.citations] == [1, 2]
    assert response.retrieved_doc_ids == ["policy-returns"]
    assert len(response.contexts) == 2
    assert response.tool_calls[0].name == "get_inventory"
    # Three decide turns plus one reflection.
    assert response.usage.prompt_tokens == 3 * 100 + 50
    assert response.usage.models == ["fake", "reflect=fake"]
    assert response.usage.total_ms > 0
    assert response.usage.estimated_cost_usd == pytest.approx(3 * 0.00001 + 0.000005)


def test_reflection_uses_main_llm_when_dedicated_reflector_is_given():
    answerer = ScriptedLLM(
        [
            _tool_turn(("search_knowledge_base", {"query": "dairy"})),
            _answer_turn("Two hours [1]."),
        ],
        reflections=[REFLECT_OK],
    )
    reflector = ScriptedLLM([], reflections=[REFLECT_OK])
    settings = Settings(tool_call_cap=2, agent_max_steps=4, retrieval_top_k=3)
    graph = AgentGraph(settings, FakeEmbedder(), FakeStore(), answerer, FakeMCP(), reflector)
    state = graph.run("Dairy window?")
    assert state["final"] == "Two hours [1]."
    assert reflector.reflections == [REFLECT_OK]


def test_tool_trace_step_carries_the_tools_message_not_its_arguments():
    llm = ScriptedLLM(
        [
            _tool_turn(("get_inventory", {"sku": "QC-X", "dark_store": "DS-BLR-001"})),
            _answer_turn("There are 35 units available [1]."),
        ]
    )
    graph = _graph(llm)
    graph.mcp.call_tool = lambda name, arguments: ToolResult(
        call=ToolCall(name=name, arguments=arguments, ok=True, latency_ms=2.0),
        structured={"found": True, "message": "Paneer at Indiranagar: 35 available."},
        text="35 available",
    )
    state = graph.run("How many units of X at Indiranagar?")
    step = next(s for s in state["trace"] if s.kind == "tool")
    assert step.detail == "Paneer at Indiranagar: 35 available."
    assert step.arguments == {"sku": "QC-X", "dark_store": "DS-BLR-001"}
    assert step.markers == [1]
