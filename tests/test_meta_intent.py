import pytest

from qc_copilot.agent.pipeline import AgentPipeline
from qc_copilot.config import Settings
from qc_copilot.evaluation.golden import load_golden_set
from qc_copilot.meta_intent import (
    SYSTEM_TRACE_LABEL,
    capability_answer,
    capability_response,
    is_meta_intent,
    normalize,
)
from qc_copilot.models import AskRequest

META_QUESTIONS = [
    "what can I ask",
    "What can I ask you?",
    "what can you do",
    "What can you do for me?",
    "help",
    "Help!",
    "what do you know",
    "What do you know?",
    "how does this work",
    "How does this work?",
    "How do you work?",
    "Hi, what can you help with?",
    "what's this?",
    "What is this tool?",
    "who are you",
    "What kind of questions can I ask?",
    "What topics do you cover?",
    "Can you help me?",
    "show me some examples",
    "What are your capabilities?",
    "Where does your data come from?",
    "how do I use this, please",
]

ORDINARY_QUESTIONS = [
    "How long does a customer have to report a problem with a dairy item?",
    "When is the delivery fee waived, and what does an order below that pay?",
    "Can a vegetarian item be substituted with a non-vegetarian one?",
    "How many units of Nandhini Fresh Paneer 200 g are available at the Indiranagar Dark Store "
    "right now?",
    "Is Nandhini Fresh Paneer 200 g in stock at the Whitefield Dark Store?",
    "What do you know about the returns window for frozen items?",
    "How does the substitution policy work for out-of-stock items?",
    "Can you help me find the reorder point for Tinytots diapers at Koramangala?",
    "What can I return after two days?",
    "What is the delivery fee?",
    "Who is responsible for cold-chain checks at a dark store?",
    "how does the refund work",
    "what can i ask the store manager about",
    "help me understand the delivery fee tiers",
]


@pytest.mark.parametrize("question", META_QUESTIONS)
def test_capability_questions_are_detected(question):
    assert is_meta_intent(question)


@pytest.mark.parametrize("question", ORDINARY_QUESTIONS)
def test_ordinary_questions_go_to_the_agent(question):
    assert not is_meta_intent(question)


def test_every_golden_question_goes_to_the_agent():
    tripped = [row.id for row in load_golden_set() if is_meta_intent(row.question)]
    assert tripped == []


def test_long_questions_never_match_even_when_they_start_like_a_meta_question():
    assert not is_meta_intent("what can you do " + "really " * 8 + "fast")
    assert not is_meta_intent("")
    assert not is_meta_intent("???")


def test_normalize_strips_filler_and_contractions():
    assert normalize("Hey, what’s this, please?") == "what is this"
    assert normalize("  HELP  ") == "help"


def test_capability_answer_names_the_tools_and_never_cites():
    text = capability_answer(["get_inventory", "search_catalog"])
    assert "`get_inventory`" in text and "`search_catalog`" in text
    assert "What I know" in text and "example chips" in text
    assert "[1]" not in text and "【" not in text
    assert "switched off" in capability_answer([])


def test_capability_response_is_a_labelled_system_answer():
    response = capability_response("what can you do", ["get_inventory"])
    assert response.mode == "system"
    assert response.refused is False
    assert response.citations == [] and response.evidence == []
    assert [step.label for step in response.trace] == [SYSTEM_TRACE_LABEL]
    assert response.guardrails is None


class _NeverRunGraph:
    class mcp:
        @staticmethod
        def tool_names():
            return ["get_inventory", "search_catalog"]

    def run(self, question):
        raise AssertionError("the agent loop must not run for a capability question")


def test_agent_pipeline_short_circuits_capability_questions():
    settings = Settings(_env_file=None, groq_api_key="g")
    pipeline = AgentPipeline(settings, _NeverRunGraph())
    response = pipeline.answer(AskRequest(question="what can I ask here?"))
    assert response.mode == "system"
    assert "`search_catalog`" in response.answer
