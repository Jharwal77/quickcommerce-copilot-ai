from qc_copilot.agent.guardrails import enforce_citations, normalize_markers, split_units
from qc_copilot.rag.prompts import REFUSAL_TEXT

VALID = {1, 2, 3}


def test_split_units_breaks_lines_and_sentences():
    text = "First fact [1]. Second fact [2].\n- Bullet one [3]\n- Bullet two"
    assert split_units(text) == [
        "First fact [1].",
        "Second fact [2].",
        "- Bullet one [3]",
        "- Bullet two",
    ]


def test_fully_cited_answer_passes_unchanged():
    report = enforce_citations("Dairy is 2 hours [1]. Cards take 5 to 7 days [2].", VALID)
    assert report.answer == "Dairy is 2 hours [1]. Cards take 5 to 7 days [2]."
    assert report.refused is False
    assert report.kept == 2
    assert report.dropped_sentences == []


def test_uncited_sentence_is_dropped():
    report = enforce_citations("Dairy is 2 hours [1]. The CEO is very tall.", VALID)
    assert report.answer == "Dairy is 2 hours [1]."
    assert report.dropped_sentences == ["The CEO is very tall."]


def test_invalid_marker_is_stripped_and_sentence_treated_as_uncited():
    report = enforce_citations("Dairy is 2 hours [1]. Refunds arrive in gold [9].", VALID)
    assert report.answer == "Dairy is 2 hours [1]."
    assert report.removed_markers == [9]
    assert report.dropped_sentences == ["Refunds arrive in gold [9]."]


def test_sentence_with_one_valid_and_one_invalid_marker_keeps_the_valid_one():
    report = enforce_citations("Two hours [1][9].", VALID)
    assert report.answer == "Two hours [1]."
    assert report.removed_markers == [9]


def test_nothing_supportable_becomes_a_refusal():
    report = enforce_citations("The CEO is tall. Refunds arrive in gold [9].", VALID)
    assert report.answer == REFUSAL_TEXT
    assert report.refused is True
    assert report.kept == 0
    assert len(report.dropped_sentences) == 2


def test_refusal_passes_through_as_refusal():
    report = enforce_citations(REFUSAL_TEXT, VALID)
    assert report.answer == REFUSAL_TEXT
    assert report.refused is True


def test_empty_answer_is_a_refusal():
    assert enforce_citations("   ", VALID).refused is True


def test_list_intro_line_is_kept_only_when_a_cited_item_follows():
    text = "At Indiranagar the paneer has:\n- 40 on hand [1]\n- 5 reserved [1]"
    report = enforce_citations(text, VALID)
    assert report.answer.splitlines()[0] == "At Indiranagar the paneer has:"
    assert report.kept == 3

    orphan = "At Indiranagar the paneer has:\n- lots of stock"
    report = enforce_citations(orphan, VALID)
    assert report.refused is True


def test_multiline_answers_preserve_line_structure():
    text = "Two hours [1].\nCards take 5 to 7 days [2].\nMade up line."
    report = enforce_citations(text, VALID)
    assert report.answer == "Two hours [1].\nCards take 5 to 7 days [2]."


def test_marker_after_the_full_stop_stays_with_its_sentence():
    assert split_units("Dairy is 2 hours. [1]") == ["Dairy is 2 hours. [1]"]
    report = enforce_citations("Dairy is 2 hours. [1]", VALID)
    assert report.answer == "Dairy is 2 hours. [1]"
    assert report.dropped_sentences == []


def test_marker_on_its_own_line_is_attached_to_the_previous_sentence():
    text = "Dairy items must be reported within **2 hours** of delivery.\n\n[1]"
    report = enforce_citations(text, VALID)
    assert report.kept == 1
    assert report.answer.startswith("Dairy items must be reported")
    assert report.answer.endswith("[1]")


def test_multiple_trailing_markers_merge():
    assert split_units("Fact. [1] [2]") == ["Fact. [1] [2]"]
    assert split_units("Fact. [1][2].") == ["Fact. [1][2]."]


def test_an_answer_made_only_of_markers_is_a_refusal():
    report = enforce_citations("[1]", VALID)
    assert report.refused is True
    assert report.answer == REFUSAL_TEXT


def test_normalize_markers_handles_every_model_variant():
    assert normalize_markers("Paneer 200 g【1】 and 400 g【 2 】") == "Paneer 200 g[1] and 400 g[2]"
    assert normalize_markers("Both facts [1, 2].") == "Both facts [1][2]."
    assert (
        normalize_markers("Odd spacing [ 3 ] and [4 ] and [ 5]")
        == "Odd spacing [3] and [4] and [5]"
    )
    assert normalize_markers("Already fine [1].") == "Already fine [1]."


def test_fullwidth_markers_count_as_citations():
    report = enforce_citations("- Nandhini Fresh Paneer 200 g【1】\n- Paneer 400 g【2】", {1, 2, 3})
    assert report.refused is False
    assert report.kept == 2
    assert "[1]" in report.answer and "【" not in report.answer
