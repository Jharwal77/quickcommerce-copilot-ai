import json

import pytest
from pydantic import ValidationError

from qc_copilot.config import get_settings
from qc_copilot.evaluation.golden import (
    EXPECTED_ROW_COUNT,
    GoldenRow,
    GoldenSetError,
    category_counts,
    load_golden_set,
)
from qc_copilot.ingest.loader import load_all_documents


@pytest.fixture(scope="module")
def rows():
    return load_golden_set(strict_count=True)


@pytest.fixture(scope="module")
def corpus_doc_ids():
    settings = get_settings()
    docs = load_all_documents(settings.policies_dir, settings.catalog_dir)
    return {d.doc_id for d in docs}


def test_golden_set_has_the_expected_size(rows):
    assert len(rows) == EXPECTED_ROW_COUNT


def test_every_expected_document_exists_in_the_corpus(rows, corpus_doc_ids):
    referenced = {doc_id for row in rows for doc_id in row.expected_doc_ids}
    assert referenced <= corpus_doc_ids, referenced - corpus_doc_ids


def test_retrieval_rows_and_tool_rows_are_both_represented(rows):
    tool_rows = [r for r in rows if r.requires_tool]
    assert len(tool_rows) >= 8
    assert len(rows) - len(tool_rows) >= 40


def test_both_implemented_tools_are_exercised(rows):
    tools = {r.expected_tool for r in rows if r.expected_tool}
    assert {"get_inventory", "search_catalog"} == {t.value for t in tools}


def test_every_category_is_covered(rows):
    counts = category_counts(rows)
    assert set(counts) == {
        "returns",
        "delivery",
        "substitutions",
        "store_ops",
        "product",
        "dark_store",
        "live_ops",
    }
    assert all(count >= 3 for count in counts.values())


def test_questions_are_phrased_as_questions(rows):
    assert all(row.question.endswith("?") for row in rows)


def test_loader_rejects_a_duplicate_id(tmp_path):
    row = {
        "id": "dup",
        "question": "What is the return window?",
        "ground_truth": "Two hours for dairy.",
        "category": "returns",
        "requires_tool": False,
        "expected_tool": None,
        "expected_doc_ids": ["policy-returns"],
    }
    path = tmp_path / "g.jsonl"
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(GoldenSetError, match="duplicate golden row ids"):
        load_golden_set(path)


def test_loader_rejects_a_tool_row_that_names_no_tool(tmp_path):
    row = {
        "id": "live-x",
        "question": "How much stock is left at the store?",
        "ground_truth": "Twelve units are available.",
        "category": "live_ops",
        "requires_tool": True,
        "expected_tool": None,
        "expected_doc_ids": [],
    }
    path = tmp_path / "g.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(GoldenSetError, match="requires a tool but names none"):
        load_golden_set(path)


def test_loader_rejects_a_retrieval_row_with_no_source(tmp_path):
    row = {
        "id": "ret-x",
        "question": "What is the return window for dairy?",
        "ground_truth": "Two hours from delivery.",
        "category": "returns",
        "requires_tool": False,
        "expected_tool": None,
        "expected_doc_ids": [],
    }
    path = tmp_path / "g.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(GoldenSetError, match="expects no documents"):
        load_golden_set(path)


def test_loader_rejects_a_tool_row_that_also_expects_documents(tmp_path):
    row = {
        "id": "live-y",
        "question": "Is this item in stock right now?",
        "ground_truth": "No, it is out of stock.",
        "category": "live_ops",
        "requires_tool": True,
        "expected_tool": "get_inventory",
        "expected_doc_ids": ["policy-returns"],
    }
    path = tmp_path / "g.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(GoldenSetError, match="also expects documents"):
        load_golden_set(path)


def test_loader_reports_the_offending_line_number(tmp_path):
    path = tmp_path / "g.jsonl"
    path.write_text("{not json}\n")
    with pytest.raises(GoldenSetError, match="line 1"):
        load_golden_set(path)


def test_loader_rejects_a_missing_file(tmp_path):
    with pytest.raises(GoldenSetError, match="not found"):
        load_golden_set(tmp_path / "absent.jsonl")


def test_row_schema_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        GoldenRow.model_validate(
            {
                "id": "x",
                "question": "A long enough question?",
                "ground_truth": "A long enough answer.",
                "category": "returns",
                "expected_doc_ids": ["policy-returns"],
                "surprise": 1,
            }
        )
