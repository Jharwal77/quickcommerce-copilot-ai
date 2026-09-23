from qc_copilot.ingest.chunker import (
    chunk_document,
    count_tokens,
    split_sections,
)
from qc_copilot.models import KBDocument, SourceKind

MARKDOWN = """# Returns Policy

Intro paragraph that sits above any section heading.

## Return windows

Dairy must be reported within 2 hours.

## Refund timelines

UPI refunds land within 24 hours.

Card refunds take 5 to 7 business days.
"""


def _doc(text: str = MARKDOWN) -> KBDocument:
    return KBDocument(
        doc_id="policy-returns",
        title="Returns Policy",
        kind=SourceKind.POLICY,
        text=text,
        metadata={"version": "1.0"},
    )


def test_split_sections_keeps_preamble_and_drops_level_one_heading():
    sections = split_sections(MARKDOWN)
    assert [heading for heading, _ in sections] == [None, "Return windows", "Refund timelines"]
    assert "Intro paragraph" in sections[0][1]
    assert "# Returns Policy" not in sections[0][1]


def test_split_sections_ignores_empty_sections():
    assert split_sections("## Empty\n\n## Filled\n\nbody") == [("Filled", "body")]


def test_every_chunk_carries_title_and_section_prefix():
    chunks = chunk_document(_doc())
    windows = [c for c in chunks if c.section == "Return windows"]
    assert len(windows) == 1
    assert windows[0].text.startswith("Returns Policy > Return windows")
    assert windows[0].citation_label == "Returns Policy - Return windows"


def test_chunks_never_merge_across_sections():
    chunks = chunk_document(_doc())
    for chunk in chunks:
        if chunk.section == "Return windows":
            assert "UPI refunds" not in chunk.text


def test_chunk_ids_are_stable_and_ordered():
    first = [c.chunk_id for c in chunk_document(_doc())]
    second = [c.chunk_id for c in chunk_document(_doc())]
    assert first == second
    assert first == ["policy-returns::000", "policy-returns::001", "policy-returns::002"]


def test_chunks_respect_the_token_budget():
    body = "\n\n".join(f"Sentence number {i} carries some policy detail." for i in range(60))
    chunks = chunk_document(_doc(f"## Long section\n\n{body}"), chunk_tokens=120, overlap_tokens=20)
    assert len(chunks) > 1
    assert all(c.token_count <= 120 for c in chunks)


def test_overlap_repeats_trailing_content_between_chunks():
    body = "\n\n".join(f"Fact {i} about refunds and returns." for i in range(40))
    chunks = chunk_document(_doc(f"## Long section\n\n{body}"), chunk_tokens=120, overlap_tokens=40)
    assert len(chunks) > 1
    tail = chunks[0].text.strip().splitlines()[-1]
    assert tail in chunks[1].text


def test_zero_overlap_produces_disjoint_chunks():
    body = "\n\n".join(f"Fact {i} about refunds." for i in range(40))
    chunks = chunk_document(_doc(f"## Long section\n\n{body}"), chunk_tokens=120, overlap_tokens=0)
    tail = chunks[0].text.strip().splitlines()[-1]
    assert tail not in chunks[1].text


def test_oversized_paragraph_is_split_by_sentence():
    paragraph = " ".join(f"This is policy sentence {i} and it is quite wordy." for i in range(60))
    chunks = chunk_document(_doc(f"## Dense\n\n{paragraph}"), chunk_tokens=120, overlap_tokens=0)
    assert len(chunks) > 1
    assert all(c.token_count <= 120 for c in chunks)


def test_overlap_must_be_smaller_than_chunk_size():
    try:
        chunk_document(_doc(), chunk_tokens=100, overlap_tokens=100)
    except ValueError as exc:
        assert "overlap_tokens" in str(exc)
    else:
        raise AssertionError("expected a ValueError")


def test_chunk_size_too_small_for_prefix_is_rejected():
    try:
        chunk_document(_doc(), chunk_tokens=20, overlap_tokens=0)
    except ValueError as exc:
        assert "too small" in str(exc)
    else:
        raise AssertionError("expected a ValueError")


def test_count_tokens_is_positive_and_monotonic():
    assert count_tokens("short") >= 1
    assert count_tokens("a much longer stretch of policy text") > count_tokens("short")
