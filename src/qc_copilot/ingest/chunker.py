"""Section-aware chunking for the knowledge base.

Policy documents carry meaning in their headings: "within 2 hours" only makes sense
under the heading it sits below. So chunks are packed inside heading boundaries and
every chunk repeats its document title and section heading. That keeps a retrieved
chunk self-describing, which is what makes the citation label and the faithfulness
check usable.
"""

from __future__ import annotations

import re
from functools import lru_cache

import tiktoken

from qc_copilot.models import Chunk, KBDocument

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@lru_cache(maxsize=1)
def _encoder() -> tiktoken.Encoding:
    # cl100k_base is not the tokenizer the serving model uses, but chunk sizing only
    # needs a stable, consistent length measure, not an exact per-model token count.
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoder().encode(text))


def split_sections(markdown: str) -> list[tuple[str | None, str]]:
    """Split markdown into (heading, body) pairs on level-2 and deeper headings."""
    sections: list[tuple[str | None, list[str]]] = [(None, [])]

    for line in markdown.splitlines():
        match = _HEADING.match(line)
        if match and len(match.group(1)) >= 2:
            sections.append((match.group(2).strip(), []))
        elif match:
            # A level-1 heading is the document title, already carried in metadata.
            continue
        else:
            sections[-1][1].append(line)

    result: list[tuple[str | None, str]] = []
    for heading, lines in sections:
        body = "\n".join(lines).strip()
        if body:
            result.append((heading, body))
    return result


def _split_units(body: str, budget: int) -> list[str]:
    """Break a section body into packable units, splitting only what is oversized."""
    units: list[str] = []
    for paragraph in (p.strip() for p in re.split(r"\n\s*\n", body)):
        if not paragraph:
            continue
        if count_tokens(paragraph) <= budget:
            units.append(paragraph)
            continue
        sentences = [s.strip() for s in _SENTENCE_END.split(paragraph) if s.strip()]
        units.extend(sentences or [paragraph])
    return units


def _overlap_tail(units: list[str], overlap_tokens: int) -> list[str]:
    if overlap_tokens <= 0:
        return []
    tail: list[str] = []
    budget = overlap_tokens
    for unit in reversed(units):
        cost = count_tokens(unit)
        if cost > budget:
            break
        tail.insert(0, unit)
        budget -= cost
    return tail


def chunk_document(
    document: KBDocument,
    chunk_tokens: int = 220,
    overlap_tokens: int = 40,
) -> list[Chunk]:
    """Chunk one document, never merging text across section boundaries."""
    if overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens must be smaller than chunk_tokens")

    chunks: list[Chunk] = []
    ordinal = 0

    for heading, body in split_sections(document.text):
        prefix = _build_prefix(document.title, heading)
        budget = chunk_tokens - count_tokens(prefix)
        if budget < 16:
            raise ValueError(
                f"chunk_tokens={chunk_tokens} is too small for the heading prefix in "
                f"{document.doc_id}"
            )

        pending: list[str] = []
        pending_tokens = 0

        for unit in _split_units(body, budget):
            cost = count_tokens(unit)
            if pending and pending_tokens + cost > budget:
                chunks.append(_make_chunk(document, heading, prefix, pending, ordinal))
                ordinal += 1
                pending = _overlap_tail(pending, overlap_tokens)
                pending_tokens = sum(count_tokens(u) for u in pending)
            pending.append(unit)
            pending_tokens += cost

        if pending:
            chunks.append(_make_chunk(document, heading, prefix, pending, ordinal))
            ordinal += 1

    return chunks


def _build_prefix(title: str, heading: str | None) -> str:
    return f"{title} > {heading}\n\n" if heading else f"{title}\n\n"


def _make_chunk(
    document: KBDocument,
    heading: str | None,
    prefix: str,
    units: list[str],
    ordinal: int,
) -> Chunk:
    text = prefix + "\n\n".join(units)
    return Chunk(
        chunk_id=f"{document.doc_id}::{ordinal:03d}",
        doc_id=document.doc_id,
        title=document.title,
        kind=document.kind,
        text=text,
        token_count=count_tokens(text),
        section=heading,
        ordinal=ordinal,
        metadata=dict(document.metadata),
    )


def chunk_documents(
    documents: list[KBDocument],
    chunk_tokens: int = 220,
    overlap_tokens: int = 40,
) -> list[Chunk]:
    return [
        chunk
        for document in documents
        for chunk in chunk_document(document, chunk_tokens, overlap_tokens)
    ]
