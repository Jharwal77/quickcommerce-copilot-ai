"""Build the Qdrant knowledge base from the policy and catalog sources.

Usage:
    python -m qc_copilot.ingest.pipeline [--recreate]
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

from qc_copilot.config import Settings, get_settings
from qc_copilot.ingest.chunker import chunk_documents
from qc_copilot.ingest.loader import load_all_documents
from qc_copilot.retrieval.embeddings import build_embedder
from qc_copilot.retrieval.store import VectorStore


@dataclass(frozen=True)
class IngestReport:
    documents: int
    chunks: int
    points_upserted: int
    collection: str
    dimension: int
    elapsed_seconds: float

    def summary(self) -> str:
        return (
            f"documents={self.documents} chunks={self.chunks} "
            f"upserted={self.points_upserted} collection={self.collection} "
            f"dim={self.dimension} in {self.elapsed_seconds:.1f}s"
        )


def build_index(settings: Settings | None = None, recreate: bool = False) -> IngestReport:
    settings = settings or get_settings()
    started = time.perf_counter()

    documents = load_all_documents(settings.policies_dir, settings.catalog_dir)
    chunks = chunk_documents(documents, settings.chunk_tokens, settings.chunk_overlap_tokens)

    embedder = build_embedder(settings)
    store = VectorStore.from_settings(settings, embedder.dimension)
    store.ensure_collection(recreate=recreate)

    vectors = embedder.encode_documents([chunk.text for chunk in chunks])
    upserted = store.upsert(chunks, vectors)

    return IngestReport(
        documents=len(documents),
        chunks=len(chunks),
        points_upserted=upserted,
        collection=settings.qdrant_collection,
        dimension=embedder.dimension,
        elapsed_seconds=time.perf_counter() - started,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and rebuild the collection instead of upserting in place.",
    )
    args = parser.parse_args()
    print(build_index(recreate=args.recreate).summary())


if __name__ == "__main__":
    main()
