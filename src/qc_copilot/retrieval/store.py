"""Qdrant-backed vector store for the knowledge base."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from qc_copilot.config import Settings
from qc_copilot.models import Chunk, RetrievedChunk, SourceKind

# Fixed namespace so a chunk_id always maps to the same point id and re-ingesting
# overwrites a chunk in place instead of duplicating it.
_POINT_NAMESPACE = uuid.UUID("6f2a1d54-9c1b-4f2e-8a77-0f3c9d2b7e11")


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))


class VectorStore:
    def __init__(self, client: QdrantClient, collection: str, dimension: int) -> None:
        self.client = client
        self.collection = collection
        self.dimension = dimension

    @classmethod
    def from_settings(cls, settings: Settings, dimension: int) -> VectorStore:
        if settings.qdrant_path:
            client = QdrantClient(path=settings.qdrant_path)
        else:
            client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key or None,
                timeout=30,
            )
        return cls(client, settings.qdrant_collection, dimension)

    def exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def ensure_collection(self, recreate: bool = False) -> None:
        if recreate and self.exists():
            self.client.delete_collection(self.collection)
        if not self.exists():
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=qmodels.VectorParams(
                    size=self.dimension,
                    distance=qmodels.Distance.COSINE,
                ),
            )
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name="kind",
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )

    def count(self) -> int:
        return int(self.client.count(self.collection, exact=True).count)

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be the same length")
        points = [
            qmodels.PointStruct(
                id=_point_id(chunk.chunk_id),
                vector=list(vector),
                payload=chunk.model_dump(mode="json"),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        for start in range(0, len(points), 128):
            self.client.upsert(self.collection, points=points[start : start + 128], wait=True)
        return len(points)

    def search(
        self,
        vector: Sequence[float],
        top_k: int,
        kinds: Sequence[SourceKind] | None = None,
    ) -> list[RetrievedChunk]:
        query_filter = None
        if kinds:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="kind",
                        match=qmodels.MatchAny(any=[k.value for k in kinds]),
                    )
                ]
            )

        response = self.client.query_points(
            collection_name=self.collection,
            query=list(vector),
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        return [
            RetrievedChunk(chunk=Chunk.model_validate(point.payload), score=float(point.score))
            for point in response.points
            if point.payload is not None
        ]
