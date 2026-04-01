"""Qdrant vector storage client for Archivist."""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np

from archivist.config import Config
from archivist.exceptions import StorageError
from archivist.log import get_logger
from archivist.models import ChunkRole, MetadataPayload

logger = get_logger("storage")


class QdrantStorage:
    """Wraps qdrant-client for chunk storage and retrieval."""

    def __init__(self, config: Config) -> None:
        self._config = config.qdrant
        self._client: Any = None
        self._collection_name = config.qdrant.collection_name

    def connect(self, vector_dimension: int) -> None:
        """Establish connection to Qdrant and ensure collection exists.

        Args:
            vector_dimension: Dimension of the embedding vectors.

        Raises:
            StorageError: If connection fails.
        """
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            kwargs: dict[str, Any] = {
                "host": self._config.host,
                "port": self._config.port,
            }
            if self._config.api_key:
                kwargs["api_key"] = self._config.api_key

            self._client = QdrantClient(**kwargs)

            # Create collection if it doesn't exist
            collections = self._client.get_collections().collections
            exists = any(c.name == self._collection_name for c in collections)

            if not exists:
                distance = getattr(Distance, self._config.distance_metric.upper(), Distance.COSINE)
                self._client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=VectorParams(size=vector_dimension, distance=distance),
                )
                logger.info("Created collection", collection=self._collection_name, dimension=vector_dimension)
            else:
                logger.info("Connected to existing collection", collection=self._collection_name)

        except Exception as e:
            raise StorageError(f"Failed to connect to Qdrant at {self._config.host}:{self._config.port}: {e}") from e

    def _ensure_connected(self) -> None:
        if self._client is None:
            raise StorageError("Not connected to Qdrant. Call connect() first.")

    def upsert_chunks(self, payloads: list[MetadataPayload], vectors: np.ndarray) -> list[str]:
        """Batch upsert chunks with their embeddings.

        Args:
            payloads: Metadata payloads for each chunk.
            vectors: Embedding vectors, shape (n_chunks, dimension).

        Returns:
            List of generated point IDs.

        Raises:
            StorageError: If upsert fails.
        """
        self._ensure_connected()
        try:
            from qdrant_client.models import PointStruct

            point_ids = [str(uuid.uuid4()) for _ in payloads]
            points = [
                PointStruct(
                    id=pid,
                    vector=vec.tolist(),
                    payload=payload.to_dict(),
                )
                for pid, payload, vec in zip(point_ids, payloads, vectors, strict=True)
            ]

            self._client.upsert(collection_name=self._collection_name, points=points)
            logger.info("Upserted chunks", count=len(points))
            return point_ids

        except StorageError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to upsert {len(payloads)} chunks: {e}") from e

    def get_family_chunks(
        self, family_slug: str, doc_type: str, version: str | None = None
    ) -> list[dict[str, Any]]:
        """Retrieve existing chunks for a document family.

        Args:
            family_slug: The document family identifier.
            doc_type: The document type.
            version: Optional specific version to filter by.

        Returns:
            List of chunk dicts with id, text, chunk_index, and version info.
        """
        self._ensure_connected()
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            conditions = [
                FieldCondition(key="family_slug", match=MatchValue(value=family_slug)),
                FieldCondition(key="doc_type", match=MatchValue(value=doc_type)),
                FieldCondition(key="chunk_role", match=MatchValue(value=ChunkRole.BASE.value)),
            ]
            if version:
                conditions.append(FieldCondition(key="version", match=MatchValue(value=version)))

            results = self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=Filter(must=conditions),
                limit=10000,
                with_payload=True,
                with_vectors=False,
            )[0]

            # Also get deltas
            delta_conditions = [
                FieldCondition(key="family_slug", match=MatchValue(value=family_slug)),
                FieldCondition(key="doc_type", match=MatchValue(value=doc_type)),
                FieldCondition(key="chunk_role", match=MatchValue(value=ChunkRole.DELTA.value)),
            ]
            if version:
                delta_conditions.append(FieldCondition(key="version", match=MatchValue(value=version)))

            delta_results = self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=Filter(must=delta_conditions),
                limit=10000,
                with_payload=True,
                with_vectors=False,
            )[0]

            chunks = []
            for point in [*results, *delta_results]:
                payload = point.payload or {}
                chunks.append({
                    "id": str(point.id),
                    "text": payload.get("text", ""),
                    "chunk_index": payload.get("chunk_index", 0),
                    "chunk_role": payload.get("chunk_role", "base"),
                    "version": payload.get("version"),
                    "version_tuple": payload.get("version_tuple"),
                    "version_range_min": payload.get("version_range_min"),
                    "version_range_max": payload.get("version_range_max"),
                })

            return chunks

        except StorageError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to query family chunks: {e}") from e

    def get_version_index(self, family_slug: str, doc_type: str) -> dict[str, Any] | None:
        """Get the version summary record for a document family."""
        self._ensure_connected()
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            results = self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=Filter(must=[
                    FieldCondition(key="family_slug", match=MatchValue(value=family_slug)),
                    FieldCondition(key="doc_type", match=MatchValue(value=doc_type)),
                    FieldCondition(key="chunk_role", match=MatchValue(value=ChunkRole.VERSION_INDEX.value)),
                ]),
                limit=1,
                with_payload=True,
                with_vectors=False,
            )[0]

            if results:
                return results[0].payload
            return None

        except Exception as e:
            raise StorageError(f"Failed to get version index: {e}") from e

    def update_version_range(self, chunk_ids: list[str], version_range_max: tuple[int, int, int]) -> None:
        """Update version_range_max for existing chunks (extend validity)."""
        self._ensure_connected()
        try:
            self._client.set_payload(
                collection_name=self._collection_name,
                payload={"version_range_max": list(version_range_max)},
                points=chunk_ids,
            )
            logger.debug("Updated version range", count=len(chunk_ids), max=version_range_max)

        except Exception as e:
            raise StorageError(f"Failed to update version range: {e}") from e

    def upsert_version_index(self, index: dict[str, Any]) -> None:
        """Create or update a version index summary record."""
        self._ensure_connected()
        try:
            from qdrant_client.models import PointStruct

            # Use a deterministic ID based on family + doc_type
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{index['family_slug']}:{index['doc_type']}"))

            self._client.upsert(
                collection_name=self._collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=[0.0] * 1,  # Placeholder — version index points don't need real vectors
                        payload={**index, "chunk_role": ChunkRole.VERSION_INDEX.value},
                    )
                ],
            )
        except Exception as e:
            raise StorageError(f"Failed to upsert version index: {e}") from e

    def check_document_exists(self, source_file: str) -> bool:
        """Check if a document has already been ingested."""
        self._ensure_connected()
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            results = self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=Filter(must=[
                    FieldCondition(key="source_file", match=MatchValue(value=source_file)),
                ]),
                limit=1,
                with_payload=False,
                with_vectors=False,
            )[0]

            return len(results) > 0

        except Exception as e:
            raise StorageError(f"Failed to check document existence: {e}") from e

    def delete_partial_ingestion(self, source_file: str) -> None:
        """Delete all chunks for a source file (cleanup after partial ingestion)."""
        self._ensure_connected()
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            self._client.delete(
                collection_name=self._collection_name,
                points_selector=Filter(must=[
                    FieldCondition(key="source_file", match=MatchValue(value=source_file)),
                ]),
            )
            logger.info("Deleted partial ingestion", source_file=source_file)

        except Exception as e:
            raise StorageError(f"Failed to delete partial ingestion: {e}") from e

    def collection_stats(self) -> dict[str, Any]:
        """Get corpus statistics."""
        self._ensure_connected()
        try:
            info = self._client.get_collection(self._collection_name)
            return {
                "total_chunks": info.points_count,
                "collection": self._collection_name,
                "status": str(info.status),
            }
        except Exception as e:
            raise StorageError(f"Failed to get collection stats: {e}") from e
