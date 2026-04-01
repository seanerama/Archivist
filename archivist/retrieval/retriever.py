"""Retrieval engine for version-aware document search."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue

from archivist.exceptions import ArchivistError, StorageError
from archivist.retrieval.models import DiffResult, FamilyInfo, SearchResult

if TYPE_CHECKING:
    from archivist.config import Config


class Retriever:
    """Version-aware document retrieval from Qdrant."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._client: QdrantClient | None = None

    def _get_client(self) -> QdrantClient:
        """Lazy-connect to Qdrant."""
        if self._client is None:
            try:
                self._client = QdrantClient(
                    host=self._config.qdrant.host,
                    port=self._config.qdrant.port,
                    api_key=self._config.qdrant.api_key,
                )
            except Exception as e:
                raise StorageError(f"Cannot connect to Qdrant: {e}") from e
        return self._client

    @property
    def _collection(self) -> str:
        return self._config.qdrant.collection_name

    def search(
        self,
        query: str,
        *,
        version: str | None = None,
        family: str | None = None,
        doc_type: str | None = None,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Search the corpus with optional version/family/doc_type filtering.

        Args:
            query: The search query text.
            version: Filter to a specific version.
            family: Filter to a document family slug.
            doc_type: Filter to a document type.
            top_k: Number of results to return.

        Returns:
            List of SearchResult ordered by relevance.
        """
        from archivist.embedding import EmbeddingBackend

        client = self._get_client()
        backend = EmbeddingBackend.create(self._config)
        query_vector = backend.embed_query(query)

        # Build Qdrant filter conditions
        conditions = []
        if family:
            conditions.append(FieldCondition(key="family_slug", match=MatchValue(value=family)))
        if doc_type:
            conditions.append(FieldCondition(key="doc_type", match=MatchValue(value=doc_type)))
        if version:
            conditions.append(FieldCondition(key="version", match=MatchValue(value=version)))

        query_filter = Filter(must=conditions) if conditions else None

        try:
            hits = client.search(
                collection_name=self._collection,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_k,
            )
        except Exception as e:
            raise StorageError(f"Search failed: {e}") from e

        results = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                SearchResult(
                    text=payload.get("text", ""),
                    score=hit.score,
                    source_file=payload.get("source_file", ""),
                    family_slug=payload.get("family_slug", ""),
                    doc_title=payload.get("doc_title", ""),
                    doc_type=payload.get("doc_type", ""),
                    version=payload.get("version"),
                    page_number=payload.get("page_number"),
                    heading_path=payload.get("heading_path"),
                    chunk_role=payload.get("chunk_role", "base"),
                )
            )
        return results

    def list_families(self) -> list[FamilyInfo]:
        """List all document families in the corpus.

        Returns:
            List of FamilyInfo with family metadata.
        """
        client = self._get_client()

        try:
            # Scroll through all points to collect family info
            families: dict[str, dict] = {}
            offset = None
            while True:
                results, offset = client.scroll(
                    collection_name=self._collection,
                    limit=100,
                    offset=offset,
                    with_payload=["family_slug", "version", "doc_type"],
                )
                if not results:
                    break
                for point in results:
                    payload = point.payload or {}
                    slug = payload.get("family_slug", "")
                    if not slug:
                        continue
                    if slug not in families:
                        families[slug] = {"versions": set(), "doc_types": set(), "doc_count": 0}
                    families[slug]["doc_count"] += 1
                    ver = payload.get("version")
                    if ver:
                        families[slug]["versions"].add(ver)
                    dt = payload.get("doc_type")
                    if dt:
                        families[slug]["doc_types"].add(dt)
                if offset is None:
                    break
        except Exception as e:
            raise StorageError(f"Failed to list families: {e}") from e

        return [
            FamilyInfo(
                family_slug=slug,
                doc_count=info["doc_count"],
                versions=sorted(info["versions"]),
                doc_types=sorted(info["doc_types"]),
            )
            for slug, info in sorted(families.items())
        ]

    def version_diff(
        self,
        family: str,
        from_version: str,
        to_version: str,
    ) -> DiffResult:
        """Show what changed between two versions of a document family.

        Args:
            family: The document family slug.
            from_version: The older version.
            to_version: The newer version.

        Returns:
            DiffResult with added, removed, and modified chunks.
        """
        client = self._get_client()

        try:
            # Get chunks for from_version
            from_chunks = self._get_version_chunks(client, family, from_version)
            to_chunks = self._get_version_chunks(client, family, to_version)
        except Exception as e:
            raise StorageError(f"Version diff failed: {e}") from e

        from_texts = {r.text for r in from_chunks}
        to_texts = {r.text for r in to_chunks}

        added = [r for r in to_chunks if r.text not in from_texts]
        removed = [r for r in from_chunks if r.text not in to_texts]

        return DiffResult(
            family=family,
            from_version=from_version,
            to_version=to_version,
            added=added,
            removed=removed,
        )

    def _get_version_chunks(
        self, client: QdrantClient, family: str, version: str
    ) -> list[SearchResult]:
        """Get all chunks for a specific family+version."""
        conditions = [
            FieldCondition(key="family_slug", match=MatchValue(value=family)),
            FieldCondition(key="version", match=MatchValue(value=version)),
        ]
        query_filter = Filter(must=conditions)

        results = []
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=self._collection,
                scroll_filter=query_filter,
                limit=100,
                offset=offset,
                with_payload=True,
            )
            if not points:
                break
            for point in points:
                payload = point.payload or {}
                results.append(
                    SearchResult(
                        text=payload.get("text", ""),
                        score=0.0,
                        source_file=payload.get("source_file", ""),
                        family_slug=payload.get("family_slug", ""),
                        doc_title=payload.get("doc_title", ""),
                        doc_type=payload.get("doc_type", ""),
                        version=payload.get("version"),
                        page_number=payload.get("page_number"),
                        heading_path=payload.get("heading_path"),
                        chunk_role=payload.get("chunk_role", "base"),
                    )
                )
            if offset is None:
                break
        return results
