"""Core retrieval engine for searching the Archivist corpus."""

from __future__ import annotations

from typing import Any

from archivist.config import Config
from archivist.embedding import get_embedding_backend
from archivist.log import get_logger
from archivist.retrieval.models import DiffResult, FamilyInfo, SearchResult
from archivist.storage import QdrantStorage
from archivist.versioning import VersionParser

logger = get_logger("retrieval")


class Retriever:
    """Searches the ingested corpus with version-aware filtering."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._storage = QdrantStorage(config)
        self._embedding = get_embedding_backend(config)
        self._version_parser = VersionParser()
        self._reranker: Any = None  # Plugged in by Stage 3

    def _ensure_connected(self) -> None:
        """Connect to storage if not already connected."""
        if self._storage._client is None:
            self._storage.connect(self._embedding.dimension)

    def search(
        self,
        query: str,
        *,
        version: str | None = None,
        family: str | None = None,
        doc_type: str | None = None,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Search the corpus with optional filters.

        Args:
            query: The search query text.
            version: Optional version string to filter results (e.g. "1.24").
            family: Optional family slug to filter results.
            doc_type: Optional document type to filter results.
            top_k: Number of results to return. Defaults to config value.

        Returns:
            List of SearchResult sorted by relevance score.
        """
        self._ensure_connected()

        if top_k is None:
            top_k = self._config.retrieval.top_k

        # Over-fetch if reranker is configured
        fetch_k = top_k * 3 if self._reranker is not None else top_k

        # Embed query
        query_vector = self._embedding.encode_query(query)
        query_list = query_vector[0].tolist()

        # Parse version filter
        version_filter = None
        if version:
            version_filter = self._version_parser.parse(version)

        # Search Qdrant
        hits = self._storage.search_vectors(
            query_list,
            family_slug=family,
            doc_type=doc_type,
            version_filter=version_filter,
            top_k=fetch_k,
        )

        # Convert to SearchResult objects
        results = [
            SearchResult(
                text=hit.get("text", ""),
                score=hit.get("score", 0.0),
                source_file=hit.get("source_file", ""),
                family_slug=hit.get("family_slug", ""),
                doc_title=hit.get("doc_title", ""),
                doc_type=hit.get("doc_type", ""),
                version=hit.get("version"),
                page_number=hit.get("page_number"),
                heading_path=hit.get("heading_path"),
                chunk_role=hit.get("chunk_role", "base"),
            )
            for hit in hits
        ]

        # Apply reranker if configured (Stage 3 hook)
        if self._reranker is not None:
            results = self._reranker.rerank(query, results, top_k)
        else:
            results = results[:top_k]

        return results

    def list_families(self) -> list[FamilyInfo]:
        """List all document families in the corpus.

        Returns:
            List of FamilyInfo with version and chunk count details.
        """
        self._ensure_connected()

        index_records = self._storage.list_all_families()

        # Aggregate by family_slug (a family can have multiple doc_types)
        families: dict[str, dict[str, Any]] = {}
        for record in index_records:
            slug = record.get("family_slug", "")
            if slug not in families:
                families[slug] = {
                    "doc_types": [],
                    "versions": [],
                    "latest_version": None,
                    "total_chunks": 0,
                }
            fam = families[slug]
            dt = record.get("doc_type", "")
            if dt and dt not in fam["doc_types"]:
                fam["doc_types"].append(dt)
            for v in record.get("versions_ingested", []):
                if v not in fam["versions"]:
                    fam["versions"].append(v)
            latest = record.get("latest_version")
            if latest:
                fam["latest_version"] = latest
            fam["total_chunks"] += record.get("version_count", 0)

        return [
            FamilyInfo(
                family_slug=slug,
                doc_types=data["doc_types"],
                versions=data["versions"],
                latest_version=data["latest_version"],
                total_chunks=data["total_chunks"],
            )
            for slug, data in sorted(families.items())
        ]

    def version_diff(
        self,
        family: str,
        from_version: str,
        to_version: str,
    ) -> list[DiffResult]:
        """Show what changed between two versions of a document family.

        Args:
            family: The document family slug.
            from_version: The older version string.
            to_version: The newer version string.

        Returns:
            List of DiffResult describing added, modified, and removed chunks.
        """
        self._ensure_connected()

        from_tuple = self._version_parser.parse(from_version)
        to_tuple = self._version_parser.parse(to_version)

        if from_tuple is None or to_tuple is None:
            return []

        chunks = self._storage.get_chunks_in_version_range(family, from_tuple, to_tuple)

        results: list[DiffResult] = []
        for chunk in chunks:
            role = chunk.get("chunk_role", "base")
            vt = chunk.get("version_tuple")
            vmax = chunk.get("version_range_max")

            # Determine change type
            if role == "delta":
                base_id = chunk.get("base_chunk_id")
                change_type = "modified" if base_id else "added"
            elif vmax is not None and tuple(vmax) < to_tuple:
                # Chunk was capped before the target version — it was removed
                change_type = "removed"
            else:
                # Base chunk still valid in range — not a change
                continue

            results.append(DiffResult(
                chunk_text=chunk.get("text", ""),
                change_type=change_type,
                source_file=chunk.get("source_file", ""),
                chunk_index=chunk.get("chunk_index", 0),
                heading_path=chunk.get("heading_path"),
            ))

        # Sort: removed first, then modified, then added
        order = {"removed": 0, "modified": 1, "added": 2}
        results.sort(key=lambda r: (order.get(r.change_type, 3), r.chunk_index))

        return results
