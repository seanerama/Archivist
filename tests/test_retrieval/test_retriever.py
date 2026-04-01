"""Tests for the Retriever class."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from archivist.config import Config
from archivist.retrieval.retriever import Retriever


@pytest.fixture
def config() -> Config:
    return Config.default()


def _make_hit(
    text: str = "chunk text",
    score: float = 0.9,
    family_slug: str = "nginx",
    doc_title: str = "Nginx Guide",
    doc_type: str = "admin_guide",
    version: str | None = "1.24",
    page_number: int | None = 1,
    heading_path: str | None = "Setup",
    chunk_role: str = "base",
    source_file: str = "guide.pdf",
) -> dict:
    return {
        "id": "abc-123",
        "text": text,
        "score": score,
        "family_slug": family_slug,
        "doc_title": doc_title,
        "doc_type": doc_type,
        "version": version,
        "page_number": page_number,
        "heading_path": heading_path,
        "chunk_role": chunk_role,
        "source_file": source_file,
    }


class TestSearch:
    @patch("archivist.retrieval.retriever.QdrantStorage")
    @patch("archivist.retrieval.retriever.get_embedding_backend")
    def test_search_returns_results(self, mock_embed_factory: MagicMock, mock_storage_cls: MagicMock, config: Config) -> None:
        mock_backend = MagicMock()
        mock_backend.encode_query.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_backend.dimension = 3
        mock_embed_factory.return_value = mock_backend

        mock_storage = mock_storage_cls.return_value
        mock_storage._client = None
        mock_storage.search_vectors.return_value = [_make_hit(), _make_hit(text="second")]

        retriever = Retriever(config)
        results = retriever.search("how to configure TLS")

        assert len(results) == 2
        assert results[0].text == "chunk text"
        assert results[0].score == 0.9
        assert results[0].family_slug == "nginx"
        mock_backend.encode_query.assert_called_once_with("how to configure TLS")

    @patch("archivist.retrieval.retriever.QdrantStorage")
    @patch("archivist.retrieval.retriever.get_embedding_backend")
    def test_search_empty_results(self, mock_embed_factory: MagicMock, mock_storage_cls: MagicMock, config: Config) -> None:
        mock_backend = MagicMock()
        mock_backend.encode_query.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_backend.dimension = 3
        mock_embed_factory.return_value = mock_backend

        mock_storage = mock_storage_cls.return_value
        mock_storage._client = None
        mock_storage.search_vectors.return_value = []

        retriever = Retriever(config)
        results = retriever.search("nonexistent topic")

        assert results == []

    @patch("archivist.retrieval.retriever.QdrantStorage")
    @patch("archivist.retrieval.retriever.get_embedding_backend")
    def test_search_with_filters(self, mock_embed_factory: MagicMock, mock_storage_cls: MagicMock, config: Config) -> None:
        mock_backend = MagicMock()
        mock_backend.encode_query.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_backend.dimension = 3
        mock_embed_factory.return_value = mock_backend

        mock_storage = mock_storage_cls.return_value
        mock_storage._client = None
        mock_storage.search_vectors.return_value = [_make_hit()]

        retriever = Retriever(config)
        retriever.search("query", family="nginx", doc_type="admin_guide", version="1.24", top_k=3)

        mock_storage.search_vectors.assert_called_once()
        call_kwargs = mock_storage.search_vectors.call_args
        assert call_kwargs.kwargs["family_slug"] == "nginx"
        assert call_kwargs.kwargs["doc_type"] == "admin_guide"
        assert call_kwargs.kwargs["version_filter"] == (1, 24, 0)
        assert call_kwargs.kwargs["top_k"] == 3

    @patch("archivist.retrieval.retriever.QdrantStorage")
    @patch("archivist.retrieval.retriever.get_embedding_backend")
    def test_search_respects_top_k_config(self, mock_embed_factory: MagicMock, mock_storage_cls: MagicMock, config: Config) -> None:
        config.retrieval.top_k = 7
        mock_backend = MagicMock()
        mock_backend.encode_query.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_backend.dimension = 3
        mock_embed_factory.return_value = mock_backend

        mock_storage = mock_storage_cls.return_value
        mock_storage._client = None
        mock_storage.search_vectors.return_value = []

        retriever = Retriever(config)
        retriever.search("query")

        call_kwargs = mock_storage.search_vectors.call_args
        assert call_kwargs.kwargs["top_k"] == 7


class TestListFamilies:
    @patch("archivist.retrieval.retriever.QdrantStorage")
    @patch("archivist.retrieval.retriever.get_embedding_backend")
    def test_list_families(self, mock_embed_factory: MagicMock, mock_storage_cls: MagicMock, config: Config) -> None:
        mock_backend = MagicMock()
        mock_backend.dimension = 3
        mock_embed_factory.return_value = mock_backend

        mock_storage = mock_storage_cls.return_value
        mock_storage._client = None
        mock_storage.list_all_families.return_value = [
            {
                "family_slug": "nginx",
                "doc_type": "admin_guide",
                "versions_ingested": ["1.22", "1.24"],
                "latest_version": "1.24",
                "version_count": 2,
            },
            {
                "family_slug": "nginx",
                "doc_type": "release_notes",
                "versions_ingested": ["1.24"],
                "latest_version": "1.24",
                "version_count": 1,
            },
        ]

        retriever = Retriever(config)
        families = retriever.list_families()

        assert len(families) == 1  # Both aggregated under "nginx"
        assert families[0].family_slug == "nginx"
        assert "admin_guide" in families[0].doc_types
        assert "release_notes" in families[0].doc_types
        assert "1.22" in families[0].versions
        assert "1.24" in families[0].versions
        assert families[0].latest_version == "1.24"

    @patch("archivist.retrieval.retriever.QdrantStorage")
    @patch("archivist.retrieval.retriever.get_embedding_backend")
    def test_list_families_empty(self, mock_embed_factory: MagicMock, mock_storage_cls: MagicMock, config: Config) -> None:
        mock_backend = MagicMock()
        mock_backend.dimension = 3
        mock_embed_factory.return_value = mock_backend

        mock_storage = mock_storage_cls.return_value
        mock_storage._client = None
        mock_storage.list_all_families.return_value = []

        retriever = Retriever(config)
        families = retriever.list_families()

        assert families == []


class TestVersionDiff:
    @patch("archivist.retrieval.retriever.QdrantStorage")
    @patch("archivist.retrieval.retriever.get_embedding_backend")
    def test_version_diff_returns_changes(self, mock_embed_factory: MagicMock, mock_storage_cls: MagicMock, config: Config) -> None:
        mock_backend = MagicMock()
        mock_backend.dimension = 3
        mock_embed_factory.return_value = mock_backend

        mock_storage = mock_storage_cls.return_value
        mock_storage._client = None
        mock_storage.get_chunks_in_version_range.return_value = [
            {
                "id": "1", "text": "added chunk", "chunk_role": "delta",
                "base_chunk_id": None, "version_tuple": [1, 24, 0],
                "version_range_min": [1, 24, 0], "version_range_max": None,
                "source_file": "guide.pdf", "chunk_index": 5, "heading_path": "New Section",
            },
            {
                "id": "2", "text": "modified chunk", "chunk_role": "delta",
                "base_chunk_id": "base-1", "version_tuple": [1, 24, 0],
                "version_range_min": [1, 22, 0], "version_range_max": None,
                "source_file": "guide.pdf", "chunk_index": 2, "heading_path": "Setup",
            },
            {
                "id": "3", "text": "removed chunk", "chunk_role": "base",
                "base_chunk_id": None, "version_tuple": [1, 22, 0],
                "version_range_min": [1, 22, 0], "version_range_max": [1, 22, 0],
                "source_file": "guide.pdf", "chunk_index": 10, "heading_path": "Old Section",
            },
        ]

        retriever = Retriever(config)
        diffs = retriever.version_diff("nginx", "1.22", "1.24")

        assert len(diffs) == 3
        change_types = [d.change_type for d in diffs]
        assert "added" in change_types
        assert "modified" in change_types
        assert "removed" in change_types

    @patch("archivist.retrieval.retriever.QdrantStorage")
    @patch("archivist.retrieval.retriever.get_embedding_backend")
    def test_version_diff_invalid_version(self, mock_embed_factory: MagicMock, mock_storage_cls: MagicMock, config: Config) -> None:
        mock_backend = MagicMock()
        mock_backend.dimension = 3
        mock_embed_factory.return_value = mock_backend

        mock_storage = mock_storage_cls.return_value
        mock_storage._client = None

        retriever = Retriever(config)
        diffs = retriever.version_diff("nginx", "not-a-version", "also-not")

        assert diffs == []

    @patch("archivist.retrieval.retriever.QdrantStorage")
    @patch("archivist.retrieval.retriever.get_embedding_backend")
    def test_version_diff_sorted(self, mock_embed_factory: MagicMock, mock_storage_cls: MagicMock, config: Config) -> None:
        mock_backend = MagicMock()
        mock_backend.dimension = 3
        mock_embed_factory.return_value = mock_backend

        mock_storage = mock_storage_cls.return_value
        mock_storage._client = None
        mock_storage.get_chunks_in_version_range.return_value = [
            {
                "id": "1", "text": "added", "chunk_role": "delta",
                "base_chunk_id": None, "version_tuple": [1, 24, 0],
                "version_range_min": [1, 24, 0], "version_range_max": None,
                "source_file": "f", "chunk_index": 0, "heading_path": None,
            },
            {
                "id": "2", "text": "removed", "chunk_role": "base",
                "base_chunk_id": None, "version_tuple": [1, 22, 0],
                "version_range_min": [1, 22, 0], "version_range_max": [1, 22, 0],
                "source_file": "f", "chunk_index": 1, "heading_path": None,
            },
        ]

        retriever = Retriever(config)
        diffs = retriever.version_diff("nginx", "1.22", "1.24")

        # Removed should come before added
        assert diffs[0].change_type == "removed"
        assert diffs[1].change_type == "added"
