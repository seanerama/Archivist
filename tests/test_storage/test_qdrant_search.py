"""Tests for QdrantStorage search extension methods."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from archivist.config import Config
from archivist.models import ChunkRole
from archivist.storage.qdrant import QdrantStorage


@pytest.fixture
def config() -> Config:
    return Config.default()


@pytest.fixture
def storage(config: Config) -> QdrantStorage:
    s = QdrantStorage(config)
    s._client = MagicMock()
    s._collection_name = "test_collection"
    return s


class TestSearchVectors:
    def test_basic_search(self, storage: QdrantStorage) -> None:
        mock_point = MagicMock()
        mock_point.id = "point-1"
        mock_point.score = 0.95
        mock_point.payload = {
            "text": "some text",
            "family_slug": "nginx",
            "doc_title": "Guide",
            "doc_type": "admin_guide",
            "version": "1.24",
            "chunk_role": "base",
            "source_file": "guide.pdf",
            "page_number": 1,
            "heading_path": "Setup",
        }

        mock_result = MagicMock()
        mock_result.points = [mock_point]
        storage._client.query_points.return_value = mock_result

        results = storage.search_vectors([0.1, 0.2, 0.3], top_k=5)

        assert len(results) == 1
        assert results[0]["text"] == "some text"
        assert results[0]["score"] == 0.95

    def test_search_with_family_filter(self, storage: QdrantStorage) -> None:
        mock_result = MagicMock()
        mock_result.points = []
        storage._client.query_points.return_value = mock_result

        storage.search_vectors([0.1], family_slug="nginx", top_k=5)

        call_args = storage._client.query_points.call_args
        query_filter = call_args.kwargs["query_filter"]
        # Check that family_slug filter is in the must conditions
        must_values = [str(c) for c in query_filter.must]
        assert any("nginx" in v for v in must_values)

    def test_search_filters_by_version(self, storage: QdrantStorage) -> None:
        mock_point_in_range = MagicMock()
        mock_point_in_range.id = "p1"
        mock_point_in_range.score = 0.9
        mock_point_in_range.payload = {
            "text": "valid",
            "version_range_min": [1, 22, 0],
            "version_range_max": [1, 26, 0],
            "chunk_role": "base",
        }

        mock_point_out_of_range = MagicMock()
        mock_point_out_of_range.id = "p2"
        mock_point_out_of_range.score = 0.8
        mock_point_out_of_range.payload = {
            "text": "too new",
            "version_range_min": [2, 0, 0],
            "version_range_max": None,
            "chunk_role": "base",
        }

        mock_result = MagicMock()
        mock_result.points = [mock_point_in_range, mock_point_out_of_range]
        storage._client.query_points.return_value = mock_result

        results = storage.search_vectors([0.1], version_filter=(1, 24, 0), top_k=10)

        assert len(results) == 1
        assert results[0]["text"] == "valid"


class TestListAllFamilies:
    def test_returns_version_index_records(self, storage: QdrantStorage) -> None:
        mock_point = MagicMock()
        mock_point.payload = {
            "family_slug": "nginx",
            "doc_type": "admin_guide",
            "versions_ingested": ["1.22", "1.24"],
            "latest_version": "1.24",
            "version_count": 2,
            "chunk_role": "version_index",
        }

        storage._client.scroll.return_value = ([mock_point], None)

        results = storage.list_all_families()

        assert len(results) == 1
        assert results[0]["family_slug"] == "nginx"

    def test_empty_corpus(self, storage: QdrantStorage) -> None:
        storage._client.scroll.return_value = ([], None)
        results = storage.list_all_families()
        assert results == []


class TestGetChunksInVersionRange:
    def test_returns_chunks_in_range(self, storage: QdrantStorage) -> None:
        mock_in_range = MagicMock()
        mock_in_range.id = "p1"
        mock_in_range.payload = {
            "text": "in range",
            "chunk_role": "delta",
            "version_tuple": [1, 24, 0],
            "version_range_min": [1, 24, 0],
            "version_range_max": None,
            "family_slug": "nginx",
        }

        mock_out_of_range = MagicMock()
        mock_out_of_range.id = "p2"
        mock_out_of_range.payload = {
            "text": "out of range",
            "chunk_role": "delta",
            "version_tuple": [2, 0, 0],
            "version_range_min": [2, 0, 0],
            "version_range_max": None,
            "family_slug": "nginx",
        }

        storage._client.scroll.return_value = ([mock_in_range, mock_out_of_range], None)

        results = storage.get_chunks_in_version_range("nginx", (1, 22, 0), (1, 26, 0))

        assert len(results) == 1
        assert results[0]["text"] == "in range"

    def test_empty_result(self, storage: QdrantStorage) -> None:
        storage._client.scroll.return_value = ([], None)
        results = storage.get_chunks_in_version_range("nginx", (1, 0, 0), (2, 0, 0))
        assert results == []
