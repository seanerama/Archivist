"""Tests for Qdrant storage client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from archivist.config import Config
from archivist.exceptions import StorageError
from archivist.models import ChunkRole, MetadataPayload
from archivist.storage.qdrant import QdrantStorage


@pytest.fixture
def storage() -> QdrantStorage:
    return QdrantStorage(Config.default())


@pytest.fixture
def mock_client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def connected_storage(storage: QdrantStorage, mock_client: MagicMock) -> QdrantStorage:
    """Storage with a mocked connected client."""
    storage._client = mock_client
    return storage


class TestQdrantStorageConnect:
    """Tests for connection and collection creation."""

    def test_connect_creates_collection_if_not_exists(self) -> None:
        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock(collections=[])

        mock_qdrant_client_mod = MagicMock()
        mock_qdrant_client_mod.QdrantClient.return_value = mock_client

        with patch.dict("sys.modules", {"qdrant_client": mock_qdrant_client_mod, "qdrant_client.models": MagicMock()}):
            storage = QdrantStorage(Config.default())
            storage.connect(vector_dimension=768)

        mock_client.create_collection.assert_called_once()

    def test_connect_skips_creation_if_exists(self) -> None:
        mock_collection = MagicMock()
        mock_collection.name = "archivist"
        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock(collections=[mock_collection])

        mock_qdrant_client_mod = MagicMock()
        mock_qdrant_client_mod.QdrantClient.return_value = mock_client

        with patch.dict("sys.modules", {"qdrant_client": mock_qdrant_client_mod, "qdrant_client.models": MagicMock()}):
            storage = QdrantStorage(Config.default())
            storage.connect(vector_dimension=768)

        mock_client.create_collection.assert_not_called()


class TestQdrantStorageOperations:
    """Tests for storage operations."""

    def test_not_connected_raises(self, storage: QdrantStorage) -> None:
        with pytest.raises(StorageError, match="Not connected"):
            storage.check_document_exists("test.pdf")

    def test_upsert_chunks(self, connected_storage: QdrantStorage, mock_client: MagicMock) -> None:
        payloads = [
            MetadataPayload(
                doc_title="test", doc_type="admin_guide", family_slug="test",
                source_file="test.pdf", format="pdf", version="1.0",
                version_tuple=(1, 0, 0), version_range_min=(1, 0, 0),
                version_range_max=None, chunk_role=ChunkRole.BASE,
                base_chunk_id=None, created_date=None,
                ingested_date="2026-04-01", metadata_complete=True,
                chunk_index=0, page_number=1, heading_path=None,
                timestamp_start=None, timestamp_end=None,
                text="test text", token_count=2,
            )
        ]
        vectors = np.array([[0.1, 0.2, 0.3]])

        ids = connected_storage.upsert_chunks(payloads, vectors)
        assert len(ids) == 1
        mock_client.upsert.assert_called_once()

    def test_check_document_exists_true(self, connected_storage: QdrantStorage, mock_client: MagicMock) -> None:
        mock_client.scroll.return_value = ([MagicMock()], None)
        assert connected_storage.check_document_exists("test.pdf") is True

    def test_check_document_exists_false(self, connected_storage: QdrantStorage, mock_client: MagicMock) -> None:
        mock_client.scroll.return_value = ([], None)
        assert connected_storage.check_document_exists("test.pdf") is False

    def test_delete_partial_ingestion(self, connected_storage: QdrantStorage, mock_client: MagicMock) -> None:
        connected_storage.delete_partial_ingestion("test.pdf")
        mock_client.delete.assert_called_once()

    def test_collection_stats(self, connected_storage: QdrantStorage, mock_client: MagicMock) -> None:
        mock_info = MagicMock()
        mock_info.points_count = 100
        mock_info.status = "green"
        mock_client.get_collection.return_value = mock_info

        mock_point = MagicMock()
        mock_point.payload = {
            "source_file": "test.pdf", "family_slug": "test", "doc_title": "Test",
            "doc_type": "other", "version": "1.0", "chunk_role": "base",
        }
        mock_client.scroll.return_value = ([mock_point], None)

        stats = connected_storage.collection_stats()
        assert stats["total_chunks"] == 100
        assert len(stats["documents"]) == 1
        assert stats["documents"][0]["source_file"] == "test.pdf"

    def test_update_version_range(self, connected_storage: QdrantStorage, mock_client: MagicMock) -> None:
        connected_storage.update_version_range(["id1", "id2"], (1, 26, 0))
        mock_client.set_payload.assert_called_once()

    def test_get_family_chunks(self, connected_storage: QdrantStorage, mock_client: MagicMock) -> None:
        mock_point = MagicMock()
        mock_point.id = "test-id"
        mock_point.payload = {
            "text": "chunk text",
            "chunk_index": 0,
            "chunk_role": "base",
            "version": "1.0",
            "version_tuple": [1, 0, 0],
            "version_range_min": [1, 0, 0],
            "version_range_max": None,
        }
        mock_client.scroll.side_effect = [([mock_point], None), ([], None)]

        chunks = connected_storage.get_family_chunks("nginx", "admin_guide")
        assert len(chunks) == 1
        assert chunks[0]["text"] == "chunk text"

    def test_get_version_index_found(self, connected_storage: QdrantStorage, mock_client: MagicMock) -> None:
        mock_point = MagicMock()
        mock_point.payload = {"family_slug": "nginx", "versions_ingested": ["1.0"]}
        mock_client.scroll.return_value = ([mock_point], None)

        result = connected_storage.get_version_index("nginx", "admin_guide")
        assert result is not None
        assert result["family_slug"] == "nginx"

    def test_get_version_index_not_found(self, connected_storage: QdrantStorage, mock_client: MagicMock) -> None:
        mock_client.scroll.return_value = ([], None)
        result = connected_storage.get_version_index("nginx", "admin_guide")
        assert result is None
