"""Tests for version index management."""

from __future__ import annotations

from unittest.mock import MagicMock

from archivist.versioning.version_index import VersionIndex


class TestVersionIndex:
    """Tests for VersionIndex."""

    def test_create_new_index(self) -> None:
        mock_storage = MagicMock()
        mock_storage.get_version_index.return_value = None

        VersionIndex.update_index(mock_storage, "nginx", "admin_guide", "1.24")

        mock_storage.upsert_version_index.assert_called_once()
        call_args = mock_storage.upsert_version_index.call_args[0][0]
        assert call_args["family_slug"] == "nginx"
        assert call_args["versions_ingested"] == ["1.24"]
        assert call_args["version_count"] == 1
        assert call_args["latest_version"] == "1.24"
        assert call_args["base_version"] == "1.24"

    def test_append_to_existing_index(self) -> None:
        mock_storage = MagicMock()
        mock_storage.get_version_index.return_value = {
            "family_slug": "nginx",
            "doc_type": "admin_guide",
            "versions_ingested": ["1.20", "1.22"],
            "version_count": 2,
            "latest_version": "1.22",
            "base_version": "1.20",
        }

        VersionIndex.update_index(mock_storage, "nginx", "admin_guide", "1.24")

        call_args = mock_storage.upsert_version_index.call_args[0][0]
        assert call_args["versions_ingested"] == ["1.20", "1.22", "1.24"]
        assert call_args["version_count"] == 3
        assert call_args["latest_version"] == "1.24"
        assert call_args["base_version"] == "1.20"

    def test_no_duplicate_versions(self) -> None:
        mock_storage = MagicMock()
        mock_storage.get_version_index.return_value = {
            "family_slug": "nginx",
            "doc_type": "admin_guide",
            "versions_ingested": ["1.24"],
            "version_count": 1,
            "latest_version": "1.24",
            "base_version": "1.24",
        }

        VersionIndex.update_index(mock_storage, "nginx", "admin_guide", "1.24")

        call_args = mock_storage.upsert_version_index.call_args[0][0]
        assert call_args["versions_ingested"] == ["1.24"]
        assert call_args["version_count"] == 1

    def test_get_index_delegates(self) -> None:
        mock_storage = MagicMock()
        mock_storage.get_version_index.return_value = {"family_slug": "nginx"}

        result = VersionIndex.get_index(mock_storage, "nginx", "admin_guide")
        assert result == {"family_slug": "nginx"}
