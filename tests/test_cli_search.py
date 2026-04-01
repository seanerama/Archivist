"""Tests for CLI search, families, and diff commands."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from archivist.cli import app
from archivist.retrieval.models import DiffResult, FamilyInfo, SearchResult

runner = CliRunner()


def _make_search_result(**overrides: object) -> SearchResult:
    defaults = {
        "text": "This is a chunk of text about configuring TLS certificates.",
        "score": 0.92,
        "source_file": "nginx-guide.pdf",
        "family_slug": "nginx",
        "doc_title": "Nginx Admin Guide",
        "doc_type": "admin_guide",
        "version": "1.24",
        "page_number": 42,
        "heading_path": "Configuration > TLS",
        "chunk_role": "base",
    }
    defaults.update(overrides)
    return SearchResult(**defaults)  # type: ignore[arg-type]


@patch("archivist.cli.load_dotenv")
@patch("archivist.cli.configure_logging")
@patch("archivist.cli.Config.load")
class TestSearchCommand:
    def test_search_no_results(self, mock_config: MagicMock, mock_log: MagicMock, mock_dotenv: MagicMock) -> None:
        mock_config.return_value = MagicMock()
        with patch("archivist.cli.Retriever" if False else "archivist.retrieval.retriever.QdrantStorage"), \
             patch("archivist.retrieval.retriever.get_embedding_backend") as mock_embed:
            mock_backend = MagicMock()
            mock_backend.dimension = 3
            mock_backend.encode_query.return_value = MagicMock(__getitem__=lambda s, i: MagicMock(tolist=lambda: [0.1]))
            mock_embed.return_value = mock_backend

            with patch("archivist.retrieval.retriever.QdrantStorage") as mock_storage_cls:
                mock_storage = mock_storage_cls.return_value
                mock_storage._client = None
                mock_storage.search_vectors.return_value = []

                result = runner.invoke(app, ["search", "nonexistent"])
                assert result.exit_code == 0
                assert "No results found" in result.output

    def test_search_with_results(self, mock_config: MagicMock, mock_log: MagicMock, mock_dotenv: MagicMock) -> None:
        mock_config.return_value = MagicMock()
        with patch("archivist.retrieval.retriever.QdrantStorage") as mock_storage_cls, \
             patch("archivist.retrieval.retriever.get_embedding_backend") as mock_embed:
            import numpy as np
            mock_backend = MagicMock()
            mock_backend.dimension = 3
            mock_backend.encode_query.return_value = np.array([[0.1, 0.2, 0.3]])
            mock_embed.return_value = mock_backend

            mock_storage = mock_storage_cls.return_value
            mock_storage._client = None
            mock_storage.search_vectors.return_value = [{
                "id": "1", "text": "TLS config text", "score": 0.92,
                "family_slug": "nginx", "doc_title": "Nginx Admin Guide",
                "doc_type": "admin_guide", "version": "1.24",
                "page_number": 42, "heading_path": "TLS", "chunk_role": "base",
                "source_file": "guide.pdf",
            }]

            result = runner.invoke(app, ["search", "TLS config"])
            assert result.exit_code == 0
            assert "Nginx Admin Guide" in result.output


@patch("archivist.cli.load_dotenv")
@patch("archivist.cli.configure_logging")
@patch("archivist.cli.Config.load")
class TestFamiliesCommand:
    def test_families_empty(self, mock_config: MagicMock, mock_log: MagicMock, mock_dotenv: MagicMock) -> None:
        mock_config.return_value = MagicMock()
        with patch("archivist.retrieval.retriever.QdrantStorage") as mock_storage_cls, \
             patch("archivist.retrieval.retriever.get_embedding_backend") as mock_embed:
            mock_backend = MagicMock()
            mock_backend.dimension = 3
            mock_embed.return_value = mock_backend

            mock_storage = mock_storage_cls.return_value
            mock_storage._client = None
            mock_storage.list_all_families.return_value = []

            result = runner.invoke(app, ["families"])
            assert result.exit_code == 0
            assert "No families found" in result.output

    def test_families_with_data(self, mock_config: MagicMock, mock_log: MagicMock, mock_dotenv: MagicMock) -> None:
        mock_config.return_value = MagicMock()
        with patch("archivist.retrieval.retriever.QdrantStorage") as mock_storage_cls, \
             patch("archivist.retrieval.retriever.get_embedding_backend") as mock_embed:
            mock_backend = MagicMock()
            mock_backend.dimension = 3
            mock_embed.return_value = mock_backend

            mock_storage = mock_storage_cls.return_value
            mock_storage._client = None
            mock_storage.list_all_families.return_value = [{
                "family_slug": "nginx",
                "doc_type": "admin_guide",
                "versions_ingested": ["1.22", "1.24"],
                "latest_version": "1.24",
                "version_count": 2,
            }]

            result = runner.invoke(app, ["families"])
            assert result.exit_code == 0
            assert "nginx" in result.output


@patch("archivist.cli.load_dotenv")
@patch("archivist.cli.configure_logging")
@patch("archivist.cli.Config.load")
class TestDiffCommand:
    def test_diff_no_changes(self, mock_config: MagicMock, mock_log: MagicMock, mock_dotenv: MagicMock) -> None:
        mock_config.return_value = MagicMock()
        with patch("archivist.retrieval.retriever.QdrantStorage") as mock_storage_cls, \
             patch("archivist.retrieval.retriever.get_embedding_backend") as mock_embed:
            mock_backend = MagicMock()
            mock_backend.dimension = 3
            mock_embed.return_value = mock_backend

            mock_storage = mock_storage_cls.return_value
            mock_storage._client = None
            mock_storage.get_chunks_in_version_range.return_value = []

            result = runner.invoke(app, ["diff", "nginx", "1.22", "1.24"])
            assert result.exit_code == 0
            assert "No differences found" in result.output

    def test_diff_with_changes(self, mock_config: MagicMock, mock_log: MagicMock, mock_dotenv: MagicMock) -> None:
        mock_config.return_value = MagicMock()
        with patch("archivist.retrieval.retriever.QdrantStorage") as mock_storage_cls, \
             patch("archivist.retrieval.retriever.get_embedding_backend") as mock_embed:
            mock_backend = MagicMock()
            mock_backend.dimension = 3
            mock_embed.return_value = mock_backend

            mock_storage = mock_storage_cls.return_value
            mock_storage._client = None
            mock_storage.get_chunks_in_version_range.return_value = [{
                "id": "1", "text": "New TLS configuration section",
                "chunk_role": "delta", "base_chunk_id": None,
                "version_tuple": [1, 24, 0],
                "version_range_min": [1, 24, 0], "version_range_max": None,
                "source_file": "guide.pdf", "chunk_index": 5, "heading_path": "TLS",
            }]

            result = runner.invoke(app, ["diff", "nginx", "1.22", "1.24"])
            assert result.exit_code == 0
            assert "Added" in result.output
            assert "1 added" in result.output
