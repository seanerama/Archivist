"""Tests for the MCP server tool registration and invocation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from archivist.config import Config
from archivist.mcp.server import (
    _format_diff,
    _format_families,
    _format_search_results,
    create_server,
)
from archivist.retrieval.models import DiffResult, FamilyInfo, SearchResult


# --- Formatter tests ---


class TestFormatSearchResults:
    def test_empty_results(self) -> None:
        assert _format_search_results([]) == "No results found."

    def test_single_result(self) -> None:
        results = [
            SearchResult(
                text="This is a test chunk with some content.",
                score=0.95,
                source_file="nginx.md",
                family_slug="nginx",
                doc_title="Nginx Guide",
                doc_type="admin_guide",
                version="1.24",
                page_number=1,
                heading_path="Setup",
                chunk_role="base",
            )
        ]
        output = _format_search_results(results)
        assert "1." in output
        assert "nginx.md" in output
        assert "v1.24" in output
        assert "0.950" in output
        assert "This is a test chunk" in output

    def test_long_text_truncated(self) -> None:
        results = [
            SearchResult(
                text="x" * 300,
                score=0.5,
                source_file="doc.md",
                family_slug="test",
                doc_title="Test",
                doc_type="other",
                version=None,
                page_number=None,
                heading_path=None,
                chunk_role="base",
            )
        ]
        output = _format_search_results(results)
        assert "..." in output

    def test_unversioned_result(self) -> None:
        results = [
            SearchResult(
                text="Some text",
                score=0.8,
                source_file="doc.md",
                family_slug="test",
                doc_title="Test",
                doc_type="other",
                version=None,
                page_number=None,
                heading_path=None,
                chunk_role="base",
            )
        ]
        output = _format_search_results(results)
        assert "unversioned" in output


class TestFormatFamilies:
    def test_empty_families(self) -> None:
        assert _format_families([]) == "No document families found."

    def test_single_family(self) -> None:
        families = [
            FamilyInfo(
                family_slug="nginx",
                doc_types=["admin_guide", "config_guide"],
                versions=["1.22", "1.24"],
                latest_version="1.24",
                total_chunks=42,
            )
        ]
        output = _format_families(families)
        assert "nginx" in output
        assert "42" in output
        assert "1.22" in output
        assert "admin_guide" in output


class TestFormatDiff:
    def test_empty_diff(self) -> None:
        output = _format_diff([], "nginx", "1.22", "1.24")
        assert "No changes found" in output

    def test_diff_with_changes(self) -> None:
        diffs = [
            DiffResult(
                chunk_text="New feature in 1.24",
                change_type="added",
                source_file="nginx.md",
                chunk_index=5,
                heading_path="Features",
            ),
            DiffResult(
                chunk_text="Deprecated in 1.22",
                change_type="removed",
                source_file="nginx.md",
                chunk_index=10,
                heading_path="Old",
            ),
        ]
        output = _format_diff(diffs, "nginx", "1.22", "1.24")
        assert "ADDED (1 chunks)" in output
        assert "New feature in 1.24" in output
        assert "REMOVED (1 chunks)" in output
        assert "Deprecated in 1.22" in output
        assert "1 added" in output
        assert "1 removed" in output


# --- Server tool registration tests ---


def _call_tool_text(result: tuple) -> str:
    """Extract text from a FastMCP call_tool result tuple."""
    content_list = result[0]
    return content_list[0].text


class TestServerCreation:
    def test_server_has_correct_name(self) -> None:
        config = Config.default()
        server = create_server(config)
        assert server.name == "archivist"

    @pytest.mark.anyio
    async def test_server_lists_tools(self) -> None:
        config = Config.default()
        server = create_server(config)
        tools = await server.list_tools()
        tool_names = {t.name for t in tools}
        assert "archivist_search" in tool_names
        assert "archivist_list_families" in tool_names
        assert "archivist_version_diff" in tool_names

    @pytest.mark.anyio
    async def test_search_tool_has_parameters(self) -> None:
        config = Config.default()
        server = create_server(config)
        tools = await server.list_tools()
        search_tool = next(t for t in tools if t.name == "archivist_search")
        schema = search_tool.inputSchema
        props = schema.get("properties", {})
        assert "query" in props
        assert "version" in props
        assert "family" in props
        assert "doc_type" in props
        assert "top_k" in props


# --- Tool invocation tests with mocked Retriever ---


class TestToolInvocation:
    @pytest.mark.anyio
    async def test_search_delegates_to_retriever(self) -> None:
        config = Config.default()
        server = create_server(config)

        mock_results = [
            SearchResult(
                text="TLS configuration guide for nginx",
                score=0.92,
                source_file="nginx_tls.md",
                family_slug="nginx",
                doc_title="Nginx TLS",
                doc_type="config_guide",
                version="1.24",
                page_number=None,
                heading_path=None,
                chunk_role="base",
            )
        ]

        with patch("archivist.mcp.server.Retriever") as MockRetriever:
            mock_instance = MagicMock()
            mock_instance.search.return_value = mock_results
            MockRetriever.return_value = mock_instance

            result = await server.call_tool("archivist_search", {"query": "TLS config"})
            text = _call_tool_text(result)

            mock_instance.search.assert_called_once_with(
                "TLS config", version=None, family=None, doc_type=None, top_k=5
            )
            assert "nginx_tls.md" in text
            assert "0.920" in text

    @pytest.mark.anyio
    async def test_search_with_filters(self) -> None:
        config = Config.default()
        server = create_server(config)

        with patch("archivist.mcp.server.Retriever") as MockRetriever:
            mock_instance = MagicMock()
            mock_instance.search.return_value = []
            MockRetriever.return_value = mock_instance

            await server.call_tool(
                "archivist_search",
                {
                    "query": "TLS",
                    "version": "1.24",
                    "family": "nginx",
                    "doc_type": "config_guide",
                    "top_k": 10,
                },
            )
            mock_instance.search.assert_called_once_with(
                "TLS", version="1.24", family="nginx", doc_type="config_guide", top_k=10
            )

    @pytest.mark.anyio
    async def test_list_families_delegates(self) -> None:
        config = Config.default()
        server = create_server(config)

        mock_families = [
            FamilyInfo(
                family_slug="nginx",
                doc_types=["admin_guide"],
                versions=["1.22", "1.24"],
                latest_version="1.24",
                total_chunks=10,
            )
        ]

        with patch("archivist.mcp.server.Retriever") as MockRetriever:
            mock_instance = MagicMock()
            mock_instance.list_families.return_value = mock_families
            MockRetriever.return_value = mock_instance

            result = await server.call_tool("archivist_list_families", {})
            text = _call_tool_text(result)

            mock_instance.list_families.assert_called_once()
            assert "nginx" in text

    @pytest.mark.anyio
    async def test_version_diff_delegates(self) -> None:
        config = Config.default()
        server = create_server(config)

        mock_diffs = [
            DiffResult(
                chunk_text="New feature",
                change_type="added",
                source_file="nginx.md",
                chunk_index=5,
                heading_path=None,
            )
        ]

        with patch("archivist.mcp.server.Retriever") as MockRetriever:
            mock_instance = MagicMock()
            mock_instance.version_diff.return_value = mock_diffs
            MockRetriever.return_value = mock_instance

            result = await server.call_tool(
                "archivist_version_diff",
                {"family": "nginx", "from_version": "1.22", "to_version": "1.24"},
            )
            text = _call_tool_text(result)

            mock_instance.version_diff.assert_called_once_with("nginx", "1.22", "1.24")
            assert "ADDED" in text
            assert "New feature" in text

    @pytest.mark.anyio
    async def test_search_error_handling(self) -> None:
        from archivist.exceptions import StorageError

        config = Config.default()
        server = create_server(config)

        with patch("archivist.mcp.server.Retriever") as MockRetriever:
            mock_instance = MagicMock()
            mock_instance.search.side_effect = StorageError("Connection refused")
            MockRetriever.return_value = mock_instance

            result = await server.call_tool("archivist_search", {"query": "test"})
            text = _call_tool_text(result)

            assert "Error:" in text
            assert "Connection refused" in text
