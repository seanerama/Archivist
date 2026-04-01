"""Tests for shared data models."""

from __future__ import annotations

import pytest

from archivist.models import (
    Chunk,
    ChunkRole,
    ClassificationResult,
    DocType,
    IngestResult,
    MetadataPayload,
    RawDocument,
    TagResult,
)


class TestRawDocument:
    """Tests for the RawDocument dataclass."""

    def test_create_basic(self) -> None:
        doc = RawDocument(text="hello", source_file="test.md", format="md")
        assert doc.text == "hello"
        assert doc.source_file == "test.md"
        assert doc.format == "md"
        assert doc.pages is None
        assert doc.native_metadata == {}

    def test_frozen(self) -> None:
        doc = RawDocument(text="hello", source_file="test.md", format="md")
        with pytest.raises(AttributeError):
            doc.text = "changed"  # type: ignore[misc]

    def test_with_pages(self) -> None:
        pages = [{"page_number": 1, "start_offset": 0, "end_offset": 100}]
        doc = RawDocument(text="content", source_file="test.pdf", format="pdf", pages=pages)
        assert doc.pages is not None
        assert len(doc.pages) == 1

    def test_with_native_metadata(self) -> None:
        doc = RawDocument(
            text="content",
            source_file="test.epub",
            format="epub",
            native_metadata={"author": "Test Author", "language": "en"},
        )
        assert doc.native_metadata["author"] == "Test Author"


class TestChunk:
    """Tests for the Chunk dataclass."""

    def test_create_basic(self) -> None:
        chunk = Chunk(text="some text", chunk_index=0, source_file="test.md", token_count=3)
        assert chunk.text == "some text"
        assert chunk.chunk_index == 0
        assert chunk.page_number is None
        assert chunk.heading_path is None

    def test_with_position_metadata(self) -> None:
        chunk = Chunk(
            text="content",
            chunk_index=5,
            source_file="doc.pdf",
            page_number=12,
            heading_path="Config > TLS",
            token_count=1,
        )
        assert chunk.page_number == 12
        assert chunk.heading_path == "Config > TLS"

    def test_with_timestamps(self) -> None:
        chunk = Chunk(
            text="spoken text",
            chunk_index=0,
            source_file="video.mp4",
            timestamp_start=60.0,
            timestamp_end=120.0,
            token_count=2,
        )
        assert chunk.timestamp_start == 60.0
        assert chunk.timestamp_end == 120.0


class TestChunkRole:
    """Tests for the ChunkRole enum."""

    def test_values(self) -> None:
        assert ChunkRole.BASE.value == "base"
        assert ChunkRole.DELTA.value == "delta"
        assert ChunkRole.VERSION_INDEX.value == "version_index"

    def test_string_comparison(self) -> None:
        assert ChunkRole.BASE == "base"


class TestDocType:
    """Tests for the DocType enum."""

    def test_all_types_exist(self) -> None:
        expected = {
            "config_guide", "admin_guide", "release_notes", "changelog",
            "quickstart", "api_reference", "architecture_guide",
            "troubleshooting", "book", "tutorial", "other",
        }
        actual = {dt.value for dt in DocType}
        assert actual == expected


class TestTagResult:
    """Tests for the TagResult dataclass."""

    def test_create(self) -> None:
        tag = TagResult(
            family_slug="nginx",
            doc_title="Nginx Admin Guide",
            vendor="Nginx Inc",
            doc_type="admin_guide",
            is_new_family=False,
            matched_existing="nginx",
            confidence=0.95,
            reasoning="Filename and content clearly indicate Nginx documentation.",
        )
        assert tag.family_slug == "nginx"
        assert tag.confidence == 0.95

    def test_new_family(self) -> None:
        tag = TagResult(
            family_slug="new-product",
            doc_title="New Product Guide",
            vendor=None,
            doc_type="other",
            is_new_family=True,
            matched_existing=None,
            confidence=0.60,
            reasoning="No existing family matches.",
        )
        assert tag.is_new_family is True
        assert tag.matched_existing is None


class TestMetadataPayload:
    """Tests for the MetadataPayload dataclass."""

    def test_to_dict(self) -> None:
        payload = MetadataPayload(
            doc_title="nginx",
            doc_type="admin_guide",
            family_slug="nginx",
            source_file="nginx_1.24.pdf",
            format="pdf",
            version="1.24",
            version_tuple=(1, 24, 0),
            version_range_min=(1, 20, 0),
            version_range_max=None,
            chunk_role=ChunkRole.BASE,
            base_chunk_id=None,
            created_date="2023-06-15",
            ingested_date="2026-04-01T14:32:00Z",
            metadata_complete=True,
            chunk_index=0,
            page_number=1,
            heading_path=None,
            timestamp_start=None,
            timestamp_end=None,
            text="some chunk text",
            token_count=3,
        )
        d = payload.to_dict()
        assert d["doc_title"] == "nginx"
        assert d["version_tuple"] == [1, 24, 0]
        assert d["version_range_max"] is None
        assert d["chunk_role"] == "base"

    def test_to_dict_none_version(self) -> None:
        payload = MetadataPayload(
            doc_title="unknown",
            doc_type="other",
            family_slug="unknown",
            source_file="unknown.txt",
            format="txt",
            version=None,
            version_tuple=None,
            version_range_min=None,
            version_range_max=None,
            chunk_role=ChunkRole.BASE,
            base_chunk_id=None,
            created_date=None,
            ingested_date="2026-04-01T14:32:00Z",
            metadata_complete=False,
            chunk_index=0,
            page_number=None,
            heading_path=None,
            timestamp_start=None,
            timestamp_end=None,
            text="text",
            token_count=1,
        )
        d = payload.to_dict()
        assert d["version_tuple"] is None
        assert d["metadata_complete"] is False


class TestClassificationResult:
    """Tests for the ClassificationResult dataclass."""

    def test_first_version(self) -> None:
        result = ClassificationResult(
            to_upsert=[],
            to_upsert_roles=[],
            to_upsert_base_ids=[],
            to_update_range=[],
            to_cap=[],
            is_first_version=True,
        )
        assert result.is_first_version is True


class TestIngestResult:
    """Tests for the IngestResult dataclass."""

    def test_defaults(self) -> None:
        result = IngestResult()
        assert result.docs_processed == 0
        assert result.docs_failed == 0
        assert result.errors == []
