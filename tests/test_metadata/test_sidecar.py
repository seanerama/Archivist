"""Tests for sidecar file I/O."""

from __future__ import annotations

from pathlib import Path

from archivist.metadata.sidecar import SidecarIO
from archivist.models import TagResult


class TestSidecarIO:
    """Tests for SidecarIO."""

    def test_sidecar_path(self) -> None:
        path = SidecarIO.sidecar_path(Path("/docs/test.pdf"))
        assert str(path) == "/docs/test.pdf.tag"

    def test_read_nonexistent_returns_none(self, tmp_path: Path) -> None:
        result = SidecarIO.read(tmp_path / "nonexistent.pdf")
        assert result is None

    def test_write_and_read_roundtrip(self, tmp_path: Path) -> None:
        source = tmp_path / "test.pdf"
        source.write_bytes(b"dummy")

        tag = TagResult(
            family_slug="nginx",
            doc_title="Nginx Admin Guide",
            vendor="Nginx Inc",
            doc_type="admin_guide",
            is_new_family=False,
            matched_existing="nginx",
            confidence=0.95,
            reasoning="Test reasoning",
        )

        SidecarIO.write(source, tag, tagger_model="qwen3:0.6b", review_mode="auto_accepted")

        # Read back
        loaded = SidecarIO.read(source)
        assert loaded is not None
        assert loaded.family_slug == "nginx"
        assert loaded.doc_title == "Nginx Admin Guide"
        assert loaded.doc_type == "admin_guide"
        assert loaded.confidence == 0.95

    def test_sidecar_file_is_valid_json(self, tmp_path: Path) -> None:
        import json

        source = tmp_path / "doc.md"
        source.write_text("content")

        tag = TagResult(
            family_slug="test", doc_title="Test", vendor=None,
            doc_type="other", is_new_family=True, matched_existing=None,
            confidence=0.7, reasoning="test",
        )
        SidecarIO.write(source, tag)

        tag_path = SidecarIO.sidecar_path(source)
        data = json.loads(tag_path.read_text())
        assert "family_slug" in data
        assert "tagged_at" in data
