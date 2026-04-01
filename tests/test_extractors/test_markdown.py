"""Tests for the markdown extractor."""

from __future__ import annotations

from pathlib import Path

import pytest

from archivist.exceptions import ExtractionError
from archivist.extractors.markdown import MarkdownExtractor


class TestMarkdownExtractor:
    """Tests for MarkdownExtractor."""

    def test_supported_extensions(self) -> None:
        ext = MarkdownExtractor()
        assert ".md" in ext.supported_extensions
        assert ".markdown" in ext.supported_extensions

    def test_extract_sample(self, fixtures_dir: Path) -> None:
        ext = MarkdownExtractor()
        doc = ext.extract(fixtures_dir / "sample.md")
        assert doc.format == "md"
        assert doc.source_file == "sample.md"
        assert "# Sample Document" in doc.text
        assert "## Configuration" in doc.text
        assert doc.pages is None
        assert doc.native_metadata == {}

    def test_extract_preserves_code_blocks(self, fixtures_dir: Path) -> None:
        ext = MarkdownExtractor()
        doc = ext.extract(fixtures_dir / "sample.md")
        assert "```yaml" in doc.text
        assert "tls:" in doc.text

    def test_extract_missing_file_raises(self, tmp_path: Path) -> None:
        ext = MarkdownExtractor()
        with pytest.raises(ExtractionError, match="Failed to read markdown"):
            ext.extract(tmp_path / "nonexistent.md")
