"""Tests for the plaintext extractor."""

from __future__ import annotations

from pathlib import Path

import pytest

from archivist.exceptions import ExtractionError
from archivist.extractors.plaintext import PlaintextExtractor


class TestPlaintextExtractor:
    """Tests for PlaintextExtractor."""

    def test_supported_extensions(self) -> None:
        ext = PlaintextExtractor()
        assert ".txt" in ext.supported_extensions

    def test_extract_sample(self, fixtures_dir: Path) -> None:
        ext = PlaintextExtractor()
        doc = ext.extract(fixtures_dir / "sample.txt")
        assert doc.format == "txt"
        assert doc.source_file == "sample.txt"
        assert "plain text document" in doc.text
        assert doc.pages is None

    def test_extract_missing_file_raises(self, tmp_path: Path) -> None:
        ext = PlaintextExtractor()
        with pytest.raises(ExtractionError, match="Failed to read text file"):
            ext.extract(tmp_path / "nonexistent.txt")
