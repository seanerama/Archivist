"""Tests for the EPUB extractor."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from archivist.exceptions import ExtractionError
from archivist.extractors.epub import EpubExtractor, _html_to_text


class TestHtmlToText:
    """Tests for HTML to text conversion."""

    def test_simple_paragraph(self) -> None:
        assert "Hello world" in _html_to_text("<p>Hello world</p>")

    def test_strips_script_tags(self) -> None:
        result = _html_to_text("<p>Text</p><script>alert('x')</script>")
        assert "alert" not in result
        assert "Text" in result

    def test_headings_add_newlines(self) -> None:
        result = _html_to_text("<h1>Title</h1><p>Body</p>")
        assert "Title" in result
        assert "Body" in result


class TestEpubExtractor:
    """Tests for EpubExtractor."""

    def test_supported_extensions(self) -> None:
        ext = EpubExtractor()
        assert ".epub" in ext.supported_extensions

    def test_extract_missing_file_raises(self, tmp_path: Path) -> None:
        ext = EpubExtractor()
        with pytest.raises(ExtractionError):
            ext.extract(tmp_path / "nonexistent.epub")

    def test_extract_returns_raw_document(self, tmp_path: Path) -> None:
        epub_path = tmp_path / "test.epub"
        epub_path.write_bytes(b"dummy")

        # Create mock ebooklib module
        mock_ebooklib = MagicMock()
        mock_ebooklib.ITEM_DOCUMENT = 9  # ebooklib constant

        # Create mock epub submodule
        mock_epub = MagicMock()

        # Mock book object
        mock_book = MagicMock()
        mock_book.get_metadata.side_effect = lambda ns, key: {
            "creator": [("Test Author", {})],
            "date": [("2024-01-01", {})],
            "publisher": [],
            "identifier": [],
            "language": [("en", {})],
            "title": [("Test Book", {})],
        }.get(key, [])

        # Mock document item
        mock_item = MagicMock()
        mock_item.get_type.return_value = 9  # ITEM_DOCUMENT
        mock_item.get_content.return_value = b"<h1>Chapter 1</h1><p>Content here</p>"
        mock_book.get_items.return_value = [mock_item]
        mock_epub.read_epub.return_value = mock_book

        # Patch sys.modules
        orig_ebooklib = sys.modules.get("ebooklib")
        orig_epub = sys.modules.get("ebooklib.epub")
        sys.modules["ebooklib"] = mock_ebooklib
        sys.modules["ebooklib.epub"] = mock_epub
        mock_ebooklib.epub = mock_epub
        try:
            ext = EpubExtractor()
            doc = ext.extract(epub_path)

            assert doc.format == "epub"
            assert doc.source_file == "test.epub"
            assert "Chapter 1" in doc.text
            assert "Content here" in doc.text
            assert doc.native_metadata["author"] == "Test Author"
            assert doc.native_metadata["language"] == "en"
        finally:
            if orig_ebooklib is not None:
                sys.modules["ebooklib"] = orig_ebooklib
            else:
                sys.modules.pop("ebooklib", None)
            if orig_epub is not None:
                sys.modules["ebooklib.epub"] = orig_epub
            else:
                sys.modules.pop("ebooklib.epub", None)
