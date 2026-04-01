"""Tests for the PDF extractor."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from archivist.exceptions import ExtractionError
from archivist.extractors.pdf import PdfExtractor


class TestPdfExtractor:
    """Tests for PdfExtractor."""

    def test_supported_extensions(self) -> None:
        ext = PdfExtractor()
        assert ".pdf" in ext.supported_extensions

    def test_extract_missing_file_raises(self, tmp_path: Path) -> None:
        ext = PdfExtractor()
        with pytest.raises(ExtractionError):
            ext.extract(tmp_path / "nonexistent.pdf")

    def test_extract_returns_raw_document(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"dummy")

        # Mock pymupdf document
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Page 1 content"
        mock_doc = MagicMock()
        mock_doc.__len__ = lambda self: 1
        mock_doc.__getitem__ = lambda self, i: mock_page
        mock_doc.metadata = {"author": "Test Author", "creationDate": "2024-01-01"}

        mock_pymupdf = MagicMock()
        mock_pymupdf.open.return_value = mock_doc

        mock_pymupdf4llm = MagicMock()
        mock_pymupdf4llm.to_markdown.return_value = "# Test PDF\n\nPage 1 content"

        # Patch in sys.modules so the local imports inside extract() pick them up
        orig_pymupdf = sys.modules.get("pymupdf")
        orig_pymupdf4llm = sys.modules.get("pymupdf4llm")
        sys.modules["pymupdf"] = mock_pymupdf
        sys.modules["pymupdf4llm"] = mock_pymupdf4llm
        try:
            # Reload the module so the import inside extract() uses our mocks
            ext = PdfExtractor()
            doc = ext.extract(pdf_path)

            assert doc.format == "pdf"
            assert doc.source_file == "test.pdf"
            assert "# Test PDF" in doc.text
            assert doc.pages is not None
            assert len(doc.pages) == 1
            assert doc.pages[0]["page_number"] == 1
            assert doc.native_metadata["author"] == "Test Author"
        finally:
            # Restore original modules
            if orig_pymupdf is not None:
                sys.modules["pymupdf"] = orig_pymupdf
            else:
                sys.modules.pop("pymupdf", None)
            if orig_pymupdf4llm is not None:
                sys.modules["pymupdf4llm"] = orig_pymupdf4llm
            else:
                sys.modules.pop("pymupdf4llm", None)
