"""Tests for the extractor registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from archivist.config import Config
from archivist.exceptions import ExtractionError
from archivist.extractors import get_extractor
from archivist.extractors.epub import EpubExtractor
from archivist.extractors.markdown import MarkdownExtractor
from archivist.extractors.pdf import PdfExtractor
from archivist.extractors.plaintext import PlaintextExtractor
from archivist.extractors.video import VideoExtractor


class TestExtractorRegistry:
    """Tests for get_extractor registry function."""

    def test_pdf_extension(self) -> None:
        ext = get_extractor(Path("doc.pdf"))
        assert isinstance(ext, PdfExtractor)

    def test_epub_extension(self) -> None:
        ext = get_extractor(Path("book.epub"))
        assert isinstance(ext, EpubExtractor)

    def test_markdown_extension(self) -> None:
        ext = get_extractor(Path("readme.md"))
        assert isinstance(ext, MarkdownExtractor)

    def test_markdown_long_extension(self) -> None:
        ext = get_extractor(Path("readme.markdown"))
        assert isinstance(ext, MarkdownExtractor)

    def test_txt_extension(self) -> None:
        ext = get_extractor(Path("notes.txt"))
        assert isinstance(ext, PlaintextExtractor)

    def test_video_extension_with_config(self) -> None:
        config = Config.default()
        ext = get_extractor(Path("video.mp4"), config=config)
        assert isinstance(ext, VideoExtractor)

    def test_video_extension_without_config_raises(self) -> None:
        with pytest.raises(ExtractionError, match="requires config"):
            get_extractor(Path("video.mp4"))

    def test_unsupported_extension_raises(self) -> None:
        with pytest.raises(ExtractionError, match="Unsupported file format"):
            get_extractor(Path("data.xyz"))

    def test_case_insensitive(self) -> None:
        ext = get_extractor(Path("DOC.PDF"))
        assert isinstance(ext, PdfExtractor)
