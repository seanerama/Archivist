"""Tests for image extraction."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from archivist.config import ImageConfig
from archivist.image.extractor import ExtractedImage, ImageExtractor
from archivist.models import RawDocument


@pytest.fixture
def image_config(tmp_path: Path) -> ImageConfig:
    return ImageConfig(cache_dir=str(tmp_path / "cache"))


@pytest.fixture
def raw_pdf_doc() -> RawDocument:
    return RawDocument(text="sample text", source_file="test.pdf", format="pdf")


@pytest.fixture
def raw_epub_doc() -> RawDocument:
    return RawDocument(text="sample text", source_file="test.epub", format="epub")


class TestExtractedImage:
    def test_frozen_dataclass(self) -> None:
        img = ExtractedImage(
            data=b"\x89PNG" + b"\x00" * 20000,
            source_file="doc.pdf",
            page_number=1,
            image_index=0,
            width=100,
            height=200,
            caption=None,
        )
        assert img.source_file == "doc.pdf"
        assert img.page_number == 1
        assert img.width == 100
        assert img.height == 200

        with pytest.raises(AttributeError):
            img.width = 999  # type: ignore[misc]


class TestImageExtractorPDF:
    def _setup_fitz_mock(self, image_data: bytes, width: int = 640, height: int = 480) -> MagicMock:
        """Create a mock fitz module with a single-page, single-image document."""
        mock_fitz = MagicMock()

        mock_page = MagicMock()
        mock_page.get_images.return_value = [(42,)]

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.extract_image.return_value = {
            "image": image_data,
            "ext": "png",
            "width": width,
            "height": height,
        }
        # Support iteration by page index via __getitem__
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)

        mock_fitz.open.return_value = mock_doc
        return mock_fitz

    def test_extract_from_pdf(
        self, image_config: ImageConfig, raw_pdf_doc: RawDocument, tmp_path: Path
    ) -> None:
        fake_image_data = b"\x89PNG" + b"\x00" * 20000

        mock_fitz = self._setup_fitz_mock(fake_image_data)
        orig = sys.modules.get("fitz")
        sys.modules["fitz"] = mock_fitz
        try:
            extractor = ImageExtractor(image_config)
            pdf_path = tmp_path / "test.pdf"
            pdf_path.touch()

            images = extractor.extract_images(pdf_path, raw_pdf_doc)

            assert len(images) == 1
            assert images[0].source_file == "test.pdf"
            assert images[0].page_number == 1
            assert images[0].width == 640
            assert images[0].height == 480
            assert images[0].image_index == 0
            mock_fitz.open.assert_called_once_with(str(pdf_path))
        finally:
            if orig is not None:
                sys.modules["fitz"] = orig
            else:
                sys.modules.pop("fitz", None)

    def test_filters_small_images(
        self, image_config: ImageConfig, raw_pdf_doc: RawDocument, tmp_path: Path
    ) -> None:
        small_data = b"\x89PNG" + b"\x00" * 100  # < min_size (10000)
        mock_fitz = self._setup_fitz_mock(small_data, width=10, height=10)

        orig = sys.modules.get("fitz")
        sys.modules["fitz"] = mock_fitz
        try:
            extractor = ImageExtractor(image_config)
            pdf_path = tmp_path / "test.pdf"
            pdf_path.touch()

            images = extractor.extract_images(pdf_path, raw_pdf_doc)
            assert len(images) == 0
        finally:
            if orig is not None:
                sys.modules["fitz"] = orig
            else:
                sys.modules.pop("fitz", None)

    def test_caching(
        self, image_config: ImageConfig, raw_pdf_doc: RawDocument, tmp_path: Path
    ) -> None:
        fake_data = b"\x89PNG" + b"\x00" * 20000
        mock_fitz = self._setup_fitz_mock(fake_data, width=100, height=100)

        orig = sys.modules.get("fitz")
        sys.modules["fitz"] = mock_fitz
        try:
            extractor = ImageExtractor(image_config)
            pdf_path = tmp_path / "test.pdf"
            pdf_path.touch()

            # First call: extracts and caches
            images1 = extractor.extract_images(pdf_path, raw_pdf_doc)
            assert len(images1) == 1

            # Second call: should load from cache (fitz.open not called again)
            mock_fitz.open.reset_mock()
            images2 = extractor.extract_images(pdf_path, raw_pdf_doc)
            assert len(images2) == 1
            mock_fitz.open.assert_not_called()
        finally:
            if orig is not None:
                sys.modules["fitz"] = orig
            else:
                sys.modules.pop("fitz", None)


class TestImageExtractorEPUB:
    def test_extract_from_epub(
        self,
        image_config: ImageConfig,
        raw_epub_doc: RawDocument,
        tmp_path: Path,
    ) -> None:
        fake_data = b"\x89PNG" + b"\x00" * 20000

        mock_item = MagicMock()
        mock_item.media_type = "image/png"
        mock_item.get_content.return_value = fake_data

        mock_book = MagicMock()
        mock_book.get_items.return_value = [mock_item]

        mock_epub = MagicMock()
        mock_epub.read_epub.return_value = mock_book

        mock_ebooklib = MagicMock()
        mock_ebooklib.epub = mock_epub  # Ensure `from ebooklib import epub` resolves correctly

        orig_ebooklib = sys.modules.get("ebooklib")
        orig_epub = sys.modules.get("ebooklib.epub")
        sys.modules["ebooklib"] = mock_ebooklib
        sys.modules["ebooklib.epub"] = mock_epub
        try:
            extractor = ImageExtractor(image_config)
            epub_path = tmp_path / "test.epub"
            epub_path.touch()

            with patch.object(ImageExtractor, "_get_image_dimensions", return_value=(320, 240)):
                images = extractor.extract_images(epub_path, raw_epub_doc)

            assert len(images) == 1
            assert images[0].source_file == "test.epub"
            assert images[0].width == 320
            assert images[0].height == 240
        finally:
            if orig_ebooklib is not None:
                sys.modules["ebooklib"] = orig_ebooklib
            else:
                sys.modules.pop("ebooklib", None)
            if orig_epub is not None:
                sys.modules["ebooklib.epub"] = orig_epub
            else:
                sys.modules.pop("ebooklib.epub", None)


class TestImageExtractorUnsupportedFormat:
    def test_returns_empty_for_unsupported(self, image_config: ImageConfig, tmp_path: Path) -> None:
        raw_doc = RawDocument(text="hello", source_file="test.md", format="md")
        extractor = ImageExtractor(image_config)
        images = extractor.extract_images(tmp_path / "test.md", raw_doc)
        assert images == []


class TestImageConfig:
    def test_defaults(self) -> None:
        config = ImageConfig()
        assert config.enabled is False
        assert config.min_size == 10000
        assert config.embedding_model == "voyage-multimodal-3.5"
        assert "pdf" in config.formats
        assert "epub" in config.formats
        assert config.cache_dir == ".archivist-cache/images"
        assert config.api_key is None
