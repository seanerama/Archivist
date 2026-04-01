"""EPUB document extractor using ebooklib."""

from __future__ import annotations

import html.parser
from pathlib import Path

from archivist.exceptions import ExtractionError
from archivist.extractors.base import BaseExtractor
from archivist.models import RawDocument


class _HTMLTextExtractor(html.parser.HTMLParser):
    """Simple HTML to text converter."""

    def __init__(self) -> None:
        super().__init__()
        self._text: list[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style"):
            self._skip = True
        elif tag in ("p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6", "li"):
            self._text.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style"):
            self._skip = False
        elif tag in ("p", "div", "h1", "h2", "h3", "h4", "h5", "h6"):
            self._text.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._text.append(data)

    def get_text(self) -> str:
        return "".join(self._text).strip()


def _html_to_text(html_content: str) -> str:
    """Convert HTML content to plain text."""
    parser = _HTMLTextExtractor()
    parser.feed(html_content)
    return parser.get_text()


class EpubExtractor(BaseExtractor):
    """Extracts text and OPF metadata from EPUB files."""

    @property
    def supported_extensions(self) -> list[str]:
        return [".epub"]

    def extract(self, path: Path) -> RawDocument:
        """Extract EPUB content with chapter structure and OPF metadata."""
        try:
            import ebooklib
            from ebooklib import epub
        except ImportError as e:
            raise ExtractionError(f"ebooklib is required for EPUB extraction: {e}") from e

        try:
            book = epub.read_epub(str(path), options={"ignore_ncx": True})

            # Extract OPF metadata
            native_metadata: dict[str, str | list[str]] = {}
            for author in book.get_metadata("DC", "creator"):
                if author[0]:
                    native_metadata["author"] = author[0]
            for date in book.get_metadata("DC", "date"):
                if date[0]:
                    native_metadata["created_date"] = date[0]
            for publisher in book.get_metadata("DC", "publisher"):
                if publisher[0]:
                    native_metadata["publisher"] = publisher[0]
            for identifier in book.get_metadata("DC", "identifier"):
                if identifier[0]:
                    native_metadata["identifier"] = identifier[0]
            for language in book.get_metadata("DC", "language"):
                if language[0]:
                    native_metadata["language"] = language[0]
            for title in book.get_metadata("DC", "title"):
                if title[0]:
                    native_metadata["title"] = title[0]

            # Extract text from document items
            chapters: list[str] = []
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    content = item.get_content().decode("utf-8", errors="replace")
                    text = _html_to_text(content)
                    if text.strip():
                        chapters.append(text)

            full_text = "\n\n".join(chapters)

            return RawDocument(
                text=full_text,
                source_file=path.name,
                format="epub",
                pages=None,
                native_metadata=native_metadata,
            )
        except ExtractionError:
            raise
        except Exception as e:
            raise ExtractionError(f"Failed to extract EPUB '{path.name}': {e}") from e
