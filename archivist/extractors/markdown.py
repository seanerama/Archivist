"""Markdown document extractor."""

from __future__ import annotations

from pathlib import Path

from archivist.exceptions import ExtractionError
from archivist.extractors.base import BaseExtractor
from archivist.models import RawDocument


class MarkdownExtractor(BaseExtractor):
    """Extracts text from Markdown files, preserving heading structure."""

    @property
    def supported_extensions(self) -> list[str]:
        return [".md", ".markdown"]

    def extract(self, path: Path) -> RawDocument:
        """Read markdown file directly — headings become natural chunk boundaries."""
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as e:
            raise ExtractionError(f"Failed to read markdown '{path.name}': {e}") from e

        return RawDocument(
            text=text,
            source_file=path.name,
            format="md",
            pages=None,
            native_metadata={},
        )
