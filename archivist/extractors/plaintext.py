"""Plaintext document extractor."""

from __future__ import annotations

from pathlib import Path

from archivist.exceptions import ExtractionError
from archivist.extractors.base import BaseExtractor
from archivist.models import RawDocument


class PlaintextExtractor(BaseExtractor):
    """Extracts text from plain text files."""

    @property
    def supported_extensions(self) -> list[str]:
        return [".txt", ".text"]

    def extract(self, path: Path) -> RawDocument:
        """Read plain text file directly."""
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as e:
            raise ExtractionError(f"Failed to read text file '{path.name}': {e}") from e

        return RawDocument(
            text=text,
            source_file=path.name,
            format="txt",
            pages=None,
            native_metadata={},
        )
