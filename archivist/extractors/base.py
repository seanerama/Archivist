"""Abstract base class for document extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from archivist.models import RawDocument


class BaseExtractor(ABC):
    """Abstract base for all document format extractors."""

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """File extensions this extractor handles (e.g. ['.pdf'])."""

    @abstractmethod
    def extract(self, path: Path) -> RawDocument:
        """Extract text and metadata from a document.

        Args:
            path: Path to the document file.

        Returns:
            RawDocument with extracted text and metadata.

        Raises:
            ExtractionError: If extraction fails.
        """
