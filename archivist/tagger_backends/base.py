"""Abstract base class for tagger backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from archivist.models import TagResult


class TaggerBackend(ABC):
    """Abstract base for LLM tagger backends (local and API)."""

    @abstractmethod
    def classify(self, filename: str, text: str, existing_families: list[str]) -> TagResult:
        """Classify a document into a family.

        Args:
            filename: Source filename (stem only, no extension).
            text: First ~1500 tokens of extracted document text.
            existing_families: List of existing family slugs from Qdrant.

        Returns:
            TagResult with classification details.
        """
