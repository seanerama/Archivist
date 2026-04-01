"""Abstract base class for embedding backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingBackend(ABC):
    """Abstract base for embedding backends (local and API)."""

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts into embedding vectors.

        Args:
            texts: List of text strings to embed.

        Returns:
            numpy array of shape (len(texts), dimension).
        """

    def encode_query(self, text: str) -> np.ndarray:
        """Encode a query string into an embedding vector.

        Some backends (e.g. Voyage) use a different input_type for queries
        vs documents. Override this method to customize query encoding.

        Args:
            text: The query string to embed.

        Returns:
            numpy array of shape (1, dimension).
        """
        return self.encode([text])

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimension of the embedding vectors produced by this backend."""
