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

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimension of the embedding vectors produced by this backend."""
