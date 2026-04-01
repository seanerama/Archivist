"""Embedding backend factory."""

from __future__ import annotations

from archivist.config import Config
from archivist.embedding.base import EmbeddingBackend
from archivist.embedding.local import LocalEmbeddingBackend
from archivist.embedding.voyage import VoyageEmbeddingBackend
from archivist.exceptions import EmbeddingError


def get_embedding_backend(config: Config) -> EmbeddingBackend:
    """Return the appropriate embedding backend based on config.

    Args:
        config: Archivist configuration.

    Returns:
        An EmbeddingBackend instance.

    Raises:
        EmbeddingError: If the backend type is not recognized.
    """
    backend_type = config.embedding.type.lower()

    if backend_type == "local":
        return LocalEmbeddingBackend(config.embedding)
    elif backend_type == "api":
        return VoyageEmbeddingBackend(config.embedding)
    else:
        raise EmbeddingError(f"Unknown embedding backend type: {backend_type}. Use 'local' or 'api'.")


__all__ = ["EmbeddingBackend", "LocalEmbeddingBackend", "VoyageEmbeddingBackend", "get_embedding_backend"]
