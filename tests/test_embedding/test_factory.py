"""Tests for embedding backend factory."""

from __future__ import annotations

import pytest

from archivist.config import Config
from archivist.embedding import get_embedding_backend
from archivist.embedding.local import LocalEmbeddingBackend
from archivist.embedding.voyage import VoyageEmbeddingBackend
from archivist.exceptions import EmbeddingError


class TestEmbeddingFactory:
    """Tests for get_embedding_backend."""

    def test_local_backend(self) -> None:
        config = Config.default()
        config.embedding.type = "local"
        backend = get_embedding_backend(config)
        assert isinstance(backend, LocalEmbeddingBackend)

    def test_api_backend(self) -> None:
        config = Config.default()
        config.embedding.type = "api"
        backend = get_embedding_backend(config)
        assert isinstance(backend, VoyageEmbeddingBackend)

    def test_unknown_backend_raises(self) -> None:
        config = Config.default()
        config.embedding.type = "unknown"
        with pytest.raises(EmbeddingError, match="Unknown embedding backend"):
            get_embedding_backend(config)
