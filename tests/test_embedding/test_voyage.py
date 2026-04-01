"""Tests for Voyage embedding backend."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from archivist.config import EmbeddingConfig
from archivist.embedding.voyage import VoyageEmbeddingBackend
from archivist.exceptions import EmbeddingError


class TestVoyageEmbeddingBackend:
    """Tests for VoyageEmbeddingBackend."""

    def test_encode_calls_api(self) -> None:
        config = EmbeddingConfig(type="api", provider="voyage", model="voyage-3.5", api_key="test-key")

        mock_result = MagicMock()
        mock_result.embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        mock_client = MagicMock()
        mock_client.embed.return_value = mock_result

        mock_voyageai = MagicMock()
        mock_voyageai.Client.return_value = mock_client

        orig = sys.modules.get("voyageai")
        sys.modules["voyageai"] = mock_voyageai
        try:
            backend = VoyageEmbeddingBackend(config)
            backend._client = mock_client

            result = backend.encode(["hello", "world"])
            assert result.shape == (2, 3)
            mock_client.embed.assert_called_once_with(
                ["hello", "world"], model="voyage-3.5", input_type="document"
            )
        finally:
            if orig is not None:
                sys.modules["voyageai"] = orig
            else:
                sys.modules.pop("voyageai", None)

    def test_no_api_key_raises(self) -> None:
        config = EmbeddingConfig(type="api", api_key=None)
        backend = VoyageEmbeddingBackend(config)
        with pytest.raises(EmbeddingError, match="API key is required"):
            backend._get_client()
