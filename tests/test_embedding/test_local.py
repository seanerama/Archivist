"""Tests for local embedding backend."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np

from archivist.config import EmbeddingConfig
from archivist.embedding.local import LocalEmbeddingBackend


class TestLocalEmbeddingBackend:
    """Tests for LocalEmbeddingBackend."""

    def test_encode_returns_correct_shape(self) -> None:
        config = EmbeddingConfig(model_name="test-model", precision="fp32", device="cpu")

        # Mock sentence_transformers
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        mock_st = MagicMock()
        mock_st.SentenceTransformer.return_value = mock_model

        orig = sys.modules.get("sentence_transformers")
        sys.modules["sentence_transformers"] = mock_st
        try:
            backend = LocalEmbeddingBackend(config)
            backend._model = mock_model
            backend._dim = 3

            result = backend.encode(["hello", "world"])
            assert result.shape == (2, 3)
            assert result.dtype == np.float32
        finally:
            if orig is not None:
                sys.modules["sentence_transformers"] = orig
            else:
                sys.modules.pop("sentence_transformers", None)

    def test_dimension_property(self) -> None:
        config = EmbeddingConfig()
        backend = LocalEmbeddingBackend(config)
        backend._dim = 768
        backend._model = MagicMock()  # Prevent lazy load

        assert backend.dimension == 768
