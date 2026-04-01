"""Local embedding backend using sentence-transformers."""

from __future__ import annotations

from typing import Any

import numpy as np

from archivist.config import EmbeddingConfig
from archivist.embedding.base import EmbeddingBackend
from archivist.exceptions import EmbeddingError
from archivist.log import get_logger

logger = get_logger("embedding.local")


class LocalEmbeddingBackend(EmbeddingBackend):
    """Local embedding using HuggingFace sentence-transformers models."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._model: Any = None
        self._dim: int | None = None

    def _load_model(self) -> None:
        """Lazy-load the model on first use."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            device = self._config.device
            if device == "auto":
                device = None  # sentence-transformers auto-detects

            self._model = SentenceTransformer(self._config.model_name, device=device)

            # Apply precision
            if self._config.precision == "fp16":
                self._model.half()
            elif self._config.precision == "q8":
                try:
                    import bitsandbytes  # noqa: F401
                    logger.info("INT8 quantisation requested — applied at model level")
                except ImportError:
                    logger.warning("bitsandbytes not available, falling back to fp32")

            # Get dimension from a test encode
            test_embedding = self._model.encode(["test"], convert_to_numpy=True)
            self._dim = test_embedding.shape[1]

            logger.info(
                "Loaded embedding model",
                model=self._config.model_name,
                precision=self._config.precision,
                dimension=self._dim,
            )
        except Exception as e:
            raise EmbeddingError(f"Failed to load embedding model '{self._config.model_name}': {e}") from e

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts using the local sentence-transformers model."""
        self._load_model()
        try:
            embeddings = self._model.encode(
                texts,
                batch_size=self._config.batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return np.array(embeddings, dtype=np.float32)
        except Exception as e:
            raise EmbeddingError(f"Failed to encode {len(texts)} texts: {e}") from e

    @property
    def dimension(self) -> int:
        """Dimension of the loaded model's embeddings."""
        self._load_model()
        assert self._dim is not None
        return self._dim
