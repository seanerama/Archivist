"""Voyage AI embedding backend."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from archivist.config import EmbeddingConfig
from archivist.embedding.base import EmbeddingBackend
from archivist.exceptions import EmbeddingError
from archivist.log import get_logger

logger = get_logger("embedding.voyage")

MAX_RETRIES = 3
BASE_DELAY = 1.0


class VoyageEmbeddingBackend(EmbeddingBackend):
    """Voyage AI API embedding backend with retry/backoff."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._client: Any = None
        self._dim: int | None = None

    def _get_client(self) -> Any:
        """Lazy-init the Voyage client."""
        if self._client is not None:
            return self._client

        try:
            import voyageai

            api_key = self._config.api_key
            if not api_key:
                raise EmbeddingError("Voyage API key is required. Set VOYAGE_API_KEY or configure api_key.")

            self._client = voyageai.Client(api_key=api_key)
            return self._client
        except ImportError as e:
            raise EmbeddingError(f"voyageai package is required: {e}") from e
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"Failed to initialize Voyage client: {e}") from e

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts using Voyage API with retry on rate limits."""
        client = self._get_client()
        model = self._config.model or "voyage-3.5"

        for attempt in range(MAX_RETRIES):
            try:
                result = client.embed(texts, model=model, input_type="document")
                embeddings = np.array(result.embeddings, dtype=np.float32)

                if self._dim is None:
                    self._dim = embeddings.shape[1]

                return embeddings

            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "rate" in error_str:
                    delay = BASE_DELAY * (2**attempt)
                    logger.warning("Rate limited, retrying", attempt=attempt + 1, delay=delay)
                    time.sleep(delay)
                    continue
                raise EmbeddingError(f"Voyage API error: {e}") from e

        raise EmbeddingError(f"Voyage API failed after {MAX_RETRIES} retries")

    @property
    def dimension(self) -> int:
        """Dimension of Voyage embeddings."""
        if self._dim is None:
            # Do a probe embed to get dimension
            self.encode(["test"])
        assert self._dim is not None
        return self._dim
