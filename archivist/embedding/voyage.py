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

MAX_RETRIES = 6
BASE_DELAY = 2.0
BATCH_SIZE = 8  # Voyage free tier has low RPM limits


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
        """Encode texts using Voyage API with batching and retry on rate limits."""
        all_embeddings: list[list[float]] = []

        # Process in small batches to avoid rate limits
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            batch_result = self._encode_batch(batch)
            all_embeddings.extend(batch_result)

        result = np.array(all_embeddings, dtype=np.float32)
        if self._dim is None:
            self._dim = result.shape[1]
        return result

    def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode a single batch with retry/backoff."""
        client = self._get_client()
        model = self._config.model or "voyage-3.5"

        for attempt in range(MAX_RETRIES):
            try:
                result = client.embed(texts, model=model, input_type="document")
                return result.embeddings

            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "rate" in error_str:
                    delay = BASE_DELAY * (2**attempt)
                    logger.warning("Rate limited, retrying", attempt=attempt + 1, delay=delay, batch_size=len(texts))
                    time.sleep(delay)
                    continue
                raise EmbeddingError(f"Voyage API error: {e}") from e

        raise EmbeddingError(f"Voyage API failed after {MAX_RETRIES} retries")

    def encode_query(self, text: str) -> np.ndarray:
        """Encode a query using Voyage with input_type='query'."""
        client = self._get_client()
        model = self._config.model or "voyage-3.5"

        for attempt in range(MAX_RETRIES):
            try:
                result = client.embed([text], model=model, input_type="query")
                arr = np.array(result.embeddings, dtype=np.float32)
                if self._dim is None:
                    self._dim = arr.shape[1]
                return arr
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
