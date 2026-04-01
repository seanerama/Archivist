"""Multimodal embedding using Voyage AI."""

from __future__ import annotations

import base64
import time
from typing import Any

import numpy as np

from archivist.config import ImageConfig
from archivist.exceptions import EmbeddingError
from archivist.image.extractor import ExtractedImage
from archivist.log import get_logger

logger = get_logger("image.embedder")

MAX_RETRIES = 6
BASE_DELAY = 2.0
BATCH_SIZE = 4  # Multimodal requests are heavier; use smaller batches


class MultimodalEmbedder:
    """Encodes images using the Voyage multimodal embedding API."""

    def __init__(self, config: ImageConfig) -> None:
        self._config = config
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy-init the Voyage client."""
        if self._client is not None:
            return self._client

        try:
            import voyageai

            api_key = self._config.api_key
            if not api_key:
                raise EmbeddingError(
                    "Voyage API key is required for multimodal embeddings. "
                    "Set VOYAGE_API_KEY or configure image.api_key."
                )

            self._client = voyageai.Client(api_key=api_key)
            return self._client
        except ImportError as e:
            raise EmbeddingError(f"voyageai package is required: {e}") from e
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"Failed to initialize Voyage client: {e}") from e

    def encode_images(self, images: list[ExtractedImage]) -> np.ndarray:
        """Encode images using the Voyage multimodal API.

        Args:
            images: List of extracted images to embed.

        Returns:
            numpy array of shape (len(images), embedding_dim).
        """
        all_embeddings: list[list[float]] = []

        for i in range(0, len(images), BATCH_SIZE):
            batch = images[i : i + BATCH_SIZE]
            batch_result = self._encode_batch(batch)
            all_embeddings.extend(batch_result)

        return np.array(all_embeddings, dtype=np.float32)

    def _encode_batch(self, images: list[ExtractedImage]) -> list[list[float]]:
        """Encode a batch of images with retry/backoff."""
        client = self._get_client()
        model = self._config.embedding_model

        # Build multimodal inputs: each image is a list containing one image content item
        inputs = []
        for img in images:
            b64 = base64.b64encode(img.data).decode("ascii")
            content = [{"type": "image", "image": b64, "media_type": "image/png"}]
            inputs.append(content)

        for attempt in range(MAX_RETRIES):
            try:
                result = client.multimodal_embed(inputs, model=model)
                return result.embeddings
            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "rate" in error_str:
                    delay = BASE_DELAY * (2**attempt)
                    logger.warning(
                        "Rate limited, retrying",
                        attempt=attempt + 1,
                        delay=delay,
                        batch_size=len(images),
                    )
                    time.sleep(delay)
                    continue
                raise EmbeddingError(f"Voyage multimodal API error: {e}") from e

        raise EmbeddingError(f"Voyage multimodal API failed after {MAX_RETRIES} retries")
