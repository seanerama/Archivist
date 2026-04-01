"""Tests for multimodal embedding."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from archivist.config import ImageConfig
from archivist.exceptions import EmbeddingError
from archivist.image.embedder import MultimodalEmbedder
from archivist.image.extractor import ExtractedImage


@pytest.fixture
def image_config() -> ImageConfig:
    return ImageConfig(api_key="test-key")


@pytest.fixture
def sample_images() -> list[ExtractedImage]:
    return [
        ExtractedImage(
            data=b"\x89PNG" + b"\x00" * 1000,
            source_file="doc.pdf",
            page_number=1,
            image_index=0,
            width=100,
            height=100,
            caption=None,
        ),
        ExtractedImage(
            data=b"\x89PNG" + b"\x00" * 2000,
            source_file="doc.pdf",
            page_number=2,
            image_index=1,
            width=200,
            height=150,
            caption="A diagram",
        ),
    ]


def _mock_voyageai_context():
    """Context manager to inject a mock voyageai into sys.modules."""
    mock_voyageai = MagicMock()
    orig = sys.modules.get("voyageai")

    class _Ctx:
        def __init__(self):
            self.mock_voyageai = mock_voyageai
            self.mock_client = MagicMock()
            mock_voyageai.Client.return_value = self.mock_client

        def __enter__(self):
            sys.modules["voyageai"] = mock_voyageai
            return self

        def __exit__(self, *args):
            if orig is not None:
                sys.modules["voyageai"] = orig
            else:
                sys.modules.pop("voyageai", None)

    return _Ctx()


class TestMultimodalEmbedder:
    def test_encode_images(
        self,
        image_config: ImageConfig,
        sample_images: list[ExtractedImage],
    ) -> None:
        with _mock_voyageai_context() as ctx:
            mock_result = MagicMock()
            mock_result.embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            ctx.mock_client.multimodal_embed.return_value = mock_result

            embedder = MultimodalEmbedder(image_config)
            embedder._client = ctx.mock_client

            vectors = embedder.encode_images(sample_images)

            assert isinstance(vectors, np.ndarray)
            assert vectors.shape == (2, 3)
            assert vectors.dtype == np.float32
            ctx.mock_client.multimodal_embed.assert_called_once()

    def test_encode_images_batching(
        self,
        image_config: ImageConfig,
    ) -> None:
        """Test that images are batched (BATCH_SIZE=4)."""
        with _mock_voyageai_context() as ctx:
            mock_result1 = MagicMock()
            mock_result1.embeddings = [[0.1] * 128] * 4
            mock_result2 = MagicMock()
            mock_result2.embeddings = [[0.2] * 128] * 1
            ctx.mock_client.multimodal_embed.side_effect = [mock_result1, mock_result2]

            images = [
                ExtractedImage(
                    data=b"\x89PNG" + b"\x00" * 100,
                    source_file="doc.pdf",
                    page_number=i,
                    image_index=i,
                    width=100,
                    height=100,
                    caption=None,
                )
                for i in range(5)
            ]

            embedder = MultimodalEmbedder(image_config)
            embedder._client = ctx.mock_client

            vectors = embedder.encode_images(images)

            assert vectors.shape == (5, 128)
            assert ctx.mock_client.multimodal_embed.call_count == 2

    def test_retry_on_rate_limit(
        self,
        image_config: ImageConfig,
        sample_images: list[ExtractedImage],
    ) -> None:
        with _mock_voyageai_context() as ctx:
            mock_result = MagicMock()
            mock_result.embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

            ctx.mock_client.multimodal_embed.side_effect = [
                Exception("429 rate limit exceeded"),
                mock_result,
            ]

            embedder = MultimodalEmbedder(image_config)
            embedder._client = ctx.mock_client

            with patch("archivist.image.embedder.time.sleep"):
                vectors = embedder.encode_images(sample_images)

            assert vectors.shape == (2, 3)
            assert ctx.mock_client.multimodal_embed.call_count == 2

    def test_missing_api_key(self) -> None:
        config = ImageConfig(api_key=None)
        embedder = MultimodalEmbedder(config)
        images = [
            ExtractedImage(
                data=b"\x89PNG",
                source_file="doc.pdf",
                page_number=1,
                image_index=0,
                width=10,
                height=10,
                caption=None,
            )
        ]
        with pytest.raises(EmbeddingError, match="API key is required"):
            embedder.encode_images(images)

    def test_non_rate_limit_error_raises(
        self,
        image_config: ImageConfig,
        sample_images: list[ExtractedImage],
    ) -> None:
        with _mock_voyageai_context() as ctx:
            ctx.mock_client.multimodal_embed.side_effect = Exception("Invalid input")

            embedder = MultimodalEmbedder(image_config)
            embedder._client = ctx.mock_client

            with pytest.raises(EmbeddingError, match="Voyage multimodal API error"):
                embedder.encode_images(sample_images)
