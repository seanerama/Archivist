"""Image extraction and multimodal embedding for Archivist."""

from archivist.image.embedder import MultimodalEmbedder
from archivist.image.extractor import ExtractedImage, ImageExtractor

__all__ = ["ExtractedImage", "ImageExtractor", "MultimodalEmbedder"]
