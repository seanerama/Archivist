"""Extractor registry — maps file extensions to extractor classes."""

from __future__ import annotations

from pathlib import Path

from archivist.config import Config
from archivist.exceptions import ExtractionError
from archivist.extractors.base import BaseExtractor
from archivist.extractors.epub import EpubExtractor
from archivist.extractors.markdown import MarkdownExtractor
from archivist.extractors.pdf import PdfExtractor
from archivist.extractors.plaintext import PlaintextExtractor
from archivist.extractors.video import VideoExtractor


def get_extractor(path: Path, config: Config | None = None) -> BaseExtractor:
    """Return the appropriate extractor for a file based on its extension.

    Args:
        path: Path to the document file.
        config: Optional config (required for video extractor).

    Returns:
        An extractor instance that can handle this file type.

    Raises:
        ExtractionError: If the file extension is not supported.
    """
    suffix = path.suffix.lower()

    # Build registry
    extractors: list[BaseExtractor] = [
        PdfExtractor(),
        EpubExtractor(),
        MarkdownExtractor(),
        PlaintextExtractor(),
    ]

    if config is not None:
        extractors.append(VideoExtractor(config))

    for extractor in extractors:
        if suffix in extractor.supported_extensions:
            return extractor

    # Check if it's a video extension but no config was provided
    if suffix in [".mp4", ".mov", ".mva", ".mp3", ".wav", ".m4a", ".webm"] and config is None:
        raise ExtractionError(f"Video extraction requires config for Whisper settings: {path.name}")

    supported = set()
    for ext in extractors:
        supported.update(ext.supported_extensions)
    raise ExtractionError(f"Unsupported file format '{suffix}' for '{path.name}'. Supported: {sorted(supported)}")


__all__ = [
    "BaseExtractor",
    "EpubExtractor",
    "MarkdownExtractor",
    "PdfExtractor",
    "PlaintextExtractor",
    "VideoExtractor",
    "get_extractor",
]
