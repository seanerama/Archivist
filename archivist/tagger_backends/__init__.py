"""Tagger backend factory."""

from __future__ import annotations

from archivist.config import Config
from archivist.exceptions import MetadataError
from archivist.tagger_backends.anthropic import AnthropicTaggerBackend
from archivist.tagger_backends.base import TaggerBackend
from archivist.tagger_backends.ollama import OllamaTaggerBackend


def get_tagger_backend(config: Config) -> TaggerBackend:
    """Return the appropriate tagger backend based on config.

    Args:
        config: Archivist configuration.

    Returns:
        A TaggerBackend instance.

    Raises:
        MetadataError: If the backend type is not recognized.
    """
    backend_type = config.tagger.type.lower()

    if backend_type == "local":
        return OllamaTaggerBackend(config.tagger)
    elif backend_type == "api":
        return AnthropicTaggerBackend(config.tagger)
    else:
        raise MetadataError(f"Unknown tagger backend type: {backend_type}. Use 'local' or 'api'.")


__all__ = ["AnthropicTaggerBackend", "OllamaTaggerBackend", "TaggerBackend", "get_tagger_backend"]
