"""Tests for Ollama tagger backend."""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock

import pytest

from archivist.config import TaggerConfig
from archivist.exceptions import MetadataError
from archivist.tagger_backends.ollama import OllamaTaggerBackend

VALID_RESPONSE = json.dumps({
    "family_slug": "nginx",
    "doc_title": "Nginx Admin Guide",
    "vendor": "Nginx Inc",
    "doc_type": "admin_guide",
    "is_new_family": False,
    "matched_existing": "nginx",
    "confidence": 0.95,
    "reasoning": "Filename and content clearly indicate Nginx documentation.",
})


class TestOllamaTaggerBackend:
    """Tests for OllamaTaggerBackend."""

    def test_classify_returns_tag_result(self) -> None:
        config = TaggerConfig(model="qwen3:0.6b")

        mock_response = MagicMock()
        mock_response.message.content = VALID_RESPONSE

        mock_client = MagicMock()
        mock_client.chat.return_value = mock_response

        mock_ollama = MagicMock()
        mock_ollama.Client.return_value = mock_client

        orig = sys.modules.get("ollama")
        sys.modules["ollama"] = mock_ollama
        try:
            backend = OllamaTaggerBackend(config)
            result = backend.classify("nginx_admin_guide", "Nginx configuration...", ["nginx", "kubernetes"])

            assert result.family_slug == "nginx"
            assert result.confidence == 0.95
            assert result.is_new_family is False
        finally:
            if orig is not None:
                sys.modules["ollama"] = orig
            else:
                sys.modules.pop("ollama", None)

    def test_parse_invalid_json_raises(self) -> None:
        config = TaggerConfig()
        backend = OllamaTaggerBackend(config)
        with pytest.raises(MetadataError, match="Failed to parse"):
            backend._parse_response("not json at all")

    def test_parse_missing_field_raises(self) -> None:
        config = TaggerConfig()
        backend = OllamaTaggerBackend(config)
        with pytest.raises(MetadataError, match="Failed to parse"):
            backend._parse_response('{"doc_title": "missing family_slug"}')
