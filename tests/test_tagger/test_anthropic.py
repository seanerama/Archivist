"""Tests for Anthropic tagger backend."""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock

import pytest

from archivist.config import TaggerConfig
from archivist.exceptions import MetadataError
from archivist.tagger_backends.anthropic import AnthropicTaggerBackend

VALID_RESPONSE = json.dumps({
    "family_slug": "cisco-firepower",
    "doc_title": "Cisco Firepower Config Guide",
    "vendor": "Cisco",
    "doc_type": "config_guide",
    "is_new_family": False,
    "matched_existing": "cisco-firepower",
    "confidence": 0.94,
    "reasoning": "FTD abbreviation matches Firepower Threat Defense.",
})


class TestAnthropicTaggerBackend:
    """Tests for AnthropicTaggerBackend."""

    def test_classify_returns_tag_result(self) -> None:
        config = TaggerConfig(type="api", model="claude-haiku-4-5-20251001", api_key="test-key")

        mock_content = MagicMock()
        mock_content.text = VALID_RESPONSE
        mock_response = MagicMock()
        mock_response.content = [mock_content]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        orig = sys.modules.get("anthropic")
        sys.modules["anthropic"] = mock_anthropic
        try:
            backend = AnthropicTaggerBackend(config)
            result = backend.classify("ftd-config-guide", "Cisco FTD...", ["cisco-firepower"])

            assert result.family_slug == "cisco-firepower"
            assert result.confidence == 0.94
        finally:
            if orig is not None:
                sys.modules["anthropic"] = orig
            else:
                sys.modules.pop("anthropic", None)

    def test_no_api_key_raises(self) -> None:
        config = TaggerConfig(type="api", api_key=None)
        backend = AnthropicTaggerBackend(config)

        mock_anthropic = MagicMock()
        orig = sys.modules.get("anthropic")
        sys.modules["anthropic"] = mock_anthropic
        try:
            with pytest.raises(MetadataError, match="API key is required"):
                backend.classify("test", "text", [])
        finally:
            if orig is not None:
                sys.modules["anthropic"] = orig
            else:
                sys.modules.pop("anthropic", None)

    def test_parse_markdown_wrapped_json(self) -> None:
        config = TaggerConfig(type="api", model="claude-haiku-4-5-20251001")
        backend = AnthropicTaggerBackend(config)
        wrapped = f"```json\n{VALID_RESPONSE}\n```"
        result = backend._parse_response(wrapped)
        assert result.family_slug == "cisco-firepower"
