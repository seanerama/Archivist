"""Anthropic Claude tagger backend."""

from __future__ import annotations

import json

from archivist.config import TaggerConfig
from archivist.exceptions import MetadataError
from archivist.log import get_logger
from archivist.models import TagResult
from archivist.tagger_backends.base import TaggerBackend
from archivist.tagger_backends.prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

logger = get_logger("tagger.anthropic")

SUPPORTED_MODELS = {"claude-haiku-4-5-20251001", "claude-haiku-3-5-20241022"}


class AnthropicTaggerBackend(TaggerBackend):
    """Tagger using Claude Haiku via Anthropic API."""

    def __init__(self, config: TaggerConfig) -> None:
        self._model = config.model
        self._api_key = config.api_key

        if self._model not in SUPPORTED_MODELS:
            logger.warning(
                "Unrecognized tagger model",
                model=self._model,
                supported=list(SUPPORTED_MODELS),
            )

    def classify(self, filename: str, text: str, existing_families: list[str]) -> TagResult:
        """Classify a document using Claude Haiku."""
        try:
            import anthropic

            if not self._api_key:
                raise MetadataError("Anthropic API key is required. Set ANTHROPIC_API_KEY.")

            client = anthropic.Anthropic(api_key=self._api_key)

            user_prompt = USER_PROMPT_TEMPLATE.format(
                existing_families=", ".join(existing_families) if existing_families else "(none)",
                filename=filename,
                text=text[:3000],
            )

            response = client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )

            content = response.content[0].text
            return self._parse_response(content)

        except MetadataError:
            raise
        except Exception as e:
            raise MetadataError(f"Anthropic tagger failed: {e}") from e

    def _parse_response(self, content: str) -> TagResult:
        """Parse JSON response into TagResult."""
        # Claude may wrap JSON in markdown code blocks
        cleaned = content.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned

        try:
            data = json.loads(cleaned)
            return TagResult(
                family_slug=data["family_slug"],
                doc_title=data.get("doc_title", ""),
                vendor=data.get("vendor"),
                doc_type=data.get("doc_type", "other"),
                is_new_family=data.get("is_new_family", True),
                matched_existing=data.get("matched_existing"),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", ""),
            )
        except (json.JSONDecodeError, KeyError) as e:
            raise MetadataError(f"Failed to parse tagger response: {e}\nResponse: {content}") from e
