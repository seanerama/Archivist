"""Ollama local LLM tagger backend."""

from __future__ import annotations

import json

from archivist.config import TaggerConfig
from archivist.exceptions import MetadataError
from archivist.log import get_logger
from archivist.models import TagResult
from archivist.tagger_backends.base import TaggerBackend
from archivist.tagger_backends.prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

logger = get_logger("tagger.ollama")


class OllamaTaggerBackend(TaggerBackend):
    """Tagger using a local LLM via Ollama."""

    def __init__(self, config: TaggerConfig) -> None:
        self._model = config.model
        self._host = config.ollama_host

    def check_model_available(self) -> None:
        """Verify the configured model is available in Ollama.

        Raises:
            MetadataError: If the model is not found or Ollama is unreachable.
        """
        try:
            import ollama

            client = ollama.Client(host=self._host)
            models = client.list()
            model_names = [m.model for m in models.models] if hasattr(models, "models") else []

            # Check if model name matches (with or without tag)
            found = any(
                self._model in name or name.startswith(self._model.split(":")[0])
                for name in model_names
            )
            if not found:
                raise MetadataError(
                    f"Model '{self._model}' not found in Ollama.\n"
                    f"  Run: ollama pull {self._model}\n"
                    f"  Or change tagger_backend.model in your config.\n"
                    f"  Available models: {model_names}"
                )
        except MetadataError:
            raise
        except Exception as e:
            raise MetadataError(f"Cannot connect to Ollama at {self._host}: {e}") from e

    def classify(self, filename: str, text: str, existing_families: list[str]) -> TagResult:
        """Classify a document using Ollama."""
        try:
            import ollama

            client = ollama.Client(host=self._host)

            user_prompt = USER_PROMPT_TEMPLATE.format(
                existing_families=", ".join(existing_families) if existing_families else "(none — this is the first)",
                filename=filename,
                text=text[:3000],  # ~1500 tokens
            )

            response = client.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                format="json",
            )

            content = response.message.content
            return self._parse_response(content)

        except MetadataError:
            raise
        except Exception as e:
            raise MetadataError(f"Ollama tagger failed: {e}") from e

    def _parse_response(self, content: str) -> TagResult:
        """Parse JSON response into TagResult."""
        try:
            data = json.loads(content)
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
