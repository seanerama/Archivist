"""High-level family tagger orchestrating classification with conflict detection."""

from __future__ import annotations

from pathlib import Path

from archivist.config import TaggerConfig
from archivist.log import get_logger
from archivist.models import TagResult
from archivist.tagger_backends.base import TaggerBackend

logger = get_logger("tagger.family")


class FamilyTagger:
    """Orchestrates document family classification with auto-accept and conflict detection."""

    def __init__(
        self,
        config: TaggerConfig,
        backend: TaggerBackend,
    ) -> None:
        self._config = config
        self._backend = backend

    def tag(
        self,
        source_path: Path,
        text: str,
        existing_families: list[str],
        filename_hint: str | None = None,
    ) -> tuple[TagResult, bool]:
        """Classify a document and determine if it should be auto-accepted or reviewed.

        Args:
            source_path: Path to the source document.
            text: First ~1500 tokens of extracted text.
            existing_families: Current family slugs from Qdrant.
            filename_hint: Optional family hint from filename parser.

        Returns:
            Tuple of (TagResult, auto_accepted: bool).
            If auto_accepted is False, the result should be queued for review.
        """
        filename = source_path.stem
        result = self._backend.classify(filename, text, existing_families)

        # Determine if auto-accept
        auto_accepted = self._should_auto_accept(result, filename_hint)

        if auto_accepted:
            logger.info("Auto-accepted tag", family=result.family_slug, confidence=result.confidence)
        else:
            logger.info(
                "Tag flagged for review",
                family=result.family_slug,
                confidence=result.confidence,
                reason=self._review_reason(result, filename_hint),
            )

        return result, auto_accepted

    def _should_auto_accept(self, result: TagResult, filename_hint: str | None) -> bool:
        """Check if a tag result should be auto-accepted."""
        if not self._config.auto_accept_tags:
            return False

        if result.confidence < self._config.auto_accept_threshold:
            return False

        if result.is_new_family and self._config.new_family_always_review:
            return False

        # Check for conflict between filename hint and tagger result
        return not (filename_hint and filename_hint != result.family_slug)

    def _review_reason(self, result: TagResult, filename_hint: str | None) -> str:
        """Generate a reason string for why the tag needs review."""
        reasons = []

        if not self._config.auto_accept_tags:
            reasons.append("auto_accept_tags is disabled")

        if result.confidence < self._config.auto_accept_threshold:
            reasons.append(f"low confidence ({result.confidence:.2f} < {self._config.auto_accept_threshold})")

        if result.is_new_family and self._config.new_family_always_review:
            reasons.append("new family requires review")

        if filename_hint and filename_hint != result.family_slug:
            reasons.append(f"filename hint '{filename_hint}' conflicts with tagger '{result.family_slug}'")

        return "; ".join(reasons) if reasons else "manual review required"
