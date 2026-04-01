"""Tests for the FamilyTagger orchestrator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from archivist.config import TaggerConfig
from archivist.metadata.family_tagger import FamilyTagger
from archivist.models import TagResult


def _make_tag_result(**kwargs: object) -> TagResult:
    defaults = {
        "family_slug": "nginx",
        "doc_title": "Nginx Admin Guide",
        "vendor": "Nginx Inc",
        "doc_type": "admin_guide",
        "is_new_family": False,
        "matched_existing": "nginx",
        "confidence": 0.95,
        "reasoning": "Test reasoning",
    }
    defaults.update(kwargs)
    return TagResult(**defaults)  # type: ignore[arg-type]


class TestFamilyTagger:
    """Tests for FamilyTagger auto-accept and conflict detection."""

    def test_auto_accept_when_high_confidence(self) -> None:
        config = TaggerConfig(auto_accept_tags=True, auto_accept_threshold=0.90)
        backend = MagicMock()
        backend.classify.return_value = _make_tag_result(confidence=0.95, is_new_family=False)

        tagger = FamilyTagger(config, backend)
        result, auto_accepted = tagger.tag(Path("nginx_admin.pdf"), "text", ["nginx"])

        assert auto_accepted is True
        assert result.family_slug == "nginx"

    def test_reject_when_low_confidence(self) -> None:
        config = TaggerConfig(auto_accept_tags=True, auto_accept_threshold=0.90)
        backend = MagicMock()
        backend.classify.return_value = _make_tag_result(confidence=0.70)

        tagger = FamilyTagger(config, backend)
        _, auto_accepted = tagger.tag(Path("ambiguous.pdf"), "text", ["nginx"])

        assert auto_accepted is False

    def test_reject_new_family_when_review_required(self) -> None:
        config = TaggerConfig(
            auto_accept_tags=True, auto_accept_threshold=0.90, new_family_always_review=True
        )
        backend = MagicMock()
        backend.classify.return_value = _make_tag_result(
            confidence=0.95, is_new_family=True, matched_existing=None
        )

        tagger = FamilyTagger(config, backend)
        _, auto_accepted = tagger.tag(Path("new_product.pdf"), "text", [])

        assert auto_accepted is False

    def test_reject_when_auto_accept_disabled(self) -> None:
        config = TaggerConfig(auto_accept_tags=False)
        backend = MagicMock()
        backend.classify.return_value = _make_tag_result(confidence=0.99)

        tagger = FamilyTagger(config, backend)
        _, auto_accepted = tagger.tag(Path("test.pdf"), "text", [])

        assert auto_accepted is False

    def test_reject_on_filename_conflict(self) -> None:
        config = TaggerConfig(auto_accept_tags=True, auto_accept_threshold=0.90)
        backend = MagicMock()
        backend.classify.return_value = _make_tag_result(
            family_slug="nginx", confidence=0.95, is_new_family=False
        )

        tagger = FamilyTagger(config, backend)
        _, auto_accepted = tagger.tag(
            Path("test.pdf"), "text", ["nginx"],
            filename_hint="apache",  # Conflict!
        )

        assert auto_accepted is False

    def test_review_reason_includes_all_factors(self) -> None:
        config = TaggerConfig(
            auto_accept_tags=True, auto_accept_threshold=0.90, new_family_always_review=True
        )
        backend = MagicMock()

        tagger = FamilyTagger(config, backend)
        result = _make_tag_result(confidence=0.70, is_new_family=True)
        reason = tagger._review_reason(result, "different-hint")

        assert "low confidence" in reason
        assert "new family" in reason
        assert "conflicts" in reason
