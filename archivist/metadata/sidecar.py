"""Read/write .tag sidecar files for persistent tagging decisions."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from archivist.models import TagResult


class SidecarIO:
    """Reads and writes .tag sidecar JSON files alongside source documents."""

    @staticmethod
    def sidecar_path(source_path: Path) -> Path:
        """Get the .tag sidecar path for a source file."""
        return source_path.with_suffix(source_path.suffix + ".tag")

    @staticmethod
    def read(source_path: Path) -> TagResult | None:
        """Read a .tag sidecar file if it exists.

        Args:
            source_path: Path to the source document.

        Returns:
            TagResult if sidecar exists, None otherwise.
        """
        tag_path = SidecarIO.sidecar_path(source_path)
        if not tag_path.exists():
            return None

        data = json.loads(tag_path.read_text())
        return TagResult(
            family_slug=data["family_slug"],
            doc_title=data.get("doc_title", ""),
            vendor=data.get("vendor"),
            doc_type=data.get("doc_type", "other"),
            is_new_family=data.get("is_new_family", False),
            matched_existing=data.get("matched_existing"),
            confidence=float(data.get("confidence", 1.0)),
            reasoning=data.get("reasoning", "loaded from sidecar"),
        )

    @staticmethod
    def write(
        source_path: Path,
        tag_result: TagResult,
        tagger_model: str = "unknown",
        review_mode: str = "auto_accepted",
    ) -> None:
        """Write a .tag sidecar file.

        Args:
            source_path: Path to the source document.
            tag_result: The tagging result to persist.
            tagger_model: Name of the model that produced the tag.
            review_mode: How the tag was accepted (auto_accepted, manual, etc.).
        """
        tag_path = SidecarIO.sidecar_path(source_path)
        data = {
            "source_file": source_path.name,
            "family_slug": tag_result.family_slug,
            "doc_title": tag_result.doc_title,
            "vendor": tag_result.vendor,
            "doc_type": tag_result.doc_type,
            "tagger_model": tagger_model,
            "confidence": tag_result.confidence,
            "review_mode": review_mode,
            "reasoning": tag_result.reasoning,
            "tagged_at": datetime.now(UTC).isoformat(),
        }
        tag_path.write_text(json.dumps(data, indent=2))
