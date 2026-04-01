"""Scan document content for version and date metadata."""

from __future__ import annotations

import re
from typing import Any

VERSION_CONTENT_PATTERNS = [
    re.compile(r"[Vv]ersion\s+(\d+\.\d+(?:\.\d+)?)"),
    re.compile(r"[Rr]elease\s+(\d+\.\d+(?:\.\d+)?)"),
    re.compile(r"v(\d+\.\d+(?:\.\d+)?)"),
]

DATE_CONTENT_PATTERNS = [
    re.compile(r"(\d{4}-\d{2}-\d{2})"),
    re.compile(r"[Cc]opyright\s+(?:\(c\)\s*)?(\d{4})"),
    re.compile(r"[Pp]ublished:?\s*(\d{4}(?:-\d{2}(?:-\d{2})?)?)"),
    re.compile(r"[Ll]ast\s+[Uu]pdated:?\s*(\d{4}(?:-\d{2}(?:-\d{2})?)?)"),
    re.compile(r"[Rr]evision:?\s*(\d+(?:\.\d+)?)"),
]


class ContentScanner:
    """Scans the beginning of a document for metadata signals."""

    def scan(self, text: str, max_tokens: int = 2000) -> dict[str, Any]:
        """Scan the first ~2000 tokens of text for version and date patterns.

        Args:
            text: Document text to scan.
            max_tokens: Approximate token limit for scanning.

        Returns:
            Dict with keys: version (str|None), date (str|None), extra (dict).
        """
        # Approximate token limit as characters
        scan_text = text[: max_tokens * 4]

        version = self._find_version(scan_text)
        date = self._find_date(scan_text)
        extra = self._find_extra(scan_text)

        return {
            "version": version,
            "date": date,
            "extra": extra,
        }

    def _find_version(self, text: str) -> str | None:
        """Find the first version pattern in text."""
        for pattern in VERSION_CONTENT_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(1)
        return None

    def _find_date(self, text: str) -> str | None:
        """Find the first date pattern in text."""
        for pattern in DATE_CONTENT_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(1)
        return None

    def _find_extra(self, text: str) -> dict[str, str]:
        """Find additional metadata signals."""
        extra: dict[str, str] = {}

        revision = re.search(r"[Rr]evision:?\s*(\S+)", text)
        if revision:
            extra["revision"] = revision.group(1)

        last_updated = re.search(r"[Ll]ast\s+[Uu]pdated:?\s*(\S+)", text)
        if last_updated:
            extra["last_updated"] = last_updated.group(1)

        return extra
