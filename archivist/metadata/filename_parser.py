"""Regex-based metadata extraction from filenames."""

from __future__ import annotations

import re
from typing import Any

# Version patterns
VERSION_PATTERNS = [
    re.compile(r"v?(\d+\.\d+\.\d+)"),       # v1.2.3 or 1.2.3
    re.compile(r"v?(\d+\.\d+)"),              # v1.2 or 1.2
    re.compile(r"r(\d+\.\d+)"),               # r9.3 (RHEL-style)
    re.compile(r"(\d{4}\.\d{1,2})"),          # 2024.04 (calendar versioning)
]

# Date patterns
DATE_PATTERNS = [
    re.compile(r"(\d{4}-\d{2}-\d{2})"),       # 2024-01-15
    re.compile(r"(\d{4}_\d{2}_\d{2})"),       # 2024_01_15
    re.compile(r"((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\d{4})", re.IGNORECASE),  # jan2024
    re.compile(r"[_-](\d{4})[_.-]?$"),        # trailing year: _2023
    re.compile(r"[_-](\d{4})[_-]"),           # embedded year: _2023_
]

# Doc type hints
DOC_TYPE_MAP = {
    "admin_guide": "admin_guide",
    "admin-guide": "admin_guide",
    "config_guide": "config_guide",
    "config-guide": "config_guide",
    "configuration_guide": "config_guide",
    "release_notes": "release_notes",
    "release-notes": "release_notes",
    "changelog": "changelog",
    "quickstart": "quickstart",
    "quick_start": "quickstart",
    "quick-start": "quickstart",
    "api_reference": "api_reference",
    "api-reference": "api_reference",
    "api_ref": "api_reference",
    "troubleshooting": "troubleshooting",
}


class FilenameParser:
    """Extracts metadata hints from document filenames."""

    def parse(self, filename: str) -> dict[str, Any]:
        """Parse a filename for version, date, and doc_type hints.

        Args:
            filename: The filename (with or without extension).

        Returns:
            Dict with keys: version (str|None), date (str|None), doc_type (str|None).
        """
        # Strip extension
        stem = filename.rsplit(".", 1)[0] if "." in filename else filename

        return {
            "version": self._extract_version(stem),
            "date": self._extract_date(stem),
            "doc_type": self._extract_doc_type(stem),
        }

    def _extract_version(self, stem: str) -> str | None:
        """Extract version string from filename stem."""
        for pattern in VERSION_PATTERNS:
            match = pattern.search(stem)
            if match:
                return match.group(1)
        return None

    def _extract_date(self, stem: str) -> str | None:
        """Extract date string from filename stem."""
        for pattern in DATE_PATTERNS:
            match = pattern.search(stem)
            if match:
                return match.group(1)
        return None

    def _extract_doc_type(self, stem: str) -> str | None:
        """Extract doc type hint from filename stem."""
        lower = stem.lower()
        for pattern, doc_type in DOC_TYPE_MAP.items():
            if pattern in lower:
                return doc_type
        return None
