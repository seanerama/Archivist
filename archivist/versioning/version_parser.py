"""Version string parsing and normalisation."""

from __future__ import annotations

import re

from archivist.models import VersionTuple

# Patterns for version extraction and normalisation
_PATTERNS = [
    re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$"),    # 1.2.3 or v1.2.3
    re.compile(r"^v?(\d+)\.(\d+)$"),             # 1.2 or v1.2
    re.compile(r"^r(\d+)\.(\d+)$"),              # r9.3
    re.compile(r"^(\d{4})\.(\d{1,2})$"),         # 2024.04
]


class VersionParser:
    """Parses and normalises version strings to comparable tuples."""

    @staticmethod
    def parse(version_str: str | None) -> VersionTuple | None:
        """Normalise a version string to a (major, minor, patch) tuple.

        Args:
            version_str: A version string like "1.24", "v2.1.3", "2024.04", "r9.3", or "N/A".

        Returns:
            A VersionTuple (major, minor, patch) or None if unparseable.
        """
        if not version_str or version_str.upper() in ("N/A", "NA", "NONE"):
            return None

        version_str = version_str.strip()

        for pattern in _PATTERNS:
            match = pattern.match(version_str)
            if match:
                groups = match.groups()
                if len(groups) == 3:
                    return (int(groups[0]), int(groups[1]), int(groups[2]))
                elif len(groups) == 2:
                    return (int(groups[0]), int(groups[1]), 0)

        return None

    @staticmethod
    def compare(a: VersionTuple, b: VersionTuple) -> int:
        """Compare two version tuples.

        Returns:
            Negative if a < b, zero if a == b, positive if a > b.
        """
        for av, bv in zip(a, b, strict=True):
            if av != bv:
                return av - bv
        return 0

    @staticmethod
    def in_range(
        version: VersionTuple,
        range_min: VersionTuple | None,
        range_max: VersionTuple | None,
    ) -> bool:
        """Check if a version falls within a range.

        Args:
            version: The version to check.
            range_min: Minimum version (inclusive). None means no lower bound.
            range_max: Maximum version (inclusive). None means no upper bound.
        """
        if range_min is not None and VersionParser.compare(version, range_min) < 0:
            return False
        return not (range_max is not None and VersionParser.compare(version, range_max) > 0)
