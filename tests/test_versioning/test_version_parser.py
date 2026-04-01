"""Tests for version parser."""

from __future__ import annotations

import pytest

from archivist.versioning.version_parser import VersionParser


class TestVersionParser:
    """Tests for VersionParser.parse."""

    @pytest.mark.parametrize(
        "input_str, expected",
        [
            ("1.24", (1, 24, 0)),
            ("v2.1.3", (2, 1, 3)),
            ("1.2.3", (1, 2, 3)),
            ("2024.04", (2024, 4, 0)),
            ("r9.3", (9, 3, 0)),
            ("v1.0", (1, 0, 0)),
            ("N/A", None),
            ("NA", None),
            (None, None),
            ("", None),
            ("not-a-version", None),
        ],
    )
    def test_parse(self, input_str: str | None, expected: tuple[int, int, int] | None) -> None:
        assert VersionParser.parse(input_str) == expected


class TestVersionComparison:
    """Tests for VersionParser.compare."""

    def test_equal(self) -> None:
        assert VersionParser.compare((1, 0, 0), (1, 0, 0)) == 0

    def test_less_than(self) -> None:
        assert VersionParser.compare((1, 0, 0), (2, 0, 0)) < 0

    def test_greater_than(self) -> None:
        assert VersionParser.compare((2, 0, 0), (1, 0, 0)) > 0

    def test_minor_version(self) -> None:
        assert VersionParser.compare((1, 24, 0), (1, 26, 0)) < 0

    def test_patch_version(self) -> None:
        assert VersionParser.compare((1, 0, 1), (1, 0, 0)) > 0


class TestVersionRange:
    """Tests for VersionParser.in_range."""

    def test_in_range(self) -> None:
        assert VersionParser.in_range((1, 24, 0), (1, 20, 0), (1, 26, 0)) is True

    def test_below_range(self) -> None:
        assert VersionParser.in_range((1, 18, 0), (1, 20, 0), (1, 26, 0)) is False

    def test_above_range(self) -> None:
        assert VersionParser.in_range((1, 28, 0), (1, 20, 0), (1, 26, 0)) is False

    def test_no_upper_bound(self) -> None:
        assert VersionParser.in_range((99, 0, 0), (1, 0, 0), None) is True

    def test_no_lower_bound(self) -> None:
        assert VersionParser.in_range((0, 1, 0), None, (1, 0, 0)) is True

    def test_at_boundary(self) -> None:
        assert VersionParser.in_range((1, 20, 0), (1, 20, 0), (1, 26, 0)) is True
        assert VersionParser.in_range((1, 26, 0), (1, 20, 0), (1, 26, 0)) is True
