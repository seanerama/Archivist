"""Tests for filename metadata parser."""

from __future__ import annotations

import pytest

from archivist.metadata.filename_parser import FilenameParser


class TestFilenameParser:
    """Tests for FilenameParser."""

    def setup_method(self) -> None:
        self.parser = FilenameParser()

    @pytest.mark.parametrize(
        "filename, expected_version",
        [
            ("nginx_1.24_admin_guide.pdf", "1.24"),
            ("kubernetes-1.29-release-notes.md", "1.29"),
            ("v2.1.3-config.pdf", "2.1.3"),
            ("RHEL_r9.3_Security_Guide.pdf", "9.3"),
            ("release_2024.04.pdf", "2024.04"),
            ("no_version_here.txt", None),
        ],
    )
    def test_extract_version(self, filename: str, expected_version: str | None) -> None:
        result = self.parser.parse(filename)
        assert result["version"] == expected_version

    @pytest.mark.parametrize(
        "filename, expected_date",
        [
            ("doc_2024-01-15.pdf", "2024-01-15"),
            ("RHEL_9.3_Security_Guide_2023.pdf", "2023"),
            ("no_date.txt", None),
        ],
    )
    def test_extract_date(self, filename: str, expected_date: str | None) -> None:
        result = self.parser.parse(filename)
        assert result["date"] == expected_date

    @pytest.mark.parametrize(
        "filename, expected_type",
        [
            ("nginx_admin_guide.pdf", "admin_guide"),
            ("release_notes_v1.2.pdf", "release_notes"),
            ("quick_start.md", "quickstart"),
            ("api_reference.pdf", "api_reference"),
            ("random_doc.txt", None),
        ],
    )
    def test_extract_doc_type(self, filename: str, expected_type: str | None) -> None:
        result = self.parser.parse(filename)
        assert result["doc_type"] == expected_type

    def test_full_parse(self) -> None:
        result = self.parser.parse("nginx_1.24_admin_guide_2023.pdf")
        assert result["version"] == "1.24"
        assert result["doc_type"] == "admin_guide"
