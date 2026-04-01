"""Tests for content scanner."""

from __future__ import annotations

from archivist.metadata.content_scanner import ContentScanner


class TestContentScanner:
    """Tests for ContentScanner."""

    def setup_method(self) -> None:
        self.scanner = ContentScanner()

    def test_find_version_pattern(self) -> None:
        text = "This is the documentation for Version 1.24.3 of nginx."
        result = self.scanner.scan(text)
        assert result["version"] == "1.24.3"

    def test_find_release_pattern(self) -> None:
        text = "Release 2.0 introduces new features."
        result = self.scanner.scan(text)
        assert result["version"] == "2.0"

    def test_find_iso_date(self) -> None:
        text = "Published: 2024-01-15\nContent here."
        result = self.scanner.scan(text)
        assert result["date"] == "2024-01-15"

    def test_find_copyright_year(self) -> None:
        text = "Copyright 2023 Nginx Inc. All rights reserved."
        result = self.scanner.scan(text)
        assert result["date"] == "2023"

    def test_find_last_updated(self) -> None:
        text = "Last Updated: 2024-03-01\nSome content."
        result = self.scanner.scan(text)
        assert result["extra"]["last_updated"] == "2024-03-01"

    def test_no_metadata_found(self) -> None:
        text = "This is a plain document with no version or date info."
        result = self.scanner.scan(text)
        assert result["version"] is None
        assert result["date"] is None

    def test_respects_token_limit(self) -> None:
        # Version appears after the scan limit
        text = "x " * 10000 + "Version 5.0"
        result = self.scanner.scan(text, max_tokens=100)
        assert result["version"] is None
