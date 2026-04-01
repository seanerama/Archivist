"""Tests for logging configuration."""

from __future__ import annotations

from archivist.log import configure_logging, get_logger


class TestLogging:
    """Tests for logging setup."""

    def test_configure_logging_does_not_raise(self) -> None:
        configure_logging("INFO")

    def test_configure_logging_debug(self) -> None:
        configure_logging("DEBUG")

    def test_get_logger_returns_bound_logger(self) -> None:
        configure_logging("INFO")
        logger = get_logger("test")
        assert logger is not None

    def test_get_logger_without_name(self) -> None:
        configure_logging("INFO")
        logger = get_logger()
        assert logger is not None

    def test_logger_can_log(self, capsys: object) -> None:
        configure_logging("DEBUG")
        logger = get_logger("test")
        # Should not raise
        logger.info("test message", key="value")
