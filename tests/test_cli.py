"""Tests for the CLI."""

from __future__ import annotations

from typer.testing import CliRunner

from archivist.cli import app

runner = CliRunner()


class TestCLI:
    """Smoke tests for the Typer CLI."""

    def test_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Archivist" in result.stdout

    def test_ingest_help(self) -> None:
        result = runner.invoke(app, ["ingest", "--help"])
        assert result.exit_code == 0
        assert "dry-run" in result.stdout

    def test_status_help(self) -> None:
        result = runner.invoke(app, ["status", "--help"])
        assert result.exit_code == 0

    def test_review_help(self) -> None:
        result = runner.invoke(app, ["review", "--help"])
        assert result.exit_code == 0

    def test_setup_help(self) -> None:
        result = runner.invoke(app, ["setup", "--help"])
        assert result.exit_code == 0
