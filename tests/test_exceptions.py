"""Tests for the exception hierarchy."""

from __future__ import annotations

import pytest

from archivist.exceptions import (
    ArchivistError,
    ChunkingError,
    ConfigError,
    EmbeddingError,
    ExtractionError,
    MetadataError,
    SetupError,
    StorageError,
    VersioningError,
)


class TestExceptionHierarchy:
    """All exceptions should subclass ArchivistError."""

    @pytest.mark.parametrize(
        "exc_class",
        [
            ConfigError,
            ExtractionError,
            MetadataError,
            ChunkingError,
            EmbeddingError,
            VersioningError,
            StorageError,
            SetupError,
        ],
    )
    def test_subclass_of_archivist_error(self, exc_class: type[ArchivistError]) -> None:
        assert issubclass(exc_class, ArchivistError)

    @pytest.mark.parametrize(
        "exc_class",
        [
            ConfigError,
            ExtractionError,
            MetadataError,
            ChunkingError,
            EmbeddingError,
            VersioningError,
            StorageError,
            SetupError,
        ],
    )
    def test_catchable_as_archivist_error(self, exc_class: type[ArchivistError]) -> None:
        with pytest.raises(ArchivistError):
            raise exc_class("test message")

    def test_exception_message(self) -> None:
        err = ConfigError("bad config")
        assert str(err) == "bad config"
