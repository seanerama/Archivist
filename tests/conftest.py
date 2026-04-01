"""Shared test fixtures for Archivist."""

from __future__ import annotations

from pathlib import Path

import pytest

from archivist.config import Config
from archivist.models import Chunk, RawDocument


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to the test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def default_config() -> Config:
    """Default Archivist configuration for tests."""
    return Config.default()


@pytest.fixture
def sample_raw_document() -> RawDocument:
    """A sample RawDocument for testing."""
    return RawDocument(
        text="# Test Document\n\nThis is a test document.\n\n## Section 1\n\nFirst section content.\n",
        source_file="test_doc_v1.2.md",
        format="md",
        pages=None,
        native_metadata={},
    )


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Sample chunks for testing."""
    return [
        Chunk(
            text="# Test Document\n\nThis is a test document with some content.",
            chunk_index=0,
            source_file="test_doc_v1.2.md",
            heading_path="Test Document",
            token_count=12,
        ),
        Chunk(
            text="## Section 1\n\nFirst section content.",
            chunk_index=1,
            source_file="test_doc_v1.2.md",
            heading_path="Test Document > Section 1",
            token_count=7,
        ),
    ]
