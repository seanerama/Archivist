"""Data models for retrieval results."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchResult:
    """A single search result with metadata."""

    text: str
    score: float
    source_file: str
    family_slug: str
    doc_title: str
    doc_type: str
    version: str | None
    page_number: int | None
    heading_path: str | None
    chunk_role: str  # "base" or "delta"


@dataclass(frozen=True)
class FamilyInfo:
    """Summary of a document family in the corpus."""

    family_slug: str
    doc_types: list[str]
    versions: list[str]
    latest_version: str | None
    total_chunks: int


@dataclass(frozen=True)
class DiffResult:
    """A chunk that changed between two versions."""

    chunk_text: str
    change_type: str  # "added", "modified", "removed"
    source_file: str
    chunk_index: int
    heading_path: str | None
