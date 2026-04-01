"""Data models for retrieval results."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SearchResult:
    """A single search result from the retrieval engine."""

    text: str
    score: float
    source_file: str
    family_slug: str
    doc_title: str
    doc_type: str
    version: str | None = None
    page_number: int | None = None
    heading_path: str | None = None
    chunk_role: str = "base"


@dataclass
class FamilyInfo:
    """Information about a document family in the corpus."""

    family_slug: str
    doc_count: int
    versions: list[str]
    doc_types: list[str]


@dataclass
class DiffResult:
    """Result of a version diff between two versions of a family."""

    family: str
    from_version: str
    to_version: str
    added: list[SearchResult] = field(default_factory=list)
    removed: list[SearchResult] = field(default_factory=list)
    modified: list[SearchResult] = field(default_factory=list)
