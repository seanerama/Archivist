"""Shared data models used across all Archivist stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

VersionTuple = tuple[int, int, int]


class ChunkRole(StrEnum):
    """Role of a chunk in the versioning system."""

    BASE = "base"
    DELTA = "delta"
    VERSION_INDEX = "version_index"


class DocType(StrEnum):
    """Constrained document type classifications."""

    CONFIG_GUIDE = "config_guide"
    ADMIN_GUIDE = "admin_guide"
    RELEASE_NOTES = "release_notes"
    CHANGELOG = "changelog"
    QUICKSTART = "quickstart"
    API_REFERENCE = "api_reference"
    ARCHITECTURE_GUIDE = "architecture_guide"
    TROUBLESHOOTING = "troubleshooting"
    BOOK = "book"
    TUTORIAL = "tutorial"
    OTHER = "other"


@dataclass(frozen=True)
class RawDocument:
    """Output of an extractor — raw text with format-specific metadata."""

    text: str
    source_file: str
    format: str  # "pdf", "epub", "md", "txt", "video"
    pages: list[dict[str, Any]] | None = None
    native_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """A single chunk of text with positional metadata."""

    text: str
    chunk_index: int
    source_file: str
    page_number: int | None = None
    heading_path: str | None = None
    timestamp_start: float | None = None
    timestamp_end: float | None = None
    token_count: int = 0


@dataclass
class TagResult:
    """Output of the LLM family tagger."""

    family_slug: str
    doc_title: str
    vendor: str | None
    doc_type: str
    is_new_family: bool
    matched_existing: str | None
    confidence: float
    reasoning: str


@dataclass
class MetadataPayload:
    """Full Qdrant payload schema for a stored chunk."""

    # Document identity
    doc_title: str
    doc_type: str
    family_slug: str
    source_file: str
    format: str

    # Version info
    version: str | None
    version_tuple: VersionTuple | None
    version_range_min: VersionTuple | None
    version_range_max: VersionTuple | None
    chunk_role: ChunkRole
    base_chunk_id: str | None

    # Dates
    created_date: str | None
    ingested_date: str
    metadata_complete: bool

    # Position
    chunk_index: int
    page_number: int | None
    heading_path: str | None
    timestamp_start: float | None
    timestamp_end: float | None

    # Content
    text: str
    token_count: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dict suitable for Qdrant payload."""
        return {
            "doc_title": self.doc_title,
            "doc_type": self.doc_type,
            "family_slug": self.family_slug,
            "source_file": self.source_file,
            "format": self.format,
            "version": self.version,
            "version_tuple": list(self.version_tuple) if self.version_tuple else None,
            "version_range_min": list(self.version_range_min) if self.version_range_min else None,
            "version_range_max": list(self.version_range_max) if self.version_range_max else None,
            "chunk_role": self.chunk_role.value,
            "base_chunk_id": self.base_chunk_id,
            "created_date": self.created_date,
            "ingested_date": self.ingested_date,
            "metadata_complete": self.metadata_complete,
            "chunk_index": self.chunk_index,
            "page_number": self.page_number,
            "heading_path": self.heading_path,
            "timestamp_start": self.timestamp_start,
            "timestamp_end": self.timestamp_end,
            "text": self.text,
            "token_count": self.token_count,
        }


@dataclass
class ClassificationResult:
    """Output of the delta engine — what to store, update, or cap."""

    to_upsert: list[Chunk]
    to_upsert_roles: list[ChunkRole]
    to_upsert_base_ids: list[str | None]
    to_update_range: list[tuple[str, VersionTuple]]  # (chunk_id, new_max)
    to_cap: list[tuple[str, VersionTuple]]  # (chunk_id, cap_version)
    is_first_version: bool


@dataclass
class IngestResult:
    """Summary of a pipeline ingestion run."""

    docs_processed: int = 0
    docs_skipped: int = 0
    docs_failed: int = 0
    chunks_created: int = 0
    chunks_updated: int = 0
    tags_auto_accepted: int = 0
    tags_flagged: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)  # (filename, error_message)
