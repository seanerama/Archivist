"""Custom exception hierarchy for Archivist."""


class ArchivistError(Exception):
    """Base exception for all Archivist errors."""


class ConfigError(ArchivistError):
    """Invalid configuration: missing fields, bad YAML, type errors."""


class ExtractionError(ArchivistError):
    """Document extraction failure: unreadable file, unsupported format, corrupt data."""


class MetadataError(ArchivistError):
    """Metadata processing failure: tagger unreachable, parse error."""


class ChunkingError(ArchivistError):
    """Unexpected failure during document chunking."""


class EmbeddingError(ArchivistError):
    """Embedding failure: model load error, API error, dimension mismatch."""


class VersioningError(ArchivistError):
    """Version processing failure: parse error, delta computation error."""


class StorageError(ArchivistError):
    """Qdrant storage failure: connection error, upsert failure."""


class SetupError(ArchivistError):
    """Setup/provisioning failure: Docker unavailable, connection refused."""
