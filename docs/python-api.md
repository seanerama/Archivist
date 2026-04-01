# Python API Reference

Archivist can be used as a library in addition to the CLI. All pipeline logic lives in the `archivist` package with no CLI dependencies.

## Configuration

```python
from pathlib import Path
from archivist.config import Config

# Load from archivist.yaml (auto-detects in current directory)
config = Config.load()

# Load from a specific path
config = Config.load(Path("/path/to/archivist.yaml"))

# Programmatic configuration
config = Config.default()
config.qdrant.host = "my-qdrant-server"
config.qdrant.port = 6333
config.embedding.type = "local"
config.embedding.model_name = "BAAI/bge-m3"
config.embedding.precision = "fp16"
```

### Config sections

| Section | Class | Key Fields |
|---------|-------|------------|
| `config.qdrant` | `QdrantConfig` | `host`, `port`, `collection_name`, `api_key` |
| `config.embedding` | `EmbeddingConfig` | `type`, `model_name`, `precision`, `device`, `batch_size` |
| `config.tagger` | `TaggerConfig` | `type`, `provider`, `model`, `auto_accept_tags`, `auto_accept_threshold` |
| `config.pipeline` | `PipelineConfig` | `chunk_size`, `chunk_overlap_pct`, `dry_run`, `overwrite_existing` |
| `config.whisper` | `WhisperConfig` | `model`, `cache_dir` |
| `config.logging` | `LoggingConfig` | `level` |

## Pipeline

```python
from pathlib import Path
from archivist.config import Config
from archivist.pipeline import Pipeline

config = Config.load()
pipeline = Pipeline(config)

# Ingest files or directories
result = pipeline.ingest([Path("./docs"), Path("specific-file.pdf")])

# Dry run (process without writing to Qdrant)
result = pipeline.ingest([Path("./docs")], dry_run=True)
```

### IngestResult

```python
result.docs_processed    # int — successfully ingested
result.docs_skipped      # int — already in database
result.docs_failed       # int — failed with errors
result.chunks_created    # int — new chunks stored
result.chunks_updated    # int — existing chunks with extended version range
result.tags_auto_accepted  # int — classification auto-accepted
result.tags_flagged      # int — classification needs review
result.errors            # list[tuple[str, str]] — (filename, error_message)
```

## Extractors

Use extractors directly for custom workflows:

```python
from pathlib import Path
from archivist.extractors import get_extractor
from archivist.config import Config

# Get the right extractor for a file
extractor = get_extractor(Path("document.pdf"))
raw_doc = extractor.extract(Path("document.pdf"))

print(raw_doc.text)             # Extracted text
print(raw_doc.format)           # "pdf"
print(raw_doc.source_file)      # "document.pdf"
print(raw_doc.native_metadata)  # {"author": "...", "created_date": "..."}
print(raw_doc.pages)            # [{"page_number": 1, "start_offset": 0, ...}]

# Video extraction requires config (for Whisper settings)
config = Config.load()
extractor = get_extractor(Path("lecture.mp4"), config=config)
raw_doc = extractor.extract(Path("lecture.mp4"))
print(raw_doc.native_metadata["segments"])  # Whisper segments with timestamps
```

## Chunking

```python
from archivist.chunking import RecursiveChunker
from archivist.config import PipelineConfig

chunker = RecursiveChunker(PipelineConfig(chunk_size=512, chunk_overlap_pct=10))
chunks = chunker.chunk(raw_doc)

for chunk in chunks:
    print(chunk.text)
    print(chunk.chunk_index)
    print(chunk.heading_path)      # For markdown: "Section > Subsection"
    print(chunk.page_number)       # For PDFs
    print(chunk.timestamp_start)   # For video
```

## Metadata

```python
from archivist.metadata import FilenameParser, ContentScanner, SidecarIO

# Extract hints from filename
parser = FilenameParser()
hints = parser.parse("nginx_1.24_admin_guide.pdf")
# {"version": "1.24", "date": None, "doc_type": "admin_guide"}

# Scan document content
scanner = ContentScanner()
meta = scanner.scan(raw_doc.text)
# {"version": "1.24.3", "date": "2023-06-15", "extra": {"revision": "3"}}

# Read/write sidecar tag files
from pathlib import Path
tag = SidecarIO.read(Path("document.pdf"))  # Returns TagResult or None
```

## Versioning

```python
from archivist.versioning import VersionParser, DeltaEngine

# Parse version strings
parser = VersionParser()
v = parser.parse("1.24.3")  # (1, 24, 3)
v = parser.parse("2024.04") # (2024, 4, 0)
v = parser.parse("N/A")     # None

# Compare versions
parser.compare((1, 24, 0), (1, 26, 0))  # negative (1.24 < 1.26)

# Check if version is in range
parser.in_range((1, 24, 0), min=(1, 20, 0), max=(1, 26, 0))  # True

# Delta classification
engine = DeltaEngine()
result = engine.classify_chunks(incoming_chunks, existing_chunks, version=(2, 0, 0))
# result.to_upsert — new/modified chunks to store
# result.to_update_range — unchanged chunk IDs to extend
# result.to_cap — removed chunk IDs to cap
```

## Embedding Backends

```python
from archivist.embedding import get_embedding_backend
from archivist.config import Config

config = Config.load()
backend = get_embedding_backend(config)

vectors = backend.encode(["text to embed", "another text"])
# numpy array of shape (2, dimension)

print(backend.dimension)  # e.g. 768, 1024
```

## Storage

```python
from archivist.storage import QdrantStorage
from archivist.config import Config

config = Config.load()
storage = QdrantStorage(config)
storage.connect(vector_dimension=768)

# Check if document already ingested
exists = storage.check_document_exists("nginx_1.24.pdf")

# Get corpus stats
stats = storage.collection_stats()
```

## Data Models

All shared types are in `archivist.models`:

- `RawDocument` — output of extractors (text, source_file, format, pages, native_metadata)
- `Chunk` — a text chunk with positional metadata
- `TagResult` — output of the LLM family tagger
- `MetadataPayload` — full Qdrant payload schema
- `ClassificationResult` — output of the delta engine
- `IngestResult` — summary of a pipeline run
- `ChunkRole` — enum: `base`, `delta`, `version_index`
- `DocType` — enum of document type classifications
- `VersionTuple` — `tuple[int, int, int]`

## Exceptions

All exceptions subclass `ArchivistError`:

```python
from archivist.exceptions import (
    ArchivistError,      # base
    ConfigError,         # invalid configuration
    ExtractionError,     # document extraction failure
    MetadataError,       # metadata/tagger failure
    ChunkingError,       # chunking failure
    EmbeddingError,      # embedding failure
    VersioningError,     # version processing failure
    StorageError,        # Qdrant operation failure
    SetupError,          # provisioning failure
)
```
