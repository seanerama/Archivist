# Archivist

Version-aware document ingestion and retrieval for technical documentation. Archivist takes your PDFs, EPUBs, markdown files, plaintext, and video/audio recordings and stores them in a Qdrant vector database with full version tracking. Every chunk knows its version, whether it's base content or a delta, and when it was written.

Search your corpus with version-aware filtering, expose it to Claude Code via MCP, and optionally extract and embed images from documents.

## Why Archivist?

Standard RAG pipelines treat every document as a flat text blob. They can't answer:

- "What changed between v1.24 and v1.26?"
- "Is this answer still valid for my version?"
- "Which documents cover Kubernetes 1.29 specifically?"

Archivist solves this by making **version metadata a first-class citizen**. It uses delta storage so only actual changes between versions are stored, and every chunk carries version range information for precise retrieval.

## Features

- **Multi-format extraction** -- PDF, EPUB, Markdown, plaintext, video/audio (via Whisper)
- **Version-aware storage** -- Delta model stores only what changed between versions
- **Version-aware search** -- Query the corpus with version filtering; only get chunks valid for your version
- **MCP server** -- Expose search to Claude Code (and other MCP clients) via stdio
- **Re-ranking** -- Optional cross-encoder or Voyage API re-ranking for better retrieval precision
- **Image extraction** -- Extract and embed images from PDFs/EPUBs using Voyage multimodal (opt-in)
- **LLM document classification** -- Automatically groups documents into families using local (Ollama) or API (Claude Haiku) models
- **Metadata review queue** -- Flags documents with missing version/date info for manual review
- **Hardware-agnostic** -- Local or API backends for both embeddings and tagging
- **Guided setup** -- Interactive wizard configures Qdrant (Docker, existing, or cloud)
- **Idempotent ingestion** -- Safe to re-run; skips already-ingested documents
- **Sidecar tag files** -- Classification decisions persist as `.tag` files alongside source documents

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- A running Qdrant instance (or Docker to spin one up)

### Install

```bash
git clone https://github.com/your-org/archivist.git
cd archivist
uv sync
```

### Setup

```bash
archivist setup
```

The wizard walks you through:
1. Connecting to Qdrant (or provisioning via Docker)
2. Choosing an embedding backend (local model or Voyage API)
3. Choosing a tagger backend (local Ollama or Anthropic API)

### Ingest

```bash
# Ingest a directory of documents
archivist ingest ./docs

# Ingest specific files
archivist ingest report.pdf guide.epub notes.md

# Preview without writing to the database
archivist ingest ./docs --dry-run
```

### Check status

```bash
archivist status
```

### Review flagged documents

```bash
archivist review
```

### Search

```bash
# Search the corpus
archivist search "How do I configure TLS?"

# Filter by version, family, or document type
archivist search "TLS setup" --version 1.24 --family nginx

# List all document families
archivist families

# Show what changed between versions
archivist diff nginx 1.22 1.24
```

### MCP Server (Claude Code integration)

Start the MCP server for use with Claude Code:

```bash
archivist mcp
```

Or add to your Claude Code MCP config (`.mcp.json` in your project root):

```json
{
  "mcpServers": {
    "archivist": {
      "command": "archivist",
      "args": ["mcp"]
    }
  }
}
```

This exposes three tools:
- `archivist_search` -- Search with optional version/family/doc_type filters
- `archivist_list_families` -- List all document families and versions
- `archivist_version_diff` -- Show changes between two versions

## Configuration

Archivist reads from `archivist.yaml` in the current directory. The setup wizard creates this file, or you can write it manually:

```yaml
qdrant:
  host: localhost
  port: 6333
  collection_name: archivist

embedding_backend:
  type: local                     # or "api"
  model_name: BAAI/bge-m3         # any HuggingFace model
  precision: fp16                 # fp32, fp16, q8
  device: auto                    # auto, cuda, cpu, mps

tagger_backend:
  type: local                     # or "api"
  provider: ollama
  model: qwen3:0.6b              # any Ollama model
  auto_accept_tags: false
  auto_accept_threshold: 0.90

pipeline:
  chunk_size: 512                 # tokens per chunk
  chunk_overlap_pct: 10

whisper:
  model: medium                   # tiny, base, small, medium, large-v3
  cache_dir: .archivist-cache

retrieval:
  top_k: 10
  reranker:
    enabled: false               # enable for better search precision
    type: local                  # "local" or "api"
    model: cross-encoder/ms-marco-MiniLM-L-6-v2

image_extraction:
  enabled: false                 # opt-in; requires Voyage API
  formats: [pdf, epub]
  min_size: 10000                # skip images smaller than 10KB
  embedding_model: voyage-multimodal-3.5

logging:
  level: INFO
```

### Environment variables

Any config value can be overridden with an `ARCHIVIST_` prefixed environment variable:

```bash
export ARCHIVIST_QDRANT_HOST=remote-server
export ARCHIVIST_LOGGING_LEVEL=DEBUG
```

API keys are stored in a `.env` file (auto-loaded, already in `.gitignore`):

```bash
# .env
VOYAGE_API_KEY=your-voyage-key
ANTHROPIC_API_KEY=your-anthropic-key
QDRANT_API_KEY=your-qdrant-cloud-key   # only if using Qdrant Cloud
```

Then reference them in `archivist.yaml` with the `env:` prefix:

```yaml
embedding_backend:
  type: api
  provider: voyage
  api_key: env:VOYAGE_API_KEY

tagger_backend:
  type: api
  provider: anthropic
  api_key: env:ANTHROPIC_API_KEY
```

No API keys are needed if using local backends (Ollama + sentence-transformers).

## Supported Formats

| Format | Extension | Extractor | Notes |
|--------|-----------|-----------|-------|
| PDF | `.pdf` | pymupdf4llm | Clean markdown output with page tracking |
| EPUB | `.epub` | ebooklib | Chapter structure + OPF metadata |
| Markdown | `.md` | native | Heading-aware chunking |
| Plaintext | `.txt` | native | Direct read |
| Video/Audio | `.mp4`, `.mov`, `.mp3`, `.wav` | faster-whisper | Transcription with caching |

## How Versioning Works

When you ingest a new version of an existing document, Archivist compares each incoming chunk against existing chunks using text similarity:

- **Unchanged** (>95% similar): The existing chunk's version range is extended. No new storage.
- **Modified** (50-95% similar): Stored as a `delta` chunk linked to the base.
- **New** (<50% similar or no match): Stored as a new `delta` chunk.
- **Removed** (in old version but not new): Existing chunk's version range is capped.

This means querying for a specific version returns only the chunks valid for that version, and storage grows only with actual differences.

## How Document Classification Works

Archivist uses an LLM to classify documents into families (e.g., "nginx admin guide", "kubernetes release notes"). The tagger:

1. Receives the filename and first ~1500 tokens of content
2. Compares against existing family slugs in the database
3. Returns a classification with confidence score

High-confidence matches can be auto-accepted. Low-confidence or new families are flagged for review. All decisions are persisted in `.tag` sidecar files so re-ingestion skips the tagger entirely.

## Python API

```python
from pathlib import Path
from archivist.config import Config
from archivist.pipeline import Pipeline

config = Config.load()

# Ingest documents
pipeline = Pipeline(config)
result = pipeline.ingest([Path("./docs")])
print(f"Processed: {result.docs_processed}")
print(f"Chunks created: {result.chunks_created}")

# Search the corpus
from archivist.retrieval import Retriever

retriever = Retriever(config)
results = retriever.search("TLS configuration", version="1.24", family="nginx")
for r in results:
    print(f"{r.doc_title} (v{r.version}, score={r.score:.3f})")
    print(f"  {r.text[:100]}...")
```

## Development

```bash
# Install with dev dependencies
uv sync

# Run tests
uv run pytest tests/ -v

# Lint
uv run ruff check archivist/ tests/

# Type check
uv run mypy archivist/
```

## Architecture

```
archivist/
├── config.py              # Configuration with YAML + env var support
├── pipeline.py            # Ingestion orchestrator
├── cli.py                 # Typer CLI (ingest, search, mcp, families, diff, status, setup)
├── models.py              # Shared data models
├── exceptions.py          # Error hierarchy
├── extractors/            # PDF, EPUB, MD, TXT, video extractors
├── metadata/              # Filename parser, content scanner, family tagger, sidecars
├── chunking/              # Recursive text splitter with format overrides
├── versioning/            # Version parser, delta engine, version index
├── embedding/             # Local (sentence-transformers) + API (Voyage) backends
├── tagger_backends/       # Local (Ollama) + API (Anthropic) backends
├── storage/               # Qdrant client + setup wizard
├── retrieval/             # Search engine with version filtering + re-ranking
├── mcp/                   # MCP server for Claude Code integration
└── image/                 # Image extraction + multimodal embedding (opt-in)
```

## License

MIT
