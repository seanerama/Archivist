# Archivist Phase 2: MCP Server + Retrieval

## Overview

Phase 2 adds a retrieval layer and MCP server on top of the Phase 1 ingestion library. This gives Claude Code (and other MCP clients) precise, version-aware retrieval over the ingested corpus.

## What Phase 1 Delivers (Complete)

- Multi-format document extraction (PDF, EPUB, MD, TXT, video)
- Version-aware delta storage in Qdrant
- LLM document classification with family tagging
- CLI: `archivist ingest`, `archivist status`, `archivist review`, `archivist setup`
- 206 passing tests

## Phase 2 Goals

1. **Query/retrieval interface** — search the corpus with version filtering
2. **MCP server** — expose retrieval as MCP tools for Claude Code
3. **Re-ranking** — improve retrieval quality with a re-ranker
4. **Image extraction + multimodal embeddings** — embed diagrams and charts alongside text

---

## Module 1: Retrieval Engine

New package: `archivist/retrieval/`

### Query Interface

```python
from archivist.retrieval import Retriever

retriever = Retriever(config)
results = retriever.search(
    query="How do I configure TLS in nginx?",
    version="1.24",           # optional: filter to specific version
    family="nginx",           # optional: filter to family
    doc_type="admin_guide",   # optional: filter to doc type
    top_k=10,                 # number of results
)
```

### What `search()` does

1. Embed the query using the same embedding backend (with `input_type="query"` for Voyage)
2. Search Qdrant with vector similarity + payload filters (version range, family, doc_type)
3. Reconstruct version-correct results: base chunks + relevant deltas for the target version
4. Return ranked results with source metadata

### Result Object

```python
@dataclass
class SearchResult:
    text: str
    score: float
    source_file: str
    family_slug: str
    doc_title: str
    doc_type: str
    version: str | None
    page_number: int | None
    heading_path: str | None
    chunk_role: str  # base or delta
```

### Version-Aware Query Logic

- If `version` specified: filter to chunks where `version_range_min <= version <= version_range_max`
- If no version: return latest version chunks (use version index to determine latest)
- Support version range queries: "what changed between 1.22 and 1.24?"
  - Return delta chunks created between those versions
  - Return chunks that were capped (removed) in that range

---

## Module 2: Re-ranking

New file: `archivist/retrieval/reranker.py`

After initial vector search returns top_k candidates, re-rank using a cross-encoder model for better precision.

### Options

- **Local**: `cross-encoder/ms-marco-MiniLM-L-6-v2` via sentence-transformers
- **API**: Voyage rerank API (`voyageai.Client.rerank()`)
- **None**: skip re-ranking (default for Phase 2 MVP)

### Config

```yaml
retrieval:
  top_k: 10
  reranker:
    enabled: false
    type: local  # or api
    model: cross-encoder/ms-marco-MiniLM-L-6-v2
```

---

## Module 3: MCP Server

New package: `archivist/mcp/`

Expose retrieval as MCP tools that Claude Code can call.

### Tools to Expose

#### `archivist_search`

Search the document corpus.

```json
{
  "name": "archivist_search",
  "description": "Search the technical documentation corpus. Returns relevant chunks with source metadata.",
  "parameters": {
    "query": "string — the search query",
    "version": "string | null — filter to a specific version (e.g. '1.24')",
    "family": "string | null — filter to a document family (e.g. 'nginx')",
    "doc_type": "string | null — filter to doc type (e.g. 'admin_guide')",
    "top_k": "int — number of results (default 5)"
  }
}
```

#### `archivist_list_families`

List all document families in the corpus.

```json
{
  "name": "archivist_list_families",
  "description": "List all document families and their versions in the corpus.",
  "parameters": {}
}
```

#### `archivist_version_diff`

Show what changed between two versions of a document.

```json
{
  "name": "archivist_version_diff",
  "description": "Show chunks that changed between two versions of a document family.",
  "parameters": {
    "family": "string — the document family slug",
    "from_version": "string — the older version",
    "to_version": "string — the newer version"
  }
}
```

### MCP Server Implementation

Use the `mcp` Python SDK:

```python
from mcp.server import Server
from mcp.types import Tool

server = Server("archivist")

@server.tool()
async def archivist_search(query: str, version: str | None = None, ...) -> str:
    retriever = Retriever(config)
    results = retriever.search(query, version=version, ...)
    return format_results(results)
```

### Running the MCP Server

```bash
# stdio mode (for Claude Code)
archivist mcp

# Or configure in Claude Code's MCP settings:
# ~/.claude/claude_desktop_config.json
{
  "mcpServers": {
    "archivist": {
      "command": "archivist",
      "args": ["mcp"],
      "env": {}
    }
  }
}
```

---

## Module 4: Image Extraction + Multimodal Embeddings

### Image Extraction

Extend PDF and EPUB extractors to pull out images:
- `pymupdf` can extract images with `page.get_images()`
- Store images in `.archivist-cache/images/` with deterministic naming
- Each image becomes an `ImageChunk` with source page/position metadata

### Multimodal Embedding

Use `voyage-multimodal-3.5` to embed images into the same vector space as text:
- Images sent as base64 to the Voyage multimodal endpoint
- Stored in the same Qdrant collection with `format: "image"` payload field
- Searchable alongside text — a query for "network topology" returns both text descriptions and topology diagrams

### Config Addition

```yaml
image_extraction:
  enabled: false          # opt-in
  formats: [pdf, epub]
  min_size: 10000         # skip tiny icons (bytes)
  embedding_model: voyage-multimodal-3.5
```

---

## Module 5: FastAPI Layer (Optional)

If a REST API is needed beyond MCP:

```
archivist/api/
├── __init__.py
├── app.py          # FastAPI app
├── routes.py       # /search, /families, /diff endpoints
└── models.py       # Pydantic request/response models
```

This imports the same `Retriever` as the MCP server — no code duplication.

```bash
archivist serve --port 8000
```

---

## New CLI Commands

| Command | Description |
|---------|-------------|
| `archivist search "query"` | CLI search (prints results to terminal) |
| `archivist mcp` | Start MCP server (stdio mode) |
| `archivist serve` | Start FastAPI server (optional) |
| `archivist families` | List all families and versions |
| `archivist diff nginx 1.22 1.24` | Show version diff |

---

## New Dependencies

| Package | Purpose |
|---------|---------|
| `mcp` | MCP server SDK |
| `voyageai` (already installed) | Rerank API + multimodal embeddings |
| `fastapi` + `uvicorn` | REST API (optional) |

---

## Suggested Build Order

1. **Retrieval engine** — search + version filtering (most critical)
2. **CLI search command** — test retrieval interactively
3. **MCP server** — expose to Claude Code
4. **Re-ranking** — improve quality
5. **Image extraction** — multimodal support
6. **FastAPI** — REST API if needed

---

## What Phase 2 Does NOT Include

- Web UI (Phase 3)
- Authentication / multi-user (Phase 3)
- Contextual retrieval / chunk context enrichment (evaluate after MCP is working)
- Hybrid search (keyword + vector) — evaluate if vector-only is sufficient first
