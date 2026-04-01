# Recovery Plan: Archivist

**Version**: 0.1.0
**Last Updated**: 2026-04-01

## Components

| Component | Type | Data at Risk | Recovery Method |
|-----------|------|-------------|-----------------|
| Qdrant | Vector database | All ingested chunks + version indices | Backup/restore or re-ingest |
| `.tag` sidecar files | Filesystem | Classification decisions | Preserved alongside source docs |
| `.archivist-cache/` | Filesystem | Whisper transcription cache | Regenerated automatically |
| `archivist.yaml` | Config file | Setup configuration | Recreate via `archivist setup` |

## Qdrant Backup & Recovery

### Docker (Local)

**Backup** (snapshot the Qdrant volume):
```bash
# Stop writes first
docker stop archivist-qdrant

# Backup the volume
docker run --rm -v archivist-qdrant-data:/data -v $(pwd)/backups:/backup \
  alpine tar czf /backup/qdrant-backup-$(date +%Y%m%d).tar.gz -C /data .

# Restart
docker start archivist-qdrant
```

**Restore**:
```bash
docker stop archivist-qdrant
docker run --rm -v archivist-qdrant-data:/data -v $(pwd)/backups:/backup \
  alpine sh -c "rm -rf /data/* && tar xzf /backup/qdrant-backup-YYYYMMDD.tar.gz -C /data"
docker start archivist-qdrant
```

**Schedule**: Back up before any bulk ingestion. Weekly for active use.

### Qdrant Cloud

Use Qdrant Cloud's built-in snapshot feature via the dashboard or API:
```bash
# Create snapshot
curl -X POST "https://your-cluster.cloud.qdrant.io/collections/archivist/snapshots" \
  -H "api-key: $QDRANT_API_KEY"
```

## Ingestion Recovery

### Scenario: Ingestion crashes mid-batch

Archivist handles this automatically:
1. Each document is ingested independently
2. Partially ingested documents are detected on re-run
3. `delete_partial_ingestion()` cleans up incomplete chunks
4. Re-running `archivist ingest ./docs` is safe (idempotent)

**Recovery steps**:
```bash
# Just re-run — Archivist skips completed docs, cleans up partials
archivist ingest ./docs
```

### Scenario: Qdrant collection corrupted

```bash
# Option 1: Re-ingest everything
# Delete the collection and re-ingest from source docs + sidecars
archivist setup --reset-collection  # (future feature)
archivist ingest ./docs

# Option 2: Restore from backup (see above)
```

### Scenario: Wrong classification applied

```bash
# Edit the .tag sidecar file to correct the classification
vim document.pdf.tag
# Change family_slug, doc_type, etc.

# Re-ingest with overwrite
archivist ingest document.pdf --overwrite-existing
```

## Health Checks

### Qdrant connectivity
```bash
# Quick check
archivist status
```

Expected output: collection name, total chunks, status "green".

### Embedding model
```bash
# Verify the model loads
python -c "from archivist.embedding import get_embedding_backend; from archivist.config import Config; b = get_embedding_backend(Config.load()); print(f'OK: dim={b.dimension}')"
```

### Ollama tagger (if using local)
```bash
ollama list  # Verify model is pulled
```

## Operational Runbooks

### Runbook: First-time setup
1. `uv sync` — install dependencies
2. `archivist setup` — configure Qdrant and backends
3. `archivist ingest ./docs --dry-run` — verify extraction works
4. `archivist ingest ./docs` — full ingestion
5. `archivist status` — verify chunks stored

### Runbook: Adding new documents
1. Place documents in your docs directory
2. `archivist ingest ./docs` — only new docs are processed
3. `archivist review` — check flagged metadata if any

### Runbook: Ingesting a new version of an existing document
1. Place the new version alongside the old one (e.g., `guide_v2.0.pdf`)
2. `archivist ingest guide_v2.0.pdf`
3. Delta engine automatically stores only changes
4. Version index is updated

### Runbook: Changing embedding model
Changing the embedding model requires re-ingesting everything (vector dimensions may differ):
1. Back up Qdrant (see above)
2. Update `archivist.yaml` with new model
3. Delete the Qdrant collection
4. `archivist ingest ./docs` — full re-ingestion

### Runbook: Disk space management
- Qdrant storage: check Docker volume usage (`docker system df -v`)
- Whisper cache: `du -sh .archivist-cache/` — safe to delete (regenerated on next ingest)
- `.tag` sidecars: small files, no cleanup needed

## Monitoring (Future)

For Phase 2 (MCP server), add:
- Qdrant collection size alerts (> 80% of allocated storage)
- Ingestion job success/failure tracking
- Query latency monitoring
- Embedding model health checks
