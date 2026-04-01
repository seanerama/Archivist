"""Pipeline orchestrator — coordinates the full ingestion flow."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from archivist.chunking import RecursiveChunker
from archivist.config import Config
from archivist.embedding import get_embedding_backend
from archivist.exceptions import ArchivistError
from archivist.extractors import get_extractor
from archivist.log import get_logger
from archivist.metadata import ContentScanner, FilenameParser, ReviewItem, ReviewQueue, SidecarIO
from archivist.metadata.family_tagger import FamilyTagger
from archivist.models import IngestResult, MetadataPayload
from archivist.storage import QdrantStorage
from archivist.tagger_backends import get_tagger_backend
from archivist.versioning import DeltaEngine, VersionIndex, VersionParser

logger = get_logger("pipeline")


class Pipeline:
    """Orchestrates the full document ingestion pipeline."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._storage = QdrantStorage(config)
        self._embedding = get_embedding_backend(config)
        self._chunker = RecursiveChunker(config.pipeline)
        self._filename_parser = FilenameParser()
        self._content_scanner = ContentScanner()
        self._version_parser = VersionParser()
        self._delta_engine = DeltaEngine()
        self._review_queue = ReviewQueue()

        # Tagger setup
        tagger_backend = get_tagger_backend(config)
        self._family_tagger = FamilyTagger(config.tagger, tagger_backend)

    def ingest(self, paths: list[Path], dry_run: bool = False) -> IngestResult:
        """Ingest documents from the given paths.

        Args:
            paths: List of file or directory paths to ingest.
            dry_run: If True, process but don't write to Qdrant.

        Returns:
            IngestResult summary.
        """
        result = IngestResult()

        console = Console()

        # Connect to storage
        if not dry_run:
            console.print("Connecting to Qdrant...")
            self._storage.connect(self._embedding.dimension)

        # Expand directories
        files = self._expand_paths(paths)
        console.print(f"Found {len(files)} document(s) to process.\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Ingesting...", total=len(files))

            for file_path in files:
                progress.update(task, description=f"Processing {file_path.name}...")
                try:
                    self._ingest_one(file_path, result, dry_run)
                except ArchivistError as e:
                    logger.error("Document failed", file=file_path.name, error=str(e))
                    result.docs_failed += 1
                    result.errors.append((file_path.name, str(e)))
                except Exception as e:
                    logger.error("Unexpected error", file=file_path.name, error=str(e))
                    result.docs_failed += 1
                    result.errors.append((file_path.name, f"Unexpected: {e}"))
                progress.advance(task)

        # Print review queue summary
        self._review_queue.render_summary()

        return result

    def _ingest_one(self, file_path: Path, result: IngestResult, dry_run: bool) -> None:
        """Ingest a single document."""
        logger.info("Processing", file=file_path.name)

        # Step 1: Check sidecar for existing tag
        existing_tag = SidecarIO.read(file_path)

        # Step 2: Check if already ingested
        if (
            not dry_run
            and not self._config.pipeline.overwrite_existing
            and self._storage.check_document_exists(file_path.name)
        ):
                logger.info("Skipping (already ingested)", file=file_path.name)
                result.docs_skipped += 1
                return

        # Step 3: Extract
        extractor = get_extractor(file_path, config=self._config)
        raw_doc = extractor.extract(file_path)

        # Step 4: Filename metadata
        filename_meta = self._filename_parser.parse(file_path.name)

        # Step 5: Content metadata
        content_meta = self._content_scanner.scan(raw_doc.text)

        # Step 6: Family tagging
        if existing_tag:
            tag_result = existing_tag
            logger.info("Using existing sidecar tag", family=tag_result.family_slug)
        else:
            tag_result, auto_accepted = self._family_tagger.tag(
                file_path, raw_doc.text[:3000], [],
                filename_hint=filename_meta.get("doc_type"),
            )
            if auto_accepted:
                result.tags_auto_accepted += 1
            else:
                result.tags_flagged += 1
                self._review_queue.add(ReviewItem(
                    source_file=file_path.name,
                    missing=[],
                    suggested_family=tag_result.family_slug,
                    confidence=tag_result.confidence,
                    reason="tagger flagged for review",
                ))

            # Write sidecar
            SidecarIO.write(file_path, tag_result, tagger_model=self._config.tagger.model)

        # Step 7: Resolve metadata (priority: sidecar > tagger > content > filename)
        version = (
            filename_meta.get("version")
            or content_meta.get("version")
        )
        created_date = (
            filename_meta.get("date")
            or content_meta.get("date")
        )
        version_tuple = self._version_parser.parse(version)

        # Check metadata completeness
        metadata_complete = version is not None and created_date is not None
        if not metadata_complete:
            missing = []
            if version is None:
                missing.append("version")
            if created_date is None:
                missing.append("created_date")
            self._review_queue.add(ReviewItem(
                source_file=file_path.name,
                missing=missing,
                detected_version=version,
                detected_date=created_date,
            ))

        # Step 8: Chunk
        chunks = self._chunker.chunk(raw_doc)
        if not chunks:
            logger.warning("No chunks produced", file=file_path.name)
            return

        if dry_run:
            logger.info("Dry run complete", file=file_path.name, chunks=len(chunks))
            result.docs_processed += 1
            return

        # Step 9: Versioning
        existing_chunks = self._storage.get_family_chunks(
            tag_result.family_slug, tag_result.doc_type, version
        )
        classification = self._delta_engine.classify_chunks(
            chunks, existing_chunks, version_tuple or (0, 0, 0)
        )

        # Step 10: Embed new/delta chunks
        if classification.to_upsert:
            texts = [c.text for c in classification.to_upsert]
            vectors = self._embedding.encode(texts)

            # Build payloads
            now = datetime.now(UTC).isoformat()
            payloads = []
            for chunk, role, base_id in zip(
                classification.to_upsert,
                classification.to_upsert_roles,
                classification.to_upsert_base_ids,
                strict=True,
            ):
                payloads.append(MetadataPayload(
                    doc_title=tag_result.doc_title,
                    doc_type=tag_result.doc_type,
                    family_slug=tag_result.family_slug,
                    source_file=file_path.name,
                    format=raw_doc.format,
                    version=version,
                    version_tuple=version_tuple,
                    version_range_min=version_tuple,
                    version_range_max=None,
                    chunk_role=role,
                    base_chunk_id=base_id,
                    created_date=created_date,
                    ingested_date=now,
                    metadata_complete=metadata_complete,
                    chunk_index=chunk.chunk_index,
                    page_number=chunk.page_number,
                    heading_path=chunk.heading_path,
                    timestamp_start=chunk.timestamp_start,
                    timestamp_end=chunk.timestamp_end,
                    text=chunk.text,
                    token_count=chunk.token_count,
                ))

            # Step 11: Upsert to Qdrant
            self._storage.upsert_chunks(payloads, vectors)
            result.chunks_created += len(payloads)

        # Update version ranges for unchanged chunks
        if classification.to_update_range:
            for chunk_id, new_max in classification.to_update_range:
                self._storage.update_version_range([chunk_id], new_max)
            result.chunks_updated += len(classification.to_update_range)

        # Cap removed chunks
        for chunk_id, cap_version in classification.to_cap:
            self._storage.update_version_range([chunk_id], cap_version)

        # Step 12: Update version index
        if version:
            VersionIndex.update_index(
                self._storage, tag_result.family_slug, tag_result.doc_type, version
            )

        result.docs_processed += 1
        logger.info(
            "Ingested",
            file=file_path.name,
            chunks_created=len(classification.to_upsert),
            chunks_unchanged=len(classification.to_update_range),
        )

    def _expand_paths(self, paths: list[Path]) -> list[Path]:
        """Expand directories into individual files."""
        files: list[Path] = []
        supported = {".pdf", ".epub", ".md", ".markdown", ".txt", ".text",
                     ".mp4", ".mov", ".mva", ".mp3", ".wav", ".m4a", ".webm"}

        for path in paths:
            if path.is_file():
                if path.suffix.lower() in supported:
                    files.append(path)
            elif path.is_dir():
                for child in sorted(path.rglob("*")):
                    if child.is_file() and child.suffix.lower() in supported:
                        files.append(child)

        return files
