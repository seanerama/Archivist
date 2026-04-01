"""Pipeline orchestrator — coordinates the full ingestion flow."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from archivist.chunking import RecursiveChunker
from archivist.config import Config
from archivist.embedding import get_embedding_backend
from archivist.exceptions import ArchivistError
from archivist.extractors import get_extractor
from archivist.log import get_logger
from archivist.metadata import ContentScanner, FilenameParser, ReviewItem, ReviewQueue, SidecarIO
from archivist.metadata.family_tagger import FamilyTagger
from archivist.models import ChunkRole, IngestResult, MetadataPayload
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

        for i, file_path in enumerate(files, 1):
            console.print(f"[bold]\\[{i}/{len(files)}] {file_path.name}[/bold]")
            try:
                self._ingest_one(file_path, result, dry_run, console=console)
            except ArchivistError as e:
                console.print(f"  [red]Error: {e}[/red]")
                result.docs_failed += 1
                result.errors.append((file_path.name, str(e)))
            except Exception as e:
                console.print(f"  [red]Unexpected error: {e}[/red]")
                result.docs_failed += 1
                result.errors.append((file_path.name, f"Unexpected: {e}"))

        # Print review queue summary
        self._review_queue.render_summary()

        return result

    def _ingest_one(
        self, file_path: Path, result: IngestResult, dry_run: bool, console: Console | None = None
    ) -> None:
        """Ingest a single document."""
        if console is None:
            console = Console()
        name = file_path.name
        logger.info("Processing", file=name)

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
        console.print(f"  Extracting {name}...")
        extractor = get_extractor(file_path, config=self._config)
        raw_doc = extractor.extract(file_path)
        console.print(f"  Extracted {len(raw_doc.text):,} chars ({raw_doc.format})")

        # Step 4: Filename metadata
        filename_meta = self._filename_parser.parse(file_path.name)

        # Step 5: Content metadata
        content_meta = self._content_scanner.scan(raw_doc.text)

        # Step 6: Family tagging
        if existing_tag:
            tag_result = existing_tag
            console.print(f"  Family: {tag_result.family_slug} (from sidecar)")
        else:
            console.print("  Classifying document...")
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

            console.print(
                f"  Family: {tag_result.family_slug} "
                f"(confidence: {tag_result.confidence:.0%}{', auto-accepted' if auto_accepted else ', needs review'})"
            )
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
        console.print("  Chunking...")
        chunks = self._chunker.chunk(raw_doc)
        if not chunks:
            console.print("  [yellow]No chunks produced[/yellow]")
            return
        console.print(f"  {len(chunks)} chunks ({sum(c.token_count for c in chunks):,} tokens)")

        # Step 8.5: Image extraction (if enabled)
        extracted_images = []
        if self._config.image.enabled and raw_doc.format in self._config.image.formats:
            from archivist.image import ImageExtractor

            img_extractor = ImageExtractor(self._config.image)
            extracted_images = img_extractor.extract_images(file_path, raw_doc)
            if extracted_images:
                console.print(f"  Extracted {len(extracted_images)} images")

        if dry_run:
            console.print("  [dim]Dry run — skipping storage[/dim]")
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
            console.print(f"  Embedding {len(texts)} chunks...")

            import numpy as np

            embed_batch_size = 32
            all_vectors: list[np.ndarray] = []
            with Progress(
                TextColumn("  "),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as embed_progress:
                embed_task = embed_progress.add_task("Embedding", total=len(texts))
                for i in range(0, len(texts), embed_batch_size):
                    batch = texts[i : i + embed_batch_size]
                    batch_vectors = self._embedding.encode(batch)
                    all_vectors.append(batch_vectors)
                    embed_progress.advance(embed_task, len(batch))

            vectors = np.vstack(all_vectors)

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
            console.print(f"  Storing {len(payloads)} chunks in Qdrant...")
            self._storage.upsert_chunks(payloads, vectors)
            result.chunks_created += len(payloads)
            console.print("  [green]Done![/green]")

        # Update version ranges for unchanged chunks
        if classification.to_update_range:
            for chunk_id, new_max in classification.to_update_range:
                self._storage.update_version_range([chunk_id], new_max)
            result.chunks_updated += len(classification.to_update_range)

        # Cap removed chunks
        for chunk_id, cap_version in classification.to_cap:
            self._storage.update_version_range([chunk_id], cap_version)

        # Step 11.5: Embed and upsert images
        if extracted_images:
            from archivist.image import MultimodalEmbedder

            console.print(f"  Embedding {len(extracted_images)} images...")
            img_embedder = MultimodalEmbedder(self._config.image)
            img_vectors = img_embedder.encode_images(extracted_images)

            now_img = datetime.now(UTC).isoformat()
            img_payloads = []
            for img in extracted_images:
                text = img.caption or f"Image from page {img.page_number} of {img.source_file}"
                img_payloads.append(MetadataPayload(
                    doc_title=tag_result.doc_title,
                    doc_type=tag_result.doc_type,
                    family_slug=tag_result.family_slug,
                    source_file=file_path.name,
                    format="image",
                    version=version,
                    version_tuple=version_tuple,
                    version_range_min=version_tuple,
                    version_range_max=None,
                    chunk_role=ChunkRole.BASE,
                    base_chunk_id=None,
                    created_date=created_date,
                    ingested_date=now_img,
                    metadata_complete=metadata_complete,
                    chunk_index=img.image_index,
                    page_number=img.page_number,
                    heading_path=None,
                    timestamp_start=None,
                    timestamp_end=None,
                    text=text,
                    token_count=0,
                    image_width=img.width,
                    image_height=img.height,
                    image_index=img.image_index,
                ))

            console.print(f"  Storing {len(img_payloads)} image embeddings in Qdrant...")
            self._storage.upsert_chunks(img_payloads, img_vectors)
            result.chunks_created += len(img_payloads)
            console.print("  [green]Images stored![/green]")

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
