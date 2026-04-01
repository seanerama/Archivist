"""Pipeline test for Stage 7: Full end-to-end pipeline.

Tests:
1. Ingest a set of test documents (MD, TXT)
2. Verify all chunks stored with correct metadata
3. Re-run ingest — verify idempotency
4. Verify review queue catches missing metadata
5. Verify dry-run produces no writes
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from archivist.config import Config
from archivist.models import ChunkRole, TagResult


def _mock_pipeline(config: Config, tmp_path: Path) -> tuple:
    """Create a Pipeline with all external dependencies mocked."""
    from archivist.pipeline import Pipeline

    mock_storage = MagicMock()
    mock_storage.check_document_exists.return_value = False
    mock_storage.get_family_chunks.return_value = []

    mock_embedding = MagicMock()
    mock_embedding.dimension = 3
    mock_embedding.encode.return_value = np.array([[0.1, 0.2, 0.3]])

    mock_tagger = MagicMock()
    mock_tagger.tag.return_value = (
        TagResult(
            family_slug="test-doc", doc_title="Test Document", vendor=None,
            doc_type="other", is_new_family=True, matched_existing=None,
            confidence=0.8, reasoning="test",
        ),
        False,  # not auto-accepted
    )

    with patch.object(Pipeline, "__init__", lambda self, c: None):
        p = Pipeline.__new__(Pipeline)
        p._config = config
        p._storage = mock_storage
        p._embedding = mock_embedding

        from archivist.chunking import RecursiveChunker
        from archivist.metadata import ContentScanner, FilenameParser, ReviewQueue
        from archivist.versioning import DeltaEngine, VersionParser

        p._chunker = RecursiveChunker(config.pipeline)
        p._filename_parser = FilenameParser()
        p._content_scanner = ContentScanner()
        p._version_parser = VersionParser()
        p._delta_engine = DeltaEngine()
        p._review_queue = ReviewQueue()
        p._family_tagger = mock_tagger

    return p, mock_storage, mock_embedding


class TestPipelineE2E:
    """End-to-end pipeline tests with mocked storage and embedding."""

    @patch("archivist.pipeline.logger")
    def test_ingest_markdown_file(self, mock_logger: MagicMock, tmp_path: Path) -> None:
        """Ingest a single markdown file and verify chunks are stored."""
        doc = tmp_path / "test_v1.2.md"
        doc.write_text("# Test Document\n\nVersion 1.2\n\n## Section 1\n\nContent here.\n")

        config = Config.default()
        pipeline, mock_storage, mock_embedding = _mock_pipeline(config, tmp_path)

        result = pipeline.ingest([doc])

        assert result.docs_processed == 1
        assert result.docs_failed == 0
        assert result.chunks_created >= 1
        mock_storage.upsert_chunks.assert_called()

        # Verify payload structure
        call_args = mock_storage.upsert_chunks.call_args
        payloads = call_args[0][0]
        assert len(payloads) >= 1
        assert payloads[0].source_file == "test_v1.2.md"
        assert payloads[0].format == "md"
        assert payloads[0].chunk_role in (ChunkRole.BASE, ChunkRole.DELTA)

    @patch("archivist.pipeline.logger")
    def test_ingest_plaintext_file(self, mock_logger: MagicMock, tmp_path: Path) -> None:
        """Ingest a plaintext file."""
        doc = tmp_path / "notes.txt"
        doc.write_text("This is a plain text document.\nVersion: 2.0\nLast updated: 2024-01-15\n")

        config = Config.default()
        pipeline, mock_storage, _ = _mock_pipeline(config, tmp_path)

        result = pipeline.ingest([doc])

        assert result.docs_processed == 1
        assert result.docs_failed == 0

    @patch("archivist.pipeline.logger")
    def test_ingest_directory(self, mock_logger: MagicMock, tmp_path: Path) -> None:
        """Ingest a directory with multiple files."""
        (tmp_path / "doc1.md").write_text("# Doc 1\n\nContent.")
        (tmp_path / "doc2.txt").write_text("Doc 2 content.")
        (tmp_path / "ignore.xyz").write_text("Not supported.")

        config = Config.default()
        pipeline, mock_storage, _ = _mock_pipeline(config, tmp_path)

        result = pipeline.ingest([tmp_path])

        assert result.docs_processed == 2  # md + txt, not xyz
        assert result.docs_failed == 0

    @patch("archivist.pipeline.logger")
    def test_idempotency_skips_existing(self, mock_logger: MagicMock, tmp_path: Path) -> None:
        """Re-ingesting a document that already exists should skip it."""
        doc = tmp_path / "existing.md"
        doc.write_text("# Existing\n\nAlready ingested.")

        config = Config.default()
        pipeline, mock_storage, _ = _mock_pipeline(config, tmp_path)
        mock_storage.check_document_exists.return_value = True

        result = pipeline.ingest([doc])

        assert result.docs_skipped == 1
        assert result.docs_processed == 0
        mock_storage.upsert_chunks.assert_not_called()

    @patch("archivist.pipeline.logger")
    def test_dry_run_no_storage_writes(self, mock_logger: MagicMock, tmp_path: Path) -> None:
        """Dry run processes but doesn't write to storage."""
        doc = tmp_path / "test.md"
        doc.write_text("# Test\n\nContent.")

        config = Config.default()
        pipeline, mock_storage, _ = _mock_pipeline(config, tmp_path)

        result = pipeline.ingest([doc], dry_run=True)

        assert result.docs_processed == 1
        mock_storage.connect.assert_not_called()
        mock_storage.upsert_chunks.assert_not_called()

    @patch("archivist.pipeline.logger")
    def test_review_queue_catches_missing_metadata(self, mock_logger: MagicMock, tmp_path: Path) -> None:
        """Documents without version/date should be flagged."""
        doc = tmp_path / "no_metadata.md"
        doc.write_text("# Document\n\nNo version or date info here.\n")

        config = Config.default()
        pipeline, mock_storage, _ = _mock_pipeline(config, tmp_path)

        pipeline.ingest([doc])

        # Review queue should have items (missing version and/or date + tagger flag)
        assert pipeline._review_queue.count >= 1

    @patch("archivist.pipeline.logger")
    def test_error_in_one_doc_does_not_stop_batch(self, mock_logger: MagicMock, tmp_path: Path) -> None:
        """One document failing should not prevent others from processing."""
        good_doc = tmp_path / "good.md"
        good_doc.write_text("# Good\n\nContent.")

        bad_doc = tmp_path / "bad.md"
        # Create a file that will cause an extraction issue
        bad_doc.write_bytes(b"\x00\x01\x02")  # binary content in .md

        config = Config.default()
        pipeline, mock_storage, _ = _mock_pipeline(config, tmp_path)

        result = pipeline.ingest([bad_doc, good_doc])

        # good.md should still process even if bad.md has issues
        assert result.docs_processed >= 1

    @patch("archivist.pipeline.logger")
    def test_sidecar_file_written(self, mock_logger: MagicMock, tmp_path: Path) -> None:
        """After ingestion, a .tag sidecar should be written."""
        doc = tmp_path / "tagged.md"
        doc.write_text("# Tagged Doc\n\nContent.")

        config = Config.default()
        pipeline, mock_storage, _ = _mock_pipeline(config, tmp_path)

        pipeline.ingest([doc])

        sidecar = tmp_path / "tagged.md.tag"
        assert sidecar.exists()

        import json

        data = json.loads(sidecar.read_text())
        assert "family_slug" in data
        assert data["family_slug"] == "test-doc"

    @patch("archivist.pipeline.logger")
    def test_sidecar_reused_on_reingest(self, mock_logger: MagicMock, tmp_path: Path) -> None:
        """If a .tag sidecar exists, the tagger should be skipped."""
        doc = tmp_path / "with_sidecar.md"
        doc.write_text("# Sidecar Test\n\nContent.")

        # Pre-create sidecar
        import json

        sidecar = tmp_path / "with_sidecar.md.tag"
        sidecar.write_text(json.dumps({
            "source_file": "with_sidecar.md",
            "family_slug": "pre-tagged",
            "doc_title": "Pre-tagged Doc",
            "vendor": None,
            "doc_type": "other",
            "confidence": 1.0,
            "reasoning": "manual tag",
        }))

        config = Config.default()
        config.pipeline.overwrite_existing = True
        pipeline, mock_storage, _ = _mock_pipeline(config, tmp_path)

        pipeline.ingest([doc])

        # Tagger should NOT have been called (sidecar was used)
        pipeline._family_tagger.tag.assert_not_called()

        # Payload should use the pre-tagged family
        call_args = mock_storage.upsert_chunks.call_args
        payloads = call_args[0][0]
        assert payloads[0].family_slug == "pre-tagged"
