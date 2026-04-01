"""Tests for the pipeline orchestrator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from archivist.config import Config
from archivist.models import Chunk, TagResult


class TestPipelineExpandPaths:
    """Tests for path expansion."""

    def test_expands_directory(self, tmp_path: Path) -> None:
        (tmp_path / "doc.md").write_text("# Hello")
        (tmp_path / "doc.txt").write_text("Hello")
        (tmp_path / "doc.xyz").write_text("unsupported")

        from archivist.pipeline import Pipeline

        with patch.object(Pipeline, "__init__", lambda self, config: None):
            p = Pipeline.__new__(Pipeline)
            files = p._expand_paths([tmp_path])

        assert len(files) == 2
        names = {f.name for f in files}
        assert "doc.md" in names
        assert "doc.txt" in names
        assert "doc.xyz" not in names

    def test_single_file(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.md"
        f.write_text("content")

        from archivist.pipeline import Pipeline

        with patch.object(Pipeline, "__init__", lambda self, config: None):
            p = Pipeline.__new__(Pipeline)
            files = p._expand_paths([f])

        assert len(files) == 1


class TestPipelineDryRun:
    """Tests for dry run mode."""

    @patch("archivist.pipeline.logger")
    def test_dry_run_does_not_connect_storage(self, mock_logger: MagicMock, tmp_path: Path) -> None:
        doc_path = tmp_path / "test.md"
        doc_path.write_text("# Test\n\nContent here.")

        config = Config.default()

        mock_storage = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.dimension = 3
        mock_embedding.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        mock_tagger_backend = MagicMock()
        mock_tagger_backend.classify.return_value = TagResult(
            family_slug="test", doc_title="Test", vendor=None,
            doc_type="other", is_new_family=True, matched_existing=None,
            confidence=0.5, reasoning="test",
        )

        from archivist.pipeline import Pipeline

        with patch.object(Pipeline, "__init__", lambda self, c: None):
            p = Pipeline.__new__(Pipeline)
            p._config = config
            p._storage = mock_storage
            p._embedding = mock_embedding
            p._chunker = MagicMock()
            p._chunker.chunk.return_value = [
                Chunk(text="test chunk", chunk_index=0, source_file="test.md", token_count=2)
            ]
            p._filename_parser = MagicMock()
            p._filename_parser.parse.return_value = {"version": None, "date": None, "doc_type": None}
            p._content_scanner = MagicMock()
            p._content_scanner.scan.return_value = {"version": None, "date": None, "extra": {}}
            p._version_parser = MagicMock()
            p._version_parser.parse.return_value = None
            p._delta_engine = MagicMock()
            p._review_queue = MagicMock()
            from archivist.metadata.family_tagger import FamilyTagger
            p._family_tagger = MagicMock(spec=FamilyTagger)
            p._family_tagger.tag.return_value = (
                TagResult(family_slug="test", doc_title="Test", vendor=None,
                          doc_type="other", is_new_family=True, matched_existing=None,
                          confidence=0.5, reasoning="test"),
                False,
            )

            result = p.ingest([doc_path], dry_run=True)

        assert result.docs_processed == 1
        mock_storage.connect.assert_not_called()
        mock_storage.upsert_chunks.assert_not_called()
