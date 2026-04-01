"""Pipeline test for Stage 6: Versioning Engine.

Tests the full versioning flow:
1. Ingest v1 of a document → all chunks stored as base
2. Ingest v2 with modifications → unchanged extended, modified stored as delta
3. Verify version index tracks both versions
"""

from __future__ import annotations

from unittest.mock import MagicMock

from archivist.models import Chunk, ChunkRole
from archivist.versioning.delta_engine import DeltaEngine
from archivist.versioning.version_index import VersionIndex
from archivist.versioning.version_parser import VersionParser


class TestVersioningPipeline:
    """End-to-end versioning flow test."""

    def test_two_version_ingestion(self) -> None:
        """Simulate ingesting v1 then v2 of the same document."""
        engine = DeltaEngine()
        parser = VersionParser()

        # === V1 Ingestion ===
        v1_version = parser.parse("1.0")
        assert v1_version == (1, 0, 0)

        v1_chunks = [
            Chunk(text="Introduction to the product. This is the overview section.", chunk_index=0,
                  source_file="guide_v1.0.md", token_count=12),
            Chunk(text="Installation guide. Follow these steps to install.", chunk_index=1,
                  source_file="guide_v1.0.md", token_count=10),
            Chunk(text="Configuration options. Set the following parameters.", chunk_index=2,
                  source_file="guide_v1.0.md", token_count=10),
        ]

        v1_result = engine.classify_chunks(v1_chunks, existing=[], version=v1_version)

        assert v1_result.is_first_version is True
        assert len(v1_result.to_upsert) == 3
        assert all(r == ChunkRole.BASE for r in v1_result.to_upsert_roles)
        assert len(v1_result.to_update_range) == 0
        assert len(v1_result.to_cap) == 0

        # Simulate what storage would have after v1
        stored_v1 = [
            {"id": f"chunk-{i}", "text": c.text, "chunk_index": c.chunk_index,
             "chunk_role": "base", "version": "1.0", "version_tuple": [1, 0, 0],
             "version_range_min": [1, 0, 0], "version_range_max": None}
            for i, c in enumerate(v1_chunks)
        ]

        # === V2 Ingestion ===
        v2_version = parser.parse("2.0")
        assert v2_version == (2, 0, 0)

        v2_chunks = [
            # Chunk 0: unchanged
            Chunk(text="Introduction to the product. This is the overview section.", chunk_index=0,
                  source_file="guide_v2.0.md", token_count=12),
            # Chunk 1: modified
            Chunk(text="Installation guide. Follow the NEW steps for v2 installation.", chunk_index=1,
                  source_file="guide_v2.0.md", token_count=11),
            # Chunk 2: removed (not present in v2)
            # Chunk 3: new
            Chunk(text="Brand new API reference section added in version 2.0.", chunk_index=2,
                  source_file="guide_v2.0.md", token_count=10),
        ]

        v2_result = engine.classify_chunks(v2_chunks, stored_v1, version=v2_version)

        assert v2_result.is_first_version is False

        # Chunk 0 should be unchanged (extend range)
        assert len(v2_result.to_update_range) >= 1
        unchanged_ids = [uid for uid, _ in v2_result.to_update_range]
        assert "chunk-0" in unchanged_ids

        # Chunk 1 (modified) or Chunk 2 (new) should be upserted as delta
        assert len(v2_result.to_upsert) >= 1
        assert any(r == ChunkRole.DELTA for r in v2_result.to_upsert_roles)

        # Chunk 2 from v1 (configuration) should be capped (not in v2 incoming)
        assert len(v2_result.to_cap) >= 1

    def test_version_index_tracks_both_versions(self) -> None:
        """Verify version index is correctly maintained across ingestions."""
        mock_storage = MagicMock()

        # First ingestion
        mock_storage.get_version_index.return_value = None
        VersionIndex.update_index(mock_storage, "test-product", "admin_guide", "1.0")

        first_call = mock_storage.upsert_version_index.call_args[0][0]
        assert first_call["versions_ingested"] == ["1.0"]
        assert first_call["version_count"] == 1
        assert first_call["base_version"] == "1.0"

        # Second ingestion
        mock_storage.get_version_index.return_value = first_call
        VersionIndex.update_index(mock_storage, "test-product", "admin_guide", "2.0")

        second_call = mock_storage.upsert_version_index.call_args[0][0]
        assert second_call["versions_ingested"] == ["1.0", "2.0"]
        assert second_call["version_count"] == 2
        assert second_call["latest_version"] == "2.0"
        assert second_call["base_version"] == "1.0"

    def test_version_parser_handles_all_formats(self) -> None:
        """Verify version parser handles all documented format varieties."""
        parser = VersionParser()

        # Standard semver
        assert parser.parse("1.24.3") == (1, 24, 3)
        # Two-part
        assert parser.parse("1.24") == (1, 24, 0)
        # With v prefix
        assert parser.parse("v2.1.3") == (2, 1, 3)
        # Calendar versioning
        assert parser.parse("2024.04") == (2024, 4, 0)
        # RHEL-style
        assert parser.parse("r9.3") == (9, 3, 0)
        # N/A
        assert parser.parse("N/A") is None

        # Comparison
        assert parser.compare((1, 24, 0), (1, 26, 0)) < 0
        assert parser.compare((2, 0, 0), (1, 99, 0)) > 0

        # Range check
        assert parser.in_range((1, 24, 0), (1, 20, 0), (1, 26, 0)) is True
        assert parser.in_range((1, 24, 0), (1, 20, 0), None) is True
        assert parser.in_range((1, 18, 0), (1, 20, 0), (1, 26, 0)) is False
