"""Tests for the delta engine."""

from __future__ import annotations

from archivist.models import Chunk, ChunkRole
from archivist.versioning.delta_engine import DeltaEngine


def _make_chunk(text: str, index: int) -> Chunk:
    return Chunk(text=text, chunk_index=index, source_file="test.md", token_count=len(text) // 4)


class TestDeltaEngine:
    """Tests for DeltaEngine.classify_chunks."""

    def test_first_version_all_base(self) -> None:
        engine = DeltaEngine()
        incoming = [_make_chunk("Hello world", 0), _make_chunk("Second chunk", 1)]

        result = engine.classify_chunks(incoming, existing=[], version=(1, 0, 0))

        assert result.is_first_version is True
        assert len(result.to_upsert) == 2
        assert all(r == ChunkRole.BASE for r in result.to_upsert_roles)
        assert all(b is None for b in result.to_upsert_base_ids)

    def test_unchanged_chunks_extend_range(self) -> None:
        engine = DeltaEngine()
        incoming = [_make_chunk("Hello world exactly the same text here", 0)]
        existing = [{"id": "existing-1", "text": "Hello world exactly the same text here", "chunk_index": 0}]

        result = engine.classify_chunks(incoming, existing, version=(1, 1, 0))

        assert result.is_first_version is False
        assert len(result.to_upsert) == 0
        assert len(result.to_update_range) == 1
        assert result.to_update_range[0][0] == "existing-1"

    def test_modified_chunk_stored_as_delta(self) -> None:
        engine = DeltaEngine()
        # ~70% similar
        incoming = [_make_chunk("Hello world with some significant changes to the content here", 0)]
        existing = [
            {"id": "existing-1", "text": "Hello world with the original content that was different", "chunk_index": 0},
        ]

        result = engine.classify_chunks(incoming, existing, version=(1, 1, 0))

        assert len(result.to_upsert) == 1
        assert result.to_upsert_roles[0] == ChunkRole.DELTA

    def test_new_chunk_no_match(self) -> None:
        engine = DeltaEngine()
        incoming = [_make_chunk("Completely different content that has nothing in common", 5)]
        existing = [{"id": "existing-1", "text": "Original content about something else entirely", "chunk_index": 0}]

        result = engine.classify_chunks(incoming, existing, version=(1, 1, 0))

        assert len(result.to_upsert) == 1
        assert result.to_upsert_roles[0] == ChunkRole.DELTA

    def test_removed_chunk_capped(self) -> None:
        engine = DeltaEngine()
        incoming: list[Chunk] = []  # No incoming chunks
        existing = [{"id": "existing-1", "text": "This chunk was removed", "chunk_index": 0}]

        result = engine.classify_chunks(incoming, existing, version=(1, 1, 0))

        assert len(result.to_cap) == 1
        assert result.to_cap[0][0] == "existing-1"

    def test_mixed_scenario(self) -> None:
        engine = DeltaEngine()

        # 3 incoming: 1 unchanged, 1 modified, 1 new
        incoming = [
            _make_chunk("This chunk is exactly the same as before and stays unchanged", 0),
            _make_chunk("This chunk has been significantly modified with new information", 1),
            _make_chunk("Brand new content not present in any previous version at all", 2),
        ]
        existing = [
            {"id": "e0", "text": "This chunk is exactly the same as before and stays unchanged", "chunk_index": 0},
            {"id": "e1", "text": "This chunk was the original version before modification happened", "chunk_index": 1},
            {"id": "e2", "text": "This old chunk will be removed from the document entirely", "chunk_index": 3},
        ]

        result = engine.classify_chunks(incoming, existing, version=(1, 2, 0))

        # e0 should be unchanged (extend range)
        assert len(result.to_update_range) >= 1
        # New/modified chunks should be upserted
        assert len(result.to_upsert) >= 1
        # e2 should be capped (not matched by any incoming)
        assert len(result.to_cap) >= 1
