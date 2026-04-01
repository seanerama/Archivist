"""Delta computation engine for version-aware chunk storage."""

from __future__ import annotations

import difflib
from typing import Any

from archivist.log import get_logger
from archivist.models import Chunk, ChunkRole, ClassificationResult, VersionTuple

logger = get_logger("versioning.delta")

# Default similarity thresholds (configurable via config in future)
UNCHANGED_THRESHOLD = 0.95
DELTA_THRESHOLD = 0.50


class DeltaEngine:
    """Classifies incoming chunks against existing family chunks using text diff."""

    def __init__(
        self,
        unchanged_threshold: float = UNCHANGED_THRESHOLD,
        delta_threshold: float = DELTA_THRESHOLD,
    ) -> None:
        self._unchanged_threshold = unchanged_threshold
        self._delta_threshold = delta_threshold

    def classify_chunks(
        self,
        incoming: list[Chunk],
        existing: list[dict[str, Any]],
        version: VersionTuple,
    ) -> ClassificationResult:
        """Classify incoming chunks against existing ones.

        Args:
            incoming: New chunks from the current document.
            existing: Existing chunks from Qdrant (dict with id, text, chunk_index, etc).
            version: The version tuple of the incoming document.

        Returns:
            ClassificationResult describing what to store, update, and cap.
        """
        if not existing:
            # First version — all chunks are base
            return ClassificationResult(
                to_upsert=incoming,
                to_upsert_roles=[ChunkRole.BASE] * len(incoming),
                to_upsert_base_ids=[None] * len(incoming),
                to_update_range=[],
                to_cap=[],
                is_first_version=True,
            )

        # Build index of existing chunks by chunk_index for positional matching
        existing_by_index = {c["chunk_index"]: c for c in existing}

        to_upsert: list[Chunk] = []
        to_upsert_roles: list[ChunkRole] = []
        to_upsert_base_ids: list[str | None] = []
        to_update_range: list[tuple[str, VersionTuple]] = []
        matched_existing_ids: set[str] = set()

        for chunk in incoming:
            best_match = self._find_best_match(chunk, existing_by_index, existing)

            if best_match is None:
                # New chunk — no positional or similarity match
                to_upsert.append(chunk)
                to_upsert_roles.append(ChunkRole.DELTA)
                to_upsert_base_ids.append(None)
                continue

            match_id, similarity = best_match
            matched_existing_ids.add(match_id)

            if similarity >= self._unchanged_threshold:
                # Unchanged — extend version range
                to_update_range.append((match_id, version))
            elif similarity >= self._delta_threshold:
                # Modified — store as delta linked to base
                to_upsert.append(chunk)
                to_upsert_roles.append(ChunkRole.DELTA)
                to_upsert_base_ids.append(match_id)
            else:
                # Too different — treat as new
                to_upsert.append(chunk)
                to_upsert_roles.append(ChunkRole.DELTA)
                to_upsert_base_ids.append(None)

        # Find removed chunks (in existing but not matched)
        previous_version = (version[0], version[1], max(0, version[2] - 1))
        to_cap: list[tuple[str, VersionTuple]] = []
        for chunk_data in existing:
            if chunk_data["id"] not in matched_existing_ids:
                to_cap.append((chunk_data["id"], previous_version))

        logger.info(
            "Delta classification complete",
            upserts=len(to_upsert),
            unchanged=len(to_update_range),
            removed=len(to_cap),
        )

        return ClassificationResult(
            to_upsert=to_upsert,
            to_upsert_roles=to_upsert_roles,
            to_upsert_base_ids=to_upsert_base_ids,
            to_update_range=to_update_range,
            to_cap=to_cap,
            is_first_version=False,
        )

    def _find_best_match(
        self,
        chunk: Chunk,
        existing_by_index: dict[int, dict[str, Any]],
        all_existing: list[dict[str, Any]],
    ) -> tuple[str, float] | None:
        """Find the best matching existing chunk using positional + similarity match."""
        # Try positional match first
        if chunk.chunk_index in existing_by_index:
            candidate = existing_by_index[chunk.chunk_index]
            similarity = self._text_similarity(chunk.text, candidate["text"])
            if similarity >= self._delta_threshold:
                return (candidate["id"], similarity)

        # Fallback: find nearest match across all existing chunks
        best_id = None
        best_sim = 0.0
        for existing_chunk in all_existing:
            sim = self._text_similarity(chunk.text, existing_chunk["text"])
            if sim > best_sim:
                best_sim = sim
                best_id = existing_chunk["id"]

        if best_id and best_sim >= self._delta_threshold:
            return (best_id, best_sim)

        return None

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        """Compute similarity ratio between two texts using difflib."""
        return difflib.SequenceMatcher(None, a, b).ratio()
