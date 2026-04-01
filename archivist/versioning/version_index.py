"""Version index management for document families."""

from __future__ import annotations

from typing import Any

from archivist.log import get_logger
from archivist.storage.qdrant import QdrantStorage

logger = get_logger("versioning.index")


class VersionIndex:
    """Maintains version summary records per document family in Qdrant."""

    @staticmethod
    def get_index(storage: QdrantStorage, family_slug: str, doc_type: str) -> dict[str, Any] | None:
        """Get the version index for a document family."""
        return storage.get_version_index(family_slug, doc_type)

    @staticmethod
    def update_index(
        storage: QdrantStorage,
        family_slug: str,
        doc_type: str,
        new_version: str,
    ) -> None:
        """Add a new version to the version index.

        Creates the index if it doesn't exist, or appends to it.
        """
        existing = storage.get_version_index(family_slug, doc_type)

        if existing:
            versions = existing.get("versions_ingested", [])
            if new_version not in versions:
                versions.append(new_version)
            index = {
                "family_slug": family_slug,
                "doc_type": doc_type,
                "versions_ingested": versions,
                "version_count": len(versions),
                "latest_version": new_version,
                "base_version": versions[0] if versions else new_version,
            }
        else:
            index = {
                "family_slug": family_slug,
                "doc_type": doc_type,
                "versions_ingested": [new_version],
                "version_count": 1,
                "latest_version": new_version,
                "base_version": new_version,
            }

        storage.upsert_version_index(index)
        logger.info("Updated version index", family=family_slug, version=new_version)
