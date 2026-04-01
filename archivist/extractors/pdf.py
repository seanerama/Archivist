"""PDF document extractor using pymupdf4llm."""

from __future__ import annotations

from pathlib import Path

from archivist.exceptions import ExtractionError
from archivist.extractors.base import BaseExtractor
from archivist.models import RawDocument


class PdfExtractor(BaseExtractor):
    """Extracts text from PDF files as clean markdown."""

    @property
    def supported_extensions(self) -> list[str]:
        return [".pdf"]

    def extract(self, path: Path) -> RawDocument:
        """Extract PDF content as markdown with page boundaries."""
        try:
            import pymupdf
            import pymupdf4llm
        except ImportError as e:
            raise ExtractionError(f"pymupdf4llm is required for PDF extraction: {e}") from e

        try:
            doc = pymupdf.open(str(path))
            page_count = len(doc)

            md_text = pymupdf4llm.to_markdown(str(path))

            # Build page boundary info
            pages = []
            offset = 0
            for i in range(page_count):
                page_text = doc[i].get_text()
                page_len = len(page_text)
                pages.append({
                    "page_number": i + 1,
                    "start_offset": offset,
                    "end_offset": offset + page_len,
                })
                offset += page_len

            # Extract native metadata from PDF info
            metadata = doc.metadata or {}
            native_metadata: dict[str, str | None] = {}
            for key in ("author", "title", "subject", "creator", "producer"):
                if metadata.get(key):
                    native_metadata[key] = metadata[key]
            if metadata.get("creationDate"):
                native_metadata["created_date"] = metadata["creationDate"]

            doc.close()

            return RawDocument(
                text=md_text,
                source_file=path.name,
                format="pdf",
                pages=pages,
                native_metadata=native_metadata,
            )
        except ExtractionError:
            raise
        except Exception as e:
            raise ExtractionError(f"Failed to extract PDF '{path.name}': {e}") from e
