"""Recursive character chunking with format-aware overrides."""

from __future__ import annotations

import re

from archivist.config import PipelineConfig
from archivist.log import get_logger
from archivist.models import Chunk, RawDocument

logger = get_logger("chunking")

# Approximate tokens-to-characters ratio
CHARS_PER_TOKEN = 4


class RecursiveChunker:
    """Chunks documents with format-aware splitting strategies."""

    def __init__(self, config: PipelineConfig) -> None:
        self._chunk_size = config.chunk_size * CHARS_PER_TOKEN
        self._overlap = int(self._chunk_size * config.chunk_overlap_pct / 100)

    def chunk(self, raw_doc: RawDocument) -> list[Chunk]:
        """Chunk a RawDocument based on its format.

        Args:
            raw_doc: The extracted document to chunk.

        Returns:
            List of Chunk objects with positional metadata.
        """
        if raw_doc.format == "md":
            return self._chunk_markdown(raw_doc)
        elif raw_doc.format == "video":
            return self._chunk_video(raw_doc)
        else:
            return self._chunk_generic(raw_doc)

    def _chunk_markdown(self, raw_doc: RawDocument) -> list[Chunk]:
        """Chunk markdown by heading boundaries, preserving code blocks."""
        sections = self._split_by_headings(raw_doc.text)
        chunks: list[Chunk] = []

        for section_text, heading_path in sections:
            if len(section_text) <= self._chunk_size:
                chunks.append(Chunk(
                    text=section_text.strip(),
                    chunk_index=len(chunks),
                    source_file=raw_doc.source_file,
                    heading_path=heading_path,
                    token_count=len(section_text) // CHARS_PER_TOKEN,
                ))
            else:
                # Sub-chunk large sections
                for sub_text in self._split_text(section_text):
                    chunks.append(Chunk(
                        text=sub_text.strip(),
                        chunk_index=len(chunks),
                        source_file=raw_doc.source_file,
                        heading_path=heading_path,
                        token_count=len(sub_text) // CHARS_PER_TOKEN,
                    ))

        return [c for c in chunks if c.text]

    def _chunk_video(self, raw_doc: RawDocument) -> list[Chunk]:
        """Chunk video transcripts by segment boundaries (~60-90s windows)."""
        segments = raw_doc.native_metadata.get("segments", [])
        if not segments:
            return self._chunk_generic(raw_doc)

        chunks: list[Chunk] = []
        current_texts: list[str] = []
        window_start = segments[0]["start"] if segments else 0.0
        current_chars = 0

        for seg in segments:
            seg_text = seg["text"]
            current_texts.append(seg_text)
            current_chars += len(seg_text)

            # Target ~60-90s or chunk_size chars
            duration = seg["end"] - window_start
            if current_chars >= self._chunk_size or duration >= 90.0:
                chunks.append(Chunk(
                    text=" ".join(current_texts).strip(),
                    chunk_index=len(chunks),
                    source_file=raw_doc.source_file,
                    timestamp_start=window_start,
                    timestamp_end=seg["end"],
                    token_count=current_chars // CHARS_PER_TOKEN,
                ))
                current_texts = []
                current_chars = 0
                window_start = seg["end"]

        # Remaining segments
        if current_texts:
            chunks.append(Chunk(
                text=" ".join(current_texts).strip(),
                chunk_index=len(chunks),
                source_file=raw_doc.source_file,
                timestamp_start=window_start,
                timestamp_end=segments[-1]["end"],
                token_count=current_chars // CHARS_PER_TOKEN,
            ))

        return [c for c in chunks if c.text]

    def _chunk_generic(self, raw_doc: RawDocument) -> list[Chunk]:
        """Chunk generic text (PDF, plaintext, EPUB) with page tracking."""
        text_parts = self._split_text(raw_doc.text)
        chunks: list[Chunk] = []

        for part in text_parts:
            page_number = None
            if raw_doc.pages:
                # Find which page this chunk starts on
                offset = raw_doc.text.find(part[:50])
                if offset >= 0:
                    for page in raw_doc.pages:
                        if page["start_offset"] <= offset < page["end_offset"]:
                            page_number = page["page_number"]
                            break

            chunks.append(Chunk(
                text=part.strip(),
                chunk_index=len(chunks),
                source_file=raw_doc.source_file,
                page_number=page_number,
                token_count=len(part) // CHARS_PER_TOKEN,
            ))

        return [c for c in chunks if c.text]

    def _split_by_headings(self, text: str) -> list[tuple[str, str]]:
        """Split markdown text by heading boundaries.

        Returns list of (section_text, heading_path) tuples.
        """
        heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
        matches = list(heading_pattern.finditer(text))

        if not matches:
            return [(text, "")]

        sections: list[tuple[str, str]] = []
        heading_stack: list[tuple[int, str]] = []

        # Text before first heading
        if matches[0].start() > 0:
            sections.append((text[: matches[0].start()], ""))

        for i, match in enumerate(matches):
            level = len(match.group(1))
            title = match.group(2).strip()

            # Update heading stack
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))

            heading_path = " > ".join(h[1] for h in heading_stack)

            # Get section text (from this heading to the next)
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_text = text[start:end]

            sections.append((section_text, heading_path))

        return sections

    def _split_text(self, text: str) -> list[str]:
        """Split text into chunks respecting natural boundaries.

        Tries to split on paragraph boundaries, then sentences, then characters.
        Never splits inside fenced code blocks.
        """
        if len(text) <= self._chunk_size:
            return [text] if text.strip() else []

        # Protect code blocks
        code_block_pattern = re.compile(r"```[\s\S]*?```", re.MULTILINE)
        protected = {}
        counter = 0

        def protect(match: re.Match[str]) -> str:
            nonlocal counter
            key = f"__CODE_BLOCK_{counter}__"
            protected[key] = match.group()
            counter += 1
            return key

        safe_text = code_block_pattern.sub(protect, text)

        # Split on paragraphs first
        paragraphs = re.split(r"\n\n+", safe_text)
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for para in paragraphs:
            para_len = len(para)
            if current_len + para_len > self._chunk_size and current:
                chunk_text = "\n\n".join(current)
                # Restore code blocks
                for key, value in protected.items():
                    chunk_text = chunk_text.replace(key, value)
                chunks.append(chunk_text)
                # Overlap: keep last paragraph
                current = [para]
                current_len = para_len
            else:
                current.append(para)
                current_len += para_len

        if current:
            chunk_text = "\n\n".join(current)
            for key, value in protected.items():
                chunk_text = chunk_text.replace(key, value)
            chunks.append(chunk_text)

        return chunks
