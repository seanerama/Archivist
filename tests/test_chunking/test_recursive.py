"""Tests for recursive chunking."""

from __future__ import annotations

from archivist.chunking.recursive import RecursiveChunker
from archivist.config import PipelineConfig
from archivist.models import RawDocument


def _make_config(chunk_size: int = 128) -> PipelineConfig:
    return PipelineConfig(chunk_size=chunk_size, chunk_overlap_pct=10)


class TestRecursiveChunker:
    """Tests for RecursiveChunker."""

    def test_small_document_single_chunk(self) -> None:
        chunker = RecursiveChunker(_make_config(chunk_size=1000))
        doc = RawDocument(text="Short text.", source_file="test.txt", format="txt")
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].text == "Short text."

    def test_markdown_splits_on_headings(self) -> None:
        text = "# Title\n\nIntro text.\n\n## Section 1\n\nFirst section.\n\n## Section 2\n\nSecond section."
        chunker = RecursiveChunker(_make_config(chunk_size=1000))
        doc = RawDocument(text=text, source_file="test.md", format="md")
        chunks = chunker.chunk(doc)

        assert len(chunks) >= 2
        # Check heading paths
        heading_paths = [c.heading_path for c in chunks]
        assert any("Section 1" in (p or "") for p in heading_paths)
        assert any("Section 2" in (p or "") for p in heading_paths)

    def test_chunk_index_sequential(self) -> None:
        text = "# A\n\nText A.\n\n## B\n\nText B.\n\n## C\n\nText C."
        chunker = RecursiveChunker(_make_config(chunk_size=1000))
        doc = RawDocument(text=text, source_file="test.md", format="md")
        chunks = chunker.chunk(doc)

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_video_chunks_by_segments(self) -> None:
        segments = [
            {"start": 0.0, "end": 30.0, "text": "First segment " * 10},
            {"start": 30.0, "end": 60.0, "text": "Second segment " * 10},
            {"start": 60.0, "end": 95.0, "text": "Third segment " * 10},
        ]
        doc = RawDocument(
            text=" ".join(s["text"] for s in segments),
            source_file="video.mp4",
            format="video",
            native_metadata={"segments": segments},
        )
        chunker = RecursiveChunker(_make_config(chunk_size=50))
        chunks = chunker.chunk(doc)

        assert len(chunks) >= 1
        assert chunks[0].timestamp_start is not None
        assert chunks[0].timestamp_end is not None

    def test_code_blocks_not_split(self) -> None:
        text = "Before code.\n\n```python\ndef foo():\n    return 'bar'\n```\n\nAfter code."
        chunker = RecursiveChunker(_make_config(chunk_size=1000))
        doc = RawDocument(text=text, source_file="test.md", format="md")
        chunks = chunker.chunk(doc)

        # The code block should be intact in some chunk
        all_text = " ".join(c.text for c in chunks)
        assert "def foo():" in all_text
        assert "return 'bar'" in all_text

    def test_generic_with_pages(self) -> None:
        text = "Page 1 content here. " * 20 + "Page 2 content here. " * 20
        pages = [
            {"page_number": 1, "start_offset": 0, "end_offset": len("Page 1 content here. " * 20)},
            {"page_number": 2, "start_offset": len("Page 1 content here. " * 20), "end_offset": len(text)},
        ]
        doc = RawDocument(text=text, source_file="test.pdf", format="pdf", pages=pages)
        chunker = RecursiveChunker(_make_config(chunk_size=30))
        chunks = chunker.chunk(doc)

        assert len(chunks) >= 1
        assert chunks[0].source_file == "test.pdf"

    def test_empty_document(self) -> None:
        doc = RawDocument(text="", source_file="empty.txt", format="txt")
        chunker = RecursiveChunker(_make_config())
        chunks = chunker.chunk(doc)
        assert len(chunks) == 0
