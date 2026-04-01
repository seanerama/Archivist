"""Tests for the review queue."""

from __future__ import annotations

from archivist.metadata.review_queue import ReviewItem, ReviewQueue


class TestReviewQueue:
    """Tests for ReviewQueue."""

    def test_empty_queue(self) -> None:
        queue = ReviewQueue()
        assert queue.count == 0
        assert queue.items == []

    def test_add_item(self) -> None:
        queue = ReviewQueue()
        queue.add(ReviewItem(source_file="test.pdf", missing=["version", "date"]))
        assert queue.count == 1
        assert queue.items[0].source_file == "test.pdf"

    def test_render_summary_no_error_when_empty(self) -> None:
        queue = ReviewQueue()
        queue.render_summary()  # Should not raise

    def test_render_summary_with_items(self) -> None:
        from io import StringIO

        from rich.console import Console

        queue = ReviewQueue()
        queue.add(ReviewItem(
            source_file="test.pdf",
            missing=["version"],
            detected_date="2024",
            reason="low confidence",
        ))

        output = StringIO()
        console = Console(file=output, force_terminal=True)
        queue.render_summary(console)

        rendered = output.getvalue()
        assert "test.pdf" in rendered
        assert "1" in rendered
