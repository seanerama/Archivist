"""Review queue for documents with missing or ambiguous metadata."""

from __future__ import annotations

from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table


@dataclass
class ReviewItem:
    """A document flagged for metadata review."""

    source_file: str
    missing: list[str]
    detected_version: str | None = None
    detected_date: str | None = None
    suggested_family: str | None = None
    confidence: float | None = None
    reason: str = ""


class ReviewQueue:
    """Collects flagged documents and renders a review summary."""

    def __init__(self) -> None:
        self._items: list[ReviewItem] = field(default_factory=list) if False else []

    def add(self, item: ReviewItem) -> None:
        """Add a document to the review queue."""
        self._items.append(item)

    @property
    def count(self) -> int:
        """Number of items in the queue."""
        return len(self._items)

    @property
    def items(self) -> list[ReviewItem]:
        """All review items."""
        return list(self._items)

    def render_summary(self, console: Console | None = None) -> None:
        """Print the review queue summary to the console."""
        if not self._items:
            return

        if console is None:
            console = Console()

        console.print()
        console.print(
            f"[bold yellow]METADATA REVIEW REQUIRED — {len(self._items)} document(s) need attention[/bold yellow]"
        )

        table = Table(show_header=True, header_style="bold")
        table.add_column("#", style="dim", width=4)
        table.add_column("File")
        table.add_column("Missing")
        table.add_column("Detected")
        table.add_column("Reason")

        for i, item in enumerate(self._items, 1):
            detected_parts = []
            if item.detected_version:
                detected_parts.append(f"version={item.detected_version}")
            if item.detected_date:
                detected_parts.append(f"date={item.detected_date}")

            table.add_row(
                str(i),
                item.source_file,
                ", ".join(item.missing),
                ", ".join(detected_parts) if detected_parts else "—",
                item.reason,
            )

        console.print(table)
