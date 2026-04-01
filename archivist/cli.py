"""Typer CLI for Archivist."""

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from archivist.config import Config
from archivist.log import configure_logging

app = typer.Typer(help="Archivist — version-aware document ingestion for technical documentation.")
console = Console()


def _load_config(config_path: str | None, verbose: bool) -> Config:
    """Load config, configure logging."""
    load_dotenv()

    path = Path(config_path) if config_path else None
    config = Config.load(path)

    level = "DEBUG" if verbose else config.logging.level
    configure_logging(level)

    return config


@app.command()
def ingest(
    path: list[str] = typer.Argument(help="File or directory paths to ingest"),  # noqa: B008
    dry_run: bool = typer.Option(False, "--dry-run", help="Process without writing to Qdrant"),
    overwrite: bool = typer.Option(False, "--overwrite-existing", help="Re-ingest already ingested documents"),
    config_path: str | None = typer.Option(None, "--config", "-c", help="Path to archivist.yaml"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """Ingest documents into the vector database."""
    config = _load_config(config_path, verbose)

    if overwrite:
        config.pipeline.overwrite_existing = True
    if dry_run:
        config.pipeline.dry_run = True

    from archivist.pipeline import Pipeline

    pipeline = Pipeline(config)
    paths = [Path(p) for p in path]
    result = pipeline.ingest(paths, dry_run=dry_run)

    # Print summary
    console.print()
    console.print("[bold]Ingestion Summary[/bold]")
    table = Table(show_header=False)
    table.add_row("Documents processed", str(result.docs_processed))
    table.add_row("Documents skipped", str(result.docs_skipped))
    table.add_row("Documents failed", str(result.docs_failed))
    table.add_row("Chunks created", str(result.chunks_created))
    table.add_row("Chunks updated", str(result.chunks_updated))
    table.add_row("Tags auto-accepted", str(result.tags_auto_accepted))
    table.add_row("Tags flagged for review", str(result.tags_flagged))
    console.print(table)

    if result.errors:
        console.print("\n[bold red]Errors:[/bold red]")
        for filename, error in result.errors:
            console.print(f"  {filename}: {error}")

    raise typer.Exit(code=1 if result.docs_failed > 0 else 0)


@app.command()
def status(
    config_path: str | None = typer.Option(None, "--config", "-c"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Show corpus statistics."""
    config = _load_config(config_path, verbose)

    from archivist.storage import QdrantStorage

    storage = QdrantStorage(config)
    try:
        storage.connect(vector_dimension=1)  # dimension doesn't matter for stats
        stats = storage.collection_stats()

        console.print("[bold]Archivist Corpus Status[/bold]")
        table = Table(show_header=False)
        table.add_row("Collection", stats.get("collection", "unknown"))
        table.add_row("Total chunks", str(stats.get("total_chunks", 0)))
        table.add_row("Status", stats.get("status", "unknown"))
        console.print(table)
    except Exception as e:
        console.print(f"[red]Cannot connect to Qdrant: {e}[/red]")
        raise typer.Exit(code=1) from None


@app.command()
def review(
    config_path: str | None = typer.Option(None, "--config", "-c"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Review flagged documents interactively."""
    _load_config(config_path, verbose)
    console.print("[yellow]Interactive review mode is not yet implemented.[/yellow]")
    console.print("Run 'archivist ingest' to see flagged documents in the review queue.")


@app.command()
def setup(
    config_path: str | None = typer.Option(None, "--config", "-c"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the interactive setup wizard."""
    config = _load_config(config_path, verbose)

    from archivist.storage import SetupWizard

    wizard = SetupWizard(config)
    output_path = Path(config_path) if config_path else Path("archivist.yaml")
    wizard.run(config_path=output_path)
