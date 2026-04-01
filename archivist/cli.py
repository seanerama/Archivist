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

        docs = stats.get("documents", [])
        families = {d["family_slug"] for d in docs if d.get("family_slug")}

        console.print("[bold]Archivist Corpus Status[/bold]\n")
        summary = Table(show_header=False)
        summary.add_row("Collection", stats.get("collection", "unknown"))
        summary.add_row("Total chunks", str(stats.get("total_chunks", 0)))
        summary.add_row("Documents", str(len(docs)))
        summary.add_row("Families", str(len(families)))
        summary.add_row("Status", stats.get("status", "unknown"))
        console.print(summary)

        if docs:
            console.print("\n[bold]Documents[/bold]\n")
            doc_table = Table(show_header=True, header_style="bold")
            doc_table.add_column("File")
            doc_table.add_column("Family")
            doc_table.add_column("Title")
            doc_table.add_column("Type")
            doc_table.add_column("Version")
            doc_table.add_column("Chunks", justify="right")

            for doc in sorted(docs, key=lambda d: d.get("family_slug", "")):
                doc_table.add_row(
                    doc["source_file"],
                    doc.get("family_slug", ""),
                    doc.get("doc_title", "")[:40],
                    doc.get("doc_type", ""),
                    doc.get("version") or "—",
                    str(doc["chunks"]),
                )
            console.print(doc_table)
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
def search(
    query: str = typer.Argument(help="Search query"),  # noqa: B008
    version: str | None = typer.Option(None, "--version", "-V", help="Filter to a specific version"),
    family: str | None = typer.Option(None, "--family", "-f", help="Filter to a document family"),
    doc_type: str | None = typer.Option(None, "--doc-type", "-t", help="Filter to a document type"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results"),
    config_path: str | None = typer.Option(None, "--config", "-c"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Search the document corpus."""
    config = _load_config(config_path, verbose)

    from archivist.retrieval import Retriever

    retriever = Retriever(config)
    results = retriever.search(query, version=version, family=family, doc_type=doc_type, top_k=top_k)

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        raise typer.Exit(code=0)

    for i, r in enumerate(results, 1):
        console.print(f"\n[bold cyan]{i}. {r.doc_title or r.source_file}[/bold cyan]")
        meta_parts = []
        if r.family_slug:
            meta_parts.append(f"family={r.family_slug}")
        if r.version:
            meta_parts.append(f"v{r.version}")
        if r.doc_type:
            meta_parts.append(r.doc_type)
        if r.page_number is not None:
            meta_parts.append(f"p.{r.page_number}")
        meta_parts.append(f"score={r.score:.3f}")
        console.print(f"   [dim]{' | '.join(meta_parts)}[/dim]")
        if r.heading_path:
            console.print(f"   [dim]{r.heading_path}[/dim]")
        # Show first ~200 chars of text
        excerpt = r.text[:200].replace("\n", " ")
        if len(r.text) > 200:
            excerpt += "..."
        console.print(f"   {excerpt}")


@app.command()
def families(
    config_path: str | None = typer.Option(None, "--config", "-c"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """List all document families and their versions."""
    config = _load_config(config_path, verbose)

    from archivist.retrieval import Retriever

    retriever = Retriever(config)
    fams = retriever.list_families()

    if not fams:
        console.print("[yellow]No families found. Ingest some documents first.[/yellow]")
        raise typer.Exit(code=0)

    table = Table(show_header=True, header_style="bold")
    table.add_column("Family")
    table.add_column("Doc Types")
    table.add_column("Versions")
    table.add_column("Latest")
    table.add_column("Chunks", justify="right")

    for f in fams:
        table.add_row(
            f.family_slug,
            ", ".join(f.doc_types),
            ", ".join(f.versions) if f.versions else "—",
            f.latest_version or "—",
            str(f.total_chunks),
        )

    console.print(table)


@app.command()
def diff(
    family: str = typer.Argument(help="Document family slug"),  # noqa: B008
    from_version: str = typer.Argument(help="Older version"),  # noqa: B008
    to_version: str = typer.Argument(help="Newer version"),  # noqa: B008
    config_path: str | None = typer.Option(None, "--config", "-c"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Show what changed between two versions of a document family."""
    config = _load_config(config_path, verbose)

    from archivist.retrieval import Retriever

    retriever = Retriever(config)
    diffs = retriever.version_diff(family, from_version, to_version)

    if not diffs:
        console.print("[yellow]No differences found.[/yellow]")
        raise typer.Exit(code=0)

    added = [d for d in diffs if d.change_type == "added"]
    modified = [d for d in diffs if d.change_type == "modified"]
    removed = [d for d in diffs if d.change_type == "removed"]

    if removed:
        console.print(f"\n[bold red]Removed ({len(removed)} chunks)[/bold red]")
        for d in removed:
            excerpt = d.chunk_text[:150].replace("\n", " ")
            console.print(f"  [red]- chunk {d.chunk_index}[/red]: {excerpt}...")

    if modified:
        console.print(f"\n[bold yellow]Modified ({len(modified)} chunks)[/bold yellow]")
        for d in modified:
            excerpt = d.chunk_text[:150].replace("\n", " ")
            console.print(f"  [yellow]~ chunk {d.chunk_index}[/yellow]: {excerpt}...")

    if added:
        console.print(f"\n[bold green]Added ({len(added)} chunks)[/bold green]")
        for d in added:
            excerpt = d.chunk_text[:150].replace("\n", " ")
            console.print(f"  [green]+ chunk {d.chunk_index}[/green]: {excerpt}...")

    console.print(f"\n[bold]Summary[/bold]: {len(added)} added, {len(modified)} modified, {len(removed)} removed")


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


@app.command()
def mcp(
    config_path: str | None = typer.Option(None, "--config", "-c", help="Path to archivist.yaml"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """Start the MCP server for Claude Code integration."""
    import asyncio

    _load_config(config_path, verbose)

    from archivist.mcp.server import run_server

    asyncio.run(run_server())
