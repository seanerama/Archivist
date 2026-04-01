"""MCP server exposing Archivist retrieval tools over stdio."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from archivist.config import Config
from archivist.exceptions import ArchivistError
from archivist.retrieval.models import DiffResult, FamilyInfo, SearchResult
from archivist.retrieval.retriever import Retriever


def _format_search_results(results: list[SearchResult]) -> str:
    """Format search results as numbered plain text."""
    if not results:
        return "No results found."

    lines = []
    for i, r in enumerate(results, 1):
        excerpt = r.text[:200].replace("\n", " ")
        if len(r.text) > 200:
            excerpt += "..."
        version_str = r.version or "unversioned"
        lines.append(
            f"{i}. [{r.source_file}] (v{version_str}, score: {r.score:.3f})\n"
            f"   {excerpt}"
        )
    return "\n\n".join(lines)


def _format_families(families: list[FamilyInfo]) -> str:
    """Format family list as a plain text table."""
    if not families:
        return "No document families found."

    header = f"{'Family':<30} {'Chunks':>6}  {'Versions':<30}  {'Types'}"
    sep = "-" * len(header)
    lines = [header, sep]
    for f in families:
        versions = ", ".join(f.versions) if f.versions else "none"
        types = ", ".join(f.doc_types) if f.doc_types else "none"
        lines.append(f"{f.family_slug:<30} {f.total_chunks:>6}  {versions:<30}  {types}")
    return "\n".join(lines)


def _format_diff(
    diffs: list[DiffResult], family: str, from_version: str, to_version: str
) -> str:
    """Format version diff as plain text grouped by change type."""
    lines = [f"Diff: {family} v{from_version} -> v{to_version}", ""]

    added = [d for d in diffs if d.change_type == "added"]
    modified = [d for d in diffs if d.change_type == "modified"]
    removed = [d for d in diffs if d.change_type == "removed"]

    if removed:
        lines.append(f"REMOVED ({len(removed)} chunks):")
        for d in removed:
            excerpt = d.chunk_text[:200].replace("\n", " ")
            if len(d.chunk_text) > 200:
                excerpt += "..."
            lines.append(f"  - chunk {d.chunk_index}: {excerpt}")
        lines.append("")

    if modified:
        lines.append(f"MODIFIED ({len(modified)} chunks):")
        for d in modified:
            excerpt = d.chunk_text[:200].replace("\n", " ")
            if len(d.chunk_text) > 200:
                excerpt += "..."
            lines.append(f"  ~ chunk {d.chunk_index}: {excerpt}")
        lines.append("")

    if added:
        lines.append(f"ADDED ({len(added)} chunks):")
        for d in added:
            excerpt = d.chunk_text[:200].replace("\n", " ")
            if len(d.chunk_text) > 200:
                excerpt += "..."
            lines.append(f"  + chunk {d.chunk_index}: {excerpt}")
        lines.append("")

    if not diffs:
        lines.append("No changes found between these versions.")

    total = f"{len(added)} added, {len(modified)} modified, {len(removed)} removed"
    lines.append(f"Summary: {total}")

    return "\n".join(lines)


def create_server(config: Config | None = None) -> FastMCP:
    """Create and configure the MCP server with retrieval tools.

    Args:
        config: Archivist configuration. If None, loads from archivist.yaml.

    Returns:
        Configured FastMCP server instance.
    """
    if config is None:
        config = Config.load()

    server = FastMCP("archivist")

    @server.tool()
    async def archivist_search(
        query: str,
        version: str | None = None,
        family: str | None = None,
        doc_type: str | None = None,
        top_k: int = 5,
    ) -> str:
        """Search the technical documentation corpus. Returns relevant chunks with source metadata.

        Args:
            query: The search query text.
            version: Filter to a specific version (e.g. '1.24').
            family: Filter to a document family (e.g. 'nginx').
            doc_type: Filter to a document type (e.g. 'admin_guide').
            top_k: Number of results to return (default 5).
        """
        try:
            retriever = Retriever(config)
            results = retriever.search(
                query,
                version=version,
                family=family,
                doc_type=doc_type,
                top_k=top_k,
            )
            return _format_search_results(results)
        except ArchivistError as e:
            return f"Error: {e}"

    @server.tool()
    async def archivist_list_families() -> str:
        """List all document families and their versions in the corpus."""
        try:
            retriever = Retriever(config)
            families = retriever.list_families()
            return _format_families(families)
        except ArchivistError as e:
            return f"Error: {e}"

    @server.tool()
    async def archivist_version_diff(
        family: str,
        from_version: str,
        to_version: str,
    ) -> str:
        """Show chunks that changed between two versions of a document family.

        Args:
            family: The document family slug.
            from_version: The older version.
            to_version: The newer version.
        """
        try:
            retriever = Retriever(config)
            diffs = retriever.version_diff(family, from_version, to_version)
            return _format_diff(diffs, family, from_version, to_version)
        except ArchivistError as e:
            return f"Error: {e}"

    return server


async def run_server() -> None:
    """Run the MCP server using stdio transport."""
    config = Config.load()
    server = create_server(config)
    await server.run_stdio_async()
