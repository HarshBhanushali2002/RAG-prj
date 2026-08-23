"""
🔬 RAG INSPECTOR — See your chunks, embeddings, and vectors up close.

This script lets you peek inside the RAG pipeline to understand
what's actually happening at each step.

Usage:
    python inspect_rag.py chunks              — See all stored chunks
    python inspect_rag.py vectors             — See actual embedding vectors (the numbers!)
    python inspect_rag.py search "your query" — See how search finds relevant chunks
    python inspect_rag.py stats               — Overview of what's in your database
    python inspect_rag.py compare "query"     — Compare similarity scores across all chunks
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
import math

from app.config import validate_config, CHROMA_DB_PATH, CHUNK_SIZE, CHUNK_OVERLAP
from app.services.vector_store import get_vector_store, COLLECTION_NAME
from app.services.embedder import embed_query

console = Console()


def show_stats():
    """Show overview of what's stored in the database."""
    store = get_vector_store()
    collection = store._collection

    all_data = collection.get(include=["metadatas", "documents"])

    if not all_data["ids"]:
        console.print("[yellow]Database is empty. Upload a PDF first with cli.py[/yellow]")
        return

    total_chunks = len(all_data["ids"])

    # Group by source file
    files = {}
    for meta in all_data["metadatas"]:
        fname = meta.get("source_file", "unknown")
        if fname not in files:
            files[fname] = {"chunks": 0, "pages": set()}
        files[fname]["chunks"] += 1
        files[fname]["pages"].add(meta.get("page_number", 0))

    # Calculate text stats
    all_texts = all_data["documents"]
    sizes = [len(t) for t in all_texts]

    console.print(Panel(
        f"[bold green]Total Chunks:[/bold green] {total_chunks}\n"
        f"[bold green]Total Documents:[/bold green] {len(files)}\n"
        f"[bold green]Chunk Size Setting:[/bold green] {CHUNK_SIZE} chars\n"
        f"[bold green]Chunk Overlap Setting:[/bold green] {CHUNK_OVERLAP} chars\n"
        f"[bold green]Avg Chunk Size:[/bold green] {sum(sizes) // len(sizes)} chars\n"
        f"[bold green]Min / Max Chunk:[/bold green] {min(sizes)} / {max(sizes)} chars\n"
        f"[bold green]DB Path:[/bold green] {CHROMA_DB_PATH}",
        title="📊 RAG Database Stats",
        border_style="cyan",
    ))

    table = Table(title="📚 Documents in Database", box=box.ROUNDED)
    table.add_column("#", style="dim")
    table.add_column("Filename", style="cyan bold")
    table.add_column("Chunks", style="green", justify="right")
    table.add_column("Pages", style="yellow", justify="right")

    for i, (fname, info) in enumerate(files.items(), 1):
        table.add_row(str(i), fname, str(info["chunks"]), str(len(info["pages"])))

    console.print(table)


def show_chunks(max_chunks: int = 20):
    """Show all stored chunks with their text content."""
    store = get_vector_store()
    collection = store._collection

    all_data = collection.get(include=["metadatas", "documents"])

    if not all_data["ids"]:
        console.print("[yellow]No chunks found. Upload a PDF first.[/yellow]")
        return

    total = len(all_data["ids"])
    showing = min(total, max_chunks)

    console.print(f"\n[bold]🔪 Showing {showing} of {total} chunks[/bold]\n")

    for i in range(showing):
        chunk_id = all_data["ids"][i]
        text = all_data["documents"][i]
        meta = all_data["metadatas"][i]

        source = meta.get("source_file", "?")
        page = meta.get("page_number", "?")
        chunk_idx = meta.get("chunk_index", "?")
        size = len(text)

        header = (
            f"Chunk #{chunk_idx} │ "
            f"📄 {source} │ "
            f"📃 Page {page} │ "
            f"📏 {size} chars"
        )

        # Color-code by size
        if size < 200:
            border_color = "red"      # Very small — might be fragmented
        elif size < 500:
            border_color = "yellow"   # Small
        elif size < 1500:
            border_color = "green"    # Good size
        else:
            border_color = "blue"     # Large

        console.print(Panel(
            text[:500] + ("\n\n[dim]... (truncated)[/dim]" if len(text) > 500 else ""),
            title=header,
            border_style=border_color,
            padding=(0, 1),
        ))


def show_vectors(max_vectors: int = 5):
    """Show the actual embedding vectors — the numbers that power semantic search."""
    store = get_vector_store()
    collection = store._collection

    # Get chunks WITH their embeddings
    all_data = collection.get(include=["metadatas", "documents", "embeddings"])

    if not all_data["ids"]:
        console.print("[yellow]No vectors found. Upload a PDF first.[/yellow]")
        return

    total = len(all_data["ids"])
    showing = min(total, max_vectors)

    console.print(f"\n[bold]🧮 Showing {showing} of {total} embedding vectors[/bold]")
    console.print(f"[dim]Each vector is a list of numbers that captures the MEANING of the text.[/dim]\n")

    for i in range(showing):
        text = all_data["documents"][i]
        vector = all_data["embeddings"][i]
        meta = all_data["metadatas"][i]

        dims = len(vector)

        # Show text preview
        console.print(f"[cyan bold]── Chunk #{meta.get('chunk_index', i)} ──[/cyan bold]")
        console.print(f"[dim]Text:[/dim] \"{text[:100]}...\"")
        console.print(f"[dim]Vector dimensions:[/dim] {dims}")

        # Show first 10 and last 5 values
        first_vals = [f"{v:+.4f}" for v in vector[:10]]
        last_vals = [f"{v:+.4f}" for v in vector[-5:]]

        console.print(f"[green]First 10 values:[/green] [{', '.join(first_vals)}, ...]")
        console.print(f"[green]Last  5 values:[/green]  [..., {', '.join(last_vals)}]")

        # Show vector stats
        vec_min = min(vector)
        vec_max = max(vector)
        vec_mean = sum(vector) / len(vector)
        vec_magnitude = math.sqrt(sum(v ** 2 for v in vector))

        table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
        table.add_column("Stat", style="bold")
        table.add_column("Value", style="yellow")
        table.add_row("Min value", f"{vec_min:.4f}")
        table.add_row("Max value", f"{vec_max:.4f}")
        table.add_row("Mean value", f"{vec_mean:.6f}")
        table.add_row("Magnitude", f"{vec_magnitude:.4f}")
        console.print(table)
        console.print()


def show_search(query: str, top_k: int = 5):
    """Show how search works — embed the query and find similar chunks."""
    store = get_vector_store()
    collection = store._collection

    console.print(f'\n[bold]🔍 Searching for: "[cyan]{query}[/cyan]"[/bold]\n')

    # Step 1: Show the query embedding
    console.print("[yellow]Step 1: Converting your question to a vector...[/yellow]")
    query_vector = embed_query(query)
    first_vals = [f"{v:+.4f}" for v in query_vector[:8]]
    console.print(f"  Query vector ({len(query_vector)} dims): [{', '.join(first_vals)}, ...]\n")

    # Step 2: Search
    console.print("[yellow]Step 2: Finding closest vectors in the database...[/yellow]\n")

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances", "embeddings"],
    )

    if not results["ids"][0]:
        console.print("[red]No results found.[/red]")
        return

    # Step 3: Show results with similarity scores
    table = Table(title="📋 Search Results (lower distance = more relevant)", box=box.ROUNDED)
    table.add_column("Rank", style="bold", justify="center")
    table.add_column("Distance", style="magenta", justify="right")
    table.add_column("Similarity", justify="right")
    table.add_column("Source", style="cyan")
    table.add_column("Page", justify="center")
    table.add_column("Text Preview", max_width=50)

    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        meta = results["metadatas"][0][i]
        text = results["documents"][0][i]

        # Convert L2 distance to a rough similarity percentage
        # L2 distance: 0 = identical, higher = less similar
        similarity = max(0, (1 - distance / 2)) * 100

        # Color code
        if similarity > 70:
            sim_style = "[bold green]"
        elif similarity > 40:
            sim_style = "[yellow]"
        else:
            sim_style = "[red]"

        table.add_row(
            f"#{i+1}",
            f"{distance:.4f}",
            f"{sim_style}{similarity:.1f}%[/]",
            meta.get("source_file", "?"),
            str(meta.get("page_number", "?")),
            text[:80].replace("\n", " ") + "...",
        )

    console.print(table)

    # Show the top result in full
    console.print(Panel(
        results["documents"][0][0],
        title="🏆 Top Result (Full Text)",
        border_style="green",
        padding=(1, 2),
    ))


def compare_all(query: str):
    """Compare the query against ALL chunks to see the full similarity landscape."""
    store = get_vector_store()
    collection = store._collection

    count = collection.count()
    if count == 0:
        console.print("[yellow]No chunks found. Upload a PDF first.[/yellow]")
        return

    console.print(f'\n[bold]📐 Comparing "[cyan]{query}[/cyan]" against all {count} chunks[/bold]\n')

    # Get query vector
    query_vector = embed_query(query)

    # Search all
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=count,
        include=["documents", "metadatas", "distances"],
    )

    # Build a visual bar chart
    console.print("[bold]Similarity Scores (higher bar = more relevant):[/bold]\n")

    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        meta = results["metadatas"][0][i]
        text = results["documents"][0][i]

        similarity = max(0, (1 - distance / 2)) * 100
        bar_length = int(similarity / 2)  # Max 50 chars

        # Color the bar
        if similarity > 70:
            color = "green"
        elif similarity > 40:
            color = "yellow"
        else:
            color = "red"

        bar = "█" * bar_length + "░" * (50 - bar_length)
        chunk_idx = meta.get("chunk_index", i)
        page = meta.get("page_number", "?")
        preview = text[:40].replace("\n", " ")

        console.print(
            f"  Chunk {chunk_idx:>3} (p{page}) [{color}]{bar}[/{color}] {similarity:.1f}%  [dim]{preview}...[/dim]"
        )

    console.print(f"\n[dim]Top-K=5 would retrieve the top 5 bars above.[/dim]")


def main():
    validate_config()

    if len(sys.argv) < 2:
        console.print("""
[bold cyan]🔬 RAG Inspector — Usage:[/bold cyan]

  [green]python inspect_rag.py stats[/green]               Overview of your database
  [green]python inspect_rag.py chunks[/green]              See all stored text chunks
  [green]python inspect_rag.py vectors[/green]             See actual embedding vectors (numbers!)
  [green]python inspect_rag.py search "query"[/green]      See how search finds relevant chunks
  [green]python inspect_rag.py compare "query"[/green]     Compare similarity across ALL chunks
        """)
        return

    command = sys.argv[1].lower()

    if command == "stats":
        show_stats()
    elif command == "chunks":
        show_chunks()
    elif command == "vectors":
        show_vectors()
    elif command == "search":
        if len(sys.argv) < 3:
            console.print('[red]Usage: python inspect_rag.py search "your question"[/red]')
            return
        show_search(" ".join(sys.argv[2:]))
    elif command == "compare":
        if len(sys.argv) < 3:
            console.print('[red]Usage: python inspect_rag.py compare "your question"[/red]')
            return
        compare_all(" ".join(sys.argv[2:]))
    else:
        console.print(f"[red]Unknown command: {command}[/red]")


if __name__ == "__main__":
    main()
