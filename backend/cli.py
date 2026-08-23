"""
🚀 RAG Document Q&A — CLI Interface

This is your first working RAG application!
Run it, upload a PDF, and ask questions about it.

Usage:
    python cli.py

Commands in the interactive loop:
    /upload <path>  — Upload and index a PDF file
    /list           — Show all indexed documents
    /delete <name>  — Delete a document from the index
    /clear          — Clear the screen
    /help           — Show available commands
    /quit           — Exit the program
    
    Or just type a question to query your documents!

WHAT YOU'RE SEEING IN ACTION:
    When you type a question, the CLI:
    1. Embeds your question → you see "Searching..."
    2. Retrieves similar chunks → you see the source documents
    3. Sends context + question to Gemini → you see the answer
    4. Shows sources → you can verify the answer!
"""

import sys
import os

# Add the backend directory to Python path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown

from app.config import validate_config, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
from app.services.pdf_parser import parse_pdf
from app.services.chunker import chunk_document, preview_chunks
from app.services.vector_store import (
    add_chunks_to_store,
    get_document_count,
    list_documents,
    delete_document,
)
from app.services.qa_chain import ask_question

console = Console()


def print_banner():
    """Show the welcome banner."""
    banner = """
╔══════════════════════════════════════════════════════════╗
║           📚 RAG Document Q&A — CLI v1.0               ║
║                                                          ║
║   Upload PDFs and ask questions in plain English!        ║
║   Type /help for commands or just ask a question.        ║
╚══════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold cyan")
    
    # Show current settings
    table = Table(title="⚙️ Current Settings", show_header=False, border_style="dim")
    table.add_column("Setting", style="bold")
    table.add_column("Value", style="green")
    table.add_row("Chunk Size", f"{CHUNK_SIZE} characters")
    table.add_row("Chunk Overlap", f"{CHUNK_OVERLAP} characters")
    table.add_row("Top-K Results", str(TOP_K))
    table.add_row("Indexed Chunks", str(get_document_count()))
    console.print(table)
    console.print()


def handle_upload(file_path: str):
    """Upload and index a PDF file."""
    file_path = file_path.strip().strip('"').strip("'")
    
    if not file_path:
        console.print("[red]Usage: /upload <path-to-pdf>[/red]")
        return
    
    if not os.path.exists(file_path):
        console.print(f"[red]❌ File not found: {file_path}[/red]")
        return
    
    if not file_path.lower().endswith(".pdf"):
        console.print("[red]❌ Only PDF files are supported (for now!)[/red]")
        return
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Step 1: Parse PDF
        task = progress.add_task("📄 Extracting text from PDF...", total=None)
        parsed = parse_pdf(file_path)
        progress.update(task, description=f"📄 Extracted {parsed.total_characters:,} chars from {len(parsed.pages)} pages")
        progress.remove_task(task)
        
        # Step 2: Chunk
        task = progress.add_task("🔪 Splitting into chunks...", total=None)
        chunks = chunk_document(parsed)
        progress.update(task, description=f"🔪 Created {len(chunks)} chunks")
        progress.remove_task(task)
        
        # Step 3: Embed and store
        task = progress.add_task("🧮 Generating embeddings & storing...", total=None)
        count = add_chunks_to_store(chunks)
        progress.update(task, description=f"💾 Stored {count} chunks in vector database")
        progress.remove_task(task)
    
    console.print(f"\n[bold green]✅ Successfully indexed '{parsed.filename}'![/bold green]")
    console.print(f"   {len(parsed.pages)} pages → {len(chunks)} chunks → {count} vectors stored")
    console.print(f"   Total indexed chunks: {get_document_count()}\n")
    
    # Show a preview of the chunks
    preview_chunks(chunks, max_display=3)


def handle_list():
    """List all indexed documents."""
    docs = list_documents()
    
    if not docs:
        console.print("[yellow]No documents indexed yet. Use /upload <path> to add one.[/yellow]")
        return
    
    table = Table(title="📚 Indexed Documents")
    table.add_column("#", style="dim")
    table.add_column("Filename", style="cyan bold")
    table.add_column("Chunks", style="green")
    table.add_column("Pages", style="yellow")
    
    for i, doc in enumerate(docs, 1):
        table.add_row(
            str(i),
            doc["filename"],
            str(doc["chunk_count"]),
            str(doc["total_pages"]),
        )
    
    console.print(table)
    console.print(f"\nTotal vectors in database: {get_document_count()}\n")


def handle_delete(filename: str):
    """Delete a document from the index."""
    filename = filename.strip()
    
    if not filename:
        console.print("[red]Usage: /delete <filename>[/red]")
        return
    
    count = delete_document(filename)
    
    if count > 0:
        console.print(f"[green]✅ Deleted {count} chunks for '{filename}'[/green]")
    else:
        console.print(f"[yellow]No document found with name '{filename}'[/yellow]")


def handle_question(question: str):
    """Ask a question about the indexed documents."""
    if get_document_count() == 0:
        console.print("[yellow]No documents indexed yet. Use /upload <path> to add a PDF first.[/yellow]")
        return
    
    console.print()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("🔍 Searching documents & generating answer...", total=None)
        result = ask_question(question)
        progress.remove_task(task)
    
    # Display the answer
    console.print(Panel(
        Markdown(result["answer"]),
        title="💬 Answer",
        border_style="green",
        padding=(1, 2),
    ))
    
    # Display sources
    if result["sources"]:
        console.print("\n[bold]📎 Sources used:[/bold]")
        for i, source in enumerate(result["sources"], 1):
            score_str = f" (relevance: {source['relevance_score']:.4f})" if source.get("relevance_score") else ""
            console.print(
                f"  {i}. [cyan]{source['source_file']}[/cyan], "
                f"Page {source['page_number']}{score_str}"
            )
            # Show a short preview of the source text
            preview = source["text"][:150].replace("\n", " ")
            console.print(f"     [dim]{preview}...[/dim]")
        console.print()


def handle_help():
    """Show available commands."""
    help_text = """
[bold cyan]Available Commands:[/bold cyan]

  [green]/upload <path>[/green]    Upload and index a PDF file
  [green]/list[/green]             Show all indexed documents
  [green]/delete <name>[/green]    Delete a document from the index
  [green]/clear[/green]            Clear the screen
  [green]/help[/green]             Show this help message
  [green]/quit[/green]             Exit the program

[bold cyan]Or just type a question![/bold cyan]
  Example: "What is the main topic of the document?"
  Example: "Summarize the key findings on page 3"
  Example: "What does the report say about revenue growth?"
    """
    console.print(help_text)


def main():
    """Main interactive loop."""
    # Validate configuration before starting
    validate_config()
    
    print_banner()
    
    while True:
        try:
            # Read user input
            user_input = console.input("[bold green]❯ [/bold green]").strip()
            
            if not user_input:
                continue
            
            # Parse commands
            if user_input.startswith("/"):
                parts = user_input.split(" ", 1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
                
                if command in ("/quit", "/exit", "/q"):
                    console.print("\n[cyan]👋 Goodbye![/cyan]")
                    break
                elif command == "/upload":
                    handle_upload(args)
                elif command == "/list":
                    handle_list()
                elif command == "/delete":
                    handle_delete(args)
                elif command == "/clear":
                    os.system("cls" if os.name == "nt" else "clear")
                    print_banner()
                elif command == "/help":
                    handle_help()
                else:
                    console.print(f"[red]Unknown command: {command}. Type /help for options.[/red]")
            else:
                # It's a question — run the RAG pipeline!
                handle_question(user_input)
                
        except KeyboardInterrupt:
            console.print("\n\n[cyan]👋 Goodbye! (Ctrl+C)[/cyan]")
            break
        except Exception as e:
            console.print(f"\n[red]❌ Error: {e}[/red]")
            console.print("[dim]Try again or type /help for options.[/dim]\n")


if __name__ == "__main__":
    main()
