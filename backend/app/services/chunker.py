"""
🔪 CHUNKER — Splits documents into smaller pieces for embedding.

WHAT YOU'RE LEARNING:
    - LLMs have context windows (input size limits)
    - Embeddings work best on focused, coherent text passages
    - You can't embed an entire 100-page PDF as one vector — it would lose meaning
    - So we split documents into "chunks" and embed each one separately

WHY CHUNKING STRATEGY MATTERS (this is one of the MOST important RAG decisions):

    Imagine searching for "What are the side effects of Drug X?"
    
    ❌ Bad chunk (too big, 5000 chars):
        Contains info about Drug X, Drug Y, Drug Z, company history...
        → The embedding represents ALL of that, so it matches many queries poorly
    
    ❌ Bad chunk (too small, 100 chars):
        "effects include nausea and headache. Patients should"
        → Cut mid-sentence! Missing the drug name entirely.
    
    ✅ Good chunk (500-1000 chars):
        "Side Effects of Drug X: Common side effects include nausea, 
        headache, and dizziness. In rare cases, patients may experience..."
        → Complete thought, focused topic, searchable.

THE SPLITTER WE USE — RecursiveCharacterTextSplitter:
    - Tries to split at paragraph breaks first (\\n\\n)
    - If chunks are still too big, splits at sentence boundaries (. ? !)
    - If still too big, splits at word boundaries (spaces)
    - Last resort: splits at character level
    - This hierarchy preserves meaning at every level
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.services.pdf_parser import ParsedDocument
from app.config import CHUNK_SIZE, CHUNK_OVERLAP
from dataclasses import dataclass


@dataclass
class Chunk:
    """
    A single chunk of text ready to be embedded.
    
    Metadata is CRITICAL — without it, you know the answer came from
    "somewhere in the document" but can't tell the user WHERE.
    """
    text: str
    metadata: dict  # source_file, page_number, chunk_index, etc.


def chunk_document(
    parsed_doc: ParsedDocument,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """
    Split a parsed document into chunks with metadata.
    
    Args:
        parsed_doc: Output from pdf_parser.parse_pdf()
        chunk_size: Override the default chunk size (for experimentation!)
        chunk_overlap: Override the default overlap
        
    Returns:
        List of Chunk objects with text and metadata
        
    EXPERIMENT:
        Call this with different chunk_size values and compare:
        - chunk_size=200:  Many small chunks, very specific but fragmented
        - chunk_size=500:  Good balance for most docs
        - chunk_size=1000: Fewer, more complete chunks
        - chunk_size=2000: Very few chunks, might mix topics
        
        Then query the same question and see how answer quality changes!
    """
    size = chunk_size or CHUNK_SIZE
    overlap = chunk_overlap or CHUNK_OVERLAP
    
    # RecursiveCharacterTextSplitter — the workhorse of RAG chunking
    # 
    # 'separators' defines the splitting hierarchy:
    #   1. "\n\n" — paragraph breaks (best split point)
    #   2. "\n"   — line breaks
    #   3. ". "   — sentence boundaries
    #   4. " "    — word boundaries
    #   5. ""     — character level (last resort)
    #
    # It tries each separator in order, using the first one that
    # produces chunks within the size limit.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=len,  # Measure chunk size by character count
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        is_separator_regex=False,
    )
    
    chunks = []
    chunk_index = 0
    
    for page in parsed_doc.pages:
        # Split this page's text into chunks
        page_chunks = splitter.split_text(page.text)
        
        for text in page_chunks:
            chunks.append(Chunk(
                text=text,
                metadata={
                    "source_file": parsed_doc.filename,
                    "page_number": page.page_number,
                    "chunk_index": chunk_index,
                    "chunk_size": len(text),
                    # These help with debugging retrieval quality
                    "total_chunks": -1,  # Will be updated after processing
                }
            ))
            chunk_index += 1
    
    # Update total_chunks count in all metadata
    for chunk in chunks:
        chunk.metadata["total_chunks"] = len(chunks)
    
    return chunks


def preview_chunks(chunks: list[Chunk], max_display: int = 5) -> None:
    """
    Print chunks nicely for debugging.
    This is your primary tool for understanding chunking quality.
    """
    from rich.console import Console
    from rich.panel import Panel
    
    console = Console()
    
    console.print(f"\n[bold green]📦 Total chunks: {len(chunks)}[/bold green]")
    
    if chunks:
        sizes = [len(c.text) for c in chunks]
        console.print(f"   Avg size: {sum(sizes) // len(sizes)} chars")
        console.print(f"   Min size: {min(sizes)} chars")
        console.print(f"   Max size: {max(sizes)} chars")
    
    console.print(f"\n[bold]Showing first {min(max_display, len(chunks))} chunks:[/bold]")
    
    for i, chunk in enumerate(chunks[:max_display]):
        header = (
            f"Chunk {i} | "
            f"Page {chunk.metadata['page_number']} | "
            f"{len(chunk.text)} chars"
        )
        # Show first 200 chars of each chunk
        preview = chunk.text[:200] + ("..." if len(chunk.text) > 200 else "")
        console.print(Panel(preview, title=header, border_style="cyan"))


# ============================================================
# 🧪 Quick test — parse a PDF and chunk it
# Usage: python -m app.services.chunker path/to/your.pdf
# ============================================================
if __name__ == "__main__":
    import sys
    from rich.console import Console
    from app.services.pdf_parser import parse_pdf
    
    console = Console()
    
    if len(sys.argv) < 2:
        console.print("[red]Usage: python -m app.services.chunker <path-to-pdf>[/red]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    # Parse the PDF
    console.print(f"\n📄 Parsing: [cyan]{pdf_path}[/cyan]")
    doc = parse_pdf(pdf_path)
    console.print(f"   Extracted {doc.total_characters:,} characters from {len(doc.pages)} pages")
    
    # Chunk with different sizes to compare
    for size in [200, 500, 1000, 2000]:
        console.print(f"\n{'='*60}")
        console.print(f"[bold yellow]🔪 Chunking with size={size}, overlap={size//5}[/bold yellow]")
        chunks = chunk_document(doc, chunk_size=size, chunk_overlap=size // 5)
        preview_chunks(chunks, max_display=2)
