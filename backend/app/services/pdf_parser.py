"""
📄 PDF PARSER — Extracts text from PDF files.

WHAT YOU'RE LEARNING:
    - PyMuPDF (imported as 'fitz') is the fastest Python PDF library
    - PDFs are NOT plain text files — they store text as positioned glyphs
    - Extraction quality varies by PDF type:
        • Digital PDFs (created from Word/LaTeX): Perfect text extraction
        • Scanned PDFs (photos of paper): Need OCR (not covered in Phase 1)
        • Mixed PDFs: Some pages digital, some scanned

WHY PyMuPDF OVER PyPDF2:
    - 10x faster for large documents
    - Better handling of multi-column layouts
    - Extracts text in correct reading order
    - Can also extract images, tables, and metadata
    
HOW IT WORKS:
    1. Open the PDF file
    2. Iterate through each page
    3. Extract text from each page
    4. Return structured data with page numbers (for source citations later!)
"""

import pymupdf  # PyMuPDF — previously imported as 'fitz', now uses its own name
from pathlib import Path
from dataclasses import dataclass


@dataclass
class PageContent:
    """
    Represents extracted text from a single PDF page.
    
    We keep page numbers so that when the LLM answers a question,
    we can tell the user "this answer came from page 5" — a crucial
    feature for document Q&A trustworthiness.
    """
    text: str
    page_number: int  # 1-indexed (humans count from 1)
    source_file: str  # Original filename


@dataclass 
class ParsedDocument:
    """
    The complete result of parsing a PDF.
    Contains all pages plus metadata about the document.
    """
    filename: str
    total_pages: int
    pages: list[PageContent]
    total_characters: int


def parse_pdf(file_path: str | Path) -> ParsedDocument:
    """
    Extract all text from a PDF file, page by page.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        ParsedDocument with all pages and metadata
        
    Raises:
        FileNotFoundError: If the PDF doesn't exist
        RuntimeError: If the PDF can't be parsed
        
    EXPERIMENT:
        Try this with different PDFs and notice:
        - Academic papers extract cleanly (they're digital)
        - Scanned receipts might return empty strings (they need OCR)
        - Two-column papers might interleave columns (layout detection is hard!)
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")
    
    if not file_path.suffix.lower() == ".pdf":
        raise ValueError(f"Not a PDF file: {file_path}")
    
    pages = []
    
    # fitz.open() loads the entire PDF into memory
    # For very large PDFs (1000+ pages), you might want streaming — but this works for 99% of use cases
    try:
        doc = pymupdf.open(str(file_path))
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF '{file_path.name}': {e}")
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # get_text() extracts text in reading order
        # Other options: get_text("html"), get_text("dict") for structured data
        text = page.get_text()
        
        # Clean up common PDF extraction artifacts
        # - Multiple spaces from column layouts
        # - Form feed characters
        # - Excessive newlines
        text = text.strip()
        
        if text:  # Skip completely empty pages (common in scanned PDFs)
            pages.append(PageContent(
                text=text,
                page_number=page_num + 1,  # Convert to 1-indexed
                source_file=file_path.name
            ))
    
    total_pages = len(doc)  # Save before closing!
    doc.close()
    
    total_chars = sum(len(p.text) for p in pages)
    
    return ParsedDocument(
        filename=file_path.name,
        total_pages=total_pages,
        pages=pages,
        total_characters=total_chars
    )


def get_full_text(parsed_doc: ParsedDocument) -> str:
    """
    Combine all pages into a single string.
    Useful for simple chunking (Phase 1).
    
    We insert page markers so the chunker can preserve
    page number metadata even after splitting.
    """
    sections = []
    for page in parsed_doc.pages:
        sections.append(f"[Page {page.page_number}]\n{page.text}")
    return "\n\n".join(sections)


# ============================================================
# 🧪 Quick test — run this file directly to test PDF parsing
# Usage: python -m app.services.pdf_parser path/to/your.pdf
# ============================================================
if __name__ == "__main__":
    import sys
    from rich.console import Console
    from rich.table import Table
    
    console = Console()
    
    if len(sys.argv) < 2:
        console.print("[red]Usage: python -m app.services.pdf_parser <path-to-pdf>[/red]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    console.print(f"\n📄 Parsing: [cyan]{pdf_path}[/cyan]\n")
    
    result = parse_pdf(pdf_path)
    
    # Display results in a nice table
    table = Table(title=f"📊 {result.filename}")
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="green")
    table.add_row("Total Pages", str(result.total_pages))
    table.add_row("Pages with Text", str(len(result.pages)))
    table.add_row("Total Characters", f"{result.total_characters:,}")
    console.print(table)
    
    # Show first 500 chars of each page
    console.print("\n[bold]📝 Preview (first 300 chars per page):[/bold]")
    for page in result.pages[:5]:  # Show first 5 pages
        console.print(f"\n[yellow]--- Page {page.page_number} ---[/yellow]")
        console.print(page.text[:300] + ("..." if len(page.text) > 300 else ""))
