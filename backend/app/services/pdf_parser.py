"""
PDF PARSER - Extracts text from PDF files page by page.
Uses PyMuPDF for fast, accurate text extraction.
"""

import pymupdf
from pathlib import Path
from dataclasses import dataclass


@dataclass
class PageContent:
    """Represents extracted text from a single PDF page."""
    text: str
    page_number: int  # 1-indexed
    source_file: str


@dataclass
class ParsedDocument:
    """The complete result of parsing a PDF."""
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
        FileNotFoundError: If the PDF does not exist
        ValueError: If file is not a PDF
        RuntimeError: If the PDF cannot be parsed
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    if not file_path.suffix.lower() == ".pdf":
        raise ValueError(f"Not a PDF file: {file_path}")

    pages = []

    try:
        doc = pymupdf.open(str(file_path))
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF '{file_path.name}': {e}")

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text().strip()

        if text:  # Skip completely empty pages
            pages.append(PageContent(
                text=text,
                page_number=page_num + 1,
                source_file=file_path.name
            ))

    total_pages = len(doc)
    doc.close()
    total_chars = sum(len(p.text) for p in pages)

    return ParsedDocument(
        filename=file_path.name,
        total_pages=total_pages,
        pages=pages,
        total_characters=total_chars
    )


def get_full_text(parsed_doc: ParsedDocument) -> str:
    """Combine all pages into a single string with page markers."""
    sections = []
    for page in parsed_doc.pages:
        sections.append(f"[Page {page.page_number}]\n{page.text}")
    return "\n\n".join(sections)
