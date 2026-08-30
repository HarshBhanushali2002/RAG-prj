"""
CHUNKER - Splits documents into smaller pieces for embedding.

Uses RecursiveCharacterTextSplitter which tries to split at:
  1. Paragraph breaks (\n\n)
  2. Line breaks (\n)
  3. Sentence boundaries (. ? !)
  4. Word boundaries (spaces)
  5. Character level (last resort)
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.services.pdf_parser import ParsedDocument
from app.config import CHUNK_SIZE, CHUNK_OVERLAP
from dataclasses import dataclass


@dataclass
class Chunk:
    """A single chunk of text ready to be embedded."""
    text: str
    metadata: dict  # source_file, page_number, chunk_index, etc.


def chunk_document(
    parsed_doc: ParsedDocument,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """
    Split a parsed document into overlapping chunks with metadata.

    Args:
        parsed_doc: Output from pdf_parser.parse_pdf()
        chunk_size: Override default chunk size (chars)
        chunk_overlap: Override default overlap (chars)

    Returns:
        List of Chunk objects with text and metadata
    """
    size = chunk_size or CHUNK_SIZE
    overlap = chunk_overlap or CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        is_separator_regex=False,
    )

    chunks = []
    chunk_index = 0

    for page in parsed_doc.pages:
        page_chunks = splitter.split_text(page.text)

        for text in page_chunks:
            chunks.append(Chunk(
                text=text,
                metadata={
                    "source_file": parsed_doc.filename,
                    "page_number": page.page_number,
                    "chunk_index": chunk_index,
                    "chunk_size": len(text),
                    "total_chunks": -1,
                }
            ))
            chunk_index += 1

    # Backfill total_chunks into every chunk's metadata
    for chunk in chunks:
        chunk.metadata["total_chunks"] = len(chunks)

    return chunks
