"""
SCHEMAS - Pydantic models for API request/response validation.

Pydantic models serve two purposes in FastAPI:
  1. INPUT VALIDATION  - wrong type or missing field returns HTTP 422 automatically
  2. RESPONSE SCHEMA   - defines the exact JSON shape and auto-generates Swagger docs
"""

from pydantic import BaseModel, Field
from typing import Optional


# REQUEST MODELS ---------------------------------------------

class QueryRequest(BaseModel):
    """Request body for POST /query."""
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The question to ask about uploaded documents",
        examples=["What is the main topic of this document?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of relevant chunks to retrieve (1-20)",
    )


# RESPONSE MODELS --------------------------------------------

class SourceChunk(BaseModel):
    """A single source chunk that contributed to the answer."""
    text: str = Field(description="Preview of the source text")
    source_file: str = Field(description="Filename of the source document")
    page_number: int | str = Field(description="Page number in the original document")
    relevance_score: Optional[float] = Field(
        default=None,
        description="Similarity score (lower = more relevant for L2 distance)",
    )


class QueryResponse(BaseModel):
    """Response body for POST /query."""
    answer: str = Field(description="The LLM-generated answer")
    question: str = Field(description="The original question (echoed back)")
    sources: list[SourceChunk] = Field(default_factory=list)
    num_sources: int = Field(description="Number of source chunks retrieved")


class UploadResponse(BaseModel):
    """Response body for POST /upload."""
    filename: str = Field(description="Name of the uploaded file")
    total_pages: int = Field(description="Number of pages in the PDF")
    total_chunks: int = Field(description="Number of chunks created")
    total_characters: int = Field(description="Total characters extracted")
    message: str = Field(description="Human-readable status message")


class DocumentInfo(BaseModel):
    """Information about a single indexed document."""
    filename: str
    chunk_count: int
    total_pages: int
    pages: list[int] = Field(default_factory=list)


class DocumentListResponse(BaseModel):
    """Response body for GET /documents."""
    documents: list[DocumentInfo] = Field(default_factory=list)
    total_documents: int
    total_chunks: int


class DeleteResponse(BaseModel):
    """Response body for DELETE /documents/{filename}."""
    filename: str
    chunks_deleted: int
    message: str


class HealthResponse(BaseModel):
    """Response body for GET /health."""
    status: str = Field(description="Service status", examples=["healthy"])
    version: str
    total_documents: int
    total_chunks: int


class ErrorResponse(BaseModel):
    """Standard error response body."""
    detail: str = Field(description="Human-readable error description")
