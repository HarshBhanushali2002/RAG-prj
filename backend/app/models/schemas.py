"""
📋 SCHEMAS — Pydantic models for API request/response validation.

WHAT YOU'RE LEARNING:
    Pydantic models serve two purposes in FastAPI:
    
    1. INPUT VALIDATION:
       - Automatically validates request bodies (wrong type? missing field? → 422 error)
       - Converts types (string "5" → int 5)
       - You never write manual validation code!
    
    2. RESPONSE SERIALIZATION:
       - Defines the exact shape of API responses
       - Auto-generates OpenAPI/Swagger documentation
       - Frontend devs can see the exact JSON structure without reading your code

WHY THIS MATTERS:
    Without Pydantic:
        @app.post("/query")
        def query(request):
            question = request.get("question")  # What if it's missing? What if it's an int?
            # Manual validation... error-prone, boring, repetitive
    
    With Pydantic:
        @app.post("/query")
        def query(request: QueryRequest):
            # request.question is guaranteed to be a non-empty string!
            # FastAPI already sent a 422 if it wasn't.
"""

from pydantic import BaseModel, Field
from datetime import datetime


# ============================================================
# REQUEST MODELS — What the client sends to us
# ============================================================

class QueryRequest(BaseModel):
    """
    Request body for POST /query.
    
    The user sends a question and optionally how many chunks to retrieve.
    """
    question: str = Field(
        ...,  # ... means required
        min_length=1,
        max_length=2000,
        description="The question to ask about uploaded documents",
        examples=["What is the main topic of this document?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,  # greater than or equal to 1
        le=20,  # less than or equal to 20
        description="Number of relevant chunks to retrieve (1-20)",
    )


# ============================================================
# RESPONSE MODELS — What we send back to the client
# ============================================================

class SourceChunk(BaseModel):
    """
    A single source chunk that contributed to the answer.
    
    This is crucial for RAG transparency — the user can verify
    that the answer actually comes from their documents.
    """
    text: str = Field(description="Preview of the source text (first 300 chars)")
    source_file: str = Field(description="Filename of the source document")
    page_number: int | str = Field(description="Page number in the original document")
    relevance_score: float | None = Field(
        default=None,
        description="Similarity score (lower = more relevant for L2 distance)",
    )


class QueryResponse(BaseModel):
    """
    Response body for POST /query.
    
    Contains the LLM's answer plus the source chunks for verification.
    """
    answer: str = Field(description="The LLM-generated answer based on document context")
    question: str = Field(description="The original question (echoed back)")
    sources: list[SourceChunk] = Field(
        default_factory=list,
        description="Source chunks used to generate the answer",
    )
    num_sources: int = Field(description="Number of source chunks retrieved")


class UploadResponse(BaseModel):
    """
    Response body for POST /upload.
    
    Reports what happened during the ingestion pipeline.
    """
    filename: str = Field(description="Name of the uploaded file")
    total_pages: int = Field(description="Number of pages in the PDF")
    total_chunks: int = Field(description="Number of chunks created")
    total_characters: int = Field(description="Total characters extracted")
    message: str = Field(description="Human-readable status message")


class DocumentInfo(BaseModel):
    """
    Information about a single indexed document.
    """
    filename: str = Field(description="Name of the document file")
    chunk_count: int = Field(description="Number of chunks stored in the vector database")
    total_pages: int = Field(description="Number of pages in the document")
    pages: list[int] = Field(
        default_factory=list,
        description="List of page numbers that have indexed content",
    )


class DocumentListResponse(BaseModel):
    """
    Response body for GET /documents.
    """
    documents: list[DocumentInfo] = Field(
        default_factory=list,
        description="List of all indexed documents",
    )
    total_documents: int = Field(description="Total number of unique documents")
    total_chunks: int = Field(description="Total number of chunks across all documents")


class DeleteResponse(BaseModel):
    """
    Response body for DELETE /documents/{filename}.
    """
    filename: str = Field(description="Name of the deleted document")
    chunks_deleted: int = Field(description="Number of chunks removed from the vector database")
    message: str = Field(description="Human-readable status message")


class HealthResponse(BaseModel):
    """
    Response body for GET /health.
    """
    status: str = Field(description="Service status", examples=["healthy"])
    version: str = Field(description="API version")
    total_documents: int = Field(description="Number of indexed documents")
    total_chunks: int = Field(description="Total chunks in vector store")


class ErrorResponse(BaseModel):
    """
    Standard error response body.
    
    FastAPI uses this for consistent error formatting.
    """
    detail: str = Field(description="Human-readable error description")
