# Route: POST /upload
# Pipeline: PDF -> parse -> chunk -> embed -> ChromaDB
"""
📤 UPLOAD ROUTES — Endpoints for document management.

WHAT YOU'RE LEARNING:
    These endpoints handle the INGESTION side of RAG:
    
    POST /upload     → Upload a PDF → parse → chunk → embed → store
    GET /documents   → List all indexed documents
    DELETE /documents/{filename} → Remove a document and all its vectors

    FastAPI concepts you'll see here:
    - UploadFile: FastAPI's file upload handler (wraps multipart form data)
    - APIRouter: Groups related endpoints (like Flask Blueprints)
    - HTTPException: Proper HTTP error responses (404, 400, 500)
    - BackgroundTasks: Run slow work after responding (not used yet, but could be)

WHY FILE UPLOADS ARE SPECIAL:
    - Regular POST endpoints accept JSON → request body parsed as dict
    - File uploads use multipart/form-data → request body is a stream of bytes
    - FastAPI handles this with UploadFile, which gives you:
      * .filename → original name of the file
      * .read()   → the raw bytes
      * .content_type → MIME type (e.g., "application/pdf")
"""

import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.config import UPLOAD_DIR
from app.models.schemas import (
    UploadResponse,
    DocumentListResponse,
    DocumentInfo,
    DeleteResponse,
)
from app.services.pdf_parser import parse_pdf
from app.services.chunker import chunk_document
from app.services.vector_store import (
    add_chunks_to_store,
    list_documents,
    delete_document,
    get_document_count,
)


# APIRouter groups related endpoints under a shared prefix/tag.
# Tags show up as sections in the Swagger docs — very helpful for organization.
router = APIRouter(tags=["Documents"])


@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload and index a PDF document",
    responses={
        400: {"description": "Invalid file type (only PDFs allowed)"},
        500: {"description": "Error during processing"},
    },
)
async def upload_document(file: UploadFile = File(..., description="PDF file to upload")):
    """
    Upload a PDF file and run the full ingestion pipeline:
    
    1. **Save** the file to the uploads directory
    2. **Parse** the PDF to extract text from each page
    3. **Chunk** the text into smaller pieces for embedding
    4. **Embed & Store** each chunk in the vector database
    
    After this endpoint returns, the document is ready for Q&A queries.
    
    **Supported formats:** PDF only (more formats coming in Phase 5)
    """
    # --- Validate file type ---
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported. Please upload a .pdf file.",
        )

    # --- Save the uploaded file ---
    # We save it to disk so we can:
    # 1. Re-process it later if we change chunking settings
    # 2. Serve it back to the frontend for preview
    # 3. Keep a record of what was uploaded
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        with open(save_path, "wb") as buffer:
            # Read the uploaded file in chunks to handle large files
            # without loading everything into memory at once
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save file: {str(e)}",
        )
    finally:
        # Always close the upload file handle
        await file.close()

    # --- Run the ingestion pipeline ---
    # This is the same pipeline from cli.py, but wrapped in an API endpoint
    try:
        # Step 1: Parse PDF → extract text from each page
        parsed_doc = parse_pdf(save_path)

        if not parsed_doc.pages:
            raise HTTPException(
                status_code=400,
                detail="The PDF appears to be empty or contains only images (no extractable text).",
            )

        # Step 2: Chunk → split text into embedding-sized pieces
        chunks = chunk_document(parsed_doc)

        # Step 3: Embed & Store → generate embeddings and save to ChromaDB
        stored_count = add_chunks_to_store(chunks)

        return UploadResponse(
            filename=parsed_doc.filename,
            total_pages=parsed_doc.total_pages,
            total_chunks=stored_count,
            total_characters=parsed_doc.total_characters,
            message=f"Successfully indexed '{parsed_doc.filename}': "
                    f"{parsed_doc.total_pages} pages → {stored_count} chunks stored.",
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Clean up the saved file if processing failed
        if os.path.exists(save_path):
            os.remove(save_path)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process PDF: {str(e)}",
        )


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List all indexed documents",
)
async def get_documents():
    """
    Returns a list of all documents that have been uploaded and indexed.
    
    For each document, shows:
    - **filename**: The original file name
    - **chunk_count**: How many chunks are stored in the vector database
    - **total_pages**: Number of pages in the document
    - **pages**: Which specific pages have indexed content
    
    Use this to see what's available for Q&A queries.
    """
    docs = list_documents()
    total_chunks = get_document_count()

    return DocumentListResponse(
        documents=[
            DocumentInfo(
                filename=doc["filename"],
                chunk_count=doc["chunk_count"],
                total_pages=doc["total_pages"],
                pages=doc.get("pages", []),
            )
            for doc in docs
        ],
        total_documents=len(docs),
        total_chunks=total_chunks,
    )


@router.delete(
    "/documents/{filename}",
    response_model=DeleteResponse,
    summary="Delete a document and its vectors",
    responses={
        404: {"description": "Document not found in vector store"},
    },
)
async def remove_document(filename: str):
    """
    Remove a document from the vector database and optionally from disk.
    
    This deletes:
    - All embedding vectors for this document from ChromaDB
    - The uploaded PDF file from the uploads directory
    
    After deletion, the document's content will no longer appear in Q&A results.
    """
    # Delete vectors from ChromaDB
    chunks_deleted = delete_document(filename)

    if chunks_deleted == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No document found with filename '{filename}'. "
                   f"Use GET /documents to see available documents.",
        )

    # Also remove the file from disk (best-effort, don't fail if already gone)
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    return DeleteResponse(
        filename=filename,
        chunks_deleted=chunks_deleted,
        message=f"Deleted {chunks_deleted} chunks for '{filename}'.",
    )

