# Route: POST /query
# Pipeline: question -> embed -> vector search -> Gemini -> response
"""
❓ QUERY ROUTES — Endpoint for asking questions about uploaded documents.

WHAT YOU'RE LEARNING:
    This is the RETRIEVAL + GENERATION side of RAG, exposed as an API:
    
    POST /query → Takes a question → retrieves relevant chunks → asks the LLM → returns answer
    
    FastAPI concepts you'll see here:
    - Response models: FastAPI uses your Pydantic model to:
      1. Validate the response data matches the schema
      2. Serialize it to JSON
      3. Generate Swagger documentation automatically
    
    - Error handling: We use HTTPException for expected errors (no docs uploaded)
      and let FastAPI's default handler catch unexpected ones (500 Internal Server Error)

WHY A SEPARATE ROUTE FILE:
    Separation of concerns:
    - upload.py handles document management (CRUD)
    - query.py handles question answering (the core RAG feature)
    
    In Phase 4, query.py will grow to support:
    - Streaming responses (SSE)
    - Conversation history
    - Follow-up questions
    
    Keeping it separate means those changes don't touch upload logic.
"""

from fastapi import APIRouter, HTTPException
from app.models.schemas import QueryRequest, QueryResponse, SourceChunk
from app.services.qa_chain import ask_question
from app.services.vector_store import get_document_count


router = APIRouter(tags=["Q&A"])


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question about uploaded documents",
    responses={
        400: {"description": "No documents uploaded yet"},
        500: {"description": "Error during question answering"},
    },
)
async def query_documents(request: QueryRequest):
    """
    Ask a question about your uploaded documents using the RAG pipeline:
    
    1. **Embed** your question into a vector
    2. **Retrieve** the most similar document chunks from ChromaDB
    3. **Augment** the LLM prompt with those chunks as context
    4. **Generate** an answer using Google Gemini
    
    The response includes:
    - **answer**: The LLM's response grounded in your documents
    - **sources**: The exact chunks used, so you can verify the answer
    - **num_sources**: How many chunks were retrieved
    
    **Tip:** Adjust `top_k` (1-20) to control how many chunks are retrieved.
    Lower values give more focused answers; higher values give more comprehensive ones.
    """
    # Check if any documents are indexed
    if get_document_count() == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents have been uploaded yet. "
                   "Use POST /upload to add a PDF first.",
        )

    try:
        # Run the full RAG pipeline
        # This calls: retrieve_context() → format_context_for_llm() → LLM
        result = ask_question(
            question=request.question,
            top_k=request.top_k,
        )

        return QueryResponse(
            answer=result["answer"],
            question=result["question"],
            sources=[
                SourceChunk(
                    text=source["text"],
                    source_file=source["source_file"],
                    page_number=source["page_number"],
                    relevance_score=source.get("relevance_score"),
                )
                for source in result["sources"]
            ],
            num_sources=len(result["sources"]),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating answer: {str(e)}",
        )

