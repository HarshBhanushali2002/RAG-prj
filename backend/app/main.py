"""
🚀 MAIN — FastAPI application entry point.

WHAT YOU'RE LEARNING:
    FastAPI is a modern Python web framework built on top of:
    - Starlette (async HTTP handling)
    - Pydantic (data validation)
    - Uvicorn (ASGI server — like Gunicorn but async)
    
    Key concepts in this file:
    
    1. CORS (Cross-Origin Resource Sharing):
       - Browsers block requests from one origin (e.g., localhost:3000) 
         to another (localhost:8000) by default — this is a SECURITY feature.
       - Our frontend (Next.js on port 3000) needs to call our API (port 8000).
       - CORSMiddleware tells the browser "it's okay, let these origins through."
       - In production, you'd restrict 'allow_origins' to your actual domain.
    
    2. Lifespan Events:
       - @asynccontextmanager with 'lifespan' runs code at startup and shutdown.
       - We use startup to validate config (fail fast if API key is missing).
    
    3. Routers:
       - We split endpoints across files (upload.py, query.py) using APIRouter.
       - app.include_router() wires them into the main application.
       - This keeps the codebase organized as it grows.

    4. Auto-Generated Docs:
       - Visit /docs for Swagger UI (interactive API testing)
       - Visit /redoc for ReDoc (beautiful read-only documentation)
       - These are generated automatically from your route decorators + Pydantic models!

HOW TO RUN:
    cd backend
    uvicorn app.main:app --reload --port 8000
    
    Then open http://localhost:8000/docs in your browser!
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import validate_config
from app.models.schemas import HealthResponse
from app.services.vector_store import get_document_count, list_documents
from app.routes import upload, query


# ============================================================
# LIFESPAN — Runs at startup and shutdown
# ============================================================
# This replaces the older @app.on_event("startup") pattern.
# The code before 'yield' runs at startup; after 'yield' runs at shutdown.

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    STARTUP:
    - Validates config (fails fast if GOOGLE_API_KEY is missing)
    - Prints a banner so you know the server is ready
    
    SHUTDOWN:
    - Cleanup (nothing to clean up yet, but the hook is here for Phase 4)
    """
    # --- STARTUP ---
    print("\n" + "=" * 60)
    print("[*] RAG Document Q&A API -- Starting up...")
    print("=" * 60)
    
    validate_config()
    
    doc_count = get_document_count()
    print(f"[i] Vector store: {doc_count} chunks indexed")
    print(f"[>] Swagger docs: http://localhost:8000/docs")
    print(f"[>] ReDoc:        http://localhost:8000/redoc")
    print("=" * 60 + "\n")
    
    yield  # Server is running
    
    # --- SHUTDOWN ---
    print("\n[*] Shutting down RAG API...")


# ============================================================
# CREATE THE FASTAPI APP
# ============================================================

app = FastAPI(
    title="RAG Document Q&A API",
    description=(
        "Upload PDF documents and ask questions about them using "
        "Retrieval-Augmented Generation (RAG).\n\n"
        "**How it works:**\n"
        "1. Upload a PDF via `POST /upload`\n"
        "2. The API extracts text, chunks it, and stores embeddings\n"
        "3. Ask questions via `POST /query`\n"
        "4. Get answers grounded in your documents, with source citations\n\n"
        "Built with FastAPI + LangChain + ChromaDB + Google Gemini."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# CORS MIDDLEWARE
# ============================================================
# This MUST be added before routes — middleware processes requests
# in the order they're added.
#
# allow_origins=["*"] means "accept requests from ANY origin."
# This is fine for development but in production you'd use:
#   allow_origins=["https://yourdomain.com"]
#
# allow_methods=["*"] allows GET, POST, PUT, DELETE, etc.
# allow_headers=["*"] allows any custom headers.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REGISTER ROUTERS
# ============================================================
# Each router adds its endpoints to the app.
# The prefix is optional — we keep it flat for simplicity.

app.include_router(upload.router)
app.include_router(query.router)


# ============================================================
# ROOT & HEALTH ENDPOINTS
# ============================================================
# These live in main.py because they're app-level, not feature-level.

@app.get(
    "/",
    summary="API Root",
    include_in_schema=False,  # Hide from Swagger (it's just a redirect hint)
)
async def root():
    """Redirect hint to the API docs."""
    return {
        "message": "RAG Document Q&A API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["System"],
)
async def health_check():
    """
    Check if the API is running and report basic stats.
    
    Use this endpoint to verify the server is up and see
    how many documents/chunks are currently indexed.
    """
    docs = list_documents()
    total_chunks = get_document_count()

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        total_documents=len(docs),
        total_chunks=total_chunks,
    )
