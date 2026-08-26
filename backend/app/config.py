"""
⚙️ CONFIG — Central configuration for the RAG pipeline.

WHAT YOU'RE LEARNING:
    - python-dotenv loads variables from a .env file into os.environ
    - Pydantic's BaseSettings validates and type-casts env vars automatically
    - This pattern keeps secrets OUT of your code (never commit .env to git!)

WHY THIS MATTERS IN PRODUCTION:
    - Every real app separates config from code (12-factor app principle)
    - Type validation catches misconfiguration early (e.g., CHUNK_SIZE="abc" fails fast)
    - Default values let you run with zero config for development
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from the backend directory
# This reads key=value pairs and sets them as environment variables
load_dotenv()

# ============================================================
# All settings in one place — change here, affects everywhere
# ============================================================

# --- API Keys ---
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

# --- Model Selection ---
LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-3.6-flash")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent  # points to backend/
CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", str(BASE_DIR / "chroma_db"))
UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", str(BASE_DIR / "uploads"))

# --- Chunking Parameters ---
# These DIRECTLY affect answer quality. Experiment with them!
# 
# CHUNK_SIZE: How many characters per chunk.
#   - Too small (200): Chunks lack context, retrieval finds fragments
#   - Too large (5000): Chunks contain mixed topics, retrieval gets noise
#   - Sweet spot: 500-1500 for most documents
#
# CHUNK_OVERLAP: How many characters overlap between adjacent chunks.
#   - Prevents important sentences from being split across chunks
#   - Rule of thumb: 10-20% of chunk size
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

# --- Retrieval ---
# TOP_K: Number of chunks to retrieve per query
#   - More chunks = more context for the LLM, but also more noise
#   - 3-5 is usually the sweet spot
TOP_K: int = int(os.getenv("TOP_K", "5"))


def validate_config():
    """
    Checks that required settings are present.
    Call this at startup to fail fast with a helpful error.
    """
    errors = []

    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your-gemini-api-key-here":
        errors.append(
            "[ERROR] GOOGLE_API_KEY is not set!\n"
            "   -> Get a free key at: https://aistudio.google.com/apikey\n"
            "   -> Copy .env.example to .env and paste your key"
        )

    if CHUNK_SIZE < 100:
        errors.append(f"[ERROR] CHUNK_SIZE={CHUNK_SIZE} is too small. Use at least 100.")

    if CHUNK_OVERLAP >= CHUNK_SIZE:
        errors.append(
            f"[ERROR] CHUNK_OVERLAP ({CHUNK_OVERLAP}) must be less than CHUNK_SIZE ({CHUNK_SIZE})"
        )

    if errors:
        print("\n" + "=" * 60)
        print("[!] CONFIGURATION ERRORS")
        print("=" * 60)
        for error in errors:
            print(f"\n{error}")
        print("\n" + "=" * 60)
        raise SystemExit(1)

    # Create directories if they don't exist
    Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
    Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
