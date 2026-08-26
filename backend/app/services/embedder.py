"""
EMBEDDER - Converts text into numerical vectors using Google Gemini.

Embeddings capture the semantic meaning of text.
Similar sentences produce similar vectors, enabling semantic search.
"""

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import GOOGLE_API_KEY, EMBEDDING_MODEL


def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    """
    Create and return the Google Gemini embedding model.

    Uses LangChain wrapper which handles:
    - Automatic batching of texts
    - Rate limiting and retries
    - Consistent output format for ChromaDB
    """
    return GoogleGenerativeAIEmbeddings(
        model=f"models/{EMBEDDING_MODEL}",
        google_api_key=GOOGLE_API_KEY,
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Convert a list of text strings into embedding vectors.

    Args:
        texts: List of text strings to embed

    Returns:
        List of embedding vectors (each is a list of floats)
    """
    model = get_embedding_model()
    return model.embed_documents(texts)


def embed_query(query: str) -> list[float]:
    """
    Convert a single query string into an embedding vector.

    Queries may use a different internal prefix than documents
    to improve retrieval accuracy (handled by LangChain automatically).

    Args:
        query: The user question string

    Returns:
        A single embedding vector (list of floats)
    """
    model = get_embedding_model()
    return model.embed_query(query)
