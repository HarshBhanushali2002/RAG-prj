"""
🧮 EMBEDDER — Converts text into numerical vectors (embeddings).

WHAT YOU'RE LEARNING:
    This is the CORE CONCEPT of RAG. Everything else is plumbing.
    
    An embedding is a list of numbers (a vector) that captures the MEANING of text.
    
    Example:
        "The cat sat on the mat"  →  [0.12, -0.45, 0.89, 0.03, ..., -0.21]  (768 numbers)
        "A feline rested on a rug" → [0.11, -0.43, 0.88, 0.05, ..., -0.20]  (768 numbers)
        "Stock prices rose today"  → [0.95, 0.12, -0.67, 0.44, ..., 0.33]  (768 numbers)
    
    Notice: The first two sentences have SIMILAR vectors (similar meaning),
    while the third has a VERY DIFFERENT vector (different meaning).
    
    This is how semantic search works:
    1. Convert your question to a vector
    2. Find stored vectors that are closest to it
    3. Those chunks are the most relevant to your question

WHY 768 DIMENSIONS?
    - Each dimension captures some aspect of meaning
    - More dimensions = more nuanced understanding, but more storage/compute
    - 768 is a sweet spot used by many models (BERT, text-embedding-004)
    - OpenAI's text-embedding-3-large uses 3072 dimensions (more expensive)

WHAT THE EMBEDDING MODEL LEARNED:
    - Google trained text-embedding-004 on billions of text pairs
    - It learned that "doctor" and "physician" should be close in vector space
    - It learned that "bank" (river) and "bank" (financial) should be far apart in context
    - You don't need to understand HOW it works — just that similar text = similar vectors
"""

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import GOOGLE_API_KEY, EMBEDDING_MODEL


def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    """
    Create and return the embedding model.
    
    We use LangChain's wrapper around Google's embedding API because:
    1. It handles batching automatically (sending 100 texts at once vs. one-by-one)
    2. It handles rate limiting and retries
    3. It returns embeddings in a consistent format that ChromaDB expects
    4. Swapping to OpenAI embeddings = changing ONE line (the import + model name)
    
    Returns:
        GoogleGenerativeAIEmbeddings instance ready to use
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
        List of embedding vectors (each is a list of 768 floats)
        
    EXPERIMENT:
        Try embedding these pairs and computing cosine similarity:
        - "How do I reset my password?" vs "I forgot my login credentials"  → HIGH similarity
        - "How do I reset my password?" vs "What is the weather today?"     → LOW similarity
    """
    model = get_embedding_model()
    return model.embed_documents(texts)


def embed_query(query: str) -> list[float]:
    """
    Convert a single query string into an embedding vector.
    
    WHY A SEPARATE FUNCTION FOR QUERIES?
        Some embedding models treat queries differently from documents.
        - Documents are embedded "as-is"
        - Queries may have a special prefix like "query: " added internally
        This improves retrieval accuracy. LangChain handles this automatically.
    
    Args:
        query: The user's question
        
    Returns:
        A single embedding vector (list of 768 floats)
    """
    model = get_embedding_model()
    return model.embed_query(query)


# ============================================================
# 🧪 Quick test — see what embeddings actually look like
# Usage: python -m app.services.embedder
# ============================================================
if __name__ == "__main__":
    from rich.console import Console
    from app.config import validate_config
    
    validate_config()
    console = Console()
    
    console.print("\n[bold]🧮 Embedding Demo[/bold]\n")
    
    # Embed some sample texts
    texts = [
        "The cat sat on the mat",
        "A feline rested on a rug",
        "Stock prices rose sharply today",
    ]
    
    console.print("[yellow]Embedding 3 sample texts...[/yellow]")
    vectors = embed_texts(texts)
    
    for i, (text, vec) in enumerate(zip(texts, vectors)):
        console.print(f'\n[cyan]Text {i}:[/cyan] "{text}"')
        console.print(f"  Vector length: {len(vec)} dimensions")
        console.print(f"  First 5 values: {[round(v, 4) for v in vec[:5]]}")
    
    # Compute cosine similarity manually to understand it
    import math
    
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """
        Cosine similarity = how much two vectors point in the same direction.
        - 1.0 = identical meaning
        - 0.0 = unrelated
        - -1.0 = opposite meaning
        """
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x ** 2 for x in a))
        norm_b = math.sqrt(sum(x ** 2 for x in b))
        return dot / (norm_a * norm_b)
    
    console.print("\n[bold]📐 Cosine Similarity (higher = more similar):[/bold]")
    console.print(f'  "cat on mat" vs "feline on rug": {cosine_similarity(vectors[0], vectors[1]):.4f}')
    console.print(f'  "cat on mat" vs "stock prices":  {cosine_similarity(vectors[0], vectors[2]):.4f}')
    console.print(f'  "feline on rug" vs "stock prices": {cosine_similarity(vectors[1], vectors[2]):.4f}')
    
    console.print("\n[green]✅ Notice: Similar sentences have HIGH similarity, unrelated ones have LOW![/green]")
