"""
💾 VECTOR STORE — Stores and retrieves document embeddings using ChromaDB.

WHAT YOU'RE LEARNING:
    A vector database is a specialized database optimized for similarity search.
    
    Regular database:  "SELECT * FROM docs WHERE title = 'Annual Report'"
                       → Exact match. Finds nothing if title is slightly different.
    
    Vector database:   "Find documents SIMILAR to 'What were last year's revenues?'"
                       → Semantic match. Finds the Annual Report even though 
                       the words don't match!

HOW CHROMADB WORKS:
    1. You store text + its embedding vector + metadata
    2. When querying, ChromaDB:
       a. Takes your query embedding
       b. Computes distance to ALL stored embeddings (using HNSW algorithm)
       c. Returns the K nearest ones
    3. Under the hood, it uses an HNSW index (Hierarchical Navigable Small World)
       - This is an approximate nearest neighbor algorithm
       - Instead of checking ALL vectors (slow), it navigates a graph structure
       - Accuracy: ~99% of brute-force, Speed: 1000x faster

WHY ChromaDB FOR LEARNING:
    - Runs embedded (no separate server process)
    - Data persists to a local directory
    - Dead-simple API: add(), query(), delete()
    - In production, you'd use Pinecone (managed), Qdrant (self-hosted), 
      or Weaviate (hybrid search) — but the CONCEPTS are identical
"""

import chromadb
from chromadb.config import Settings
from app.services.embedder import get_embedding_model
from app.services.chunker import Chunk
from app.config import CHROMA_DB_PATH, TOP_K
from langchain_community.vectorstores import Chroma
from langchain.schema import Document


# We use a single collection name for simplicity.
# In production, you might use one collection per user or per document set.
COLLECTION_NAME = "rag_documents"


def get_vector_store() -> Chroma:
    """
    Get or create the ChromaDB vector store with LangChain integration.
    
    WHY USE LANGCHAIN'S CHROMA WRAPPER?
        - ChromaDB has its own API, but LangChain's wrapper:
          1. Automatically generates embeddings when you add documents
          2. Returns LangChain Document objects (consistent with the rest of the pipeline)
          3. Integrates with LangChain's retriever interface
          4. Makes swapping to Pinecone/Qdrant a one-line change
    
    Returns:
        Chroma vector store instance (persistent — data survives restarts)
    """
    embedding_model = get_embedding_model()
    
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=CHROMA_DB_PATH,
    )


def add_chunks_to_store(chunks: list[Chunk], doc_id: str | None = None) -> int:
    """
    Add document chunks to the vector store.
    
    This is the INGESTION step:
    1. Takes your text chunks
    2. Generates an embedding for each chunk (via Gemini API)
    3. Stores text + embedding + metadata in ChromaDB
    
    Args:
        chunks: List of Chunk objects from the chunker
        doc_id: Optional document identifier for filtering later
        
    Returns:
        Number of chunks successfully stored
    """
    store = get_vector_store()
    
    # Convert our Chunk objects to LangChain Documents
    # LangChain Documents have: page_content (str) + metadata (dict)
    documents = []
    for chunk in chunks:
        metadata = chunk.metadata.copy()
        if doc_id:
            metadata["doc_id"] = doc_id
        
        documents.append(Document(
            page_content=chunk.text,
            metadata=metadata,
        ))
    
    # add_documents() does three things:
    # 1. Calls the embedding model for each document's text
    # 2. Generates unique IDs for each document  
    # 3. Stores everything in ChromaDB
    #
    # The embeddings are generated in batches automatically (efficient!)
    store.add_documents(documents)
    
    return len(documents)


def search_similar(query: str, top_k: int | None = None) -> list[Document]:
    """
    Find the most similar chunks to a query.
    
    This is the RETRIEVAL step — the "R" in RAG:
    1. Converts your question into an embedding
    2. Finds the K closest embeddings in the store
    3. Returns those chunks (with their metadata)
    
    Args:
        query: The user's question in plain English
        top_k: Number of results to return (default from config)
        
    Returns:
        List of LangChain Document objects, most relevant first
    """
    store = get_vector_store()
    k = top_k or TOP_K
    
    # similarity_search() does:
    # 1. Embeds the query using the same embedding model
    # 2. Finds the K nearest vectors using cosine similarity
    # 3. Returns the corresponding documents
    results = store.similarity_search(query, k=k)
    
    return results


def search_with_scores(query: str, top_k: int | None = None) -> list[tuple[Document, float]]:
    """
    Like search_similar(), but also returns the similarity score.
    
    Scores help you understand retrieval quality:
    - Score close to 0: Very relevant (ChromaDB uses L2 distance, lower = closer)
    - Score > 1.0: Probably not relevant
    
    This is useful for:
    - Debugging: "Why did it retrieve this irrelevant chunk?"
    - Filtering: "Only use chunks with score < 0.5"
    - Confidence: "All scores are high → the doc probably doesn't contain the answer"
    """
    store = get_vector_store()
    k = top_k or TOP_K
    
    return store.similarity_search_with_score(query, k=k)


def get_document_count() -> int:
    """Return the total number of chunks stored in the vector database."""
    store = get_vector_store()
    return store._collection.count()


def delete_document(source_file: str) -> int:
    """
    Delete all chunks belonging to a specific document.
    
    Args:
        source_file: The filename to delete (e.g., "report.pdf")
        
    Returns:
        Number of chunks deleted
    """
    store = get_vector_store()
    collection = store._collection
    
    # Get all chunks from this document
    results = collection.get(where={"source_file": source_file})
    
    if results["ids"]:
        collection.delete(ids=results["ids"])
        return len(results["ids"])
    
    return 0


def list_documents() -> list[dict]:
    """
    List all unique documents in the vector store.
    
    Returns:
        List of dicts with document info (filename, chunk_count, etc.)
    """
    store = get_vector_store()
    collection = store._collection
    
    # Get all metadata
    all_data = collection.get(include=["metadatas"])
    
    if not all_data["metadatas"]:
        return []
    
    # Group by source file
    docs = {}
    for metadata in all_data["metadatas"]:
        filename = metadata.get("source_file", "unknown")
        if filename not in docs:
            docs[filename] = {
                "filename": filename,
                "chunk_count": 0,
                "pages": set(),
            }
        docs[filename]["chunk_count"] += 1
        page = metadata.get("page_number")
        if page:
            docs[filename]["pages"].add(page)
    
    # Convert sets to sorted lists for JSON serialization
    result = []
    for doc in docs.values():
        doc["pages"] = sorted(doc["pages"])
        doc["total_pages"] = len(doc["pages"])
        result.append(doc)
    
    return result


# ============================================================
# 🧪 Quick test — store and search
# Usage: python -m app.services.vector_store
# ============================================================
if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table
    from app.config import validate_config
    
    validate_config()
    console = Console()
    
    console.print("\n[bold]💾 Vector Store Demo[/bold]\n")
    
    # Create some test documents
    test_chunks = [
        Chunk(text="Python is a programming language created by Guido van Rossum.", 
              metadata={"source_file": "test.pdf", "page_number": 1, "chunk_index": 0, "total_chunks": 3, "chunk_size": 60}),
        Chunk(text="Machine learning is a subset of artificial intelligence that learns from data.",
              metadata={"source_file": "test.pdf", "page_number": 1, "chunk_index": 1, "total_chunks": 3, "chunk_size": 75}),
        Chunk(text="The capital of France is Paris. It is known for the Eiffel Tower.",
              metadata={"source_file": "test.pdf", "page_number": 2, "chunk_index": 2, "total_chunks": 3, "chunk_size": 63}),
    ]
    
    console.print("[yellow]Adding 3 test chunks to vector store...[/yellow]")
    count = add_chunks_to_store(test_chunks)
    console.print(f"[green]✅ Stored {count} chunks[/green]\n")
    
    # Search
    queries = [
        "Who created Python?",
        "What is AI?",
        "Tell me about France",
    ]
    
    for query in queries:
        console.print(f'[cyan]🔍 Query: "{query}"[/cyan]')
        results = search_with_scores(query, top_k=2)
        
        table = Table()
        table.add_column("Score", style="bold")
        table.add_column("Text", max_width=60)
        
        for doc, score in results:
            table.add_row(f"{score:.4f}", doc.page_content[:60])
        
        console.print(table)
        console.print()
