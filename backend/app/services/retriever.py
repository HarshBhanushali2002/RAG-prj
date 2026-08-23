"""
🔍 RETRIEVER — Finds relevant document chunks for a given question.

WHAT YOU'RE LEARNING:
    The retriever is the bridge between the user's question and the LLM.
    
    Without retrieval (plain LLM):
        User: "What was our Q3 revenue?"
        LLM: "I don't have access to your financial data." ← Useless!
    
    With retrieval (RAG):
        User: "What was our Q3 revenue?"
        Retriever finds: [chunk about Q3 financials from your uploaded PDF]
        LLM: "Based on your annual report, Q3 revenue was $4.2M." ← Useful!

    The retriever's job is simple but critical:
    1. Take the user's question
    2. Find the most relevant chunks from the vector store
    3. Format them as context for the LLM

WHY THIS IS A SEPARATE MODULE:
    In production, retrieval gets sophisticated:
    - Phase 1 (now): Simple similarity search
    - Phase 4: Hybrid search (keyword + semantic)
    - Phase 4: Re-ranking (use a cross-encoder to re-score results)
    - Advanced: Multi-query retrieval (rephrase the question multiple ways)
    
    By isolating retrieval, we can upgrade it without touching the rest of the pipeline.
"""

from langchain.schema import Document
from app.services.vector_store import search_similar, search_with_scores


def retrieve_context(
    query: str,
    top_k: int = 5,
    include_scores: bool = False,
) -> list[dict]:
    """
    Retrieve relevant document chunks for a question.
    
    Args:
        query: The user's question in plain English
        top_k: Number of chunks to retrieve
        include_scores: If True, include relevance scores
        
    Returns:
        List of dicts with 'text', 'metadata', and optionally 'score'
        
    EXPERIMENT:
        Try the same query with top_k=1, 3, 5, 10 and observe:
        - top_k=1: Very focused but might miss relevant info
        - top_k=3: Good balance for most queries
        - top_k=5: Good for complex questions that span multiple sections
        - top_k=10: Risk of including irrelevant chunks that confuse the LLM
    """
    if include_scores:
        results = search_with_scores(query, top_k=top_k)
        return [
            {
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score),
            }
            for doc, score in results
        ]
    else:
        results = search_similar(query, top_k=top_k)
        return [
            {
                "text": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in results
        ]


def format_context_for_llm(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a clean context string for the LLM.
    
    WHY FORMAT MATTERS:
        The LLM needs to distinguish between:
        1. Different chunks (they might even contradict each other)
        2. The chunk text vs. metadata (source, page number)
        
        Clear formatting helps the LLM:
        - Cite sources correctly
        - Handle conflicting information
        - Distinguish between chunks from different documents
    
    Args:
        chunks: List of chunk dicts from retrieve_context()
        
    Returns:
        Formatted string ready to inject into the LLM prompt
    """
    if not chunks:
        return "No relevant documents found."
    
    formatted_chunks = []
    
    for i, chunk in enumerate(chunks, 1):
        source = chunk["metadata"].get("source_file", "Unknown")
        page = chunk["metadata"].get("page_number", "?")
        
        # Each chunk is clearly labeled so the LLM can reference it
        formatted = (
            f"[Source {i}: {source}, Page {page}]\n"
            f"{chunk['text']}"
        )
        formatted_chunks.append(formatted)
    
    return "\n\n---\n\n".join(formatted_chunks)
