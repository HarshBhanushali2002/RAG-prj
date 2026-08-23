"""
🤖 QA CHAIN — The "Generation" part of Retrieval-Augmented Generation.

WHAT YOU'RE LEARNING:
    This is where the magic happens. The QA chain:
    1. Takes the user's question
    2. Takes the retrieved context (chunks from the vector store)
    3. Constructs a prompt that says "Answer this question using ONLY this context"
    4. Sends it to the LLM (Gemini)
    5. Returns the answer
    
    The KEY INSIGHT of RAG:
    → The LLM doesn't need to "know" the answer from its training data
    → We GIVE it the answer (in the context) and ask it to synthesize/rephrase
    → This is why RAG dramatically reduces hallucinations!

THE SYSTEM PROMPT — The Most Important Part:
    The system prompt controls the LLM's behavior. Our prompt says:
    - "Answer based ONLY on the provided context"  ← Prevents hallucination
    - "If the answer isn't in the context, say so"  ← Honest about limitations
    - "Cite your sources"                           ← Traceability
    
    A bad system prompt = a bad RAG system, regardless of retrieval quality.

WHY LangChain OVER DIRECT API CALLS:
    You COULD call the Gemini API directly. But LangChain's chain:
    - Handles prompt templating (cleaner code)
    - Supports streaming (for Phase 4)
    - Makes model swapping trivial
    - Adds retry logic and error handling
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.config import GOOGLE_API_KEY, LLM_MODEL, TOP_K
from app.services.retriever import retrieve_context, format_context_for_llm


# ============================================================
# THE RAG SYSTEM PROMPT
# ============================================================
# This is arguably the most important piece of code in the entire project.
# Every word matters. Experiment with it!
#
# THINGS TO TRY:
# - Remove "If the context doesn't contain the answer..." → Watch it hallucinate
# - Remove "Cite the source..." → Watch it give unverifiable answers
# - Add "Respond in bullet points" → Changes output format
# - Add "Respond in Spanish" → Changes output language
# ============================================================

RAG_SYSTEM_PROMPT = """You are a helpful document assistant. Your job is to answer questions 
based ONLY on the provided context from uploaded documents.

RULES:
1. Answer the question using ONLY the information in the context below.
2. If the context doesn't contain enough information to answer the question, 
   say "I couldn't find enough information in the uploaded documents to answer this question."
3. Cite your sources by mentioning the source document and page number.
4. Be concise but thorough. Include relevant details from the context.
5. If multiple sources provide information, synthesize them into a coherent answer.
6. Do NOT make up information that isn't in the context.

CONTEXT FROM UPLOADED DOCUMENTS:
{context}
"""

RAG_USER_PROMPT = """Question: {question}

Please provide a detailed answer based on the document context above."""


def get_llm() -> ChatGoogleGenerativeAI:
    """
    Create and return the LLM instance.
    
    Temperature controls randomness:
    - 0.0: Deterministic, same answer every time (best for factual Q&A)
    - 0.3: Slight variation (good for natural-sounding answers)
    - 1.0: Very creative (bad for document Q&A, good for brainstorming)
    
    We use 0.1 — mostly deterministic but slightly natural.
    """
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.1,  # Low temperature = factual, consistent answers
        max_output_tokens=2048,
    )


def ask_question(question: str, top_k: int | None = None) -> dict:
    """
    The full RAG pipeline in one function:
    Query → Retrieve → Augment → Generate
    
    Args:
        question: The user's question in plain English
        top_k: Number of chunks to retrieve (default from config)
        
    Returns:
        Dict with:
        - 'answer': The LLM's response
        - 'sources': The retrieved chunks used (for transparency)
        - 'question': The original question (for logging)
        
    THIS IS THE CORE OF RAG:
        1. RETRIEVE: Find relevant chunks
        2. AUGMENT: Add them to the prompt as context
        3. GENERATE: Let the LLM synthesize an answer
    """
    k = top_k or TOP_K
    
    # ---- STEP 1: RETRIEVE ----
    # Find the most relevant chunks from our vector store
    chunks = retrieve_context(question, top_k=k, include_scores=True)
    
    if not chunks:
        return {
            "answer": "No documents have been uploaded yet. Please upload a PDF first.",
            "sources": [],
            "question": question,
        }
    
    # ---- STEP 2: AUGMENT ----
    # Format the chunks into a context string
    context = format_context_for_llm(chunks)
    
    # Build the prompt using LangChain's prompt template
    # This injects the context and question into our system prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("human", RAG_USER_PROMPT),
    ])
    
    # ---- STEP 3: GENERATE ----
    # Create the chain: prompt → LLM → parse output as string
    llm = get_llm()
    chain = prompt | llm | StrOutputParser()
    
    # Run the chain
    answer = chain.invoke({
        "context": context,
        "question": question,
    })
    
    # ---- RETURN RESULTS ----
    # We return the sources so the UI can show WHERE the answer came from
    return {
        "answer": answer,
        "sources": [
            {
                "text": chunk["text"][:300],  # Preview, not full chunk
                "source_file": chunk["metadata"].get("source_file", "Unknown"),
                "page_number": chunk["metadata"].get("page_number", "?"),
                "relevance_score": chunk.get("score", None),
            }
            for chunk in chunks
        ],
        "question": question,
    }
