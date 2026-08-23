# RAG Document Q&A

A full-stack application to upload PDF documents and ask questions about them in plain English, powered by Retrieval-Augmented Generation (RAG).

## Quick Start

### 1. Setup
```bash
cd backend
pip install -r requirements.txt
```

### 2. Configure API Key
```bash
# Copy the environment template
copy .env.example .env

# Edit .env and add your Gemini API key
# Get a free key at: https://aistudio.google.com/apikey
```

### 3. Run the CLI
```bash
python cli.py
```

### 4. Use It!
```
❯ /upload path/to/your/document.pdf
❯ What is the main topic of this document?
❯ Summarize the key findings
❯ What does page 5 say about revenue?
```

## Project Structure
```
backend/
├── cli.py                    # Interactive CLI (Phase 1)
├── app/
│   ├── config.py             # Environment & settings
│   └── services/
│       ├── pdf_parser.py     # PDF text extraction
│       ├── chunker.py        # Text splitting
│       ├── embedder.py       # Vector embeddings
│       ├── vector_store.py   # ChromaDB operations
│       ├── retriever.py      # Similarity search
│       └── qa_chain.py       # LLM Q&A generation
├── uploads/                  # Stored PDFs
└── chroma_db/                # Vector database
```

## Tech Stack
- **Python 3.11+** — AI/ML ecosystem
- **LangChain** — RAG orchestration
- **Google Gemini** — LLM & embeddings
- **ChromaDB** — Vector database
- **PyMuPDF** — PDF parsing
- **Rich** — Beautiful CLI output
