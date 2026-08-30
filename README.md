# RAG Document Q&A

A full-stack application to upload PDF documents and ask questions about them in plain English, powered by **Retrieval-Augmented Generation (RAG)**.

## How It Works

```
PDF Upload -> Text Extraction -> Chunking -> Embedding -> ChromaDB Storage
                                                              |
User Question -> Embedding -> Similarity Search -> Gemini LLM -> Answer + Sources
```

## Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/HarshBhanushali2002/RAG-prj.git
cd RAG-prj/backend
pip install -r requirements.txt
```

### 2. Configure API Key
```bash
# Copy the environment template
copy .env.example .env

# Edit .env and set GOOGLE_API_KEY=your_key_here
# Get a free key at: https://aistudio.google.com/apikey
```

### 3. Run the API Server
```bash
uvicorn app.main:app --reload --port 8000
# Open http://localhost:8000/docs for interactive Swagger UI
```

### 4. Or Use the CLI
```bash
python cli.py
# Then: /upload path/to/document.pdf
# Then: What is the main topic of this document?
```

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| POST | /upload | Upload a PDF and index it |
| POST | /query | Ask a question about indexed documents |
| GET | /documents | List all indexed documents |
| DELETE | /documents/{filename} | Remove a document from the index |
| GET | /health | Health check + stats |

## Project Structure

```
backend/
├── cli.py                    # Interactive CLI
├── requirements.txt          # Python dependencies
├── app/
│   ├── main.py               # FastAPI app + CORS + lifespan
│   ├── config.py             # Environment & settings
│   ├── models/
│   │   └── schemas.py        # Pydantic request/response models
│   ├── routes/
│   │   ├── upload.py         # POST /upload endpoint
│   │   └── query.py          # POST /query endpoint
│   └── services/
│       ├── pdf_parser.py     # PDF text extraction (PyMuPDF)
│       ├── chunker.py        # Text splitting (LangChain)
│       ├── embedder.py       # Vector embeddings (Gemini)
│       ├── vector_store.py   # ChromaDB operations
│       ├── retriever.py      # Similarity search + context formatting
│       └── qa_chain.py       # LLM Q&A generation
├── uploads/                  # Stored PDF files
└── chroma_db/                # Persistent vector database
```

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| LLM + Embeddings | Google Gemini | Answering + text-embedding-004 |
| RAG Orchestration | LangChain | Pipeline wiring |
| Vector Database | ChromaDB | Similarity search |
| PDF Parsing | PyMuPDF | Text extraction |
| API | FastAPI | REST endpoints + Swagger docs |
| CLI | Rich | Beautiful terminal output |

## Environment Variables

```
GOOGLE_API_KEY=your_gemini_api_key
CHROMA_DB_PATH=./chroma_db
EMBEDDING_MODEL=text-embedding-004
LLM_MODEL=gemini-1.5-flash
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=5
```
