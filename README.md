# Synapse Engine

A "second brain" application that lets you chat with your own documents using a RAG (Retrieval-Augmented Generation) pipeline. Built with FastAPI, ChromaDB, Mistral, and VoyageAI embeddings, with a Streamlit frontend.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      User Interface                      │
│                  Streamlit (port 8501)                   │
└────────────────────────┬─────────────────────────────────┘
                         │ HTTP  X-API-Key
┌────────────────────────▼─────────────────────────────────┐
│                    FastAPI (port 8000)                    │
│                   POST /ask  (secured)                   │
└────────────────────────┬─────────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │      QA Service     │
              │    (RAG pipeline)   │
              └──────┬──────────────┘
                     │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
   ┌───────────┐ ┌───────┐ ┌──────────────┐
   │ ChromaDB  │ │Mistral│ │  VoyageAI    │
   │(retrieval)│ │ (LLM) │ │ (embeddings) │
   └───────────┘ └───────┘ └──────────────┘

Ingestion pipeline:
  Documents (PDF, TXT, MD)
       │
  TextSplitter (chunk_size=1000, overlap=200)
       │
  VoyageAI embeddings
       │
  ChromaDB (persisted)
```

## Tech Stack

| Component | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Vector store | ChromaDB |
| Embeddings | VoyageAI (`voyage-large-2`) |
| LLM | Mistral (`mistral-large-latest`) |
| Orchestration | LangChain |
| Containerization | Docker |

## Prerequisites

- Python 3.11+
- A [VoyageAI](https://www.voyageai.com/) API key
- A [Mistral](https://console.mistral.ai/) API key

## Installation

```bash
git clone https://github.com/FloDudu/synapse-engine.git
cd synapse-engine

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```env
MISTRAL_API_KEY=your_mistral_key
VOYAGE_API_KEY=your_voyageai_key
APP_SECRET_KEY=your_random_secret_key   # used to authenticate API calls

# Optional
VECTOR_DB_PATH=./chroma_db
COLLECTION_NAME=synapse_knowledge
```

## Usage

### 1. Ingest documents

```bash
python -m src.ingest ./data
```

Supports `.pdf`, `.txt`, and `.md` files. Documents are chunked, embedded, and stored in ChromaDB.

### 2. Start the API

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### 3. Start the frontend

```bash
streamlit run src/interface/app.py --server.port 8501
```

Or use the convenience scripts:

```bash
# Linux/macOS
./start.sh

# Windows
start.bat
```

### Query the API directly

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_secret_key" \
  -d '{"question": "What are the key points in the documents?"}'
```

```json
{
  "answer": "Based on the documents, ..."
}
```

Interactive API docs available at `http://localhost:8000/docs`.

## Docker

```bash
docker build -t synapse-engine .

docker run -p 8000:8000 -p 8501:8501 \
  -e MISTRAL_API_KEY=your_key \
  -e VOYAGE_API_KEY=your_key \
  -e APP_SECRET_KEY=your_secret \
  -v $(pwd)/chroma_db:/app/chroma_db \
  synapse-engine
```

> Mounting `chroma_db` as a volume ensures document embeddings persist across container restarts.

## Running Tests

```bash
pytest
```

## Project Structure

```
synapse-engine/
├── src/
│   ├── config.py                    # Settings from environment
│   ├── main.py                      # FastAPI app (lifespan, routes)
│   ├── ingest.py                    # CLI ingestion script
│   ├── core/
│   │   └── qa_service.py            # RAG chain (LangChain LCEL)
│   ├── infrastructure/
│   │   ├── ai_factory.py            # Factory for LLM + embeddings
│   │   └── vector_store_manager.py  # Strategy pattern over ChromaDB
│   ├── interface/
│   │   └── app.py                   # Streamlit frontend
│   └── utils/
│       └── document_loader.py       # Document loading + chunking
└── tests/
    ├── infrastructure/
    │   ├── test_ai_factory.py
    │   └── test_vector_store_manager.py
    └── utils/
        └── test_document_loader.py
```

## Design Decisions

**Strategy pattern on `VectorStoreManager`** — the vector store backend is abstracted behind `VectorStoreStrategy`, making it straightforward to swap ChromaDB for another store (FAISS, Pinecone) without touching the rest of the codebase.

**Factory pattern for AI components** — `AIFactory` centralises model instantiation and key validation, which simplifies mocking in tests and keeps configuration out of business logic.

**API key authentication** — all endpoints require an `X-API-Key` header validated against `APP_SECRET_KEY`. The app refuses to start if this variable is not set.
