# AI Multi-Agent Research Assistant

A production-ready multi-agent RAG (Retrieval-Augmented Generation) pipeline for semantic search and reasoning across documents using LangChain, LangGraph, FastAPI, and ChromaDB.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  FastAPI Backend                 │
│  ┌───────────┐  ┌───────────┐  ┌─────────────┐ │
│  │  /upload   │  │  /query   │  │  /history    │ │
│  └─────┬─────┘  └─────┬─────┘  └──────┬──────┘ │
│        │              │               │         │
│  ┌─────▼──────────────▼───────────────▼───────┐ │
│  │           LangGraph Orchestrator            │ │
│  │  ┌──────────┐ ┌──────────┐ ┌─────────────┐ │ │
│  │  │ Retriever│ │ Reasoner │ │ Synthesizer │ │ │
│  │  │  Agent   │ │  Agent   │ │   Agent     │ │ │
│  │  └────┬─────┘ └────┬─────┘ └──────┬──────┘ │ │
│  └───────┼─────────────┼──────────────┼────────┘ │
│          │             │              │           │
│  ┌───────▼─────────────▼──────────────▼────────┐ │
│  │              ChromaDB (Vector Store)         │ │
│  └─────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────┐ │
│  │         Conversational Memory (Redis)        │ │
│  └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

## Features

- **Multi-Agent Pipeline**: Three specialized agents (Retriever, Reasoner, Synthesizer) orchestrated via LangGraph
- **RAG Pipeline**: Semantic search over uploaded documents using ChromaDB vector store
- **Conversational Memory**: Redis-backed chat history with session management
- **Document Support**: PDF, TXT, MD, DOCX file ingestion with chunking
- **Streaming Responses**: Server-Sent Events (SSE) for real-time response streaming
- **Session Management**: Multi-user support with isolated conversation contexts

## Tech Stack

- **Framework**: FastAPI + Uvicorn
- **AI Orchestration**: LangChain, LangGraph
- **Vector Store**: ChromaDB
- **Embeddings**: HuggingFace sentence-transformers
- **Memory**: Redis
- **LLM**: OpenAI GPT-4 (configurable)
- **Containerization**: Docker + Docker Compose

## Quick Start

### 1. Clone & Configure
```bash
git clone https://github.com/suraj7880314386/ai-research-assistant.git
cd ai-research-assistant
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

### 2. Run with Docker
```bash
docker-compose up --build
```

### 3. Run Locally
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 4. Test
```bash
# Upload a document
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@your_document.pdf"

# Ask a question
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarize the key findings", "session_id": "user-123"}'

# Get chat history
curl http://localhost:8000/api/v1/history/user-123
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/documents/upload` | Upload and index a document |
| GET | `/api/v1/documents` | List all indexed documents |
| DELETE | `/api/v1/documents/{doc_id}` | Remove a document |
| POST | `/api/v1/query` | Ask a question (with RAG) |
| GET | `/api/v1/query/stream` | Stream a response (SSE) |
| GET | `/api/v1/history/{session_id}` | Get conversation history |
| DELETE | `/api/v1/history/{session_id}` | Clear session history |
| GET | `/api/v1/health` | Health check |

## Project Structure

```
ai-research-assistant/
├── app/
│   ├── main.py                 # FastAPI app entry point
│   ├── config.py               # Configuration & env vars
│   ├── api/
│   │   ├── routes.py           # API route definitions
│   │   └── schemas.py          # Pydantic request/response models
│   ├── agents/
│   │   ├── orchestrator.py     # LangGraph multi-agent orchestrator
│   │   ├── retriever.py        # Retriever agent (semantic search)
│   │   ├── reasoner.py         # Reasoner agent (analysis & logic)
│   │   └── synthesizer.py      # Synthesizer agent (final response)
│   └── core/
│       ├── document_loader.py  # Document ingestion & chunking
│       ├── vector_store.py     # ChromaDB operations
│       └── memory.py           # Conversational memory management
├── tests/
│   └── test_api.py             # API tests
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

## License

MIT
