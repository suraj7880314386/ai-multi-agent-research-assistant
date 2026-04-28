"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api.routes import router

# ─── Logging Setup ────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Lifespan (startup/shutdown) ──────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    logger.info("=" * 60)
    logger.info("  AI Multi-Agent Research Assistant")
    logger.info("=" * 60)
    logger.info(f"  LLM Model:    {settings.llm_model}")
    logger.info(f"  Embeddings:   {settings.embedding_model}")
    logger.info(f"  Vector Store: ChromaDB ({settings.chroma_persist_dir})")
    logger.info(f"  Memory:       Redis ({settings.redis_url})")
    logger.info(f"  Chunk Size:   {settings.chunk_size} (overlap: {settings.chunk_overlap})")
    logger.info("=" * 60)

    # Pre-warm vector store connection
    from app.core.vector_store import vector_store
    stats = vector_store.get_stats()
    logger.info(f"  Vector Store Status: {stats['status']}")
    logger.info(f"  Indexed Chunks: {stats['total_chunks']}")
    logger.info("=" * 60)

    yield  # App is running

    # Shutdown
    logger.info("Shutting down Research Assistant...")


# ─── FastAPI App ──────────────────────────────────────────

app = FastAPI(
    title="AI Multi-Agent Research Assistant",
    description=(
        "A multi-agent RAG pipeline for semantic search and reasoning "
        "across documents using LangChain, LangGraph, and ChromaDB."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(router)


# ─── Root ─────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name": "AI Multi-Agent Research Assistant",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
