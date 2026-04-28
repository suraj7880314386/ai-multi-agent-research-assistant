"""Pydantic models for API request/response validation."""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from enum import Enum


# ─── Enums ────────────────────────────────────────────────

class AgentRole(str, Enum):
    RETRIEVER = "retriever"
    REASONER = "reasoner"
    SYNTHESIZER = "synthesizer"


class DocumentStatus(str, Enum):
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


# ─── Request Models ───────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="User question")
    session_id: str = Field(default="default", description="Session ID for conversation memory")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of documents to retrieve")
    include_sources: bool = Field(default=True, description="Include source references")


class StreamQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(default="default")
    top_k: int = Field(default=5, ge=1, le=20)


# ─── Response Models ──────────────────────────────────────

class SourceDocument(BaseModel):
    content: str
    metadata: dict
    relevance_score: float


class AgentStep(BaseModel):
    agent: AgentRole
    input_summary: str
    output_summary: str
    duration_ms: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument] = []
    agent_trace: List[AgentStep] = []
    session_id: str
    query_duration_ms: float


class DocumentResponse(BaseModel):
    doc_id: str
    filename: str
    status: DocumentStatus
    num_chunks: int
    created_at: datetime


class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime


class HistoryResponse(BaseModel):
    session_id: str
    messages: List[ChatMessage]
    total_messages: int


class HealthResponse(BaseModel):
    status: str
    vector_store: str
    memory: str
    documents_indexed: int


class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None
