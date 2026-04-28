"""API route definitions for the Research Assistant."""

import logging
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.api.schemas import (
    QueryRequest,
    QueryResponse,
    DocumentResponse,
    DocumentListResponse,
    HistoryResponse,
    HealthResponse,
    SourceDocument,
    AgentStep,
    ChatMessage,
    DocumentStatus,
)
from app.agents.orchestrator import orchestrator
from app.core.document_loader import document_loader
from app.core.vector_store import vector_store
from app.core.memory import memory_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Research Assistant"])


# ─── Document Endpoints ───────────────────────────────────

@router.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and index a document for RAG retrieval."""
    try:
        # Save file
        file_path = await document_loader.save_upload(file)

        # Process: extract text → chunk → index
        doc_id, chunks = document_loader.process_document(file_path, file.filename)

        # Add to vector store
        vector_store.add_documents(chunks)

        # Get document info
        doc_info = document_loader.get_document(doc_id)

        logger.info(f"Document uploaded: {file.filename} → {len(chunks)} chunks")

        return DocumentResponse(
            doc_id=doc_id,
            filename=file.filename,
            status=DocumentStatus.INDEXED,
            num_chunks=doc_info["num_chunks"],
            created_at=doc_info["created_at"],
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """List all indexed documents."""
    docs = document_loader.list_documents()
    return DocumentListResponse(
        documents=[
            DocumentResponse(
                doc_id=d["doc_id"],
                filename=d["filename"],
                status=DocumentStatus(d["status"]),
                num_chunks=d["num_chunks"],
                created_at=d["created_at"],
            )
            for d in docs
        ],
        total=len(docs),
    )


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document and its chunks from the vector store."""
    try:
        # Remove from vector store
        deleted_chunks = vector_store.delete_by_doc_id(doc_id)

        # Remove from document tracker
        document_loader.remove_document(doc_id)

        return {
            "doc_id": doc_id,
            "deleted_chunks": deleted_chunks,
            "message": "Document removed successfully",
        }
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")


# ─── Query Endpoints ──────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Ask a question — runs the full multi-agent RAG pipeline:
    Memory → Retriever → Reasoner → Synthesizer → Memory
    """
    try:
        result = orchestrator.run(
            question=request.question,
            session_id=request.session_id,
            top_k=request.top_k,
            include_sources=request.include_sources,
        )

        # Format response
        sources = [
            SourceDocument(
                content=s["content"],
                metadata=s["metadata"],
                relevance_score=s["relevance_score"],
            )
            for s in result.get("sources", [])
        ]

        agent_trace = [
            AgentStep(
                agent=step["agent"],
                input_summary=step["input_summary"],
                output_summary=step["output_summary"],
                duration_ms=step["duration_ms"],
            )
            for step in result.get("agent_trace", [])
        ]

        return QueryResponse(
            answer=result["answer"],
            sources=sources,
            agent_trace=agent_trace,
            session_id=result["session_id"],
            query_duration_ms=result["query_duration_ms"],
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@router.get("/query/stream")
async def stream_query(
    question: str = Query(..., min_length=1),
    session_id: str = Query(default="default"),
    top_k: int = Query(default=5, ge=1, le=20),
):
    """Stream a response using Server-Sent Events (SSE)."""

    async def event_generator():
        try:
            # Send status updates
            yield f"data: {{\"type\": \"status\", \"message\": \"Loading memory...\"}}\n\n"

            result = orchestrator.run(
                question=question,
                session_id=session_id,
                top_k=top_k,
                include_sources=True,
            )

            # Stream agent trace
            for step in result.get("agent_trace", []):
                import json
                yield f"data: {{\"type\": \"agent_step\", \"data\": {json.dumps(step)}}}\n\n"

            # Stream final answer in chunks
            answer = result["answer"]
            chunk_size = 50
            for i in range(0, len(answer), chunk_size):
                chunk = answer[i : i + chunk_size]
                import json
                yield f"data: {{\"type\": \"answer_chunk\", \"content\": {json.dumps(chunk)}}}\n\n"

            # Send completion
            import json
            yield f"data: {{\"type\": \"done\", \"total_ms\": {result['query_duration_ms']}}}\n\n"

        except Exception as e:
            import json
            yield f"data: {{\"type\": \"error\", \"message\": {json.dumps(str(e))}}}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ─── History Endpoints ────────────────────────────────────

@router.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str, limit: int = Query(default=50, ge=1, le=200)):
    """Get conversation history for a session."""
    messages = memory_manager.get_history(session_id, limit=limit)

    return HistoryResponse(
        session_id=session_id,
        messages=[
            ChatMessage(
                role=msg["role"],
                content=msg["content"],
                timestamp=msg["timestamp"],
            )
            for msg in messages
        ],
        total_messages=len(messages),
    )


@router.delete("/history/{session_id}")
async def clear_history(session_id: str):
    """Clear conversation history for a session."""
    cleared = memory_manager.clear_session(session_id)
    return {
        "session_id": session_id,
        "cleared": cleared,
        "message": "Session history cleared" if cleared else "Session not found",
    }


# ─── Health Check ─────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check."""
    vs_stats = vector_store.get_stats()
    memory_status = memory_manager.get_status()

    return HealthResponse(
        status="healthy",
        vector_store=vs_stats["status"],
        memory=memory_status,
        documents_indexed=vs_stats["total_chunks"],
    )
