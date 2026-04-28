"""Tests for the Research Assistant API."""

import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, MagicMock
from app.main import app


@pytest.fixture
def client():
    """Create async test client."""
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


# ─── Health Check ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_check(client):
    """Test health endpoint returns 200."""
    async with client as c:
        response = await c.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "vector_store" in data
    assert "memory" in data


# ─── Root Endpoint ────────────────────────────────────────

@pytest.mark.asyncio
async def test_root(client):
    """Test root endpoint."""
    async with client as c:
        response = await c.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "AI Multi-Agent Research Assistant"


# ─── Document Upload ──────────────────────────────────────

@pytest.mark.asyncio
async def test_upload_unsupported_file(client):
    """Test that unsupported file types are rejected."""
    async with client as c:
        response = await c.post(
            "/api/v1/documents/upload",
            files={"file": ("test.exe", b"fake content", "application/octet-stream")},
        )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_upload_text_file(client):
    """Test uploading a text file."""
    content = b"This is a test document about artificial intelligence and machine learning."
    async with client as c:
        response = await c.post(
            "/api/v1/documents/upload",
            files={"file": ("test.txt", content, "text/plain")},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test.txt"
    assert data["status"] == "indexed"
    assert data["num_chunks"] >= 1


# ─── Document Listing ────────────────────────────────────

@pytest.mark.asyncio
async def test_list_documents(client):
    """Test listing documents."""
    async with client as c:
        response = await c.get("/api/v1/documents")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert "total" in data


# ─── History ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_empty_history(client):
    """Test getting history for a new session."""
    async with client as c:
        response = await c.get("/api/v1/history/new-session-xyz")
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "new-session-xyz"
    assert data["total_messages"] == 0


@pytest.mark.asyncio
async def test_clear_history(client):
    """Test clearing session history."""
    async with client as c:
        response = await c.delete("/api/v1/history/test-session")
    assert response.status_code == 200


# ─── Query (mocked LLM) ──────────────────────────────────

@pytest.mark.asyncio
async def test_query_no_documents(client):
    """Test querying when no documents are indexed returns graceful message."""
    async with client as c:
        response = await c.post(
            "/api/v1/query",
            json={
                "question": "What is quantum computing?",
                "session_id": "test-empty",
                "top_k": 3,
            },
        )
    # Should succeed but with a "no docs found" type response
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["session_id"] == "test-empty"
