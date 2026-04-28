"""Conversational memory management with Redis backend."""

import json
import logging
from datetime import datetime
from typing import List, Dict, Optional

from app.config import settings

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Manages conversational memory per session.
    Uses Redis for persistence with TTL-based expiry.
    Falls back to in-memory dict if Redis is unavailable.
    """

    def __init__(self):
        self._redis = None
        self._fallback_store: Dict[str, List[dict]] = {}
        self._use_redis = True

    def _get_redis(self):
        """Lazy-connect to Redis."""
        if self._redis is None:
            try:
                import redis

                self._redis = redis.from_url(
                    settings.redis_url, decode_responses=True
                )
                self._redis.ping()
                logger.info("Connected to Redis for memory storage")
            except Exception as e:
                logger.warning(f"Redis unavailable ({e}), using in-memory fallback")
                self._use_redis = False
        return self._redis

    def _session_key(self, session_id: str) -> str:
        """Generate Redis key for a session."""
        return f"chat_memory:{session_id}"

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Add a message to the session history."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if self._use_redis:
            try:
                r = self._get_redis()
                if r:
                    key = self._session_key(session_id)
                    r.rpush(key, json.dumps(message))
                    r.expire(key, settings.memory_ttl_seconds)
                    return
            except Exception as e:
                logger.warning(f"Redis write failed: {e}, falling back to memory")
                self._use_redis = False

        # Fallback: in-memory
        if session_id not in self._fallback_store:
            self._fallback_store[session_id] = []
        self._fallback_store[session_id].append(message)

    def get_history(self, session_id: str, limit: int = 50) -> List[dict]:
        """Retrieve conversation history for a session."""
        if self._use_redis:
            try:
                r = self._get_redis()
                if r:
                    key = self._session_key(session_id)
                    messages = r.lrange(key, -limit, -1)
                    return [json.loads(m) for m in messages]
            except Exception as e:
                logger.warning(f"Redis read failed: {e}")
                self._use_redis = False

        # Fallback
        messages = self._fallback_store.get(session_id, [])
        return messages[-limit:]

    def get_context_window(self, session_id: str, max_messages: int = 10) -> str:
        """
        Get recent conversation history formatted for LLM context.
        Returns a string of recent exchanges.
        """
        history = self.get_history(session_id, limit=max_messages)

        if not history:
            return ""

        formatted = []
        for msg in history:
            role_label = "Human" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role_label}: {msg['content']}")

        return "\n".join(formatted)

    def clear_session(self, session_id: str) -> bool:
        """Clear all history for a session."""
        if self._use_redis:
            try:
                r = self._get_redis()
                if r:
                    return bool(r.delete(self._session_key(session_id)))
            except Exception:
                pass

        if session_id in self._fallback_store:
            del self._fallback_store[session_id]
            return True
        return False

    def get_status(self) -> str:
        """Check memory backend status."""
        if self._use_redis:
            try:
                r = self._get_redis()
                if r and r.ping():
                    return "redis:connected"
            except Exception:
                pass
        return "in-memory:active"


# Singleton
memory_manager = MemoryManager()
