"""Retriever Agent: Performs semantic search over the vector store."""

import time
import logging
from typing import List, Dict, Any

from langchain.schema import Document

from app.core.vector_store import vector_store

logger = logging.getLogger(__name__)


class RetrieverAgent:
    """
    Agent 1: Retriever
    Responsible for finding the most relevant document chunks
    for a given query using semantic similarity search.
    """

    def __init__(self):
        self.name = "retriever"

    def _rewrite_query(self, question: str, context: str = "") -> str:
        """
        Optionally rewrite the query for better retrieval.
        Uses conversation context to resolve references like 'it', 'that', etc.
        """
        if not context:
            return question

        # Simple heuristic: if question contains pronouns, prepend context hint
        ambiguous_words = {"it", "this", "that", "they", "them", "those", "these"}
        words = set(question.lower().split())

        if words & ambiguous_words:
            # Take the last user message from context as disambiguation
            lines = context.strip().split("\n")
            last_topic = ""
            for line in reversed(lines):
                if line.startswith("Human:"):
                    last_topic = line.replace("Human:", "").strip()
                    break
            if last_topic:
                return f"{last_topic} — {question}"

        return question

    def run(
        self,
        question: str,
        top_k: int = 5,
        conversation_context: str = "",
    ) -> Dict[str, Any]:
        """
        Execute retrieval.

        Args:
            question: User's question
            top_k: Number of results to retrieve
            conversation_context: Recent conversation history for query rewriting

        Returns:
            Dict with retrieved documents, scores, and metadata
        """
        start = time.time()

        # Rewrite query if needed
        search_query = self._rewrite_query(question, conversation_context)

        logger.info(f"[Retriever] Searching for: {search_query[:80]}...")

        # Perform semantic search
        results = vector_store.search(query=search_query, top_k=top_k)

        # Format results
        retrieved_docs = []
        for doc, score in results:
            retrieved_docs.append(
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": round(score, 4),
                }
            )

        duration_ms = (time.time() - start) * 1000

        logger.info(
            f"[Retriever] Found {len(retrieved_docs)} results in {duration_ms:.1f}ms"
        )

        return {
            "agent": self.name,
            "search_query": search_query,
            "documents": retrieved_docs,
            "num_results": len(retrieved_docs),
            "duration_ms": round(duration_ms, 2),
        }


# Singleton
retriever_agent = RetrieverAgent()
