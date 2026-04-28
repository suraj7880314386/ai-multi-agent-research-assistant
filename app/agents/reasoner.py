"""Reasoner Agent: Analyzes retrieved documents and performs logical reasoning."""

import time
import logging
from typing import Dict, Any, List

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from app.config import settings

logger = logging.getLogger(__name__)

REASONER_SYSTEM_PROMPT = """You are a Research Reasoning Agent. Your job is to:

1. Analyze retrieved document chunks and determine their relevance to the question.
2. Identify key facts, arguments, and evidence from the sources.
3. Detect contradictions or gaps across sources.
4. Perform logical reasoning to connect information from multiple chunks.
5. Produce a structured analysis that a Synthesizer agent can use to write the final answer.

Output Format:
- KEY_FINDINGS: List the most important facts/insights (bullet points)
- EVIDENCE: Quote or paraphrase the most relevant passages with source references
- CONTRADICTIONS: Note any conflicting information across sources
- REASONING: Your logical analysis connecting the findings to the question
- CONFIDENCE: Rate your confidence (HIGH/MEDIUM/LOW) and explain why
- GAPS: What information is missing or unclear

Be precise. Cite chunk indices when referencing specific passages."""


class ReasonerAgent:
    """
    Agent 2: Reasoner
    Takes retrieved documents and performs analytical reasoning
    to extract insights, identify patterns, and build logical connections.
    """

    def __init__(self):
        self.name = "reasoner"
        self._llm = None

    def _get_llm(self) -> ChatOpenAI:
        """Lazy-load the LLM."""
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                api_key=settings.openai_api_key,
            )
        return self._llm

    def _format_documents_for_analysis(self, documents: List[Dict]) -> str:
        """Format retrieved documents into a structured context block."""
        if not documents:
            return "No relevant documents were retrieved."

        formatted = []
        for i, doc in enumerate(documents):
            source = doc.get("metadata", {}).get("filename", "Unknown")
            score = doc.get("relevance_score", 0)
            content = doc.get("content", "")
            formatted.append(
                f"[Chunk {i+1}] (Source: {source}, Relevance: {score:.2f})\n{content}"
            )

        return "\n\n---\n\n".join(formatted)

    def run(
        self,
        question: str,
        retrieved_documents: List[Dict],
        conversation_context: str = "",
    ) -> Dict[str, Any]:
        """
        Execute reasoning over retrieved documents.

        Args:
            question: Original user question
            retrieved_documents: Output from RetrieverAgent
            conversation_context: Recent conversation for continuity

        Returns:
            Dict with structured analysis
        """
        start = time.time()

        logger.info(
            f"[Reasoner] Analyzing {len(retrieved_documents)} documents for: {question[:60]}..."
        )

        # Format the input
        docs_context = self._format_documents_for_analysis(retrieved_documents)

        # Build the prompt
        user_prompt = f"""QUESTION: {question}

CONVERSATION CONTEXT:
{conversation_context if conversation_context else "No prior conversation."}

RETRIEVED DOCUMENTS:
{docs_context}

Analyze these documents and provide your structured reasoning."""

        # Call LLM
        llm = self._get_llm()
        messages = [
            SystemMessage(content=REASONER_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        response = llm.invoke(messages)
        analysis = response.content

        duration_ms = (time.time() - start) * 1000

        logger.info(f"[Reasoner] Analysis complete in {duration_ms:.1f}ms")

        return {
            "agent": self.name,
            "analysis": analysis,
            "num_sources_analyzed": len(retrieved_documents),
            "duration_ms": round(duration_ms, 2),
        }


# Singleton
reasoner_agent = ReasonerAgent()
