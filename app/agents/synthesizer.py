"""Synthesizer Agent: Generates the final user-facing response."""

import time
import logging
from typing import Dict, Any, List

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from app.config import settings

logger = logging.getLogger(__name__)

SYNTHESIZER_SYSTEM_PROMPT = """You are a Research Synthesis Agent. Your job is to produce the final answer for a user.

You receive:
1. The original question
2. A structured analysis from a Reasoning Agent (with findings, evidence, contradictions, and confidence)
3. Conversation history for context continuity

Your output must:
- Directly answer the question in clear, well-structured prose
- Cite sources naturally (e.g., "According to [filename]...")
- Acknowledge uncertainty when confidence is LOW
- Be concise but thorough — aim for completeness without unnecessary padding
- If the documents don't contain enough information, say so honestly
- Maintain conversational continuity (reference prior exchanges if relevant)

Do NOT:
- Make up information not found in the analysis
- Use overly academic or verbose language
- Repeat the question back unnecessarily
- Add disclaimers about being an AI"""


class SynthesizerAgent:
    """
    Agent 3: Synthesizer
    Takes the reasoner's analysis and produces the final
    user-facing response with proper citations and formatting.
    """

    def __init__(self):
        self.name = "synthesizer"
        self._llm = None

    def _get_llm(self) -> ChatOpenAI:
        """Lazy-load the LLM."""
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=settings.llm_model,
                temperature=0.4,  # Slightly more creative for synthesis
                api_key=settings.openai_api_key,
            )
        return self._llm

    def run(
        self,
        question: str,
        reasoner_analysis: str,
        source_documents: List[Dict],
        conversation_context: str = "",
    ) -> Dict[str, Any]:
        """
        Generate the final response.

        Args:
            question: Original user question
            reasoner_analysis: Structured analysis from ReasonerAgent
            source_documents: Retrieved documents for citation
            conversation_context: Chat history

        Returns:
            Dict with final answer and metadata
        """
        start = time.time()

        logger.info(f"[Synthesizer] Generating response for: {question[:60]}...")

        # Build source reference list
        sources_summary = "\n".join(
            f"- {doc['metadata'].get('filename', 'Unknown')} (relevance: {doc['relevance_score']:.2f})"
            for doc in source_documents[:5]
        )

        user_prompt = f"""QUESTION: {question}

CONVERSATION HISTORY:
{conversation_context if conversation_context else "No prior conversation."}

REASONER'S ANALYSIS:
{reasoner_analysis}

AVAILABLE SOURCES:
{sources_summary}

Generate a clear, well-cited answer for the user."""

        # Call LLM
        llm = self._get_llm()
        messages = [
            SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        response = llm.invoke(messages)
        answer = response.content

        duration_ms = (time.time() - start) * 1000

        logger.info(f"[Synthesizer] Response generated in {duration_ms:.1f}ms")

        return {
            "agent": self.name,
            "answer": answer,
            "duration_ms": round(duration_ms, 2),
        }


# Singleton
synthesizer_agent = SynthesizerAgent()
