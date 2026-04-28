"""LangGraph Orchestrator: Coordinates the multi-agent RAG pipeline."""

import time
import logging
from typing import Dict, Any, TypedDict, List, Annotated
from operator import add

from langgraph.graph import StateGraph, END

from app.agents.retriever import retriever_agent
from app.agents.reasoner import reasoner_agent
from app.agents.synthesizer import synthesizer_agent
from app.core.memory import memory_manager

logger = logging.getLogger(__name__)


# ─── State Definition ─────────────────────────────────────

class PipelineState(TypedDict):
    """State that flows through the multi-agent pipeline."""

    # Input
    question: str
    session_id: str
    top_k: int
    include_sources: bool

    # Intermediate
    conversation_context: str
    retrieved_documents: List[Dict]
    retriever_output: Dict
    reasoner_output: Dict
    synthesizer_output: Dict

    # Output
    final_answer: str
    sources: List[Dict]
    agent_trace: Annotated[List[Dict], add]
    total_duration_ms: float


# ─── Node Functions ───────────────────────────────────────

def load_memory_node(state: PipelineState) -> Dict:
    """Node 1: Load conversation context from memory."""
    session_id = state["session_id"]
    context = memory_manager.get_context_window(session_id, max_messages=10)

    logger.info(f"[Orchestrator] Loaded memory for session: {session_id}")

    return {"conversation_context": context}


def retriever_node(state: PipelineState) -> Dict:
    """Node 2: Run the Retriever Agent."""
    result = retriever_agent.run(
        question=state["question"],
        top_k=state["top_k"],
        conversation_context=state.get("conversation_context", ""),
    )

    trace_entry = {
        "agent": "retriever",
        "input_summary": f"Query: {state['question'][:80]}...",
        "output_summary": f"Retrieved {result['num_results']} documents",
        "duration_ms": result["duration_ms"],
    }

    return {
        "retriever_output": result,
        "retrieved_documents": result["documents"],
        "agent_trace": [trace_entry],
    }


def reasoner_node(state: PipelineState) -> Dict:
    """Node 3: Run the Reasoner Agent."""
    result = reasoner_agent.run(
        question=state["question"],
        retrieved_documents=state["retrieved_documents"],
        conversation_context=state.get("conversation_context", ""),
    )

    trace_entry = {
        "agent": "reasoner",
        "input_summary": f"Analyzing {len(state['retrieved_documents'])} documents",
        "output_summary": f"Analysis complete ({len(result['analysis'])} chars)",
        "duration_ms": result["duration_ms"],
    }

    return {
        "reasoner_output": result,
        "agent_trace": [trace_entry],
    }


def synthesizer_node(state: PipelineState) -> Dict:
    """Node 4: Run the Synthesizer Agent."""
    result = synthesizer_agent.run(
        question=state["question"],
        reasoner_analysis=state["reasoner_output"]["analysis"],
        source_documents=state["retrieved_documents"],
        conversation_context=state.get("conversation_context", ""),
    )

    trace_entry = {
        "agent": "synthesizer",
        "input_summary": "Generating final response from analysis",
        "output_summary": f"Response generated ({len(result['answer'])} chars)",
        "duration_ms": result["duration_ms"],
    }

    # Build source list
    sources = []
    if state.get("include_sources", True):
        sources = [
            {
                "content": doc["content"][:200] + "...",
                "metadata": doc["metadata"],
                "relevance_score": doc["relevance_score"],
            }
            for doc in state["retrieved_documents"][:5]
        ]

    return {
        "synthesizer_output": result,
        "final_answer": result["answer"],
        "sources": sources,
        "agent_trace": [trace_entry],
    }


def save_memory_node(state: PipelineState) -> Dict:
    """Node 5: Save the exchange to conversational memory."""
    session_id = state["session_id"]

    memory_manager.add_message(session_id, "user", state["question"])
    memory_manager.add_message(session_id, "assistant", state["final_answer"])

    logger.info(f"[Orchestrator] Saved to memory for session: {session_id}")
    return {}


# ─── Conditional Edge ─────────────────────────────────────

def should_reason(state: PipelineState) -> str:
    """
    Decide whether to proceed to reasoning or skip if no documents found.
    """
    docs = state.get("retrieved_documents", [])
    if not docs:
        return "no_docs"
    return "has_docs"


def no_docs_response(state: PipelineState) -> Dict:
    """Handle the case where no relevant documents were found."""
    answer = (
        "I couldn't find any relevant information in the indexed documents to answer "
        f"your question: \"{state['question']}\"\n\n"
        "This could mean:\n"
        "- No documents have been uploaded yet\n"
        "- The uploaded documents don't contain information related to your query\n\n"
        "Try uploading relevant documents first, or rephrase your question."
    )

    trace_entry = {
        "agent": "synthesizer",
        "input_summary": "No documents retrieved — generating fallback",
        "output_summary": "No-docs fallback response",
        "duration_ms": 0,
    }

    return {
        "final_answer": answer,
        "sources": [],
        "agent_trace": [trace_entry],
    }


# ─── Build the Graph ──────────────────────────────────────

def build_pipeline() -> StateGraph:
    """Construct the LangGraph multi-agent pipeline."""

    workflow = StateGraph(PipelineState)

    # Add nodes
    workflow.add_node("load_memory", load_memory_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("reasoner", reasoner_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("save_memory", save_memory_node)
    workflow.add_node("no_docs_response", no_docs_response)

    # Define edges
    workflow.set_entry_point("load_memory")
    workflow.add_edge("load_memory", "retriever")

    # Conditional: if no docs retrieved, skip reasoning
    workflow.add_conditional_edges(
        "retriever",
        should_reason,
        {
            "has_docs": "reasoner",
            "no_docs": "no_docs_response",
        },
    )

    workflow.add_edge("reasoner", "synthesizer")
    workflow.add_edge("synthesizer", "save_memory")
    workflow.add_edge("no_docs_response", "save_memory")
    workflow.add_edge("save_memory", END)

    return workflow.compile()


# ─── Executor ─────────────────────────────────────────────

class Orchestrator:
    """Main executor for the multi-agent pipeline."""

    def __init__(self):
        self._pipeline = None

    def _get_pipeline(self):
        """Lazy-build the pipeline."""
        if self._pipeline is None:
            self._pipeline = build_pipeline()
            logger.info("[Orchestrator] Pipeline built successfully")
        return self._pipeline

    def run(
        self,
        question: str,
        session_id: str = "default",
        top_k: int = 5,
        include_sources: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute the full multi-agent pipeline.

        Args:
            question: User's question
            session_id: Session identifier for memory
            top_k: Number of documents to retrieve
            include_sources: Whether to include source citations

        Returns:
            Dict with answer, sources, agent trace, and timing
        """
        start = time.time()

        pipeline = self._get_pipeline()

        # Initialize state
        initial_state = {
            "question": question,
            "session_id": session_id,
            "top_k": top_k,
            "include_sources": include_sources,
            "conversation_context": "",
            "retrieved_documents": [],
            "retriever_output": {},
            "reasoner_output": {},
            "synthesizer_output": {},
            "final_answer": "",
            "sources": [],
            "agent_trace": [],
            "total_duration_ms": 0,
        }

        # Execute
        result = pipeline.invoke(initial_state)

        total_ms = (time.time() - start) * 1000

        logger.info(f"[Orchestrator] Pipeline complete in {total_ms:.1f}ms")

        return {
            "answer": result["final_answer"],
            "sources": result.get("sources", []),
            "agent_trace": result.get("agent_trace", []),
            "session_id": session_id,
            "query_duration_ms": round(total_ms, 2),
        }


# Singleton
orchestrator = Orchestrator()
