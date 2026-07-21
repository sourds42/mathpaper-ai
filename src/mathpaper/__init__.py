"""MathPaper AI — agentic RAG for explaining math in research papers."""
from .agents import PlanningAgent, AgentState, Intent
from .retrieval import HybridRetriever, load_demo_corpus
from .llm import call_llm

__all__ = ["PlanningAgent", "AgentState", "Intent",
           "HybridRetriever", "load_demo_corpus", "call_llm"]
__version__ = "0.1.0"
