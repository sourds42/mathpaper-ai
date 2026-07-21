"""
Orchestration tests with a mocked LLM (no API key needed).

Verifies the behaviors that make this a real agentic system rather than a
fixed pipeline:
  - dynamic routing: simple queries skip agents they don't need
  - the verify -> math-knowledge -> re-verify loop (anti-hallucination)
  - memory rewriting follow-up questions into self-contained ones

Run:  pytest tests/ -v
"""
import json

import pytest

from mathpaper import agents
from mathpaper.retrieval import HybridRetriever, load_demo_corpus


class MockLLM:
    """Deterministic stand-in for a real model. Fresh instance per test so the
    verify-loop's one-shot 'insufficient then sufficient' behavior resets."""

    def __init__(self):
        self.count = 0
        self._verified_once = False

    def __call__(self, system, prompt, model="small"):
        self.count += 1
        if "Classify the user's question" in system:
            q = prompt.lower()
            if "lambda" in q or "represent" in q:
                return json.dumps({"intent": "variable_lookup", "expertise": "undergraduate"})
            if "derived" in q or "derivation" in q:
                return json.dumps({"intent": "derivation", "expertise": "researcher"})
            return json.dumps({"intent": "equation_explanation", "expertise": "undergraduate"})
        if "Rewrite the follow-up" in system:
            return "Why is sigma squared in the KL term of Equation (5)?"
        if "decide if the evidence is sufficient" in system.lower():
            if self._verified_once:
                return json.dumps({"sufficient": True, "missing_concepts": []})
            self._verified_once = True
            return json.dumps({"sufficient": False, "missing_concepts": ["KL divergence"]})
        if "mathematical reference" in system:
            return "KL divergence measures how one distribution diverges from another."
        if "every factual claim" in system:
            return json.dumps({"valid": True, "unsupported": []})
        return "MOCK ANSWER citing [chunk_3]."


@pytest.fixture
def planner(monkeypatch):
    """Patch the LLM the agents module uses, then build a fresh planner."""
    mock = MockLLM()
    monkeypatch.setattr(agents, "call_llm", mock)
    p = agents.PlanningAgent(HybridRetriever(load_demo_corpus()))
    p._mock = mock
    return p


def test_variable_lookup_skips_heavy_agents(planner):
    """'What does lambda represent?' should take the short pipeline."""
    s = planner.run("What does lambda represent?")
    assert not any(
        t.startswith(("EvidenceVerifier", "MathKnowledge", "CitationValidator"))
        for t in s.trace
    ), s.trace


def test_verify_loop_invokes_math_knowledge(planner):
    """A missing prerequisite should trigger the Math Knowledge Agent, then pass."""
    s = planner.run("Why is KL divergence minimized in Equation (5)?")
    assert any(t.startswith("MathKnowledge") for t in s.trace), s.trace
    assert s.verified


def test_memory_rewrites_followup(planner):
    """A follow-up should be resolved into a self-contained question."""
    planner.run("Explain Equation (8).")           # seed a prior turn
    s = planner.run("Why is the second term squared?")
    assert "sigma squared" in s.resolved_question, s.resolved_question


def test_answer_is_produced(planner):
    s = planner.run("Why use cross entropy instead of mean squared error?")
    assert s.answer
    assert "[chunk_" in s.answer
