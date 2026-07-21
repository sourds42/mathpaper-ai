"""
MathPaper AI — Agentic RAG for explaining math in research papers.
Each agent has one job. The Planner decides which agents run (dynamic orchestration).

LLM calls are routed through `call_llm()` — plug in any provider there.
"""

from dataclasses import dataclass, field
from enum import Enum
import json
import re

from .retrieval import HybridRetriever


# ----------------------------------------------------------------------
# LLM adapter — provider chosen via LLM_PROVIDER env var (see llm.py).
# Works with Groq, Gemini, OpenRouter, GitHub Models, Ollama, or Anthropic.
# ----------------------------------------------------------------------
from .llm import call_llm  # noqa: E402


# ----------------------------------------------------------------------
# Robust JSON parsing for model output.
# Smaller / reasoning models don't always obey "reply only JSON": DeepSeek-R1
# wraps output in <think>...</think>, some models add prose or ```json fences,
# and tiny models occasionally emit syntactically invalid JSON. safe_json
# repairs what it can and falls back to a caller-supplied default so one bad
# response never crashes the pipeline.
# ----------------------------------------------------------------------
def safe_json(raw: str, default: dict) -> dict:
    if raw is None:
        return dict(default)
    # try as-is first
    try:
        return json.loads(raw)
    except Exception:
        pass
    # strip reasoning tags + code fences, then extract the first {...} object
    cleaned = re.sub(r"<think>.*?</think>", "", str(raw), flags=re.DOTALL)
    cleaned = cleaned.replace("```json", "").replace("```", "")
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    # give up gracefully — return the safe default so the pipeline continues
    return dict(default)


# ----------------------------------------------------------------------
# Shared state passed between agents
# ----------------------------------------------------------------------
class Intent(str, Enum):
    VARIABLE_LOOKUP = "variable_lookup"
    EQUATION_EXPLANATION = "equation_explanation"
    CONCEPT_COMPARISON = "concept_comparison"
    DERIVATION = "derivation"
    SUMMARY = "summary"


@dataclass
class AgentState:
    question: str
    intent: Intent | None = None
    expertise: str = "undergraduate"
    resolved_question: str = ""          # after memory resolution
    evidence: list[dict] = field(default_factory=list)
    external_knowledge: list[dict] = field(default_factory=list)
    verified: bool = False
    missing: list[str] = field(default_factory=list)
    answer: str = ""
    trace: list[str] = field(default_factory=list)   # which agents ran

    def log(self, agent: str, note: str = ""):
        self.trace.append(f"{agent}: {note}" if note else agent)


# ----------------------------------------------------------------------
# Agents
# ----------------------------------------------------------------------
class QueryAnalyzerAgent:
    SYSTEM = (
        "Classify the user's question about a research paper. "
        'Reply ONLY with JSON: {"intent": one of '
        '["variable_lookup","equation_explanation","concept_comparison",'
        '"derivation","summary"], "expertise": one of '
        '["beginner","undergraduate","researcher"]}'
    )

    def run(self, state: AgentState) -> AgentState:
        raw = call_llm(self.SYSTEM, state.question, model="small")
        data = safe_json(raw, {"intent": "equation_explanation",
                               "expertise": "undergraduate"})
        try:
            state.intent = Intent(data.get("intent", "equation_explanation"))
        except ValueError:
            state.intent = Intent.EQUATION_EXPLANATION
        state.expertise = data.get("expertise", "undergraduate")
        state.log("QueryAnalyzer", state.intent.value)
        return state


class MemoryAgent:
    """Rewrites follow-ups ('why is the second term squared?') into
    self-contained questions using conversation history."""

    def __init__(self):
        self.history: list[dict] = []
        self.active_context = {"equation": None, "paper": None}

    def run(self, state: AgentState) -> AgentState:
        if not self.history:
            state.resolved_question = state.question
        else:
            system = (
                "Rewrite the follow-up question so it is fully self-contained, "
                "using the conversation history. Reply with the question only."
            )
            prompt = f"History: {json.dumps(self.history[-6:])}\n\nFollow-up: {state.question}"
            state.resolved_question = call_llm(system, prompt, model="small")
        state.log("Memory", state.resolved_question[:60])
        return state

    def commit(self, state: AgentState):
        self.history.append({"q": state.resolved_question, "a": state.answer[:500]})


class PaperRetrievalAgent:
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def run(self, state: AgentState, extra_query: str | None = None) -> AgentState:
        query = extra_query or state.resolved_question or state.question
        hits = self.retriever.search(query, k=5)
        state.evidence.extend(h for h in hits if h not in state.evidence)
        state.log("PaperRetrieval", f"{len(hits)} chunks for '{query[:40]}'")
        return state


class MathKnowledgeAgent:
    """Fills prerequisite gaps the paper assumes (e.g. KL divergence definition)."""

    SYSTEM = (
        "You are a mathematical reference. Give a precise, textbook-style "
        "definition of the requested concept in 3-5 sentences with the key formula."
    )

    def run(self, state: AgentState) -> AgentState:
        for concept in state.missing:
            definition = call_llm(self.SYSTEM, concept, model="strong")
            state.external_knowledge.append({"concept": concept, "text": definition})
        state.log("MathKnowledge", f"filled {len(state.missing)} gaps")
        state.missing = []
        return state


class EvidenceVerificationAgent:
    SYSTEM = (
        "Given a question and retrieved evidence, decide if the evidence is "
        "sufficient to answer WITHOUT guessing. Check: are all symbols defined? "
        "are prerequisite concepts present? "
        'Reply ONLY JSON: {"sufficient": bool, "missing_concepts": [..]}'
    )

    def run(self, state: AgentState) -> AgentState:
        prompt = (
            f"Question: {state.resolved_question}\n\n"
            f"Paper evidence: {json.dumps([e['text'] for e in state.evidence])}\n\n"
            f"External knowledge: {json.dumps([k['text'] for k in state.external_knowledge])}"
        )
        data = safe_json(call_llm(self.SYSTEM, prompt, model="strong"),
                         {"sufficient": True, "missing_concepts": []})
        state.verified = data.get("sufficient", True)
        state.missing = data.get("missing_concepts", [])
        state.log("EvidenceVerifier", "sufficient" if state.verified else f"missing: {state.missing}")
        return state


class ExplanationGeneratorAgent:
    def run(self, state: AgentState) -> AgentState:
        system = (
            f"Explain for a {state.expertise}-level reader. Use ONLY the provided "
            "evidence. Cite chunk ids like [chunk_3]. Break derivations into steps. "
            "If evidence does not support a claim, say so instead of guessing."
        )
        prompt = (
            f"Question: {state.resolved_question}\n\n"
            f"Evidence: {json.dumps(state.evidence)}\n\n"
            f"Background: {json.dumps(state.external_knowledge)}"
        )
        state.answer = call_llm(system, prompt, model="strong")
        state.log("Generator")
        return state


class CitationValidationAgent:
    SYSTEM = (
        "Check that every factual claim in the answer is supported by the cited "
        'evidence chunks. Reply ONLY JSON: {"valid": bool, "unsupported": [..]}'
    )

    def run(self, state: AgentState) -> AgentState:
        prompt = f"Answer: {state.answer}\n\nEvidence: {json.dumps(state.evidence)}"
        data = safe_json(call_llm(self.SYSTEM, prompt, model="strong"),
                         {"valid": True, "unsupported": []})
        valid = data.get("valid", True)
        state.log("CitationValidator", "valid" if valid else "rejected")
        if not valid:
            state.verified = False           # forces another retrieval cycle
            state.missing = data.get("unsupported", [])
        return state


# ----------------------------------------------------------------------
# Planner — dynamic orchestration (the fix for the latency problem)
# ----------------------------------------------------------------------
class PlanningAgent:
    """Maps intent -> minimal agent pipeline. Simple queries skip 4 agents."""

    PIPELINES = {
        Intent.VARIABLE_LOOKUP:      ["retrieve", "generate"],
        Intent.SUMMARY:              ["retrieve", "generate"],
        Intent.EQUATION_EXPLANATION: ["memory", "retrieve", "verify", "generate"],
        Intent.CONCEPT_COMPARISON:   ["memory", "retrieve", "verify", "generate", "cite"],
        Intent.DERIVATION:           ["memory", "retrieve", "verify", "generate", "cite"],
    }
    MAX_RETRIEVAL_CYCLES = 3

    def __init__(self, retriever: HybridRetriever):
        self.memory = MemoryAgent()
        self.agents = {
            "analyze": QueryAnalyzerAgent(),
            "memory": self.memory,
            "retrieve": PaperRetrievalAgent(retriever),
            "math": MathKnowledgeAgent(),
            "verify": EvidenceVerificationAgent(),
            "generate": ExplanationGeneratorAgent(),
            "cite": CitationValidationAgent(),
        }

    def run(self, question: str, on_step=None) -> AgentState:
        """on_step(label) is called before each agent runs, for live progress UIs.
        It's optional — existing callers pass nothing and behavior is unchanged."""
        def notify(label):
            if on_step:
                on_step(label)

        state = AgentState(question=question)
        notify("Query Analyzer")
        state = self.agents["analyze"].run(state)
        plan = self.PIPELINES[state.intent]
        notify("Planner")
        state.log("Planner", f"plan={plan}")

        if "memory" not in plan:
            state.resolved_question = question

        labels = {
            "memory": "Memory", "retrieve": "Paper Retrieval",
            "verify": "Evidence Verifier", "generate": "Explanation Generator",
            "cite": "Citation Validator",
        }
        for step in plan:
            if step == "verify":
                # verification loop: retrieve more / fill math gaps until sufficient
                for _ in range(self.MAX_RETRIEVAL_CYCLES):
                    notify("Evidence Verifier")
                    state = self.agents["verify"].run(state)
                    if state.verified:
                        break
                    if state.missing:
                        notify("Math Knowledge")
                        state = self.agents["math"].run(state)
                    else:
                        notify("Paper Retrieval")
                        state = self.agents["retrieve"].run(
                            state, extra_query=state.resolved_question
                        )
            else:
                notify(labels.get(step, step))
                state = self.agents[step].run(state)

        self.memory.commit(state)
        return state


if __name__ == "__main__":
    from .retrieval import load_demo_corpus
    planner = PlanningAgent(HybridRetriever(load_demo_corpus()))
    result = planner.run("Why is KL divergence minimized in Equation (5)?")
    print("\n".join(result.trace))
    print("\n--- ANSWER ---\n", result.answer)
