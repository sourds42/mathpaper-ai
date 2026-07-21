# MathPaper AI

**An agentic RAG system that explains mathematical concepts, derivations, and proofs from research papers.**

Instead of one LLM doing everything, MathPaper AI decomposes each question into a
team of specialized agents — a planner coordinates retrieval, prerequisite-filling,
memory, evidence verification, explanation, and citation checking. This modular
design improves accuracy, reduces hallucination, and cuts latency by invoking only
the agents a given query actually needs.

![CI](https://github.com/sourds42/maths-paper-rag/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sourds42/maths-paper-rag/blob/main/MathPaper_AI_Colab.ipynb)

---

## Why multi-agent?

A plain RAG pipeline fails on math papers for four reasons: equations get split
from their explanations during chunking; papers assume prerequisite knowledge they
never define; the model hallucinates when retrieval is incomplete; and follow-up
questions lose context. Each agent targets one of these failure modes.

```
                         User Question
                              │
                     Query Analyzer Agent      (intent + expertise level)
                              │
                       Planning Agent          (chooses the minimal pipeline)
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
  Paper Retrieval       Math Knowledge         Memory Agent
     Agent (hybrid)     Agent (fills gaps)     (multi-turn context)
        │                     │
        └──────────┬──────────┘
                   │
        Evidence Verification Agent    (blocks unsupported answers → re-retrieve)
                   │
        Explanation Generator Agent    (grounded, LaTeX, expertise-adapted)
                   │
        Citation Validation Agent      (rejects unsupported claims)
                   │
              Final Answer
```

**Dynamic orchestration** is the key idea: a simple variable lookup runs 3 agents;
a full derivation runs all 7. The planner picks the path per query.

---

## Quickstart

### Zero setup (no API key, runs offline)

```bash
pip install -e ".[dev]"
pytest tests/ -v        # orchestration tests with a mocked LLM
python evaluate.py      # retrieval benchmark + charts in results/
```

### Run on Google Colab (no local hardware needed)

Open `MathPaper_AI_Colab.ipynb` in Colab (badge above). It clones the repo,
installs everything, and has ready-to-run cells for: the offline benchmark, the
free-cloud-API pipeline (Groq), and local models on Colab's free T4 GPU. Ideal if
your own machine is low on RAM/VRAM.

### End-to-end with a free LLM provider

The agent layer is provider-agnostic (`src/mathpaper/llm.py`). Get a free key —
Groq and Google AI Studio need no credit card — then:

```bash
export LLM_PROVIDER=groq
export GROQ_API_KEY=gsk_your_key
python -m mathpaper.agents        # runs the full pipeline on a sample question
```

Supported providers: `groq`, `gemini`, `openrouter`, `github`, `ollama` (local),
`anthropic`. See [SETUP.md](SETUP.md) for keys and free-tier limits.

### Interactive demo

`demo/demo.jsx` is a self-contained React component: paste it into a Claude.ai
artifact to watch the agents light up in real time, with LaTeX-rendered equations
and clickable citations.

---

## Results

Reproducible mini-benchmark (15-chunk corpus, 16 labeled queries). Full write-up
and honest caveats in [RESULTS.md](RESULTS.md).

| Configuration | Recall@3 | MRR |
|---|---|---|
| Fixed chunks + dense (baseline) | 0.875 | 0.818 |
| Equation-anchored + dense | 0.938 | 0.856 |
| Equation-anchored + BM25 | 0.938 | 0.859 |
| Equation-anchored + hybrid (RRF) | 0.938 | 0.856 |

Dynamic orchestration cuts the most common query type (variable lookup) from 7
agent invocations to 3 — about 57% fewer LLM calls — while keeping the full
pipeline for derivations.

---

## Project layout

```
src/mathpaper/
  agents.py       8 agents + the Planner (dynamic orchestration, verify loop)
  retrieval.py    hybrid dense + BM25 retrieval (RRF), toy corpus
  llm.py          provider-agnostic LLM adapter (stdlib only)
tests/
  test_agents.py  orchestration tests with a mocked LLM (no key needed)
evaluate.py       retrieval + orchestration benchmark
demo/demo.jsx     interactive React demo with LaTeX rendering
results/          benchmark output (json + charts)
```

---

## How it works (interview notes)

- **Agents** are LLMs with focused roles that make decisions (the planner routes,
  the verifier blocks). **Tools** are what they call (vector search, BM25,
  reranker, external references). **Memory** is shared working state (`AgentState`)
  plus conversation history for follow-ups.
- **Model routing:** classification and memory use a small/fast model; planning,
  verification, and explanation use a stronger reasoning model — a deliberate
  cost/quality split, configurable per provider.
- **Testing strategy:** agent *routing* is tested separately from LLM *quality*
  with a mocked LLM, so the pipeline is debuggable and CI runs with no API key.

---

## License

MIT — see [LICENSE](LICENSE).
