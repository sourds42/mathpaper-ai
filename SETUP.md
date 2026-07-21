# Running MathPaper AI with a free LLM provider

The agent layer is provider-agnostic. Pick any provider below, set two env
vars, and run. No code changes needed to switch.

## 1. Get a free key (~2 min, no credit card)

| Provider | Where | Free tier |
|---|---|---|
| **Groq** (recommended, fast) | console.groq.com | ~30 req/min, Llama 3.3 70B |
| **Gemini** (best quality) | aistudio.google.com | 250k tokens/min, Gemini Flash |
| **OpenRouter** (variety) | openrouter.ai | ~30 free models |
| **GitHub Models** (free Claude) | github.com/marketplace/models | monthly budget, uses GitHub login |
| **Ollama** (fully offline) | ollama.com | no key, download model |

## 2. Set env vars and run

```bash
# Groq
export LLM_PROVIDER=groq
export GROQ_API_KEY=gsk_your_key
python llm.py          # smoke test -> should print "Adapter is working."
python agents.py       # full pipeline on a sample question

# Gemini
export LLM_PROVIDER=gemini
export GEMINI_API_KEY=your_key

# OpenRouter
export LLM_PROVIDER=openrouter
export OPENROUTER_API_KEY=sk-or-your_key

# GitHub Models
export LLM_PROVIDER=github
export GITHUB_TOKEN=ghp_your_token

# Ollama (local, no key)
ollama pull qwen2.5:3b && ollama pull qwen2.5:7b
export LLM_PROVIDER=ollama
```

## 3. What runs with no key at all

```bash
python evaluate.py       # retrieval benchmark + charts
python test_agents.py    # orchestration tests (mocked LLM)
```

## Notes
- The small/strong split is per-provider: classification & memory use the
  small model, planning/verification/explanation use the strong one.
- Free tiers train on your prompts (except Groq) and change monthly — don't
  send private data.
