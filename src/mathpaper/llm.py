"""
Provider-agnostic LLM adapter for MathPaper AI.

Pick a provider with the LLM_PROVIDER env var (default: groq). Set the matching
API key env var. The rest of the system calls `call_llm(system, prompt, model)`
and never needs to know which provider is behind it.

    export LLM_PROVIDER=groq
    export GROQ_API_KEY=gsk_...
    python agents.py

model="small"  -> fast/cheap model (classification, memory rewrite)
model="strong" -> reasoning model (planning, verification, explanation)

No SDKs required — everything goes over plain HTTPS with urllib, so the only
dependency is the Python standard library.
"""

import json
import os
import urllib.request
import urllib.error

# ----------------------------------------------------------------------
# Provider registry: endpoint + model names for the small/strong roles.
# All except Anthropic use the OpenAI-compatible /chat/completions shape.
# ----------------------------------------------------------------------
PROVIDERS = {
    "groq": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "key_env": "GROQ_API_KEY",
        "small": "llama-3.1-8b-instant",
        "strong": "llama-3.3-70b-versatile",
        "style": "openai",
    },
    "gemini": {
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "key_env": "GEMINI_API_KEY",
        "small": "gemini-2.5-flash-lite",
        "strong": "gemini-2.5-flash",
        "style": "openai",
    },
    "openrouter": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "key_env": "OPENROUTER_API_KEY",
        "small": "meta-llama/llama-3.3-70b-instruct:free",
        "strong": "deepseek/deepseek-r1:free",
        "style": "openai",
    },
    "github": {
        "url": "https://models.github.ai/inference/chat/completions",
        "key_env": "GITHUB_TOKEN",
        "small": "openai/gpt-4o-mini",
        "strong": "openai/gpt-4o",
        "style": "openai",
    },
    "ollama": {  # fully local, no key
        "url": "http://localhost:11434/v1/chat/completions",
        "key_env": None,
        "small": "qwen2.5:3b",
        "strong": "qwen2.5:7b",
        "style": "openai",
    },
    "anthropic": {
        "url": "https://api.anthropic.com/v1/messages",
        "key_env": "ANTHROPIC_API_KEY",
        "small": "claude-haiku-4-5-20251001",
        "strong": "claude-sonnet-4-6",
        "style": "anthropic",
    },
}

PROVIDER = os.environ.get("LLM_PROVIDER", "groq").lower()


def _post(url, headers, payload):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{url} -> {e.code}: {e.read().decode()[:300]}")


def call_llm(system: str, prompt: str, model: str = "small") -> str:
    if PROVIDER not in PROVIDERS:
        raise ValueError(f"Unknown LLM_PROVIDER={PROVIDER}. Options: {list(PROVIDERS)}")
    cfg = PROVIDERS[PROVIDER]
    model_name = cfg[model]                       # "small" or "strong" -> real id
    key = os.environ.get(cfg["key_env"], "") if cfg["key_env"] else ""
    if cfg["key_env"] and not key:
        raise RuntimeError(f"Set {cfg['key_env']} for provider '{PROVIDER}'.")

    if cfg["style"] == "anthropic":
        headers = {
            "content-type": "application/json",
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        }
        data = _post(cfg["url"], headers, {
            "model": model_name, "max_tokens": 1200, "system": system,
            "messages": [{"role": "user", "content": prompt}],
        })
        return data["content"][0]["text"]

    # OpenAI-compatible shape (groq / gemini / openrouter / github / ollama)
    headers = {"content-type": "application/json"}
    if key:
        headers["authorization"] = f"Bearer {key}"
    data = _post(cfg["url"], headers, {
        "model": model_name, "max_tokens": 1200,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    })
    return data["choices"][0]["message"]["content"]


if __name__ == "__main__":
    # Smoke test: confirms the selected provider + key work end to end.
    print(f"Provider: {PROVIDER}")
    try:
        out = call_llm("Reply with exactly one word.", "Say 'ready'.", model="small")
        print("Small model replied:", out.strip())
        print("Adapter is working.")
    except Exception as e:
        print("Adapter test failed:", e)
