"""
Tools the agents can call. Currently: a reference-lookup tool that fetches real
definitions from Wikipedia, so the Math Knowledge Agent grounds prerequisite
concepts in an external source instead of inventing them from LLM memory.

Wikipedia's REST summary API needs no key and returns a clean extract:
    https://en.wikipedia.org/api/rest_v1/page/summary/<Title>

Design notes:
- Network calls are wrapped so a failure (offline, rate limit, missing page)
  never crashes the pipeline — the caller gets None and can fall back.
- We try the concept as-is, then a couple of light normalizations, because
  math concepts often live under slightly different titles ("Kullback-Leibler
  divergence" vs "KL divergence").
"""

import json
import urllib.parse
import urllib.request

_API = "https://en.wikipedia.org/api/rest_v1/page/summary/"
_SEARCH = "https://en.wikipedia.org/w/api.php"
_UA = {"User-Agent": "MathPaperAI/0.1 (educational RAG demo)"}


def _get(url: str, timeout: int = 8):
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _search_title(concept: str) -> str | None:
    """Use Wikipedia search to resolve the best page title for a concept."""
    params = urllib.parse.urlencode({
        "action": "query", "list": "search", "srsearch": concept,
        "format": "json", "srlimit": 1,
    })
    try:
        data = _get(f"{_SEARCH}?{params}")
        hits = data.get("query", {}).get("search", [])
        return hits[0]["title"] if hits else None
    except Exception:
        return None


def wikipedia_lookup(concept: str, max_chars: int = 600) -> dict | None:
    """Return {"concept", "text", "source"} for a concept, or None if not found.

    Tries the summary endpoint directly, then falls back to search-then-summary
    so slightly-off concept names still resolve.
    """
    candidates = [concept]
    resolved = _search_title(concept)
    if resolved and resolved not in candidates:
        candidates.append(resolved)

    for title in candidates:
        try:
            data = _get(_API + urllib.parse.quote(title.replace(" ", "_")))
        except Exception:
            continue
        extract = (data.get("extract") or "").strip()
        # skip disambiguation pages and empty extracts
        if extract and data.get("type") != "disambiguation":
            page = data.get("content_urls", {}).get("desktop", {}).get("page", "")
            return {
                "concept": concept,
                "text": extract[:max_chars],
                "source": page or f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title)}",
            }
    return None
