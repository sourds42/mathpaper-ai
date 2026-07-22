"""
Tools the agents can call.

Main tool: `math_reference_lookup()` — fetches REAL definitions for a
mathematical concept from external references, so the Math Knowledge Agent
grounds prerequisites in a citable source instead of inventing them.

Sources (tried in order, all keyless):
  1. Wikipedia                  — broad coverage, clean extracts
  2. Encyclopedia of Mathematics — rigorous, Springer-backed math reference
  3. ProofWiki                  — definitions + proofs
  4. Wikibooks                  — textbook-style treatments
  5. Wolfram MathWorld          — concise formal definitions (HTML meta)
  6. arXiv                      — related papers, as further reading

Sources 1-4 are MediaWiki sites, so one generic client covers them all.
Every network call degrades gracefully: a failure returns None and the caller
falls back to the next source (and finally to the LLM).
"""

import json
import re
import urllib.parse
import urllib.request

_UA = {"User-Agent": "MathPaperAI/0.1 (educational RAG demo)"}

# ---- MediaWiki-backed sources -----------------------------------------
MEDIAWIKI_SOURCES = {
    "wikipedia": {
        "api": "https://en.wikipedia.org/w/api.php",
        "base": "https://en.wikipedia.org/wiki/",
        "label": "Wikipedia",
    },
    "encyclopediaofmath": {
        "api": "https://encyclopediaofmath.org/api.php",
        "base": "https://encyclopediaofmath.org/wiki/",
        "label": "Encyclopedia of Mathematics",
    },
    "proofwiki": {
        "api": "https://proofwiki.org/w/api.php",
        "base": "https://proofwiki.org/wiki/",
        "label": "ProofWiki",
    },
    "wikibooks": {
        "api": "https://en.wikibooks.org/w/api.php",
        "base": "https://en.wikibooks.org/wiki/",
        "label": "Wikibooks",
    },
}

# order tried by math_reference_lookup()
DEFAULT_ORDER = ["wikipedia", "encyclopediaofmath", "proofwiki", "mathworld", "wikibooks"]


def _get_json(url: str, timeout: int = 8):
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _get_text(url: str, timeout: int = 8) -> str:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode(errors="ignore")


def _strip_wikitext(text: str) -> str:
    """Crude wikitext -> plain text, for sites without the TextExtracts API."""
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)          # templates
    text = re.sub(r"\[\[([^\]|]*\|)?([^\]]*)\]\]", r"\2", text)  # links
    text = re.sub(r"'''?|<[^>]+>", "", text)            # bold/italic/html
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def mediawiki_lookup(source_key: str, concept: str, max_chars: int = 700) -> dict | None:
    """Search a MediaWiki site for a concept and return its intro text."""
    cfg = MEDIAWIKI_SOURCES.get(source_key)
    if not cfg:
        return None

    # 1) resolve the best page title via search
    q = urllib.parse.urlencode({
        "action": "query", "list": "search", "srsearch": concept,
        "format": "json", "srlimit": 1,
    })
    try:
        data = _get_json(f"{cfg['api']}?{q}")
        hits = data.get("query", {}).get("search", [])
        if not hits:
            return None
        title = hits[0]["title"]
    except Exception:
        return None

    # 2) try TextExtracts (clean plain-text intro)
    q = urllib.parse.urlencode({
        "action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1,
        "titles": title, "format": "json", "redirects": 1,
    })
    extract = ""
    try:
        data = _get_json(f"{cfg['api']}?{q}")
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            extract = (page.get("extract") or "").strip()
            break
    except Exception:
        extract = ""

    # 3) fall back to raw wikitext for sites without TextExtracts
    if not extract:
        q = urllib.parse.urlencode({
            "action": "query", "prop": "revisions", "rvprop": "content",
            "rvslots": "main", "titles": title, "format": "json", "redirects": 1,
        })
        try:
            data = _get_json(f"{cfg['api']}?{q}")
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                content = (page.get("revisions", [{}])[0]
                           .get("slots", {}).get("main", {}).get("*", ""))
                extract = _strip_wikitext(content)
                break
        except Exception:
            return None

    if not extract:
        return None
    return {
        "concept": concept,
        "text": extract[:max_chars],
        "source": cfg["base"] + urllib.parse.quote(title.replace(" ", "_")),
        "source_name": cfg["label"],
    }


def mathworld_lookup(concept: str, max_chars: int = 700) -> dict | None:
    """Wolfram MathWorld has no public API; pull the page's meta description."""
    slug = "".join(w.capitalize() for w in re.findall(r"[A-Za-z0-9]+", concept))
    url = f"https://mathworld.wolfram.com/{slug}.html"
    try:
        html = _get_text(url)
    except Exception:
        return None
    m = re.search(r'<meta\s+name="description"\s+content="([^"]+)"', html, re.I)
    if not m:
        return None
    text = re.sub(r"\s+", " ", m.group(1)).strip()
    if not text:
        return None
    return {"concept": concept, "text": text[:max_chars],
            "source": url, "source_name": "Wolfram MathWorld"}


def arxiv_lookup(concept: str, max_results: int = 3) -> list[dict]:
    """Related papers as further reading. Returns [] on failure."""
    q = urllib.parse.urlencode({
        "search_query": f"all:{concept}", "start": 0, "max_results": max_results,
    })
    try:
        xml = _get_text(f"http://export.arxiv.org/api/query?{q}")
    except Exception:
        return []
    out = []
    for entry in re.findall(r"<entry>(.*?)</entry>", xml, re.DOTALL):
        title = re.search(r"<title>(.*?)</title>", entry, re.DOTALL)
        link = re.search(r"<id>(.*?)</id>", entry, re.DOTALL)
        if title and link:
            out.append({
                "title": re.sub(r"\s+", " ", title.group(1)).strip(),
                "source": link.group(1).strip(),
                "source_name": "arXiv",
            })
    return out


def math_reference_lookup(concept: str, order: list[str] | None = None) -> dict | None:
    """Try each reference source in turn; return the first real definition found."""
    for key in (order or DEFAULT_ORDER):
        hit = (mathworld_lookup(concept) if key == "mathworld"
               else mediawiki_lookup(key, concept))
        if hit and len(hit["text"].split()) >= 8:   # skip stubs
            return hit
    return None


# backwards-compatible alias (earlier version exposed only Wikipedia)
def wikipedia_lookup(concept: str, max_chars: int = 700) -> dict | None:
    return mediawiki_lookup("wikipedia", concept, max_chars=max_chars)
