"""
Reference-free evaluation of MathPaper AI answers.

We can't assume a gold-answer set for an arbitrary uploaded paper, so these are
*reference-free* metrics computed against the retrieved evidence — the same idea
behind faithfulness/groundedness scores in Ragas/DeepEval, implemented locally so
the demo needs no extra API:

  - citation_validity : fraction of [chunk_N] citations that point to real chunks
  - citation_coverage : does the answer cite at least one source? (0/1)
  - groundedness      : token overlap between answer content and the cited chunks
                        (proxy for "is the answer supported by the evidence")
  - hallucination_flag: 1 if the answer cites a chunk id that doesn't exist
  - length            : answer word count (for context, not quality)

These are heuristics, not ground truth — good for *comparing models on the same
question*, which is exactly the demo's job.
"""

import re

_CITE_RE = re.compile(r"\[(chunk_\d+)\]")
_WORD_RE = re.compile(r"[a-zA-Z]{3,}")

# very small stoplist so overlap reflects content, not filler
_STOP = {
    "the", "and", "for", "that", "this", "with", "are", "was", "which", "from",
    "have", "has", "not", "but", "can", "its", "into", "than", "then", "because",
    "used", "use", "using", "between", "each", "also", "more", "these", "those",
    "such", "when", "where", "what", "why", "how", "does", "term", "terms",
}


def _content_tokens(text: str) -> set:
    return {w.lower() for w in _WORD_RE.findall(text) if w.lower() not in _STOP}


def score_answer(answer: str, evidence: list[dict]) -> dict:
    """Score one answer against the chunks that were retrieved for it."""
    answer = answer or ""
    valid_ids = {c["id"] for c in evidence}
    cited = _CITE_RE.findall(answer)
    cited_set = set(cited)

    real = [c for c in cited if c in valid_ids]
    citation_validity = (len(real) / len(cited)) if cited else 0.0
    citation_coverage = 1.0 if cited else 0.0
    hallucination_flag = 1 if any(c not in valid_ids for c in cited_set) else 0

    # groundedness: overlap of answer content tokens with the CITED chunks'
    # tokens (fall back to all evidence if nothing was cited)
    cited_chunks = [c for c in evidence if c["id"] in cited_set] or evidence
    evidence_tokens = set()
    for c in cited_chunks:
        evidence_tokens |= _content_tokens(c["text"])
    ans_tokens = _content_tokens(answer)
    if ans_tokens:
        groundedness = len(ans_tokens & evidence_tokens) / len(ans_tokens)
    else:
        groundedness = 0.0

    return {
        "citation_validity": round(citation_validity, 3),
        "citation_coverage": round(citation_coverage, 3),
        "groundedness": round(groundedness, 3),
        "hallucination_flag": hallucination_flag,
        "n_citations": len(cited_set),
        "length": len(answer.split()),
    }


def composite(scores: dict) -> float:
    """A single 0-1 quality proxy blending the sub-metrics (higher = better)."""
    return round(
        0.4 * scores["citation_validity"]
        + 0.2 * scores["citation_coverage"]
        + 0.4 * scores["groundedness"]
        - 0.2 * scores["hallucination_flag"],
        3,
    )
