"""
Turn an uploaded PDF research paper into a retrieval corpus, so the system works
on any paper — not just the built-in demo.

Pipeline: extract text (PyMuPDF) -> split into paragraph-ish blocks ->
equation-anchored chunking (keep lines with math glued to their surrounding
explanation) -> emit chunks in the same {id, section, text} shape the retriever
expects.

Note: this is a lightweight extractor. It handles text and inline math well; it
does NOT perfectly reconstruct complex typeset equations (that needs a heavy
parser like Nougat). Good enough for a live demo on most papers.
"""

import re


def _extract_pages(pdf_path: str) -> list[str]:
    """Return a list of page texts. Requires PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise RuntimeError(
            "PyMuPDF is required for PDF upload. Install with: pip install pymupdf"
        ) from e
    doc = fitz.open(pdf_path)
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return pages


# crude section detector: short line, title-ish, common heading words
_SECTION_RE = re.compile(
    r"^\s*(\d+\.?\s+)?(abstract|introduction|related work|background|method|"
    r"methods|approach|model|experiments?|results?|evaluation|discussion|"
    r"conclusion|appendix|references)\b",
    re.IGNORECASE,
)
# lines that look like they contain math
_MATH_HINT = re.compile(
    r"(=|\\|\^|_|\bequation\b|\blemma\b|\btheorem\b|\bproof\b|"
    r"[α-ωΑ-Ω]|\b(sigma|beta|lambda|mu|epsilon|theta|gamma|delta|phi|psi)\b)",
    re.IGNORECASE,
)


def _clean(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def pdf_to_corpus(pdf_path: str, max_words: int = 120, min_words: int = 12) -> list[dict]:
    """Extract and chunk a PDF into retrieval chunks.

    Equation-anchored: when a paragraph contains math, we keep it intact (don't
    split mid-equation) and attach the following explanatory sentence if the
    paragraph is short — mirroring the demo corpus design.
    """
    pages = _extract_pages(pdf_path)
    full = _clean("\n".join(pages))

    # Split into blocks on blank lines. If the PDF has no blank-line paragraph
    # breaks (common), fall back to treating each non-empty line as a block so
    # section detection and chunking still work.
    blocks = [b.strip() for b in re.split(r"\n\s*\n", full) if b.strip()]
    if len(blocks) <= 1:
        blocks = [ln.strip() for ln in full.split("\n") if ln.strip()]

    chunks: list[dict] = []
    section = "body"
    cid = 0
    buffer = ""

    def flush(buf, sec):
        nonlocal cid
        buf = buf.strip()
        if len(buf.split()) >= min_words:
            chunks.append({"id": f"chunk_{cid}", "section": sec, "text": buf})
            cid += 1

    for block in blocks:
        first_line = block.split("\n", 1)[0].strip()
        if _SECTION_RE.match(first_line) and len(first_line.split()) <= 6:
            # new section heading — flush what we have, switch section
            flush(buffer, section)
            buffer = ""
            section = first_line.lower()
            continue

        block = block.replace("\n", " ").strip()
        candidate = (buffer + " " + block).strip() if buffer else block

        # flush when the buffer would exceed the target size. Math blocks get a
        # little extra room (so a short equation stays with its explanation) but
        # still flush eventually rather than absorbing the whole paper.
        limit = max_words + (40 if _MATH_HINT.search(block) else 0)
        if buffer and len(candidate.split()) > limit:
            flush(buffer, section)
            buffer = block
        else:
            buffer = candidate

    flush(buffer, section)

    # fallback: if extraction produced almost nothing (scanned PDF), say so
    if not chunks:
        raise RuntimeError(
            "No extractable text found. This may be a scanned/image PDF that needs OCR."
        )
    return chunks


def corpus_summary(corpus: list[dict]) -> str:
    """One-line human summary of an ingested corpus."""
    sections = sorted({c["section"] for c in corpus})
    words = sum(len(c["text"].split()) for c in corpus)
    return (f"{len(corpus)} chunks · ~{words} words · "
            f"sections: {', '.join(sections[:8])}"
            + (" …" if len(sections) > 8 else ""))
