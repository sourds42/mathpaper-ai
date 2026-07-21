"""Tests PDF ingestion: extraction, section detection, equation-anchored chunking."""
import fitz
import pytest
from mathpaper.ingest import pdf_to_corpus, corpus_summary


@pytest.fixture
def sample_pdf(tmp_path):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72),
        "Abstract\nWe study gradient descent for convex optimization.\n"
        "Method\nEquation (1): theta = theta - lambda * grad(L). Here lambda is "
        "the learning rate.\nWe minimize the loss L using stochastic updates.\n"
        "Results\nOur method converges faster than baseline SGD.\n", fontsize=11)
    p = tmp_path / "paper.pdf"
    doc.save(str(p)); doc.close()
    return str(p)


def test_extracts_chunks(sample_pdf):
    corpus = pdf_to_corpus(sample_pdf, max_words=40, min_words=6)
    assert len(corpus) >= 2
    assert all("id" in c and "text" in c and "section" in c for c in corpus)


def test_detects_sections(sample_pdf):
    corpus = pdf_to_corpus(sample_pdf, max_words=40, min_words=6)
    sections = {c["section"] for c in corpus}
    assert "method" in sections


def test_summary_runs(sample_pdf):
    corpus = pdf_to_corpus(sample_pdf, max_words=40, min_words=6)
    assert "chunks" in corpus_summary(corpus)
