"""Tests the multi-source reference lookup with a mocked network layer."""
from mathpaper import tools


def _fake_json(search_hit=True, extract="A precise definition of the concept in question here."):
    def inner(url, timeout=8):
        if "list=search" in url:
            return {"query": {"search": ([{"title": "Some Title"}] if search_hit else [])}}
        return {"query": {"pages": {"1": {"extract": extract}}}}
    return inner


def test_mediawiki_lookup_returns_definition(monkeypatch):
    monkeypatch.setattr(tools, "_get_json", _fake_json())
    hit = tools.mediawiki_lookup("wikipedia", "KL divergence")
    assert hit is not None
    assert hit["source_name"] == "Wikipedia"
    assert hit["source"].startswith("https://en.wikipedia.org/wiki/")


def test_lookup_falls_through_to_next_source(monkeypatch):
    def selective(url, timeout=8):
        if "list=search" in url:
            if "en.wikipedia.org" in url:
                return {"query": {"search": []}}          # wikipedia misses
            return {"query": {"search": [{"title": "T"}]}}  # next source hits
        return {"query": {"pages": {"1": {"extract":
                "A rigorous definition from a mathematics encyclopedia source."}}}}
    monkeypatch.setattr(tools, "_get_json", selective)
    monkeypatch.setattr(tools, "_get_text", lambda u, timeout=8: "")
    hit = tools.math_reference_lookup("obscure concept")
    assert hit is not None
    assert hit["source_name"] != "Wikipedia"


def test_network_failure_returns_none(monkeypatch):
    def boom(url, timeout=8):
        raise OSError("no network")
    monkeypatch.setattr(tools, "_get_json", boom)
    monkeypatch.setattr(tools, "_get_text", boom)
    assert tools.math_reference_lookup("anything") is None


def test_stub_results_are_skipped(monkeypatch):
    monkeypatch.setattr(tools, "_get_json", _fake_json(extract="Short."))
    monkeypatch.setattr(tools, "_get_text", lambda u, timeout=8: "")
    assert tools.math_reference_lookup("x") is None   # too short -> treated as stub


def test_arxiv_parses_entries(monkeypatch):
    xml = "<entry><title>A Paper On Divergence</title><id>http://arxiv.org/abs/1234</id></entry>"
    monkeypatch.setattr(tools, "_get_text", lambda u, timeout=8: xml)
    papers = tools.arxiv_lookup("divergence")
    assert papers and papers[0]["source"] == "http://arxiv.org/abs/1234"
