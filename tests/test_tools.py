"""Tests the reference-lookup tool with a mocked network layer (no real HTTP)."""
from mathpaper import tools


def _install_fake(monkeypatch, extract="A definition.", typ="standard", found=True):
    def fake_get(url, timeout=8):
        if "list=search" in url:
            return {"query": {"search": ([{"title": "Some Title"}] if found else [])}}
        return {"extract": extract, "type": typ,
                "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/X"}}}
    monkeypatch.setattr(tools, "_get", fake_get)


def test_lookup_returns_definition(monkeypatch):
    _install_fake(monkeypatch, extract="KL divergence measures distribution difference.")
    hit = tools.wikipedia_lookup("KL divergence")
    assert hit is not None
    assert "divergence" in hit["text"].lower()
    assert hit["source"].startswith("https://")


def test_disambiguation_is_skipped(monkeypatch):
    _install_fake(monkeypatch, typ="disambiguation")
    # disambiguation on the direct title, but search fallback also returns same -> None
    assert tools.wikipedia_lookup("Mercury") is None


def test_network_failure_returns_none(monkeypatch):
    def boom(url, timeout=8):
        raise OSError("no network")
    monkeypatch.setattr(tools, "_get", boom)
    assert tools.wikipedia_lookup("anything") is None
