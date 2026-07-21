"""Tests reference-free answer scoring."""
from mathpaper.evaluation import score_answer, composite

EV = [{"id": "chunk_3", "text": "We minimize KL divergence to regularize the latent space."},
      {"id": "chunk_1", "text": "Generative models must produce valid structures."}]


def test_good_answer_scores_high():
    ans = "We minimize KL divergence to regularize the latent space [chunk_3]."
    sc = score_answer(ans, EV)
    assert sc["citation_validity"] == 1.0
    assert sc["hallucination_flag"] == 0
    assert composite(sc) > 0.7


def test_hallucinated_citation_flagged():
    ans = "This cites [chunk_99] which does not exist."
    sc = score_answer(ans, EV)
    assert sc["hallucination_flag"] == 1
    assert sc["citation_validity"] == 0.0
    assert composite(sc) < 0.3


def test_uncited_answer_low_coverage():
    ans = "KL divergence regularizes the latent space."
    sc = score_answer(ans, EV)
    assert sc["citation_coverage"] == 0.0
