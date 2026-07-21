"""
Hybrid retrieval: dense (vector) + BM25 (lexical), fused with Reciprocal Rank
Fusion, optionally reranked. In production the dense side is bge-large +
ChromaDB; here TF-IDF cosine keeps the module dependency-free so the
benchmark runs anywhere.
"""

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_demo_corpus() -> list[dict]:
    """Chunks from a toy VAE paper — equation-anchored chunking:
    each equation stays with its explanation."""
    return [
        {"id": "chunk_0", "section": "abstract", "text": "We propose a variational autoencoder for molecular generation trained with a composite objective balancing reconstruction and regularization."},
        {"id": "chunk_1", "section": "intro", "text": "Generative models for molecules must produce valid structures. Prior work uses GANs but suffers from mode collapse."},
        {"id": "chunk_2", "section": "method", "text": "Equation (3): L_rec = -E_q[log p(x|z)] is the reconstruction loss, the expected negative log-likelihood of the data under the decoder."},
        {"id": "chunk_3", "section": "method", "text": "Equation (5): L = L_rec + beta * D_KL(q(z|x) || p(z)). We minimize KL divergence between the approximate posterior q(z|x) and the prior p(z) to regularize the latent space."},
        {"id": "chunk_4", "section": "method", "text": "The hyperparameter beta controls the trade-off between reconstruction fidelity and latent regularization. We set beta = 0.5 by validation."},
        {"id": "chunk_5", "section": "method", "text": "The symbol lambda denotes the learning-rate decay coefficient in the scheduler, set to 0.95 per epoch."},
        {"id": "chunk_6", "section": "method", "text": "Equation (8): z = mu + sigma * epsilon, epsilon ~ N(0, I). The reparameterization trick allows gradients to flow through the sampling step."},
        {"id": "chunk_7", "section": "method", "text": "In Equation (8) the second term is sigma * epsilon; sigma is squared in the KL term of Equation (5) because the KL between Gaussians depends on the variance sigma^2, not the standard deviation."},
        {"id": "chunk_8", "section": "training", "text": "We use cross entropy for atom-type prediction rather than mean squared error because atom types are categorical; cross entropy matches the multinomial likelihood."},
        {"id": "chunk_9", "section": "training", "text": "Training runs for 200 epochs with Adam, batch size 128, on a single A100."},
        {"id": "chunk_10", "section": "results", "text": "Our model achieves 94.2% validity, beating the GAN baseline at 87.1%."},
        {"id": "chunk_11", "section": "results", "text": "Ablation: removing the KL term of Equation (5) collapses the latent space and validity drops to 71%."},
        {"id": "chunk_12", "section": "related", "text": "beta-VAE introduced the weighting of the divergence term to encourage disentangled representations."},
        {"id": "chunk_13", "section": "appendix", "text": "Derivation of Equation (5): starting from the ELBO, log p(x) >= E_q[log p(x|z)] - D_KL(q(z|x)||p(z)); maximizing the ELBO equals minimizing L."},
        {"id": "chunk_14", "section": "appendix", "text": "For Gaussian q and p, the divergence term has closed form 0.5 * sum(sigma^2 + mu^2 - 1 - log sigma^2)."},
    ]


class BM25Retriever:
    def __init__(self, corpus):
        self.corpus = corpus
        self.bm25 = BM25Okapi([c["text"].lower().split() for c in corpus])

    def search(self, query, k=5):
        scores = self.bm25.get_scores(query.lower().split())
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [self.corpus[i] for i in order]


class DenseRetriever:
    """Stand-in for bge-large embeddings + vector DB."""
    def __init__(self, corpus):
        self.corpus = corpus
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.mat = self.vec.fit_transform([c["text"] for c in corpus])

    def search(self, query, k=5):
        sims = cosine_similarity(self.vec.transform([query]), self.mat)[0]
        order = sims.argsort()[::-1][:k]
        return [self.corpus[i] for i in order]


class HybridRetriever:
    """Reciprocal Rank Fusion of dense + BM25 rankings."""
    def __init__(self, corpus, rrf_k=60):
        self.corpus = corpus
        self.dense = DenseRetriever(corpus)
        self.bm25 = BM25Retriever(corpus)
        self.rrf_k = rrf_k

    def search(self, query, k=5):
        scores = {}
        for retriever in (self.dense, self.bm25):
            for rank, chunk in enumerate(retriever.search(query, k=10)):
                scores[chunk["id"]] = scores.get(chunk["id"], 0) + 1 / (self.rrf_k + rank + 1)
        by_id = {c["id"]: c for c in self.corpus}
        top = sorted(scores, key=scores.get, reverse=True)[:k]
        return [by_id[i] for i in top]
