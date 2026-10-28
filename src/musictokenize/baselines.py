"""Phase 3 baselines. Each has .score(context_ids) -> np.ndarray over the vocab."""
from collections import defaultdict

import numpy as np

from .data import END, PAD, START, UNK

FEATURES = ["genre", "mood", "tempo", "decade", "language", "vocals"]


def _counts(train_seqs, V, only_kept=False, skip_bursts=False):
    c = np.zeros(V)
    for s in train_seqs:
        if skip_bursts and s["is_burst"]:
            continue
        for i, k in zip(s["ids"][1:-1], s["skips"][1:-1]):
            if not (only_kept and k):
                c[i] += 1
    return c


class Random:
    name = "random"

    def __init__(self, V, seed=42):
        self.V, self.rng = V, np.random.default_rng(seed)

    def score(self, ctx):
        return self.rng.random(self.V)


class Popularity:
    """Most-played songs in train, ignoring context."""
    name = "popularity"

    def __init__(self, train_seqs, V):
        self.s = _counts(train_seqs, V)

    def score(self, ctx):
        return self.s


class KeepPopularity:
    """Songs most often played through (not skipped) in non-burst train sessions."""
    name = "keep_popularity"

    def __init__(self, train_seqs, V):
        self.s = _counts(train_seqs, V, only_kept=True, skip_bursts=True)

    def score(self, ctx):
        return self.s


class Bigram:
    """P(next | last) from train transitions, Dirichlet-smoothed toward popularity."""
    name = "bigram_markov"

    def __init__(self, train_seqs, V, alpha=5.0):
        pop = _counts(train_seqs, V)
        self.pop = pop / pop.sum()
        self.alpha = alpha
        self.nxt = defaultdict(lambda: defaultdict(int))
        for s in train_seqs:
            ids = s["ids"]
            for a, b in zip(ids[:-1], ids[1:]):
                self.nxt[a][b] += 1
        self.V = V

    def score(self, ctx):
        row = self.nxt.get(ctx[-1])
        out = self.alpha * self.pop
        if row:
            n = sum(row.values())
            for b, c in row.items():
                out[b] += c
            return out / (n + self.alpha)
        return out


class Item2Vec:
    """word2vec (skip-gram) over train sessions; score = cosine to the mean of the last songs."""
    name = "item2vec"

    def __init__(self, train_seqs, V, vocab, dim=64, window=5, epochs=15, last_k=3, seed=42):
        from gensim.models import Word2Vec

        special = {vocab[t] for t in (PAD, UNK, START, END)}
        sents = [[str(i) for i in s["ids"] if i not in special] for s in train_seqs]
        m = Word2Vec(sents, vector_size=dim, window=window, sg=1, negative=10, min_count=1,
                     epochs=epochs, workers=1, seed=seed)
        self.E = np.zeros((V, dim))
        self.known = np.zeros(V, dtype=bool)
        for w in m.wv.index_to_key:
            self.E[int(w)] = m.wv[w]
            self.known[int(w)] = True
        n = np.linalg.norm(self.E, axis=1, keepdims=True)
        self.E = np.divide(self.E, n, out=np.zeros_like(self.E), where=n > 0)
        self.pop = _counts(train_seqs, V)
        self.k = last_k

    def score(self, ctx):
        idx = [i for i in ctx[-self.k:] if self.known[i]]
        if not idx:
            return self.pop
        v = self.E[idx].mean(0)
        return self.E @ v


class FeatureNN:
    """Nearest neighbours on library features (genre, mood, tempo, decade, language, vocals, BPM).

    Only songs with features can be scored; if the last song has none, falls back to popularity.
    """
    name = "feature_knn"

    def __init__(self, train_seqs, V, vocab, songs):
        by_id = songs.set_index("track_id")
        self.F = -np.ones((V, len(FEATURES)), dtype=int)
        self.bpm = np.full(V, np.nan)
        cats = {f: {v: j for j, v in enumerate(sorted(songs[f].dropna().unique()))} for f in FEATURES}
        for tid, i in vocab.items():
            if tid in by_id.index and by_id.at[tid, "in_library"]:
                row = by_id.loc[tid]
                for j, f in enumerate(FEATURES):
                    if row[f] == row[f]:
                        self.F[i, j] = cats[f][row[f]]
                self.bpm[i] = row["bpm"]
        self.has = (self.F >= 0).any(1)
        self.pop = _counts(train_seqs, V)
        self.tie = self.pop / (self.pop.max() + 1) * 1e-3  # popularity breaks ties

    def score(self, ctx):
        last = ctx[-1]
        if not self.has[last]:
            return self.pop
        both = (self.F >= 0) & (self.F[last] >= 0)
        match = ((self.F == self.F[last]) & both).sum(1) / len(FEATURES)
        bpm = np.nan_to_num(1 - np.abs(self.bpm - self.bpm[last]) / 100, nan=0).clip(0, 1) * 0.1
        return np.where(self.has, match + bpm, -1) + self.tie
