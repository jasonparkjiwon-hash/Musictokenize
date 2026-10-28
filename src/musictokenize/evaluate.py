"""Shared offline evaluation for every model (Phase 3 onwards).

A scorer is any object with `score(context_ids: list[int]) -> np.ndarray` of length len(vocab):
higher = more likely next. Context always starts with <START>. Special tokens are masked out.
Sequence models may instead/also define `score_sequence(ids) -> array [len(ids)-1, V]` whose row
t-1 equals score(ids[:t]); it is used when present so a session costs one forward pass.

Protocol, on validation sequences, one position per play:
- target = the song actually played next. Positions whose target is <UNK> are skipped
  (the model cannot name it); `coverage` reports the share kept.
- hit@k / NDCG@k: is the target in the top k? Ties are scored by expected rank, so models that
  output many equal scores are not rewarded or punished for arbitrary ordering.
- skip_rate_hits@k: skip rate of the plays the model would have recommended (target in top k).
  This is the offline estimate of "skip rate of top picks": we only know how I reacted to songs
  I actually heard, so it is measured on hits. Compare it with `base_skip_rate`.
- pick_skip@k: average smoothed TRAIN skip rate of the model's k picks (does it suggest songs
  I usually skip?). Needs no hit, so it covers all positions.
Burst sessions (rapid-skip browsing) are reported separately: headline numbers exclude them.
"""
import numpy as np

from .data import SPECIALS

K = 10


def train_skip_rates(train_seqs: list[dict], vocab_size: int, prior_strength: float = 5.0):
    """Smoothed per-song skip rate from non-burst train plays (burst plays are masked)."""
    plays = np.zeros(vocab_size)
    skips = np.zeros(vocab_size)
    for s in train_seqs:
        if s["is_burst"]:
            continue
        for i, k in zip(s["ids"][1:-1], s["skips"][1:-1]):
            plays[i] += 1
            skips[i] += k
    base = skips.sum() / max(plays.sum(), 1)
    return (skips + prior_strength * base) / (plays + prior_strength), base


def _expected_hit_ndcg(greater: int, ties: int, k: int) -> tuple[float, float]:
    """Hit/NDCG for a target tied with `ties - 1` others, `greater` strictly above it."""
    if greater >= k:
        return 0.0, 0.0
    pos = greater + np.arange(min(ties, k - greater))
    return len(pos) / ties, float((1.0 / np.log2(pos + 2)).sum() / ties)


def evaluate(scorer, val_seqs: list[dict], vocab: dict, skip_rate: np.ndarray, k: int = K) -> dict:
    V = len(vocab)
    mask = np.zeros(V, dtype=bool)
    mask[[vocab[t] for t in SPECIALS]] = True
    unk = vocab["<UNK>"]
    rows = []  # (burst, hit, ndcg, target_skip, pick_skip)
    n_total = 0
    batched = hasattr(scorer, "score_sequence")
    for s in val_seqs:
        ids = s["ids"]
        all_scores = np.asarray(scorer.score_sequence(ids), dtype=float) if batched else None
        for t in range(1, len(ids) - 1):
            n_total += 1
            target = ids[t]
            if target == unk:
                continue
            sc = (all_scores[t - 1] if batched else np.asarray(scorer.score(ids[:t]), dtype=float)).copy()
            sc[mask] = -np.inf
            greater = int((sc > sc[target]).sum())
            ties = int((sc == sc[target]).sum())
            hit, ndcg = _expected_hit_ndcg(greater, ties, k)
            top = np.argpartition(-sc, k)[:k]
            rows.append((s["is_burst"], hit, ndcg, s["skips"][t], skip_rate[top].mean()))
    a = np.array(rows, dtype=float)
    burst, hit, ndcg, tskip, pick = a.T
    nb = burst == 0

    def stats(m):
        h = hit[m]
        return {
            "n": int(m.sum()),
            "expected_hits": float(h.sum()),  # sample size behind skip_rate_hits
            f"hit@{k}": float(h.mean()),
            f"ndcg@{k}": float(ndcg[m].mean()),
            f"hit@{k}_kept_targets": float(hit[m & (tskip == 0)].mean()),
            f"skip_rate_hits@{k}": float((hit[m] * tskip[m]).sum() / max(h.sum(), 1e-9)),
            f"pick_skip@{k}": float(pick[m].mean()),
            "base_skip_rate": float(tskip[m].mean()),
        }

    return {
        "coverage": len(rows) / n_total,
        "no_burst": stats(nb),
        "all": stats(np.ones_like(nb)),
    }
