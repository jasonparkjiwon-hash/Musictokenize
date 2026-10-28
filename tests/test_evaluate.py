import numpy as np
import pandas as pd
import pytest

from musictokenize.baselines import Bigram, FeatureNN, KeepPopularity, Popularity
from musictokenize.evaluate import _expected_hit_ndcg, evaluate, train_skip_rates

# ids: 0 PAD, 1 UNK, 2 START, 3 END, songs 4..9
VOCAB = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3, **{f"s{i}": i for i in range(4, 10)}}
V = len(VOCAB)


def seq(songs, skips=None, burst=False):
    skips = skips or [0] * len(songs)
    return {"ids": [2, *songs, 3], "skips": [0, *skips, 0], "skip_types": [0] * (len(songs) + 2),
            "is_burst": burst}


class Fixed:
    """Scores by a fixed vector regardless of context."""
    def __init__(self, s):
        self.s = np.array(s, float)

    def score(self, ctx):
        return self.s


def test_expected_hit_ndcg_ties():
    assert _expected_hit_ndcg(0, 1, 10) == (1.0, 1.0)
    assert _expected_hit_ndcg(10, 1, 10) == (0.0, 0.0)
    hit, _ = _expected_hit_ndcg(5, 10, 10)  # 5 of 10 tie positions fall inside top 10
    assert hit == pytest.approx(0.5)


def test_perfect_scorer_hits_and_masks_specials():
    val = [seq([4, 5])]
    # specials score highest but must be masked; song 5 ranked top for every position
    sc = Fixed([99, 99, 99, 99, 1, 2, 0, 0, 0, 0])
    r = evaluate(sc, val, VOCAB, np.full(V, 0.5), k=1)
    # position 1: target 4 (rank 2nd) -> miss; position 2: target 5 -> hit
    assert r["no_burst"]["hit@1"] == pytest.approx(0.5)
    assert r["coverage"] == 1.0


def test_unk_targets_skipped_and_burst_separated():
    val = [seq([4, 1]), seq([4, 4, 4], burst=True)]
    r = evaluate(Fixed([0, 0, 0, 0, 5, 0, 0, 0, 0, 0]), val, VOCAB, np.full(V, 0.5), k=1)
    assert r["coverage"] == pytest.approx(4 / 5)  # one UNK target skipped
    assert r["no_burst"]["n"] == 1 and r["all"]["n"] == 4


def test_skip_rate_hits_and_pick_skip():
    val = [seq([4, 4], skips=[1, 0])]
    rate = np.full(V, 0.5); rate[4] = 0.9
    r = evaluate(Fixed([0, 0, 0, 0, 5, 0, 0, 0, 0, 0]), val, VOCAB, rate, k=1)["no_burst"]
    assert r["skip_rate_hits@1"] == pytest.approx(0.5)  # hits at both plays, one skipped
    assert r["pick_skip@1"] == pytest.approx(0.9)


def test_train_skip_rates_ignore_bursts():
    train = [seq([4, 4], skips=[1, 1]), seq([5] * 50, skips=[1] * 50, burst=True)]
    rate, base = train_skip_rates(train, V, prior_strength=1)
    assert base == 1.0 and rate[5] == base  # burst plays masked: song 5 has no data, gets the prior


def test_popularity_keep_and_bigram():
    train = [seq([4, 5, 4, 5], skips=[0, 1, 0, 1]), seq([4, 5], skips=[0, 1])]
    assert Popularity(train, V).score([2])[4] == 3
    keep = KeepPopularity(train, V).score([2])
    assert keep[4] == 3 and keep[5] == 0
    b = Bigram(train, V, alpha=1.0).score([2, 4])
    assert b.argmax() == 5  # 4 is always followed by 5
    assert Bigram(train, V).score([2, 9]).argmax() == 4  # unseen context backs off to popularity


def test_feature_knn_prefers_same_features_and_falls_back():
    songs = pd.DataFrame({
        "track_id": ["s4", "s5", "s6", "s7"], "in_library": [True, True, True, False],
        "genre": ["Rap", "Rap", "Jazz", None], "mood": ["Dark", "Dark", "Calm", None],
        "tempo": ["Fast"] * 3 + [None], "decade": ["2020s"] * 3 + [None],
        "language": ["En"] * 3 + [None], "vocals": ["Vocal"] * 3 + [None],
        "bpm": [150.0, 148.0, 60.0, None]})
    knn = FeatureNN([seq([4, 5, 6, 7])], V, VOCAB, songs)
    s = knn.score([2, 4])
    assert s[5] > s[6] and s[7] == -1 + knn.tie[7]  # no-feature song is last
    assert (knn.score([2, 7]) == knn.pop).all()  # last song has no features -> popularity


def test_item2vec_scores_cooccurring_songs_higher():
    from musictokenize.baselines import Item2Vec

    train = [seq([4, 5] * 20), seq([6, 7] * 20)]
    m = Item2Vec(train, V, VOCAB, dim=16, window=2, epochs=30, last_k=1)
    s = m.score([2, 4])
    assert s[5] > s[6] and s[5] > s[7]
    assert (m.score([2, 9]) == m.pop).all()  # song 9 never seen -> popularity fallback
