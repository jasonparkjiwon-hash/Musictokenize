import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from musictokenize.evaluate import evaluate  # noqa: E402
from musictokenize.model import GRUScorer, SongGRU  # noqa: E402
from musictokenize.train import IGNORE, collate, make_windows, run_epoch, split_dev  # noqa: E402

VOCAB = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3, **{f"s{i}": i for i in range(4, 10)}}
V = len(VOCAB)


def seq(songs, burst=False, ts="2026-01-01"):
    return {"ids": [2, *songs, 3], "skips": [0] * (len(songs) + 2),
            "skip_types": [0] * (len(songs) + 2), "is_burst": burst,
            "ts": pd.Timestamp(ts, tz="UTC")}


def test_windows_cover_each_target_once_and_respect_bursts():
    s = seq(list(range(4, 10)) * 2)  # ids length 14 -> 13 targets
    w = make_windows([s], max_len=5)
    assert sum(len(x) - 1 for x in w) == 13
    assert all(len(x) <= 6 for x in w)
    assert make_windows([seq([4, 5], burst=True)], 5, include_bursts=False) == []
    assert len(make_windows([seq([4, 5], burst=True)], 5, include_bursts=True)) == 1


def test_collate_pads_and_ignores_unk_targets():
    x, y = collate([[2, 4, 1, 3], [2, 5]], unk=1)
    assert x.shape == (2, 3)
    assert y.tolist() == [[4, IGNORE, 3], [5, IGNORE, IGNORE]]  # UNK and padding ignored


def test_split_dev_keeps_sessions_whole():
    seqs = [seq([4], ts="2026-01-01"), seq([4], ts="2026-05-20"), seq([4], ts="2026-06-01")]
    fit, dev = split_dev(seqs, dev_months=1)
    assert len(fit) == 1 and len(dev) == 2


def test_model_shapes_and_weight_tying():
    m = SongGRU(V, 8, 16, tie_weights=True)
    assert m(torch.tensor([[2, 4, 5]])).shape == (1, 3, V)
    assert m.out.weight is m.embed.weight


def test_scorer_sequence_matches_per_position():
    m = SongGRU(V, 8, 16, dropout=0.0)
    sc = GRUScorer(m)
    ids = [2, 4, 5, 6, 3]
    seq_scores = sc.score_sequence(ids)
    for t in range(1, len(ids) - 1):
        assert np.allclose(seq_scores[t - 1], sc.score(ids[:t]), atol=1e-5)


def test_gru_learns_a_deterministic_cycle():
    torch.manual_seed(0)
    train = [seq([4, 5, 6, 7] * 6) for _ in range(20)]
    w = make_windows(train, max_len=16)
    m = SongGRU(V, 16, 32, dropout=0.0, tie_weights=False)
    opt = torch.optim.Adam(m.parameters(), lr=0.02)
    rng = np.random.default_rng(0)
    first = run_epoch(m, w, 1, 8, opt, shuffle=True, rng=rng)
    for _ in range(25):
        last = run_epoch(m, w, 1, 8, opt, shuffle=True, rng=rng)
    assert last < first * 0.3
    r = evaluate(GRUScorer(m), [seq([4, 5, 6, 7] * 3)], VOCAB, np.full(V, 0.5), k=1)
    assert r["no_burst"]["hit@1"] > 0.9
