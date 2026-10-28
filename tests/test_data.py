import json

import pandas as pd
import pytest

from musictokenize.data import (
    END,
    START,
    UNK,
    add_track_counts,
    assign_sessions,
    build_vocab,
    drop_single_hard_skips,
    flag_bursts,
    label_skips,
    load_history,
    skip_label_agreement,
    time_split,
    to_sequences,
)


def plays(rows):
    """rows: (track_id, 'YYYY-MM-DD HH:MM', ms_played[, skipped_flag])"""
    df = pd.DataFrame(
        [
            {
                "track_id": r[0],
                "ts": pd.Timestamp(r[1], tz="UTC"),
                "ms_played": r[2],
                "skipped_flag": r[3] if len(r) > 3 else None,
            }
            for r in rows
        ]
    )
    df["skipped_flag"] = df["skipped_flag"].astype("boolean")
    return df.sort_values("ts").reset_index(drop=True)


def test_session_split_on_gap():
    df = plays([("a", "2026-01-01 10:00", 1), ("b", "2026-01-01 10:20", 1),
                ("c", "2026-01-01 11:00", 1)])  # 40 min after b -> new session
    assert assign_sessions(df, 30)["session_id"].tolist() == [0, 0, 1]


def test_session_gap_exactly_threshold_stays_together():
    df = plays([("a", "2026-01-01 10:00", 1), ("b", "2026-01-01 10:30", 1)])
    assert assign_sessions(df, 30)["session_id"].tolist() == [0, 0]


def test_session_split_across_midnight_and_single_play():
    df = plays([("a", "2026-01-01 23:50", 1), ("b", "2026-01-02 00:10", 1),
                ("c", "2026-01-03 09:00", 1)])
    assert assign_sessions(df, 30)["session_id"].tolist() == [0, 0, 1]


def test_skip_label_flag_or_short():
    df = plays([("a", "2026-01-01 10:00", 200_000, False),
                ("b", "2026-01-01 10:05", 10_000, False),   # short, flag false -> skip
                ("c", "2026-01-01 10:10", 200_000, True),   # long, flag true -> skip
                ("d", "2026-01-01 10:15", 200_000)])        # no flag (standard export)
    assert label_skips(df, 30_000)["skip"].tolist() == [False, True, True, False]


def test_skip_agreement_without_flag():
    df = plays([("a", "2026-01-01 10:00", 1)])
    assert skip_label_agreement(df) == {"has_flag": False}


def test_vocab_cutoff_and_specials():
    df = plays([("a", f"2026-01-01 10:0{i}", 1) for i in range(3)] + [("b", "2026-01-01 11:00", 1)])
    v = build_vocab(df, min_count=3)
    assert "a" in v and "b" not in v
    assert v[UNK] == 1 and len(v) == 5  # 4 specials + a


def test_time_split_keeps_sessions_whole_and_is_chronological():
    df = plays([("a", "2026-01-01 10:00", 1), ("a", "2026-03-01 10:00", 1),
                ("a", "2026-03-01 10:10", 1), ("a", "2026-06-30 10:00", 1)])
    df = label_skips(assign_sessions(df))
    train, val = time_split(df, val_months=2)
    assert len(train) == 3 and len(val) == 1
    assert train["ts"].max() < val["ts"].min()
    assert not set(train["session_id"]) & set(val["session_id"])


def test_to_sequences_wraps_and_maps_unk():
    df = plays([("a", "2026-01-01 10:00", 100_000), ("z", "2026-01-01 10:05", 1_000)])
    df = label_skips(assign_sessions(df))
    vocab = {"<PAD>": 0, UNK: 1, START: 2, END: 3, "a": 4}
    (seq,) = to_sequences(df, vocab)
    assert seq["ids"] == [2, 4, 1, 3]
    assert seq["skips"] == [0, 0, 1, 0]


def test_load_history_both_formats(tmp_path):
    ext = [{"ts": "2026-01-01T10:00:00Z", "ms_played": 5000, "skipped": True,
            "ip_addr": "1.2.3.4", "master_metadata_track_name": "T1",
            "master_metadata_album_artist_name": "A1", "master_metadata_album_album_name": "Al",
            "spotify_track_uri": "spotify:track:1"},
           {"ts": "2026-01-01T10:05:00Z", "ms_played": 9000, "master_metadata_track_name": None}]
    std = [{"endTime": "2026-01-02 09:00", "artistName": "A2", "trackName": "T2", "msPlayed": 1}]
    (tmp_path / "Streaming_History_Audio_2026_0.json").write_text(json.dumps(ext))
    (tmp_path / "StreamingHistory0.json").write_text(json.dumps(std))
    (tmp_path / "Payments.json").write_text("[]")
    df = load_history(tmp_path)
    assert df["track_id"].tolist() == ["spotify:track:1", "A2||T2"]  # podcast row dropped
    assert "ip_addr" not in df.columns


def test_load_history_empty_dir_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_history(tmp_path)


def test_hard_soft_keep_labels():
    df = plays([("a", "2026-01-01 10:00", 200_000, False),   # keep
                ("b", "2026-01-01 10:05", 4_999, False),     # hard
                ("c", "2026-01-01 10:10", 5_000, False),     # soft (exactly 5s is not hard)
                ("d", "2026-01-01 10:15", 200_000, True)])   # flagged, long -> soft
    out = label_skips(df, 30_000, 5_000)
    assert out["skip_type"].tolist() == [0, 2, 1, 1]
    assert out["skip"].tolist() == [False, True, True, True]


def test_drop_single_hard_skips_only_drops_once_heard_hard_skips():
    df = plays([("once_hard", "2026-01-01 10:00", 1_000),    # dropped
                ("once_long", "2026-01-01 10:01", 200_000),  # kept: heard fully
                ("twice", "2026-01-01 10:02", 1_000),        # kept: appears twice
                ("twice", "2026-01-01 10:03", 1_000)])
    out = drop_single_hard_skips(add_track_counts(label_skips(df)))
    assert out["track_id"].tolist() == ["once_long", "twice", "twice"]
    assert out["track_hard_skips"].tolist() == [0, 2, 2]


def test_sequences_carry_skip_types():
    df = plays([("a", "2026-01-01 10:00", 100_000), ("a", "2026-01-01 10:05", 1_000),
                ("a", "2026-01-01 10:06", 10_000)])
    df = label_skips(assign_sessions(df))
    vocab = {"<PAD>": 0, UNK: 1, START: 2, END: 3, "a": 4}
    (seq,) = to_sequences(df, vocab)
    assert seq["skip_types"] == [0, 0, 2, 1, 0]


def test_flag_bursts():
    rows = [("a", f"2026-01-01 10:{i:02d}", 1_000) for i in range(6)]            # session 0: burst
    rows += [("b", f"2026-01-01 12:{i:02d}", 200_000) for i in range(6)]         # session 1: listening
    rows += [("c", f"2026-01-01 14:{i:02d}", 1_000) for i in range(3)]           # session 2: too short
    df = flag_bursts(label_skips(assign_sessions(plays(rows))), min_plays=5, min_hard_rate=0.7)
    assert df.groupby("session_id")["is_burst"].first().tolist() == [True, False, False]


def test_sequences_carry_burst_flag():
    rows = [("a", f"2026-01-01 10:{i:02d}", 1_000) for i in range(4)]
    df = flag_bursts(label_skips(assign_sessions(plays(rows))), min_plays=3, min_hard_rate=0.7)
    vocab = {"<PAD>": 0, UNK: 1, START: 2, END: 3, "a": 4}
    (seq,) = to_sequences(df, vocab)
    assert seq["is_burst"] is True
