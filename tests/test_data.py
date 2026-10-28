import json

import pandas as pd
import pytest

from musictokenize.data import (
    END,
    START,
    UNK,
    assign_sessions,
    build_vocab,
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
