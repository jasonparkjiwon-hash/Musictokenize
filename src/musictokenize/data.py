"""Phase 1: streaming history -> labelled plays -> sessions -> vocab -> train/val sequences.

Handles both Spotify exports:
- extended: Streaming_History_Audio_*.json (has track URI and a `skipped` flag)
- standard: StreamingHistory*.json (artist + track name only, no ID, no skipped flag)

Only the columns listed in KEEP_COLS are ever read out of the raw files; IP address, country,
username etc. are dropped immediately.
"""
import json
from pathlib import Path

import pandas as pd

PAD, UNK, START, END = "<PAD>", "<UNK>", "<START>", "<END>"
SPECIALS = [PAD, UNK, START, END]

COLS = ["track_id", "track", "artist", "album", "ts", "ms_played", "skipped_flag"]


def _from_extended(rec: dict) -> dict | None:
    if not rec.get("master_metadata_track_name"):  # podcasts, audiobooks, unknown
        return None
    uri = rec.get("spotify_track_uri")
    artist = rec.get("master_metadata_album_artist_name")
    track = rec["master_metadata_track_name"]
    return {
        "track_id": uri or f"{artist}||{track}",
        "track": track,
        "artist": artist,
        "album": rec.get("master_metadata_album_album_name"),
        "ts": rec["ts"],
        "ms_played": rec.get("ms_played"),
        "skipped_flag": rec.get("skipped"),
    }


def _from_standard(rec: dict) -> dict | None:
    if not rec.get("trackName"):
        return None
    return {
        "track_id": f"{rec.get('artistName')}||{rec['trackName']}",
        "track": rec["trackName"],
        "artist": rec.get("artistName"),
        "album": None,
        "ts": rec["endTime"],
        "ms_played": rec.get("msPlayed"),
        "skipped_flag": None,
    }


def load_history(raw_dir: str | Path) -> pd.DataFrame:
    """Read every history JSON under raw_dir into one time-sorted table of plays."""
    rows = []
    for path in sorted(Path(raw_dir).rglob("*.json")):
        name = path.name
        if name.startswith("Streaming_History_Audio"):
            parse = _from_extended
        elif name.startswith("StreamingHistory"):
            parse = _from_standard
        else:  # other export files (payments, search, library...) are ignored
            continue
        with open(path, encoding="utf-8") as f:
            for rec in json.load(f):
                row = parse(rec)
                if row is not None:
                    rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No streaming history JSON found under {raw_dir}")
    df = pd.DataFrame(rows, columns=COLS)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, format="mixed")
    df["ms_played"] = df["ms_played"].fillna(0).astype("int64")
    df["skipped_flag"] = df["skipped_flag"].astype("boolean")
    df = df.drop_duplicates(subset=["track_id", "ts", "ms_played"])
    return df.sort_values("ts", kind="stable").reset_index(drop=True)


def label_skips(df: pd.DataFrame, skip_ms_threshold: int = 30_000) -> pd.DataFrame:
    """skip = Spotify's skipped flag is true OR the play lasted under the threshold."""
    out = df.copy()
    flag = out["skipped_flag"].fillna(False).astype(bool)
    out["skip"] = flag | (out["ms_played"] < skip_ms_threshold)
    return out


def skip_label_agreement(df: pd.DataFrame, skip_ms_threshold: int = 30_000) -> dict:
    """Answers the open question 'is 30s right?': compare the flag with the threshold rule.

    Only meaningful for the extended export (the standard one has no flag).
    """
    has_flag = df["skipped_flag"].notna().any()
    if not has_flag:
        return {"has_flag": False}
    flag = df["skipped_flag"].fillna(False).astype(bool)
    short = df["ms_played"] < skip_ms_threshold
    return {
        "has_flag": True,
        "n": len(df),
        "flag_rate": float(flag.mean()),
        "short_rate": float(short.mean()),
        "flag_and_short": int((flag & short).sum()),
        "flag_only": int((flag & ~short).sum()),
        "short_only": int((~flag & short).sum()),
    }


def assign_sessions(df: pd.DataFrame, gap_minutes: int = 30) -> pd.DataFrame:
    """New session whenever the gap since the previous play's start exceeds gap_minutes.

    Rows must be time-sorted. ts is the end of the play in both exports, so the gap is
    measured between consecutive ts values.
    """
    out = df.copy()
    gap = out["ts"].diff() > pd.Timedelta(minutes=gap_minutes)
    out["session_id"] = gap.cumsum().astype("int64")
    return out


def time_split(df: pd.DataFrame, val_months: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on older data, validate on the last val_months. Whole sessions stay on one side."""
    cutoff = df["ts"].max() - pd.DateOffset(months=val_months)
    session_start = df.groupby("session_id")["ts"].transform("min")
    is_val = session_start >= cutoff
    return df[~is_val].copy(), df[is_val].copy()


def build_vocab(train: pd.DataFrame, min_count: int = 3) -> dict[str, int]:
    """Specials first, then every train track with >= min_count plays (most played first).

    Built from train only so validation never leaks into the vocabulary.
    """
    counts = train["track_id"].value_counts()
    kept = counts[counts >= min_count].index.tolist()
    return {tok: i for i, tok in enumerate(SPECIALS + kept)}


def to_sequences(df: pd.DataFrame, vocab: dict[str, int]) -> list[dict]:
    """One dict per session: song ids and skip labels, wrapped in <START> ... <END>.

    Labels for the <START>/<END> positions are 0 (not a skip) and should be masked by callers.
    """
    unk = vocab[UNK]
    seqs = []
    for sid, g in df.groupby("session_id", sort=True):
        ids = [vocab[START]] + [vocab.get(t, unk) for t in g["track_id"]] + [vocab[END]]
        skips = [0] + g["skip"].astype(int).tolist() + [0]
        seqs.append({"session_id": int(sid), "ids": ids, "skips": skips, "ts": g["ts"].iloc[0]})
    return seqs
