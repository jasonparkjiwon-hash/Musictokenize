"""Build data/processed/ from data/raw/. Usage: python -m musictokenize.process"""
import json

import pandas as pd

from .config import ROOT, load_config, set_seed
from .data import (
    UNK,
    assign_sessions,
    build_vocab,
    label_skips,
    load_history,
    skip_label_agreement,
    time_split,
    to_sequences,
)


def main() -> None:
    cfg = load_config()
    set_seed(cfg["seed"])
    d = cfg["data"]
    raw, out = ROOT / cfg["paths"]["raw"], ROOT / cfg["paths"]["processed"]
    out.mkdir(parents=True, exist_ok=True)

    plays = load_history(raw)
    agreement = skip_label_agreement(plays, d["skip_ms_threshold"])
    plays = assign_sessions(label_skips(plays, d["skip_ms_threshold"]), d["session_gap_minutes"])
    train, val = time_split(plays, d["val_months"])
    vocab = build_vocab(train, d["min_play_count"])

    plays.to_parquet(out / "plays.parquet")
    (out / "vocab.json").write_text(json.dumps(vocab))
    pd.to_pickle(
        {"train": to_sequences(train, vocab), "val": to_sequences(val, vocab)},
        out / "sequences.pkl",
    )

    unk_id = vocab[UNK]
    val_unk = 1 - val["track_id"].isin(vocab).mean() if len(val) else float("nan")
    print(f"plays={len(plays)} sessions={plays['session_id'].nunique()} "
          f"train={len(train)} val={len(val)}")
    print(f"vocab={len(vocab)} (unk id {unk_id}); val plays outside vocab: {val_unk:.1%}")
    print(f"skip rate: train={train['skip'].mean():.1%} "
          f"val={val['skip'].mean() if len(val) else float('nan'):.1%}")
    print(f"skip label agreement: {agreement}")


if __name__ == "__main__":
    main()
