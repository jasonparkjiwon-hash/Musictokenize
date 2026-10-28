"""Phase 4: train the GRU next-song model on my own sessions.

Usage:
    python -m musictokenize.train            # one run with config.yaml, then report on val
    python -m musictokenize.train --sweep    # grid from config `sweep`, pick best on DEV, report val once

Early stopping and sweep selection use a DEV slice (the last `dev_months` of train), so the
validation split is touched once, for the final chosen model.
"""
import argparse
import copy
import csv
import itertools
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .config import ROOT, load_config, set_seed
from .evaluate import evaluate, train_skip_rates
from .model import GRUScorer, SongGRU

IGNORE = -100


def split_dev(train_seqs: list[dict], dev_months: int):
    """Last dev_months of train (by session start) become dev; whole sessions stay together."""
    cutoff = max(s["ts"] for s in train_seqs) - pd.DateOffset(months=dev_months)
    fit = [s for s in train_seqs if s["ts"] < cutoff]
    dev = [s for s in train_seqs if s["ts"] >= cutoff]
    return fit, dev


def make_windows(seqs: list[dict], max_len: int, include_bursts: bool = True) -> list[list[int]]:
    """Cut sessions into chunks of max_len+1 ids; consecutive chunks share one id so every
    next-song target is predicted exactly once. Hidden state is not carried across chunks."""
    out = []
    for s in seqs:
        if s["is_burst"] and not include_bursts:
            continue
        ids = s["ids"]
        for i in range(0, len(ids) - 1, max_len):
            w = ids[i:i + max_len + 1]
            if len(w) >= 2:
                out.append(w)
    return out


def collate(windows: list[list[int]], unk: int):
    """Pad to the longest window. Targets that are padding or <UNK> are ignored in the loss."""
    T = max(len(w) for w in windows) - 1
    x = torch.zeros(len(windows), T, dtype=torch.long)
    y = torch.full((len(windows), T), IGNORE, dtype=torch.long)
    for r, w in enumerate(windows):
        n = len(w) - 1
        x[r, :n] = torch.tensor(w[:-1])
        tgt = torch.tensor(w[1:])
        y[r, :n] = torch.where(tgt == unk, torch.tensor(IGNORE), tgt)
    return x, y


def run_epoch(model, windows, unk, batch_size, opt=None, grad_clip=1.0, shuffle=False, rng=None):
    """One pass. Trains if opt is given, else evaluates. Returns mean CE per scored target."""
    model.train(opt is not None)
    order = rng.permutation(len(windows)) if shuffle else np.arange(len(windows))
    total, count = 0.0, 0
    with torch.set_grad_enabled(opt is not None):
        for i in range(0, len(order), batch_size):
            x, y = collate([windows[j] for j in order[i:i + batch_size]], unk)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                                   ignore_index=IGNORE, reduction="sum")
            n = int((y != IGNORE).sum())
            if opt is not None and n:
                opt.zero_grad()
                (loss / n).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
            total += float(loss)
            count += n
    return total / max(count, 1)


def fit(cfg: dict, fit_seqs, dev_seqs, vocab, skip_rate, log_path=None, verbose=True):
    """Train with early stopping on dev loss. Returns (best_model, history)."""
    t, m = cfg["train"], cfg["model"]
    set_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    unk, V = vocab["<UNK>"], len(vocab)
    model = SongGRU(V, m["embed_dim"], m["hidden_dim"], m["num_layers"], m["dropout"],
                    m["tie_weights"])
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    fit_w = make_windows(fit_seqs, t["max_len"], t["include_bursts"])
    dev_w = make_windows(dev_seqs, t["max_len"], True)
    best, best_state, bad, history = float("inf"), None, 0, []
    for epoch in range(1, t["epochs"] + 1):
        t0 = time.time()
        tr = run_epoch(model, fit_w, unk, t["batch_size"], opt, t["grad_clip"], True, rng)
        dv = run_epoch(model, dev_w, unk, t["batch_size"])
        nb = evaluate(GRUScorer(model), dev_seqs, vocab, skip_rate)["no_burst"]
        row = {"epoch": epoch, "train_loss": tr, "dev_loss": dv, "dev_hit@10": nb["hit@10"],
               "dev_ndcg@10": nb["ndcg@10"], "dev_pick_skip@10": nb["pick_skip@10"],
               "secs": time.time() - t0}
        history.append(row)
        if verbose:
            print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in row.items()))
        if dv < best:
            best, best_state, bad = dv, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= t["patience"]:
                break
    model.load_state_dict(best_state)
    if log_path:
        with open(log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0]))
            w.writeheader()
            w.writerows(history)
    return model.eval(), history


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    torch.set_num_threads(4)
    proc, res = ROOT / cfg["paths"]["processed"], ROOT / "results"
    res.mkdir(exist_ok=True)
    vocab = json.loads((proc / "vocab.json").read_text())
    seqs = pd.read_pickle(proc / "sequences.pkl")
    fit_seqs, dev_seqs = split_dev(seqs["train"], cfg["train"]["dev_months"])
    skip_rate, _ = train_skip_rates(fit_seqs, len(vocab))
    print(f"fit sessions={len(fit_seqs)} dev sessions={len(dev_seqs)} val sessions={len(seqs['val'])}")

    if args.sweep:
        keys = list(cfg["sweep"])
        runs = []
        for combo in itertools.product(*(cfg["sweep"][k] for k in keys)):
            c = copy.deepcopy(cfg)
            for k, v in zip(keys, combo):
                (c["train"] if k in c["train"] else c["model"])[k] = v
            model, hist = fit(c, fit_seqs, dev_seqs, vocab, skip_rate, verbose=False)
            best = min(hist, key=lambda r: r["dev_loss"])
            runs.append({"params": dict(zip(keys, combo)), "epochs": len(hist), **best})
            print(runs[-1]["params"], f"dev_loss={best['dev_loss']:.4f} "
                  f"dev_hit@10={best['dev_hit@10']:.4f} at epoch {best['epoch']}", flush=True)
        win = min(runs, key=lambda r: r["dev_loss"])
        (res / "gru_sweep.json").write_text(json.dumps(runs, indent=2))
        print("winner (by dev loss):", win["params"])
        for k, v in win["params"].items():
            (cfg["train"] if k in cfg["train"] else cfg["model"])[k] = v

    model, hist = fit(cfg, fit_seqs, dev_seqs, vocab, skip_rate, res / "gru_log.csv")
    ckpt_dir = ROOT / cfg["train"]["checkpoint_dir"]
    ckpt_dir.mkdir(exist_ok=True)
    torch.save({"state": model.state_dict(), "cfg": cfg, "vocab_size": len(vocab)},
               ckpt_dir / "gru.pt")
    r = evaluate(GRUScorer(model), seqs["val"], vocab, skip_rate)
    (res / "gru.json").write_text(json.dumps({"cfg": {"model": cfg["model"], "train": cfg["train"]},
                                              "val": r}, indent=2))
    nb = r["no_burst"]
    print(f"VAL (non-burst, n={nb['n']}): hit@10={nb['hit@10']:.3f} ndcg@10={nb['ndcg@10']:.3f} "
          f"kept-hit@10={nb['hit@10_kept_targets']:.3f} "
          f"skip_hits={nb['skip_rate_hits@10']:.3f}(n~{nb['expected_hits']:.0f}) "
          f"pick_skip={nb['pick_skip@10']:.3f}")


if __name__ == "__main__":
    main()
