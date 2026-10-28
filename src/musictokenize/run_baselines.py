"""Evaluate all baselines on the validation split. Usage: python -m musictokenize.run_baselines"""
import json

import pandas as pd

from .baselines import Bigram, FeatureNN, Item2Vec, KeepPopularity, Popularity, Random
from .config import ROOT, load_config, set_seed
from .evaluate import evaluate, train_skip_rates


def main() -> None:
    cfg = load_config()
    set_seed(cfg["seed"])
    proc = ROOT / cfg["paths"]["processed"]
    vocab = json.loads((proc / "vocab.json").read_text())
    seqs = pd.read_pickle(proc / "sequences.pkl")
    songs = pd.read_parquet(proc / "songs.parquet")
    train, val, V = seqs["train"], seqs["val"], len(vocab)
    skip_rate, base = train_skip_rates(train, V)

    models = [Random(V, cfg["seed"]), Popularity(train, V), KeepPopularity(train, V),
              Bigram(train, V), Item2Vec(train, V, vocab, seed=cfg["seed"]),
              FeatureNN(train, V, vocab, songs)]
    results = {}
    for m in models:
        results[m.name] = evaluate(m, val, vocab, skip_rate)
        nb = results[m.name]["no_burst"]
        print(f"{m.name:16s} hit@10={nb['hit@10']:.3f} ndcg@10={nb['ndcg@10']:.3f} "
              f"kept-hit@10={nb['hit@10_kept_targets']:.3f} "
              f"skip_hits={nb['skip_rate_hits@10']:.3f}(n~{nb['expected_hits']:.0f}) pick_skip={nb['pick_skip@10']:.3f}")
    first = next(iter(results.values()))
    print(f"coverage={first['coverage']:.3f} n={first['no_burst']['n']} "
          f"val base skip rate (non-burst)={first['no_burst']['base_skip_rate']:.3f} "
          f"train base={base:.3f}")
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "baselines.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
