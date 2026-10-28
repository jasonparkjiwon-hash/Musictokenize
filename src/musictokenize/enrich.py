"""Write data/processed/songs.parquet. Usage: python -m musictokenize.enrich"""
import pandas as pd

from .config import ROOT, load_config
from .metadata import build_songs, load_library


def main() -> None:
    cfg = load_config()
    raw, out = ROOT / cfg["paths"]["raw"], ROOT / cfg["paths"]["processed"]
    plays = pd.read_parquet(out / "plays.parquet")
    songs, rep = build_songs(plays, load_library(raw / "library.xlsx"))
    songs.to_parquet(out / "songs.parquet")
    print(f"songs={len(songs)} in_library={int(songs['in_library'].sum())}")
    print(f"library match: {rep['library_played']}/{rep['library_songs']} "
          f"({rep['library_match_rate']:.1%}); library songs are "
          f"{rep['plays_in_library_share']:.1%} of plays")
    print("unmatched library ids (sample):", rep["library_unplayed_ids"][:5])


if __name__ == "__main__":
    main()
