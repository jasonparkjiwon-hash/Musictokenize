"""Phase 2: join the library workbook to the listening history by Spotify track ID.

The workbook's `Spotify link` column gives the track ID, which maps exactly onto
`spotify:track:<id>` in the extended export, so no fuzzy matching is needed.
Tempo/mood/genre in `Categories` are model-estimated (see `genre_source`): treat as noisy.
"""
from pathlib import Path

import pandas as pd

META_COLS = {
    "Genre": "genre", "Style (Spotify tag)": "style", "Decade": "decade", "Mood": "mood",
    "Tempo": "tempo", "BPM": "bpm", "Vocals": "vocals", "Language": "language",
    "Genre based on": "genre_source",
}


def load_library(xlsx: str | Path) -> pd.DataFrame:
    """One row per library song, keyed by track_id, with its category features."""
    x = pd.ExcelFile(xlsx)
    songs, cats = x.parse("All Songs"), x.parse("Categories")
    if not (songs["#"].values == cats["#"].values).all():
        raise ValueError("'All Songs' and 'Categories' rows are not aligned on '#'")
    lib = songs[["Playlists", "In # playlists", "Spotify link", "Year"]].copy()
    lib.columns = ["playlists", "n_playlists", "link", "year"]
    lib = pd.concat([lib, cats[list(META_COLS)].rename(columns=META_COLS)], axis=1)
    ids = lib["link"].str.extract(r"track/(\w+)")[0]
    lib["track_id"] = "spotify:track:" + ids
    lib = lib[ids.notna()].drop(columns="link")
    return lib.drop_duplicates("track_id").reset_index(drop=True)


def build_songs(plays: pd.DataFrame, library: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """One row per played song: name/artist/album, play stats, library features, in_library flag.

    Also returns a match report. Library songs never played are reported but not added
    (they have no listening signal and are outside the vocabulary).
    """
    g = plays.groupby("track_id")
    songs = g.agg(track=("track", "first"), artist=("artist", "first"), album=("album", "first"),
                  plays=("ts", "size"), skips=("skip", "sum"), hard_skips=("hard_skip", "sum"))
    songs["skip_rate"] = songs["skips"] / songs["plays"]
    songs = songs.join(library.set_index("track_id"), how="left")
    songs["in_library"] = songs["genre"].notna() | songs["playlists"].notna()
    songs["n_playlists"] = songs["n_playlists"].fillna(0).astype(int)
    songs = songs.reset_index()
    played = set(plays["track_id"])
    lib_ids = set(library["track_id"])
    report = {
        "library_songs": len(lib_ids),
        "library_played": len(lib_ids & played),
        "library_match_rate": len(lib_ids & played) / len(lib_ids),
        "played_songs": len(played),
        "played_in_library": int(songs["in_library"].sum()),
        "plays_in_library_share": float(plays["track_id"].isin(lib_ids).mean()),
        "library_unplayed_ids": sorted(lib_ids - played)[:20],
    }
    return songs, report
