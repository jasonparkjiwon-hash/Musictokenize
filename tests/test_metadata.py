import pandas as pd
import pytest

from musictokenize.metadata import build_songs, load_library


def make_xlsx(path, links, misaligned=False):
    n = len(links)
    songs = pd.DataFrame({"#": range(1, n + 1), "Title": "t", "Artist(s)": "a", "Album": "al",
                          "Year": 2020.0, "Length": "1:00", "Playlists": "Rap",
                          "In # playlists": 1, "Genres": None, "Spotify link": links})
    cats = pd.DataFrame({"#": range(1, n + 1) if not misaligned else range(5, 5 + n),
                         "Title": "t", "Artist(s)": "a", "Genre": "Rap", "Style (Spotify tag)": None,
                         "Decade": "2020s", "Mood": "Mellow & sad", "Tempo": "Slow", "BPM": 90.0,
                         "Vocals": "Vocal", "Language": "English", "Genre based on": "Spotify tag",
                         "Note on the year": None, "Release year used": 2020.0})
    with pd.ExcelWriter(path) as w:
        songs.to_excel(w, sheet_name="All Songs", index=False)
        cats.to_excel(w, sheet_name="Categories", index=False)


def test_load_library_maps_ids_and_dedupes(tmp_path):
    make_xlsx(tmp_path / "l.xlsx", ["https://open.spotify.com/track/AAA", "https://open.spotify.com/track/AAA",
                                    "https://open.spotify.com/track/BBB"])
    lib = load_library(tmp_path / "l.xlsx")
    assert lib["track_id"].tolist() == ["spotify:track:AAA", "spotify:track:BBB"]
    assert lib.loc[0, "genre"] == "Rap" and lib.loc[0, "bpm"] == 90.0


def test_load_library_rejects_misaligned_sheets(tmp_path):
    make_xlsx(tmp_path / "l.xlsx", ["https://open.spotify.com/track/AAA"], misaligned=True)
    with pytest.raises(ValueError):
        load_library(tmp_path / "l.xlsx")


def test_build_songs_join_and_report(tmp_path):
    make_xlsx(tmp_path / "l.xlsx", ["https://open.spotify.com/track/AAA", "https://open.spotify.com/track/ZZZ"])
    plays = pd.DataFrame({
        "track_id": ["spotify:track:AAA", "spotify:track:AAA", "spotify:track:CCC"],
        "track": ["A", "A", "C"], "artist": "x", "album": "y",
        "ts": pd.to_datetime(["2026-01-01"] * 3, utc=True),
        "skip": [True, False, True], "hard_skip": [True, False, False]})
    songs, rep = build_songs(plays, load_library(tmp_path / "l.xlsx"))
    s = songs.set_index("track_id")
    assert s.loc["spotify:track:AAA", "in_library"] and not s.loc["spotify:track:CCC", "in_library"]
    assert s.loc["spotify:track:AAA", "skip_rate"] == 0.5
    assert rep["library_played"] == 1 and rep["library_songs"] == 2
    assert rep["plays_in_library_share"] == pytest.approx(2 / 3)
