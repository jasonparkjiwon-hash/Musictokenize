# Status

Handoff file for new sessions. Read this first, then `CLAUDE.md` and the relevant phase in `PLAN.md`.
Update it at the end of every phase, in the same commit as the work.

## Current state
- **Last completed:** Phase 1 (code, tests, and a successful run on the real extended export)
- **Next:** Phase 2 or 3 when the user says to start (see data findings below first)
- **Branch:** `claude/new-session-pae0ve`

## Phase log
| Phase | Status | Notes |
|---|---|---|
| 0 Repo setup | Done | Layout, .gitignore, pyproject, config.yaml, CLAUDE.md, config/seed helpers |
| 1 Data to sequences | Done | `data.py`, `process.py`, 11 tests pass, ran on real export. Playlist/library sequences deferred (see decisions) |
| 2 Metadata enrichment | Not started | Optional |
| 3 Baselines | Not started | |
| 4 Pretraining | Not started | |
| 5 Skip-aware fine-tuning | Not started | |
| 6 Generation | Not started | |
| 7 Feedback loop | Not started | |
| 8 Interface | Not started | |
| 9 Evaluation and writeup | Not started | |

## Facts a new session needs
- Raw Spotify export is NOT in the repo (gitignored). The container is ephemeral: the user must re-supply the zip in a new session (only the 11 `Streaming_History_Audio_*.json` files are needed in `data/raw/`; the Video files and PDF are ignored).
  there before starting Phase 1; if not, ask the user for it.
- PyTorch is not installed yet (Phase 0 didn't need it). Run `pip install -e ".[dev]"`.
- Run tests with `pytest` from the repo root.
- Do only the phase the user asks for. One phase = one commit. Don't open a PR unless asked.

## Data findings (real export, run 2026-10-28; aggregates only)
- 136,530 plays (134,590 after dropping one-off hard skips), 2023-07 to 2026-10, 4,249 sessions (median 14 plays, longest 3,532 - check that one).
- Split: train 121,191 / val 15,339. Vocab 4,554 incl. specials; 15.3% of val plays are outside it (UNK).
- 11,120 distinct tracks; 4,932 heard once. Top 50 tracks are only 7.3% of plays (not a replay-my-favourites listener).
- **Skip types after the hard/soft split:** keep 30.3%, soft 24.7%, hard 45.0%. 1,940 one-off hard-skipped plays dropped. Val plays outside vocab now 14.3%.
- **Longest session (3,532 plays) is a rapid-skip burst:** 2.6h, ~1 play per 2.7s, 100% skipped, 2,613 distinct songs. 68 sessions over 200 plays hold 26% of all plays; they look like browsing/auditioning, not listening. Now flagged as `is_burst` (see decisions).
- **Skip rate is ~70%** (train 69.9%, val 72.2%). 46% of plays are under 5s, 53% under 10s: heavy browsing/skipping. Skip-rate-of-top-picks is the headline metric, so a random baseline is already ~70% skips.
- Flag vs 30s rule: flag 68.4%, under-30s 60.0%. 79,525 both; 13,849 flagged but over 30s; 2,436 under 30s but not flagged.

## Open questions (from PLAN.md)
- Does the Excel list include Spotify track IDs?
- Which external datasets are still available?
- Is the 30 second skip threshold right? Data says the flag marks ~14k plays the threshold misses, so a longer threshold or flag-only could be tried. Also consider a stricter 'hard skip' (<5s) vs 'soft skip'. User to decide.
- (Resolved 2026-10-28: hard skip = under 5s, soft skip = other skips.) Plays under ~5s may be browsing, not a judgement on the song. Should they be down-weighted or dropped from training?

## Decisions and deviations
Record every choice that isn't spelled out in `PLAN.md`, and every place the work departs from it.
Add a row when it happens, newest last. Don't rewrite old rows; if a decision is reversed, add a
new row that says so. "Decided by" is `user` or `claude` (a default Claude picked; the user can
overrule it).

| Date | Phase | Decision or deviation | Why | Decided by |
|---|---|---|---|---|
| 2026-10-26 | 0 | Work is developed on branch `claude/new-session-pae0ve`, not a branch per phase | Session setup designates this branch | user |
| 2026-10-26 | 0 | Do only the phase the user asks for, then stop and wait | User wants to trigger each phase themselves | user |
| 2026-10-26 | 0 | Added `STATUS.md` (this file), not in PLAN.md's Phase 0 layout | Lets a new session resume without chat history | user |
| 2026-10-26 | 0 | `data/processed/` is gitignored too, not just `data/raw/` | Processed tables are derived from personal listening history | claude |
| 2026-10-26 | 0 | Config lives in a single `config.yaml` read with PyYAML | Simplest option for "a config file"; Phase 4 asks for config-driven training | claude |
| 2026-10-26 | 0 | PyTorch listed as a dependency but not installed in the cloud container yet | Phase 0 needs no model code; avoids a large download | claude |
| 2026-10-26 | 0 | `PLAN.md` copied into the repo unchanged | Plan is the source of truth for later sessions | claude |
| 2026-10-28 | 1 | No raw export was available, so Phase 1 was built against the documented Spotify schemas and tested on synthetic data | Don't block on data; real-data run is the first step next session | claude |
| 2026-10-28 | 1 | Supports both extended (`Streaming_History_Audio_*.json`) and standard (`StreamingHistory*.json`) exports; standard has no URI or skipped flag, so its track ID is `artist||track` and skip is threshold-only | Plan lists standard export as backup | claude |
| 2026-10-28 | 1 | Skip = `skipped` flag true OR under 30s played | Matches CLAUDE.md; `skip_label_agreement` prints how the two rules differ, to answer the open 30s question | claude |
| 2026-10-28 | 1 | Time split keeps whole sessions on one side (a session goes to val if it starts in the last 2 months); vocab built from train only | Avoids leakage across the boundary | claude |
| 2026-10-28 | 1 | Podcasts/audiobooks (no track name) dropped; only needed columns are read, so IP, country, username never enter the tables | Privacy rule | claude |
| 2026-10-28 | 1 | DEFERRED: converting playlists and library into extra sequences | Needs the real `Playlist*.json` / `YourLibrary.json` layout to be seen first | claude |
| 2026-10-28 | 1 | Outputs: `plays.parquet`, `vocab.json`, `sequences.pkl` (train/val lists of dicts with ids, skips, session start ts) in `data/processed/` | Simple, gitignored | claude |
| 2026-10-28 | 1 | Real run used the 11 `Streaming_History_Audio_*.json` files; Video history ignored | Plan is about songs | claude |
| 2026-10-28 | 1 | Hard skip = played under 5s (`hard_skip_ms`); soft skip = flagged or under 30s but not hard; `skip_type` 0/1/2 added to plays and sequences. `skip` (0/1) kept as hard or soft | User asked to distinguish hard vs soft | user |
| 2026-10-28 | 1 | Per-song totals (`track_plays`, `track_hard_skips`) stored in plays.parquet; plays of songs heard exactly once in all history AND hard-skipped are dropped (`drop_single_hard_skips`, switchable in config) | User rule. My reading: "played once in the whole history" = one play total, and that play was under 5s | user |
| 2026-10-28 | 1 | Sessions are assigned before the drop, so a dropped play still counts as activity for the 30-minute gap | Sessions reflect real listening time | claude |
| 2026-10-28 | 1 | Burst session = >= 50 plays and >= 70% hard skips (`burst_min_plays`, `burst_min_hard_rate`). Bursts stay in sequences for next-song training; every sequence carries `is_burst` and the skip head (Phase 5) must mask burst sequences | User accepted recommendation: weight bursts out of skip training, keep for sequence modelling. Thresholds are my choice, tuned to match the ~26% of plays seen in 200+ play sessions | user (rule) / claude (thresholds) |
