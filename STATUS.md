# Status

Handoff file for new sessions. Read this first, then `CLAUDE.md` and the relevant phase in `PLAN.md`.
Update it at the end of every phase, in the same commit as the work.

## Current state
- **Last completed:** Phase 1 (code and tests done; NOT yet run on the real export)
- **Next:** user drops the export in `data/raw/`, run `python -m musictokenize.process` and review the printed stats; then Phase 2 or 3 when the user says to start
- **Branch:** `claude/new-session-pae0ve`

## Phase log
| Phase | Status | Notes |
|---|---|---|
| 0 Repo setup | Done | Layout, .gitignore, pyproject, config.yaml, CLAUDE.md, config/seed helpers |
| 1 Data to sequences | Done, pending real data | `data.py`, `process.py`, 11 tests pass. Playlist/library sequences deferred (see decisions) |
| 2 Metadata enrichment | Not started | Optional |
| 3 Baselines | Not started | |
| 4 Pretraining | Not started | |
| 5 Skip-aware fine-tuning | Not started | |
| 6 Generation | Not started | |
| 7 Feedback loop | Not started | |
| 8 Interface | Not started | |
| 9 Evaluation and writeup | Not started | |

## Facts a new session needs
- Raw Spotify export is NOT in the repo (gitignored). It goes in `data/raw/`. Phase 1 has never run on it.
  there before starting Phase 1; if not, ask the user for it.
- PyTorch is not installed yet (Phase 0 didn't need it). Run `pip install -e ".[dev]"`.
- Run tests with `pytest` from the repo root.
- Do only the phase the user asks for. One phase = one commit. Don't open a PR unless asked.

## Open questions (from PLAN.md)
- Does the Excel list include Spotify track IDs?
- Which external datasets are still available?
- Is the 30 second skip threshold right? Compare it against the skipped flag.

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
