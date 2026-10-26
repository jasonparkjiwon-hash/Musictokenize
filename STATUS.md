# Status

Handoff file for new sessions. Read this first, then `CLAUDE.md` and the relevant phase in `PLAN.md`.
Update it at the end of every phase, in the same commit as the work.

## Current state
- **Last completed:** Phase 0 (repo setup)
- **Next:** Phase 1 (data to sequences), only when the user says to start
- **Branch:** `claude/new-session-pae0ve`

## Phase log
| Phase | Status | Notes |
|---|---|---|
| 0 Repo setup | Done | Layout, .gitignore, pyproject, config.yaml, CLAUDE.md, config/seed helpers |
| 1 Data to sequences | Not started | |
| 2 Metadata enrichment | Not started | Optional |
| 3 Baselines | Not started | |
| 4 Pretraining | Not started | |
| 5 Skip-aware fine-tuning | Not started | |
| 6 Generation | Not started | |
| 7 Feedback loop | Not started | |
| 8 Interface | Not started | |
| 9 Evaluation and writeup | Not started | |

## Facts a new session needs
- Raw Spotify export is NOT in the repo (gitignored). It goes in `data/raw/`. Check whether it is
  there before starting Phase 1; if not, ask the user for it.
- PyTorch is not installed yet (Phase 0 didn't need it). Run `pip install -e ".[dev]"`.
- Run tests with `pytest` from the repo root.
- Do only the phase the user asks for. One phase = one commit. Don't open a PR unless asked.

## Open questions (from PLAN.md)
- Does the Excel list include Spotify track IDs?
- Which external datasets are still available?
- Is the 30 second skip threshold right? Compare it against the skipped flag.

## Decisions and deviations
- None yet.
