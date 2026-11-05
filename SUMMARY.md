# Musictokenize: Summary

**Status:** Only Phase 0 (repo setup) is done. Phases 1–9 haven't started, so there are no experimental results or lessons yet. The "lessons learned" below come from the plan's risk list and from how the project is run. Nothing was learned by running a model.

## Goal
Build a tiny language model where the vocabulary is songs instead of words. Given a seed song and optionally the last few songs played, it returns a sequence you are unlikely to skip. Tempo and genre don't matter. The only question is whether you'd skip this song after that one.

The model learns two things:
- P(next song | previous songs), from listening sessions.
- P(not skipped | context, candidate), from your own skip labels.

**Success metric:** the skip rate of its top picks on held-out data, using a time-based split on the last 2 months. Secondary metrics are hit@10 and NDCG. Everything is compared against three baselines: random, bigram Markov and item2vec.

## Process
The work is split into 10 phases, one session and one commit each. The user triggers each phase by hand.

| Phase | What happens |
|---|---|
| 0 | Repo setup (**done**) |
| 1 | Turn the Spotify export into sessions with skip labels, a vocab and a time-based split |
| 2 | Optional metadata enrichment (artist, genre, tempo) |
| 3 | Baselines and a shared eval script |
| 4 | GRU pretraining on external playlists, then fine-tuning on your own history |
| 5 | Non-skip head trained on keep/skip labels |
| 6 | Generation: sampling and beam search, reranked by non-skip probability, with repeat and artist controls |
| 7 | Feedback loop: rating CLI and retraining |
| 8 | `suggest` CLI |
| 9 | Ablations and writeup |

**Conventions:**
- Skip means the skipped flag is set or under 30s was played.
- Sessions split on a 30-minute gap.
- The vocab is songs with 3+ plays, plus `<UNK>`, `<START>` and `<END>`.
- Python and PyTorch, with settings in `config.yaml` and fixed seeds.
- `data/raw/` and `data/processed/` are gitignored, and API caches are never committed.
- `STATUS.md` is the handoff file between sessions.

**Phase 0 delivered:** the repo layout, `.gitignore`, `pyproject`, `config.yaml`, `CLAUDE.md`, config and seed helpers, and `PLAN.md`. PyTorch is listed as a dependency but not installed yet.

## Lessons and risks (expected, not yet validated)
- **Small data:** about 37k plays a year is enough for baselines and a small GRU, not much more. Repetition is the real limit, since many songs appear only once or twice. The plan is to keep the model tiny, use dropout and lean on pretraining.
- **ID matching:** external datasets may not align with your song IDs, so log the match rate early.
- **Playlist order is weak:** treat pretraining as "goes together", not strict order.
- **Popularity skew:** external data favors popular songs, and fine-tuning on your skips corrects for it.
- **Skip is noisy:** you may skip because of mood, place or time of day, not the song itself.
- **Replaying your top 50:** tune the exploration knob and track how many unseen songs get suggested.
- **Process lessons so far:** a `STATUS.md` handoff file lets a new session resume without chat history. Doing one phase per session keeps the work reviewable. Recording every decision and deviation in `STATUS.md` keeps the plan honest.

## Open questions
- Does the Excel list include Spotify track IDs?
- Which external datasets are still available?
- Is the 30s skip threshold right? Compare it against the skipped flag.

**Next step:** start Phase 1. First check that the raw Spotify export is in `data/raw/`.
