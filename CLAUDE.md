# Musictokenize

**PROJECT CLOSED (failed at proof of concept, 2026-10-28). Do not continue unless the user reopens it.**

**Start every session by reading `STATUS.md`** (current phase, next step, open questions).
Update it at the end of each phase.

## Goal
A tiny language model where the vocabulary is songs. Given a seed song (and optionally recent
songs), return a sequence I am unlikely to skip. Tempo and genre don't matter; only "would I skip
this after that?" does. Full roadmap: `PLAN.md` (one phase per session, one commit per phase).

## Privacy rules
- Never commit personal data. `data/raw/` and `data/processed/` are gitignored; keep it that way.
- Cache external API results (MusicBrainz, Last.fm) locally; never commit the cache.

## Evaluation metric
Primary: skip rate of top picks on held-out data (time-based split, last 2 months).
Secondary: hit@10 and NDCG for next-song prediction. Every model is compared against the
baselines (random, bigram Markov, item2vec) with the shared eval script.

## Conventions
- Python + PyTorch; settings live in `config.yaml`; seeds fixed via `set_seed`.
- Skip = skipped flag or under 30s played. Sessions split on a 30 minute gap.
- Vocab = songs with 3+ plays, plus `<UNK>`, `<START>`, `<END>`.
- Setup: `pip install -e ".[dev]"`; tests: `pytest`.
