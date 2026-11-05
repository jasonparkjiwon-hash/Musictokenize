# Musictokenize: Summary

**Outcome:** Closed on 2026-10-28 as a failed proof of concept, after Phases 0–4 of 9. Phases 5–9 were not started.

## Goal
Build a tiny language model where the vocabulary is songs instead of words. Given a seed song (and optionally recent songs), return a sequence the user is unlikely to skip. Tempo and genre don't matter; only "would I skip this after that?" does.

The model was meant to learn:
- P(next song | previous songs) from listening sessions.
- P(not skipped | context, candidate) from the user's own skip labels.

**Proof-of-concept question:** can a model trained on the user's own history suggest songs they are less likely to skip than simple baselines?

**Metrics:** skip rate of top picks on held-out data (time-based split, last 2 months); secondary hit@10 and NDCG@10. Compared against random, popularity, bigram Markov, item2vec and feature kNN.

## Process (what was done)
| Phase | Result |
|---|---|
| 0 Repo setup | Layout, gitignore for personal data, pyproject, `config.yaml`, seeds, `CLAUDE.md`, `STATUS.md` handoff file |
| 1 Data to sequences | Parsed the Spotify extended export (136,530 plays, 2023-07 to 2026-10) into 4,249 sessions with skip labels, vocab (3+ plays) and a time-based split (train 121k / val 15k plays) |
| 2 Metadata | Joined the user's library workbook by Spotify track ID: 2,838 of 2,897 songs matched (98%). Only ~31% of played songs have genre/tempo features |
| 3 Baselines | Random, popularity, keep-popularity, bigram, item2vec, feature kNN, with a shared eval script |
| 4 GRU | 1-layer GRU next-song model on own sessions only (external pretraining dropped by user), with a small sweep. Did not beat the baselines |
| 5–9 | Not started |

Conventions: skip = skipped flag or under 30s played; hard skip = under 5s; sessions split on a 30-minute gap; sessions that are rapid-skip "browsing bursts" (50+ plays, 70%+ hard skips) are masked from skip training.

## Results (validation, burst sessions excluded)
| Model | hit@10 | NDCG@10 | pick_skip@10 (lower is better) |
|---|---|---|---|
| random | 0.002 | 0.001 | 0.654 |
| popularity | 0.014 | 0.007 | 0.331 |
| keep_popularity | 0.015 | 0.007 | **0.301** |
| bigram_markov | 0.030 | 0.017 | 0.530 |
| item2vec | 0.031 | 0.021 | 0.643 |
| feature_knn | **0.037** | **0.025** | 0.579 |
| GRU (first run) | 0.037 | 0.021 | 0.534 |
| GRU (sweep winner) | 0.027 | 0.013 | 0.497 |

`pick_skip@10` is the mean historical skip rate of the model's 10 picks. It is not a true counterfactual, because skips are only observed for songs actually played.

## Why it was closed
- The GRU (val hit@10 0.027–0.037) did not beat feature kNN (0.037) or clearly beat bigram (0.030).
- Next-song models, GRU included, pick songs skipped 50–58% of the time. Context-blind "popular songs I usually keep" picks are skipped only 30–33%. The simple baseline already wins on the metric that matters.
- The data is high entropy and small for this: about 134k plays, 4.5k vocab songs, ~70% skips, 14% of validation plays outside the vocab, and about a quarter of plays in rapid-skip bursts.
- The skip head (Phase 5) was never tried, so the hypothesis that scoring candidates for non-skip beats popularity is untested.

## Lessons learned
1. **Predicting what you play next is not predicting what you keep.** Context-aware models won on hit@10, but their hits were skipped about as often as random, because much of what follows a song in this history is browsing.
2. **Build the strongest simple baseline first.** A context-blind keep-rate ranker beat everything on the real metric. Any future model has to beat that, not random.
3. **The user skips a lot.** ~70% of plays are skips, 46% are under 5s. Random already scores ~70% skip, so the metric has little headroom unless the target is defined carefully.
4. **Data limits bite early.** ~5k songs heard once, 69% of distinct played songs outside the library (discovery), and heavy overfitting (train loss ~6.3–6.9 vs dev ~7.9).
5. **Browsing bursts need handling.** One 3,532-play session was 100% skips at one play per 2.7s. Flagging bursts and masking them from skip training was necessary.
6. **Don't tune on validation.** Validation was viewed twice. A held-out dev slice (last month of train) was used for selection, and the sweep winner was reported even though an earlier run scored better on val.
7. **Playlists weren't usable as sequences.** The library workbook has membership but no order, so pretraining and extra sequences were dropped.
8. **Process helped.** One phase per session, a `STATUS.md` handoff file and a decisions log made it cheap to resume and to stop.

## If ever reopened
- Use a context-blind keep-rate ranker plus light personalisation as the real baseline.
- Run Phase 5, the skip-aware head, since it is the untested idea.
- Use a non-burst, larger dev set for model selection.
- Use order-preserving playlist data (e.g. Exportify with added-at) for more sequences.

## Open questions left
- Which external datasets are still available?
- Is the 30s skip threshold right? The flag marks ~14k plays the threshold misses.

Source: `STATUS.md`, `PLAN.md`, `results/` in this repo. Only aggregate numbers are included; no personal listening data.
