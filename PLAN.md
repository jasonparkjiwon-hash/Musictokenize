# Personal Song-Sequence Model: Project Plan

## Goal
Given a seed song (and optionally the last few songs played), return a sequence of songs I am unlikely to skip. Tempo and genre don't matter; only "would I skip this after that?" does.

**Core framing:** a tiny language model where the vocabulary is songs instead of tokens.
- Learns P(next song | previous songs) from listening sessions.
- Learns P(not skipped | previous songs, candidate) from my own skip labels.
- Pretrain on broad playlist data, then fine-tune on my history, then refine with feedback.

## Data

| Source | Use |
|---|---|
| Extended streaming history (requested) | Main training data: plays, timestamps, ms played, skip flags |
| Standard export (past year) | Backup if extended is delayed |
| My playlists and library | Extra sequences and positive signal |
| Excel list (library songs: tempo, genre, artist) | Optional metadata features; check whether it has track IDs |
| External playlists/sessions (ListenBrainz dumps, Last.fm, 30Music, Million Playlist Dataset if still hosted) | Pretraining transitions; verify availability first |
| Web metadata (MusicBrainz, Last.fm tags) | Optional artist/genre embeddings for rare songs; cache all API results |

**Skip from the export:** payment, customer service, family plan, location, voice, Spotify for Artists. Keep `data/raw/` in `.gitignore` (personal data).

**Scale check:** ~2500 min/week is about 37k plays per year. Enough for baselines and a small GRU. A transformer is a stretch goal. Repetition (not volume) is the real limit: many songs appear only 1-2 times.

## Architecture (high level)
1. **Song embedding table** (vocab = songs with 3+ plays, plus `<UNK>`, `<START>`, `<END>`).
2. **Sequence encoder:** GRU first, optional 2-layer transformer later.
3. **Two heads:**
   - Next-song head (pretraining): cross-entropy, sampled softmax if vocab is large.
   - Non-skip head (fine-tuning): scores a candidate given context, trained on keep vs skip labels.
4. **Optional inputs:** artist/genre embeddings, time of day.
5. **Discriminator = my skips.** Hand ratings and hard-coded rules are add-ons.

## Phases
Each phase is one Claude Code session and one commit.

### Phase 0: Repo setup
- Layout: `data/raw` (gitignored), `data/processed`, `src/`, `notebooks/`, `tests/`, `PLAN.md`, `CLAUDE.md`.
- Python, PyTorch, a config file, fixed random seeds.
- `CLAUDE.md` states the goal, the privacy rule (never commit raw data), and the eval metric.

**Prompt:** "Set up this repo per PLAN.md Phase 0. Add .gitignore for data/raw, a pyproject, and a CLAUDE.md summarizing the goal and rules."

### Phase 1: Data to sequences
- Parse streaming history into one table: track ID, artist, album, timestamp, ms played.
- Label each play **skip** (skipped flag, or under ~30s played) or **keep**.
- Split into sessions when the gap between plays exceeds 30 minutes.
- Build the vocab with a 3+ play cutoff; the rest map to `<UNK>`.
- Time-based split: train on older data, validate on the last 2 months.
- Convert playlists and library into extra sequences.

**Prompt:** "Implement Phase 1: parse the export in data/raw into sessions with skip labels, build the vocab and a time-based split, and write tests for session splitting."

### Phase 2: Metadata enrichment (optional but cheap)
- Join the Excel list and cached web metadata into `songs.parquet`.
- Match by track ID if available; otherwise fuzzy name + artist, and log unmatched songs.
- Output artist, genre, tempo, and a "missing" flag.

**Prompt:** "Implement Phase 2: join my Excel song list with the streaming history, report match rate, and write songs.parquet."

### Phase 3: Baselines
- Random song from my library.
- Bigram Markov chain ("what usually follows A").
- Item2vec (word2vec on sessions).
- Feature-only nearest neighbors (if Phase 2 is done).
- **Metric for everything:** skip rate of top picks on held-out data, plus hit@10 and NDCG for next-song prediction.

**Prompt:** "Implement Phase 3 baselines and a shared eval script that reports skip rate, hit@10, and NDCG on the validation split."

### Phase 4: Pretraining
- Pretrain on external playlist/session transitions, matched to my vocabulary by ID or name + artist.
- Then continue on my own sessions and playlists.
- GRU first; track loss and hit@10; small hyperparameter sweep; dropout for the small data.
- Stretch: 2-layer transformer.

**Prompt:** "Implement Phase 4: a GRU next-song model with pretraining then fine-tuning stages, config-driven, logging metrics each epoch."

### Phase 5: Skip-aware fine-tuning
- Train the non-skip head: positives are songs played through after a given context, negatives are skipped ones.
- Compare candidate scoring versus the baselines on held-out skip rate.
- Add time of day as an optional input and test whether it helps.

**Prompt:** "Implement Phase 5: add a non-skip head, train it on keep/skip labels, and report held-out skip rate versus baselines."

### Phase 6: Generation
- Seed song, then autoregressive sampling with temperature, top-k, and beam search.
- Rerank each step by predicted non-skip probability.
- Controls: repeat penalty, no repeats within a window, exploration knob (familiar versus unseen songs).
- Optional hard rules: max 2 songs per artist, a blacklist of skip-prone songs.

**Prompt:** "Implement Phase 6: sequence generation with rerank by non-skip probability and the controls in PLAN.md."

### Phase 7: Feedback loop
- CLI rating tool: thumbs up/down on generated sequences (extra labels beyond skips).
- Loop: generate, rate, retrain discriminator, fine-tune generator (pairwise preference loss or REINFORCE), repeat.
- Version every checkpoint and keep a held-out set of ratings to confirm real improvement.

**Prompt:** "Implement Phase 7: a rating CLI that stores feedback, plus a retrain step that updates the discriminator and generator with versioned checkpoints."

### Phase 8: Interface
- CLI: `suggest "song name" --n 20`.
- Optional: export the result as a Spotify playlist.

**Prompt:** "Implement Phase 8: the suggest CLI with output of song, artist, and predicted keep probability."

### Phase 9: Evaluation and writeup
- Ablations: with versus without pretraining, IDs only versus IDs plus features, with versus without feedback.
- Compare all against baselines and report the real-world skip rate when I use it for a week.
- README with an architecture diagram and lessons learned.

## Risks
- **Small data:** keep the model tiny, use dropout, lean on pretraining.
- **ID matching:** external playlists may not align with my song IDs; log the match rate early.
- **Playlist order is weak:** many playlists are loose collections, so treat pretraining as "goes together" rather than strict order.
- **Popularity skew:** external data favors popular songs; fine-tuning on my skips corrects for it.
- **Replaying my top 50:** tune the exploration knob and track how many unseen songs get suggested.
- **Skip is noisy:** I may skip due to context (location, mood, time of day), not the song itself.

## Open questions
- Does the Excel list include Spotify track IDs?
- Which external datasets are still available?
- Is the 30-second skip threshold right? Compare it against the skipped flag.
