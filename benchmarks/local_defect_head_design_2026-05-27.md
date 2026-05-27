# Localized-defect head — design (task #33, 2026-05-27)

## Why (the corruption-gate finding)

The corruption-corpus gate (codec-corpus#7) proved that NO global mean-pooled
perceptual metric (v47-recal, QAT, V39) can rank a localized 8×8 structural
corruption below an honest q20 encode — the 8×8 defect is globally negligible
(recal scores it ~74–96 vs q20's 44). zensim correctly measures *global*
perceptual difference; the regression-test use case wants "is there ANY
structural defect," which is a LOCAL property. Whole-image corruptions pass
the gate; localized ones (sq8/sq16: 0%) don't. The fix is a LOCAL signal.

## Approach A (recommended, cheap): tile-min pooling

Score zensim on a grid of overlapping tiles; report `min(tile_scores)` (or a
low percentile, e.g. p2). A localized 8×8 corruption fully corrupts one tile →
that tile's score craters → `min` reflects it. An honest q20 encode degrades
ALL tiles moderately → `min ≈ global score`. So `min(tile)` ranks localized
corruption BELOW honest q20 — the gate.

- Tile size: sweep (64×64, 32×32) with 50% overlap. Smaller tiles catch
  smaller defects but raise per-tile noise. The corruption corpus's region
  axis (whole→8×8) is the calibration grid for the tile size.
- No retraining — wraps the existing zensim metric. `zensim::compute` per tile.
- Output: a SECOND score `zensim_local = min/p2 over tiles`, alongside the
  global `zensim`. Regression tests gate on `zensim_local`; quality dials use
  the global `zensim`.
- Cost: ~(image/tile)² metric calls. For 576×576 / 64×64 = 81 tiles ×
  ~ms each = manageable; SIMD + the strided-row API (CLAUDE.md) make it cheap.

## Approach B (heavier): structural-defect classifier

Train a small classifier (on the corruption corpus + honest-lq anchors) that
outputs P(structurally-broken). Compose: `final = global_zensim − BIG·P(broken)`.
Needs the corpus as training data (codec-corpus#7 / PR#8 has the generators).
More expressive but more infra; do A first.

## Validation

Re-run `scripts/v_next/corruption_gate_eval.py` with the tile-min score: the
gate `min_tile(corruption) < min_tile(q20)` should now PASS on the localized
families (sq8/sq16) that the global metric failed. Target: ≥90% gate pass
across families × regions × severities, including the subtle end.

## Open questions

- Tile-min vs p2 vs a learned pooling — tile-min is most sensitive (catches
  the single worst tile) but noisiest; p2 is a robust compromise. Sweep on the
  corpus.
- Overlap stride — 50% overlap halves the chance a defect straddles a tile
  boundary + dilutes; finer overlap costs more calls.
- Interaction with the negative tail — a tiled localized corruption should
  push `min_tile` negative (the corrupted tile's score), giving the
  regression-test "broken < honest-lq < identity" ordering the use case needs.

## Status: design only (the corpus + gate harness are ready:
scripts/v_next/corruption_gate_eval.py, /mnt/v/output/zensim/corruption_gate,
codec-corpus PR#8). Implementation deferred behind the QAT verification.
