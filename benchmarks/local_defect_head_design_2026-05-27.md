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

## VALIDATION (2026-05-27, tile-min scorer built + run)

Built `zensim-validate/src/bin/score_tiles_with_bake.rs` (in-process: load
ref+dist once, tile, score each tile via the bake, report global/min/p2/p5/
median). `corruption_gate_eval.py` gained a `TILE_MIN=1` mode (+ `TILE_SIZE`).
Run on the QAT-native bake + gb82/dog corpus:

| | global metric | **tile-min, tile=64** | tile-min, tile=32 |
|---|--:|--:|--:|
| overall gate (corruption<q20) | 17.3% | **36.9%** | 0.0% |
| sq8 localized | 0% | 11% | 0% |
| whole | 35.6% | 47% | 0% |

Spot-check (tile=64): channel_invert sq8 global 71.9 (FAILED) → **min 11.1**
(PASS, < q20-min 31.1); block_zero sq8 75.3 → **14.7** (PASS). The corrupted
tile craters. identity min 97.7; chan_invert WHOLE min 0.0.

**Findings:**
1. **Tile-min fixes structurally-significant localized defects** (channel
   swap, block zero/garbage — the real decoder bugs): the corrupted tile
   craters far below honest content. 17%→37% overall, sq8 0%→pass for these.
2. **tile=64 is the granularity sweet spot.** tile=32 → 0% because the honest
   q20 anchor's OWN worst 32×32 tile craters to ~0 → the min-to-min bar
   becomes impossible. Honest compression has locally-bad tiles too; the tile
   must be large enough that honest content's worst tile stays moderate
   (q20-min ~31 at tile=64).
3. **Subtle / sub-perceptual stay high** (1-px bit-flip min 94.8; low-opacity
   overlay; tiny tone shift) — arguably correct (near-imperceptible even
   locally), or needs a different signal than perceptual tiling.
4. **Gate definition is a calibration choice.** min-to-min(q20) at tile=64
   gives 37%. Alternatives to evaluate: min-to-GLOBAL(q20) (compare the
   broken decode's worst tile to the honest decode's overall score), or an
   absolute threshold (min_tile < ~20 catches channel/block, excludes q20's
   31). The corruption corpus is the calibration grid.

## Multi-content (2026-05-27): tile-min is CONTENT-DEPENDENT

Re-ran on screen content (gb82-sc/codec_wiki.png, text/UI):

| | gb82/dog (photo) | codec_wiki (screen) |
|---|--:|--:|
| gate global | 17% | 24% |
| gate tile-min (tile=64) | 37% | 24% (no gain) |

Tile-min doubled the PHOTO gate but gave NO gain on SCREEN. Cause: screen
content's q20 anchor has LOCALLY-bad tiles (text edges / flat-region ringing
compress poorly → a 64×64 tile craters), so the honest q20's min-tile is
already low → the min-to-min bar is too low for localized corruptions to beat.
On continuous-tone photo, q20 degrades uniformly → min-tile stays moderate →
the bar works.

**Implication:** the min-to-min(q20) gate is fragile across content. The
shipped local-defect head should use a CONTENT-ROBUST gate — min-tile <
absolute threshold T (calibrated so honest-lq's min-tile > T across content),
or min-tile(corruption) vs GLOBAL(honest) — NOT min-to-min. The corpus
(multi-content) is the calibration set for T.

V39-broken-at-identity (scores 0) and QAT-identity-correct (97.7) BOTH
generalize across photo + screen (+ 8 refs) — robust.

## Next (shipped head)

- Promote tile-min into a first-class Rust API (a `ZensimLocal` profile or a
  `zensim::compute_local` returning global + min-tile), wired into
  zensim-regress so a broken decode's tile-min gates the test.
- Tune tile size (48–96) + the gate definition on the full corpus + multiple
  refs. Optionally Approach B (a defect classifier trained on the corpus) for
  the subtle families tile-min can't catch.

Scorer + gate harness ready: `score_tiles_with_bake`,
`corruption_gate_eval.py TILE_MIN=1`, results in
`benchmarks/corruption_gate_tilemin_2026-05-27.md`.
