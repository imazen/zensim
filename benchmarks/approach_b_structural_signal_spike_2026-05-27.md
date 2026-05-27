# Approach-B spike: content-robust localized-defect signal (task #33, 2026-05-27)

## Hypothesis

Perceptual tiling (tile-min / global−min gap) FAILED on screen content
(`local_defect_head_design_2026-05-27.md`): honest screen q20 has
locally-bad tiles (text-edge ringing / flat-region artifacts compress
poorly), so the min-to-min bar is too low for a localized defect to beat.
The fix must recognize the DEFECT vs honest compression, not just localized
perceptual difference.

**Hypothesis (error-vs-activity decorrelation):** honest compression error
correlates with local source activity — busy/edge regions quantize harder,
flat regions stay clean. A structural decoder bug injects error NOT explained
by the source's local activity: large error in a tile whose source activity
doesn't justify it. The discriminator is, per tile, how far the error lies
ABOVE the honest error-vs-activity cloud — **relative to the source's own
activity, hence content-robust.**

## Method (spike: `scripts/v_next/structural_signature_spike.py`)

Per non-overlapping 64×64 tile, vs the clean ref (BT.601 YCbCr):
- **mean signal**: tile RMSE (luma + chroma)
- **maxpix signal**: max over the tile's pixels of `|Δluma| + |Δchroma|`
  (catches thin/hard defects a tile-mean dilutes — a 1px line)
- **activity**: source luma std in the tile

Build a per-activity-bin honest bar (8 quantile bins) = p95 of each signal
over a sample of honest q20 anchors. A defect's score = max over its tiles of
`(signal − honest_p95[tile's activity bin])`. **Gate = either signal clears
its honest-q20 p95 bar.** Localized regions only (sq8/sq16/sq64/frac4),
structural-bug families only (channel/block/chroma/composite/overlay).

## Result — beats tile-min on BOTH content types, content-robust

| gate | PHOTO (gb82/dog) | SCREEN (codec_wiki) |
|---|--:|--:|
| prior **tile-min** (perceptual) | 37% | **24%** |
| mean-excess (activity-decorrelation) | 52.9% | **47.5%** |
| maxpix-excess | 63.7% | 17.9% |
| **combined (either)** | **68.8%** | **47.5%** |

- **Mean-excess ~2× tile-min on screen** (47.5% vs 24%) and is the
  content-robust workhorse — comparable photo/screen (53/48) unlike
  tile-min's lopsided 37/24.
- **Maxpix-excess complements on photo** (53→69%): honest photo has *zero*
  anomalous max-pixel error (bar 0.00) so any pixel spike is a defect. On
  screen it adds nothing — honest text-edge ringing spikes single pixels
  (bar 71), so maxpix can't discriminate there.

Per-family combined gate (photo): block_garbage/channel_invert/channel_max_r/
composite_*/overlay_glyph/overlay_rect all 92%; overlay_line (thin) 75%
(maxpix rescued it). Screen: block/channel_invert/composite/overlay_rect 58–75%.

## Honest gaps — where it still misses (and why)

Misses are dominated by **op20** (the defect blended 20% with honest content):
photo 45/75 misses, screen 58/126 misses are op20. A 20%-opacity localized
perturbation is genuinely faint — these are arguably SHOULD-miss
(near-imperceptible). op100 misses are few (photo 8, screen 27).

Two real residual gaps at op100:
1. **channel_swap on achromatic regions** (channel_swap_rg 0% screen): swapping
   R↔G where the source is gray (R=G=B, i.e. black text on white) produces
   ZERO error — the swap is *invisible*. Arguably a correct miss; only a
   chroma-aware "this region SHOULD be neutral but the swap created a
   detectable hue under any non-gray pixel" signal would catch the rest.
2. **chroma_boundary 0% both**: pure chroma misalignment. The current
   `|Δluma| + sqrt(chroma)/...` weighting under-weights chroma; a chroma-only
   channel (or a higher chroma weight) is the obvious next lever.

## Verdict + next

**Approach B's direction is VALIDATED.** A content-robust signal —
error measured RELATIVE to the source's own local activity — is decisively
better than perceptual tiling on the content type (screen) that defeated
tile-min, and better on photo too. The spike is a 2-signal hand-built
discriminator at ~50–69%; a *trained* structural-defect head on these
features (+ a chroma-specific channel + op-level-aware calibration) is the
path to the design doc's ≥90% target. The honest ceiling: op20 faint blends
and invisible swap-on-gray are arguably outside any metric's remit (they're
near-imperceptible by construction).

Concrete next chunks (none ship a public API — all measurement/infra):
1. Add a **chroma-only** signal channel (close chroma_boundary + colored-region
   swaps). Cheap — one more block-reduce on the chroma error map.
2. **op-level-stratified** gate report (op100 vs op50 vs op20) to separate
   "real defect missed" from "faint blend correctly ignored."
3. If 1–2 push op100 to ≥90%, promote the signal into a Rust
   `score_tiles_with_bake`-style binary (still no public zensim API change),
   then propose the `ZensimLocal` API to the user.

Spike: `scripts/v_next/structural_signature_spike.py`. Logs:
`/tmp/struct-sig-{screen,photo}.log` (this run). Corpus:
`/mnt/v/output/zensim/corruption_gate{,_screen}/` (672 entries each,
codec-corpus#7 generators).
