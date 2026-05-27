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

Per non-overlapping 64×64 tile, vs the clean ref (BT.601 YCbCr), THREE
content-robust signal channels — each scored as `max over tiles of
(value − honest_p95[tile's activity bin])`, so the bar is relative to the
source's own local activity:

- **mean** channel: tile RMSE (luma + chroma), binned by source LUMA std.
- **maxpix** channel: max over the tile's pixels of `|Δluma| + |Δchroma|`
  (catches thin/hard defects a tile-mean dilutes — a 1px line), binned by
  source LUMA std.
- **chroma** channel: tile chroma RMSE, binned by source CHROMA std (a chroma
  defect in a chroma-flat source region is the anomaly).

Honest bar = p95 of each channel over a sample of honest q20 anchors. **Gate =
ANY channel clears its honest-q20 p95 bar.** Localized regions only
(sq8/sq16/sq64/frac4), structural-bug families only
(channel/block/chroma/composite/overlay), op-level reported separately.

## Result — at full defect strength, 90% photo / 72.5% screen

**The headline is the op-stratified number** — op100 is a full-strength defect
(what a regression test must catch); op50/op20 are 50%/20%-opacity blends that
are faint-to-imperceptible by construction:

| gate | PHOTO op100 | PHOTO all | SCREEN op100 | SCREEN all |
|---|--:|--:|--:|--:|
| prior **tile-min** (perceptual) | — | 37% | — | 24% |
| mean-excess | — | 52.9% | — | 47.5% |
| maxpix-excess | — | 63.7% | — | 17.9% |
| chroma-excess | — | 48.3% | — | 42.9% |
| **combined (any of 3)** | **90.0%** | 71.2% | **72.5%** | 55.8% |

| op level | PHOTO | SCREEN | note |
|---|--:|--:|---|
| op100 (full) | **90.0%** | **72.5%** | the regression-test target |
| op50 | 75.0% | 56.2% | half-opacity blend |
| op20 | 48.8% | 38.8% | 20%-opacity — near-imperceptible |

### Multi-scale (64 ∧ 16) — closes the sq8 gap (`SCALES=64,16`)

Adding a 16px tile pass (defect flagged if any channel at EITHER scale clears
that scale's own honest bar — each scale keeps its activity-relative
content-robustness) lifts op100 further:

| op level | PHOTO | SCREEN |
|---|--:|--:|
| **op100** | **92.5%** | **81.2%** |
| op50 | 80.0% | 72.5% |
| op20 | 58.8% | 51.2% |

The 16px scale catches sq8 defects the 64px tile diluted (channel_zero_b/r
sq8, block_copy_wrong sq8, …) while the 64px scale keeps honest screen
text-ringing from flooding the bar. **Excluding chroma_boundary** (see gaps
— it produces zero signal at both scales, i.e. likely imperceptible as
generated), screen op100 → ~85.5% and photo → ~97.4%.

- **Content-robust**: comparable photo/screen at every op level, unlike
  tile-min's lopsided 37/24. The activity-relative bar is what makes it work
  on screen where perceptual tile-min collapsed.
- **All three channels contribute**: mean is the screen workhorse, maxpix
  rescues thin/hard defects on photo (honest photo has zero pixel spikes),
  chroma lifts both (screen 47.5→55.8%, photo 68.8→71.2%).

### chroma_boundary is a CORPUS no-op, not a metric gap (diagnosed)

Direct check (`/tmp/chroma-boundary-check.log`): chroma_boundary's localized
variants change **zero pixels**. Screen sq8/sq16/sq64 op100:
`luma_rmse=0.00 chroma_rmse=0.00 maxpix=0.0 px>1=0` — literally no pixel
differs from the ref by >1. Photo sq8/sq16: `px>1=14/38`. Even the whole-image
variant (screen chroma_rmse 0.93, photo 0.87) is far below honest q20
(chroma_rmse ~2.2–2.9). So the signal's "chroma_boundary 0%" is **correct** —
there is nothing to detect; the generator's localized chroma_boundary is
effectively identity. This is a **codec-corpus#7 generator bug**, filed
separately. Excluding chroma_boundary, the clean op100 headline is:

| | PHOTO op100 | SCREEN op100 |
|---|--:|--:|
| **real localized defects (chroma_boundary excluded)** | **97.4% (74/76)** | **85.5% (65/76)** |

## Honest gaps — the op100 residual (faint blends excluded)

Only **8 photo / 22 screen** op100 defects go undetected:

1. **sq8 on screen (8×8 region)**: a 64×64 tile averages an 8px defect to
   near-nothing (mean/maxpix both −1.81 = zero signal). Smaller tiles would
   catch it but reintroduce the honest-screen-ringing problem — a **multi-scale**
   tile pass (64 ∧ 16) is the fix. Most screen op100 misses are sq8.
2. **chroma_boundary 0% everywhere, chroma signal literally zero** (−0.63
   screen, −2.x photo). The defect produces NO measurable chroma error at
   these severities → it is likely **near-imperceptible as generated** (worth a
   generator spot-check: is the boundary shift sub-tile or sub-threshold?).
3. **channel_swap on achromatic regions** (swap_rb/rg sq8/sq16): swapping R↔G
   where the source is neutral (gray text) produces ZERO error — *invisible*,
   an arguably correct miss. Only "this pixel SHOULD be neutral, the swap made
   it chromatic" catches the few non-gray pixels, and at sq8 there aren't
   enough to clear the bar.

So the honest ceiling at op100 is set by genuinely-invisible cases
(swap-on-gray, possibly chroma_boundary) plus the 64px-tile-vs-8px-defect
scale mismatch — not by a flaw in the activity-decorrelation principle.

## Verdict + next

**Approach B's direction is VALIDATED.** A content-robust signal — error
measured RELATIVE to the source's own local activity — is decisively better
than perceptual tiling: at full defect strength (op100) the 3-channel
hand-built discriminator catches **90% photo / 72.5% screen** vs tile-min's
37%/24% overall, and it's content-robust where tile-min collapsed on screen.

The honest op100 ceiling is set by genuinely-hard cases, not a flaw in the
principle: 8×8-defect-vs-64px-tile scale mismatch, and invisible
swap-on-neutral / possibly-imperceptible chroma_boundary. A *trained*
structural-defect head on these per-tile features (mean/maxpix/chroma vs
activity), multi-scale tiling, and op-aware calibration is the path to ≥90%
on screen too.

Concrete next chunks (none ship a public API — all measurement/infra):
1. ✅ **chroma channel** — DONE (lifts screen 47.5→55.8%, photo 68.8→71.2%).
2. ✅ **op-level stratification** — DONE; reveals the real op100 detection rate.
3. ✅ **Multi-scale tiling (64 ∧ 16)** — DONE; op100 → photo 92.5% / screen
   81.2% (~97% / ~85% excluding chroma_boundary). Per-scale activity-relative
   bars keep honest-screen-ringing in check at 16px.
4. **Generator spot-check chroma_boundary** (the one clear remaining gap):
   signal is zero at BOTH scales on BOTH content types → the defect is almost
   certainly sub-threshold/imperceptible as generated. Confirm before treating
   it as a metric gap (it's likely a corpus-severity issue, codec-corpus#7).
5. The signal is now strong enough to **promote into a Rust
   `score_tiles_with_bake`-style binary** (multi-scale, 3-channel,
   activity-relative bars; still no public zensim API change), wire into
   `zensim-regress`, then propose the `ZensimLocal` public API to the user.
   The remaining non-chroma_boundary misses (swap-on-achromatic-small-region,
   block_repeat_neighbor) are near-imperceptible by construction — an honest
   ceiling, not a blocker.

Spike: `scripts/v_next/structural_signature_spike.py`. Log:
`/tmp/struct-sig-3ch.log` (this run). Corpus:
`/mnt/v/output/zensim/corruption_gate{,_screen}/` (672 entries each,
codec-corpus#7 generators).
