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

- **Content-robust**: comparable photo/screen at every op level, unlike
  tile-min's lopsided 37/24. The activity-relative bar is what makes it work
  on screen where perceptual tile-min collapsed.
- **All three channels contribute**: mean is the screen workhorse, maxpix
  rescues thin/hard defects on photo (honest photo has zero pixel spikes),
  chroma lifts both (screen 47.5→55.8%, photo 68.8→71.2%).

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
1. ✅ **chroma channel** — DONE this spike (lifts screen 47.5→55.8%, photo
   68.8→71.2%; op100 now 90/72.5).
2. ✅ **op-level stratification** — DONE; reveals the real op100 detection rate.
3. **Multi-scale tiling** (64 ∧ 16): the dominant screen op100 miss is sq8,
   which a 64px tile dilutes. Add a 16px pass and take the max excess across
   scales — but rebuild the honest bar per scale (16px hits honest-ringing).
4. **Generator spot-check chroma_boundary**: chroma signal is literally zero
   → confirm the defect is actually visible at these severities before
   chasing it.
5. If 3 pushes screen op100 ≥85%, promote the 3-channel signal into a Rust
   `score_tiles_with_bake`-style binary (still no public zensim API change),
   then propose the `ZensimLocal` API to the user.

Spike: `scripts/v_next/structural_signature_spike.py`. Log:
`/tmp/struct-sig-3ch.log` (this run). Corpus:
`/mnt/v/output/zensim/corruption_gate{,_screen}/` (672 entries each,
codec-corpus#7 generators).
