# HDR-VDC viewing-condition study — **ppd/distance front-end adaptation PAYS (+0.077, p<1e-4); front-end LUMINANCE conditioning does NOT (−0.018 n.s.) — luminance conditioning is now dead at both levels** (2026-07-29)

**VERDICT (registered rule): the second clause fires — "conditioning pays
only jointly with ppd adaptation" — and the registered (iii)−(ii)
decomposition attributes the ENTIRE gain to the ppd/distance term.**
On 464 human pwcmp-JOD rows (116 HDR-AV1 videos × 4 viewing conditions,
HDR-VDC / QoMEX 2024):

- **Front-end luminance conditioning does NOT pay.** Feeding the actual
  display luminance into the chunk-2 PQ display model (measured 700-nit
  peak + the experiment's documented ÷8-in-linear dimming shader for dim
  rows) moves the probe's pooled cross-condition SROCC **−0.0178**
  (CI95 [−0.0530, +0.0159], p(Δ≤0)=0.85 — n.s.; the mirrored harm rule
  does not fire either), and does not shrink the alignment gap (+0.0605 →
  +0.0632). The null is axis-robust: it reproduces on the isolated
  within-luminance pools (near −0.013, far −0.026) and even
  WITHIN-condition at dim (dim-near probe 0.728 → 0.683), where no
  cross-luminance anchor caveat applies. score228: +0.0021 n.s.
  **Combined with the feature-level closure (`csfw_g6_loo_2026-07-29.md`
  G6 FAIL + `hdr_dmean_commensurability_2026-07-29.md` no-value →
  family closed), luminance conditioning is now measured dead at BOTH
  levels of the pipeline.**
- **Front-end ppd/distance adaptation pays decisively.** Scoring far-
  viewing rows (120 ppd) on the registered 2× Lanczos downscale of the
  4K display frame lifts pooled cross-condition SROCC **probe +0.0767
  (CI95 [+0.0405, +0.1170], p(Δ≤0)=0.0000)** and **score228 +0.0974
  (CI95 [+0.0710, +0.1261], p=0.0000)**, and flips the alignment gap
  negative (probe +0.0605 → **−0.0229**; score228 +0.0661 → −0.0263):
  pooled cross-condition ordering under leg (iii) is BETTER than the
  mean within-condition ordering — the adaptation adds genuine
  cross-condition information instead of just preserving within-condition
  rank. The (iii)−(ii) decomposition is +0.0945/+0.0952 — larger than
  (iii)−(i) itself because the luminance term subtracts. Consistent with
  the paper's own ANOVA (distance η² 0.101 = medium; luminance η² 0.015
  = small).
- **Q1: the 944 probe transfers materially to HDR-AV1 VIDEO** — its
  first video-domain read: within-condition SROCC 0.68–0.77 (all 4
  conditions clear the registered 0.60 band), per-content mean 0.92–0.95
  (the pwcmp comparison unit). The fixed score228 readout is *stronger*
  than the probe on every cell (0.75–0.82 pooled-within-condition) — the
  UPIQ-SDR ridge head adds nothing over the fixed readout on this axis.
- **Q3 (diagnostic): BANDVIS-8 alone (OOF 0.700) ≈ full-944 (0.716) —
  fourth independent BANDVIS-is-the-carrier read**, first on video;
  only-append2 (20 lanes) 0.746 BEATS full-944 and minus-folded720
  (0.770) beats everything (small-n dilution reproduced on a fifth
  domain); and the **HDR-gated highlight bins are live and are the
  strongest single lanes on HDR-AV1** (HL_BIN1 s3 −0.672, s2 −0.646 —
  their first live-regime read; structural-0 on SDR).

Either way it landed with numbers: the CSF-family closure now has its
constructive complement — the viewing-condition axis that pays at the
front end is GEOMETRY (ppd), not luminance.

## Protocol, gates, provenance

Pre-registered BEFORE any number:
`/mnt/v/output/zensim/hdrvdc-conditions-2026-07-29/PROTOCOL.md` (3 verdict
looks + 1 diagnostic; pilot after registration, before extraction; zero
deviations — the registered chain ran as written). Build: zensim
origin/main **6b3505a5** (`hdrvdc_features_extract` examples-only commit
on `6b7a73cb`; feature-value code untouched). Gates ALL PASS: the UPIQ-SDR
944 ridge probe reconstruction reproduced the recorded hdr-dmean head
BIT-IDENTICALLY (λ=100 grid-edge, 689 cols, SDR CV 0.9362861433052736,
UPIQ-HDR 0.7596713929714486 / 0.7688482648531633 / 0.9346082397263838);
coverage **116/116 distorted videos × 8 frames × 5 configs = 4,640
extractions, 0 drops, 0 non-finite**; ref/test frame counts equal for all
116; all 148 streams uniform (AV1 yuv420p10le tv bt2020nc smpte2084
bt2020).

Data: labels `/mnt/v/datasets/hdr-vdc/HDR_VDC_JOD_Scores.csv` (528 rows;
the 64 is_reference rows are identity FR pairs pinned at JOD 10.0 —
excluded, 464 registered rows); videos Tower
`/mnt/tower/input/datasets/hdr-vdc/HDR-VDC.zip` (3.9 GB, sha256-verified
`fe38d5a4…`), staged batch-wise through `~/tmp` (per-content decode →
extract → delete; peak-RSS 5.45 GiB, min-avail 44.7 GiB, 1,697 s wall for
the full pipeline). Pointer: `zenpapers:datasets/HDR-VDC.pointer.md`.

Stimulus reconstruction (registered): ffmpeg/dav1d → full-range rgb48 PQ
code values (swscale bt2020 matrix, accurate_rnd+full_chroma_int) →
Lanczos-a3 upscale to the 3840×2160 display frame (the experiment
displayed ALL content at 4K via a Lanczos shader) → far leg additionally
Lanczos ×0.5 → 8 uniform frames/clip (LYB precedent: faithful for
mean-pooled statistics; compression+upscaling are sustained), mean
temporal pooling (ridge is linear ⇒ pooling features ≡ pooling
predictions; score228 = mean of per-frame scores).

The three registered configurations (all: declared-HDR streaming route,
mode 944, `HdrEncoding::Pq{..}` — the chunk-2 front-end display model
`min(EOTF_PQ, peak) + black + refl` under test):
**(i) BLIND** = raw code values, 4K frame, `Pq{1000}` (the route's
documented standard-HDR default) — one score per video for all 4
condition rows. **(ii) luminance-AWARE** = `Pq{700}` (measured LG G2
peak); dim rows get the experiment's documented dimming shader
(PQ-decode spec-peak → linear ÷8 → PQ-re-encode, f64; applied to both
sides — confirmed against the official `gfxdisp/HDR-VDC` display-model
example, whose measured G2 LUT is exactly what the zensim front-end
model replaces here). **(iii) luminance+ppd-AWARE** = (ii) + far rows
(120 ppd) scored on the 1920×1080 downscale (registered mapping:
metric-native ≈ 60 ppd assumption ⇒ ×0.5; near 60 ppd ⇒ ×1).

## Q1 — within-condition ranking (n=116/condition; scene-cluster CI)

| condition | probe (i) | probe (ii) | probe (iii) | s228 (i) | s228 (ii) | s228 (iii) |
|---|--:|--:|--:|--:|--:|--:|
| bright-near | 0.7362 | 0.7358 | 0.7358 | 0.7911 | 0.7884 | 0.7884 |
| bright-far | 0.7430 | 0.7439 | **0.7660** | 0.8021 | 0.8010 | **0.8195** |
| dim-near | 0.7279 | 0.6829 | 0.6829 | 0.7645 | 0.7487 | 0.7487 |
| dim-far | 0.7130 | 0.6970 | 0.7086 | 0.7632 | 0.7615 | **0.7839** |

Per-content SROCC (the pwcmp unit; n=5..8 rows per content — small-n):
mean 0.91–0.95 across all cells, min 0.38–0.86. Bootstrap CI95s ≈ ±0.10
(16 content clusters). Within-condition, luminance-aware configs (ii)
never help and cost up to −0.045 at dim-near; the far-row downscale
(iii) helps within bright-far/dim-far too (+0.02 both scorers).

## Q2 — cross-condition commensurability (the verdict look; 464 rows)

| leg | probe pooled | probe gap | s228 pooled | s228 gap |
|---|--:|--:|--:|--:|
| (i) blind | 0.6695 | +0.0605 | 0.7141 | +0.0661 |
| (ii) lum-aware | 0.6517 | +0.0632 | 0.7162 | +0.0587 |
| (iii) lum+ppd-aware | **0.7462** | **−0.0229** | **0.8114** | **−0.0263** |

(gap = mean within-condition SROCC − pooled; positive = condition
misalignment the metric fails to explain.)

Paired deltas (cluster bootstrap over 16 contents, 10k, seed 20260729):

| Δ | probe | p(Δ≤0) | s228 | p(Δ≤0) |
|---|--:|--:|--:|--:|
| (ii)−(i) | −0.0178 [−0.0530,+0.0159] | 0.8547 | +0.0021 [−0.0229,+0.0270] | 0.4223 |
| (iii)−(i) | **+0.0767** [+0.0405,+0.1170] | **0.0000** | **+0.0974** [+0.0710,+0.1261] | **0.0000** |
| (iii)−(ii) | +0.0945 [+0.0753,+0.1162] | 0.0000 | +0.0952 [+0.0752,+0.1161] | 0.0000 |

Axis decompositions (registered): cross-DISTANCE pools (directly
measured axis) — bright: probe 0.680 (i) → 0.784 (iii), s228 0.731 →
0.831; dim: probe 0.670 → 0.719, s228 0.710 → 0.792. Cross-LUMINANCE
pools (anchor-linked axis): near probe 0.723 (i) → 0.710 (ii); far
0.725 → 0.700 — luminance-awareness slightly HURTS even on its own
axis. Headroom: per-video JOD spread across its 4 conditions mean 1.50
JOD (q25 1.07, q75 1.79, max 3.28) — there is real condition-driven
variance, and the ppd leg captures much of the distance part of it.

## Q3 — diagnostic family attribution (no verdict weight)

Scene-disjoint nested CV (outer GroupKFold(4) by content) on leg-(ii)
features, target = per-condition z-scored JOD (condition main effect
removed):

| set | OOF SROCC | Δ vs full |
|---|--:|--:|
| full944 | 0.7160 | — |
| minus_folded720 | **0.7703** | +0.054 |
| only_append2 (20) | 0.7461 | +0.030 |
| minus_append | 0.7366 | +0.021 |
| only_folded720 | 0.7335 | +0.018 |
| only_append (204) | 0.7292 | +0.013 |
| minus_append2 | 0.7130 | −0.003 |
| only_BANDVIS8 | 0.7004 | −0.016 |

Zero-fit single lanes (pooled |SROCC|, error polarity): **HL_BIN1 s3
0.672 / s2 0.646** (highlight-error bins — live on the HDR route for the
first time; HL_BIN2 constant-0 on this corpus, flagged not read),
BANDVIS_GAIN s3 0.611 / s2 0.529, BANDVIS_LOSS s0 0.375 (per-condition
0.45–0.54), LUMA_MEAN_REF 0.083 (weak, scale-replicated). BANDVIS on
HDR-AV1 = the would-be fourth carrier read and it holds at the set level
(BANDVIS-8 ≈ full-944), with the honest temporal caveat: 8-frame
mean-pooled; LYB's full-temporal validation showed mean pooling faithful
for sustained artifacts, but AV1 banding can be transient and no
full-temporal leg was run here.

## Caveats

1. **Cross-luminance JOD is anchor-linked** (the experiment collected no
   cross-luminance comparisons; bright↔dim comparability rests on the
   common reference node). The luminance-conditioning null does NOT rest
   on that linkage — it reproduces within-condition and within-axis —
   but pooled-across-luminance magnitudes carry the caveat.
2. **The ppd mapping rests on the registered metric-native ≈ 60 ppd
   assumption** (calibration corpora are desktop-viewed SDR; no exact
   figure exists). No alternative scale factors were evaluated (no axis
   mining). A luminance-blind + ppd-aware combination was NOT registered
   or extracted; the (iii)−(ii) decomposition brackets it.
3. Rank statistics only; n=16 content clusters; JOD label CI95 width is
   itself mean 0.93 JOD. Per-content cells are n=5..8.
4. score228 outperforms the UPIQ-SDR ridge probe on every Q1 cell — on
   this axis the probe head is not the value-add; both carried the same
   verdict directionally.
5. The dim leg replaces the official measured G2 LUT with zensim's
   analytic front-end model (hard clip at peak + fixed black/reflection
   0.005/0.398 cd/m² vs the experiment's ~0-lux room) — registered
   approximation; the ÷8 shader itself is exact (round-trip 1e-13).
6. Resampling chain approximations (swscale vs GPU sampler chroma
   upsampling, PNG 16-bit quantization) are common-mode across legs.

## Product implication

The HDR route's display-model declaration (`Pq{peak_nits}`) is a
correctness feature, not a quality-conditioning lever — consumers need
not chase exact display luminance for ranking accuracy (Δ n.s.). What
IS worth exposing is VIEWING-GEOMETRY adaptation: scoring at the
viewer's effective ppd (here: 2× downscale for 120 ppd) bought +0.08..
+0.10 pooled SROCC on real cross-condition human data — and costs LESS
compute (far legs run at quarter pixels). Any future viewing-condition
surface should be a ppd/distance knob, not a nits knob.
