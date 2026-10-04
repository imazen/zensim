# NEIGHSTEER — neighbour-aware steering: why the map misranks heavy-JPEG blocks, and the fix (2026-10-04)

User, 2026-10-04 00:15 MT: "work on neighbor aware steering solutions", after JPEGSTEER (`jpegsteer_2026-10-04.md`) showed both
by_v2fy and v2 + basic failing 8×8 steering at KADID's heaviest JPEG level. Development diagnostics on TRAIN-role KADID content;
nothing here qualifies a product change. Instruments: `diffmap_block_coherence` (with the 2026-10-04 `ZENSIM_REPAIR_ALPHA` /
`ZENSIM_REPAIR_SOURCE` options, zensim dfd24c3f) on the COSTSET strict-route bakes, densified and stamped Rev4
(`/var/tmp/steercheck/gate3/rev4dense`), prepared steering, `ZENSIM_FORMULA_REV=4`, single thread.

## 1. What fails

At KADID JPEG level 05, block 8, both sets pass 2–3 of 12 cases: M3f (rank of the map's `refinement_gain` against the true score
change from copying the reference into the block) has median ≈ 0.65, while M2 (the same rank with the TRUE per-block feature changes
pushed through the gradient) is ≈ 1.000. The features and the head are fine; the map's prediction of the feature changes is not.
Two properties of the failing cases:

* **Repairs can lower the score.** 47–58 % of the 8×8 reference repairs at level 05 have ΔS < 0: a pristine block pasted among heavily
  blocked neighbours creates a seam. The frozen-signal map has no term for a seam it never measured.
* **The error tracks the neighbourhood.** On I61 level 05 (by_v2fy, s5101) the per-block log prediction ratio correlates −0.47 with the
  neighbours' mean true ΔS and −0.64 with the block's own ΔS.

## 2. Where the error lives (per-feature oracle)

`ZENSIM_V2_DIAG` records, per block and per feature ID, the observed `s_k·Δf_k` and the map's predicted density for that ID. Replacing the
predicted per-feature gains with the observed ones on chosen subsets gives the M3 the map WOULD reach if that subset were exact.
48 cases (2 sets × 3 seeds × KADID I01/I21/I41/I61 × levels 03, 05), block 8, medians:

| set, level | map today | exact v2 at scales 1–3 | exact v1 basic at 1–3 | exact all at 1–3 | exact all scales (= M2) |
|---|---:|---:|---:|---:|---:|
| by_v2fy, 03 | 0.813 | 0.983 | 0.833 | **0.998** | 1.000 |
| by_v2fy, 05 | 0.650 | 0.966 | 0.640 | **0.994** | 1.000 |
| v2 + basic, 03 | 0.833 | 0.931 | 0.842 | 0.947 | 1.000 |
| v2 + basic, 05 | 0.644 | 0.903 | 0.681 | 0.897 | 1.000 |

Per scale on the I61 case: the map's per-scale predicted block gain correlates 0.92 with the observed at scale 0, then 0.64, 0.50, 0.30 at
scales 1–3, and under-predicts the coarse totals (|P| 47 vs |O| 82 at scale 2, 25 vs 63 at scale 3). The largest single error is
HF_MAG_LOSS at scales 2–3 (predicted ≈ 1/7 of observed); the v2 soft-peak slots (9–11) have no density at all; BANDING at scale 3 has the
wrong sign. Mechanism: an 8×8 block is 4×4, 2×2, 1×1 coarse pixels at scales 1–3, inside 11×11 coarse blur windows that are mostly
unrepaired neighbours, so a block's effect at coarse scales is a neighbourhood effect the frozen signals cannot represent.
v2 + basic keeps a residual at scale 0 (it reads X/B at scale 0; by_v2fy reads Y only there).

## 3. What does not fix it

**Partial repair** (`ZENSIM_REPAIR_ALPHA`, dist + α(ref − dist), 96 cases, block 8): M3f gets WORSE as α falls (by_v2fy level 03:
0.81 at α = 1, 0.71 at 0.5, 0.57 at 0.25; level 05: 0.65, 0.65, 0.54), and 30–50 % of blended blocks still lose score (half-blended
blocks carry ghost edges). Small-step linearity is not the missing piece, and blending is not an encoder-like intervention.

## 4. The realistic JPEG intervention

`gen_jpeg_distortion --subsampling 444 --decoded-out` encodes 8 KADID references (I01, I11, …, I71) with zenjpeg at q 10/30/60/85;
with 4:4:4 every decoded 8×8 block depends only on its own coefficients, so `ZENSIM_REPAIR_SOURCE=<q_hi decode>` on the q_lo decode
is an exact "spend more bits on this block" intervention (data `neighsteer_swap_2026-10-04.tsv`, 3 seeds × 8 images per row):

| set | swap | block | pass | M3f median | M2 median | share of |ΔS| that is loss | Σ single-block ΔS (median) |
|---|---|---:|---:|---:|---:|---:|---:|
| by_v2fy | q10→30 | 8 | 4/24 | 0.66 | 1.000 | 41 % | +8.85 |
| by_v2fy | q10→30 | 16 | 13/24 | 0.71 | | 31 % | +11.87 |
| by_v2fy | q10→30 | 32 | 20/24 | 0.83 | | 30 % | +7.86 |
| by_v2fy | q30→60 | 8 | 1/24 | 0.64 | 1.000 | 50 % | +0.09 |
| by_v2fy | q30→60 | 16 | 8/24 | 0.68 | | 48 % | +0.57 |
| by_v2fy | q30→60 | 32 | 14/24 | 0.76 | | 58 % | −1.42 |
| by_v2fy | q60→85 | 8 | 15/24 | 0.70 | 1.000 | 50 % | +0.10 |
| v2 + basic | q10→30 | 8 | 9/24 | 0.69 | 1.000 | 41 % | +9.69 |
| v2 + basic | q30→60 | 8 | 7/24 | 0.66 | 1.000 | 49 % | +0.60 |
| v2 + basic | q60→85 | 8 | 17/24 | 0.71 | 1.000 | 51 % | −0.35 |

The map ranks larger blocks better (q10→30: 4 → 13 → 20 of 24 passing at 8/16/32). Half of the |ΔS| mass of single-block (and
even 32×32) upgrades at q30→60 and q60→85 is LOSS: by the model, upgrading an isolated block lowers the score about as often as it
raises it. Whether that is a seam/inhomogeneity penalty or non-monotone model behaviour per block is not settled here; an
independent judge (ssimulacra2 / butteraugli on the same composites) is the next check.

**Per-feature oracle on six swap cases** (by_v2fy, `ZENSIM_V2_DIAG` with `ZENSIM_REPAIR_SOURCE`): exact v2 + v1 terms at scales 1–3
lift the swap ranking from 0.59–0.73 to 0.984–0.998, with scale 0 still served by the full-repair density. The coarse-scale engine is
therefore enough for realistic JPEG interventions too, PROVIDED it is given the candidate pixels (an API decision, §5).

## 6. Allocation: does steering by the map pay off?

`benchmarks/neighsteer_2026-10-04/alloc_compose.py` upgrades 25 % of the 8×8 blocks of each q_lo decode to q_hi, chosen by a ranking,
and the whole-image score change is measured (`--block 1024`, `neighsteer_alloc_2026-10-04.tsv`; by_v2fy s5101, 8 images, medians):

| swap | upgrade all blocks | 25 % by true single-block ΔS | 25 % by today's map | 25 % random | 25 % by lowest map gain |
|---|---:|---:|---:|---:|---:|
| q10→30 | +18.41 | +16.36 | **+13.97** (81 % of all) | +4.55 | −4.99 |
| q30→60 | +2.67 | +8.60 | **+5.80** (228 % of all) | +0.05 | −5.35 |
| q60→85 | +3.55 | +6.96 | **+5.05** (131 % of all) | +0.27 | −4.31 |

* Today's map already steers usefully: its top quarter gets 3–100× the random quarter's gain, and at q30–85 it beats upgrading
  everything because it avoids the blocks whose upgrade the model scores as a loss.
* The gains are close to additive over the chosen blocks (the oracle quarter ≈ the sum of the positive single-block gains); the full
  upgrade looks small only because harmful single-block upgrades cancel helpful ones.
* **The prize for the neighbour-aware engine:** a perfect single-block ranking earns 17–48 % more score than today's map from the
  same 25 % budget (+16.4 vs +14.0, +8.6 vs +5.8, +7.0 vs +5.1). Sign accuracy matters as much as rank: the controller must be told
  which upgrades hurt.
* Bits are not equalised here (blocks differ in cost); an RD-correct test needs per-block byte costs from the encoder.

## 5. Design

Two parts, both neighbour-aware:

1. **Prediction — exact local recompute at the coarse scales.** For a query rectangle R and candidate distorted pixels inside it,
   recompute exactly, at scales 1–3, every v2 per-pixel term whose window touches a changed coarse pixel, from the planes the prepared
   steering session already retains (`FoldRetention`, `feature_v2.rs:11361`: per scale and channel `pyr_src`, `pyr_dst`, `mu1`, `mu2`,
   `ssq`, `s12`, `act`, `bs2`, plus the exact f64 pooled cells). The pyramid is a pinned 2×2 box (`blur::downscale_2x_into`), so
   changed coarse pixels are computable exactly; every v2 pool (Σ/n, central moments, soft-peak ΣW·v/ΣW, masked/IW with fixed weights,
   edge width from mean gradients) updates exactly from base accumulators plus local deltas. Old and new local terms are computed by the
   same routine so strip-geometry rounding cancels. Scale 0 keeps the existing density (correlation 0.92 there). The oracle (§2) bounds
   this at ≈ 0.98 (v2 only) to ≈ 0.998 (with v1 basic at scales 1–3) for by_v2fy. The same engine answers ARBITRARY candidate pixels
   (an encoder's re-quantized block), which the reference-repair map cannot.
   Footprint per 8×8 query: at scale s the changed region is ⌈8/2^s⌉² coarse pixels, the recomputed region that plus a 5-pixel blur
   halo (plus ±1 for gradients): ≈ 196 + 144 + 121 coarse pixels at scales 1–3. Cost to be measured, not estimated.
2. **Policy — group queries.** Because isolated upgrades can lower the score at low quality, a controller should be able to ask for the
   gain of upgrading a group of blocks together (non-additive). The local engine handles a union of rectangles directly; the 16×16 /
   32×32 swap results (§4) size how much grouping helps.

Phase 1 (lane NEIGHSTEER, `~/tmp/zensim-paper/rev4/NEIGHSTEER_brief.md`): the v2 coarse engine as a crate-private module with a
golden test against full recomputation, an env-gated integration into prepared steering's `refinement_gain` (default off and
byte-identical), and the measured M3f / cost on this file's panels. No public API change without the user's decision; the candidate-pixel
and group-query surfaces are proposed separately once Phase 1 is measured.
