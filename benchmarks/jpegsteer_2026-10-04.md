# JPEGSTEER — spatial steering of by_v2fy and v2 + basic on JPEG at 8×8 and 16×16 blocks (2026-10-04)

Measured 2026-10-03 23:45 – 2026-10-04 00:05 MT. Development diagnostic, not a qualification. Data:
`jpegsteer_2026-10-04.tsv` (96 rows); raw per-block JSONs on tower `output/zensim/jpegsteer-2026-10-04/`.

> **Correction (2026-10-04 03:00 MT).** The bakes below are the COSTSET strict-route human-only bakes, later found to be
> non-monotone under per-block JPEG upgrades (`neighsteer_2026-10-04.md`, correction banner and §7). Rerun with adopted-recipe
> bakes — featpot cells `sel:59f0bbc2f290@h32:H128:cv16:cf98` (by_v2fy) and `set:v2+basic@h32:H128:cv16:cf98`, fold without_kadid,
> seeds 0/1/2 (labelled 5101/5103/5107), densified + stamped Rev4 — on the same pairs (`jpegsteer_cf98_2026-10-04.tsv`):
>
> | set (cf98) | panel | pass | M3f median (min) | M2 min |
> |---|---|---:|---|---:|
> | by_v2fy | KADID JPEG level 03, block 8 | 12/12 | 0.91 (0.91) | 0.999 |
> | by_v2fy | KADID JPEG level 05, block 8 | **10/12** | 0.84 (0.55) | 1.000 |
> | v2 + basic | KADID JPEG level 03, block 8 | 12/12 | 0.93 (0.89) | 0.999 |
> | v2 + basic | KADID JPEG level 05, block 8 | **9/12** | 0.81 (0.64) | 1.000 |
> | by_v2fy | owner (level 03, block 32) | 12/12 | 0.97 (0.96) | 0.999 |
> | v2 + basic | owner (level 03, block 32) | 12/12 | 0.98 (0.94) | 0.999 |
>
> The heavy-JPEG failure reported below (3/12, 2/12) is mostly a property of the COSTSET bakes; with adopted-recipe models both
> sets steer 8×8 blocks at KADID's heaviest JPEG level in 9–10 of 12 cases.

## Setup

* Tool: `diffmap_block_coherence` (prepared steering, `ZENSIM_PREPARED_STEERING=1`), binary sha256 prefix `6a165c4ab20a291e`
  (the COSTSET3 build; KERNOPT/KERNOPT2 were verified to leave steering panels byte-identical).
* Bakes: the COSTSET strict-route human bakes (H128, seeds 5101/5103/5107), densified and stamped Rev4
  (`/var/tmp/steercheck/gate3/rev4dense`), `ZENSIM_FORMULA_REV=4`. These are uncurated-recipe bakes, not the featpot cf98 cells.
* Pairs: KADID-10k distortion type 10 (JPEG) on I01, I21, I41, I61 (the owner panel's references), levels 03 and 05
  (05 is KADID's heaviest JPEG level), blocks 8 and 16. 2 sets × 3 seeds × 4 images × 2 levels × 2 blocks = 96 cases.
* Gate as the broad/owner panels: M2 ≥ 0.99 and M3f ≥ 0.70. M2 = SROCC of the gradient-linearized prediction against the
  true score change from copying the reference into a block; M3f = SROCC of the map's finite-rectangle `refinement_gain`
  against that change; SSE bar = SROCC of per-block SSE against it. Neither establishes an encoder RD gain.

## Results

| set | block | JPEG level | pass | M3f median (min) | M2 min | SSE-bar median |
|---|---:|---:|---:|---|---:|---:|
| by_v2fy | 8 | 03 | 12/12 | 0.81 (0.79) | 1.000 | 0.07 |
| by_v2fy | 8 | 05 | 3/12 | 0.65 (0.57) | 0.981 | 0.29 |
| by_v2fy | 16 | 03 | 12/12 | 0.85 (0.80) | 1.000 | 0.12 |
| by_v2fy | 16 | 05 | 6/12 | 0.73 (0.57) | 0.978 | 0.34 |
| v2 + basic | 8 | 03 | 12/12 | 0.83 (0.80) | 0.995 | 0.03 |
| v2 + basic | 8 | 05 | 2/12 | 0.64 (0.36) | 0.998 | 0.18 |
| v2 + basic | 16 | 03 | 12/12 | 0.89 (0.83) | 0.991 | 0.12 |
| v2 + basic | 16 | 05 | 6/12 | 0.68 (0.54) | 0.996 | 0.24 |

Existing panels for context (same bakes, Rev4 dense): owner panel (these four KADID JPEG level-03 pairs, block 32) by_v2fy
12/12, M3f median 0.91 (min 0.87); v2 + basic 10/12 (both misses M2 0.986). Broad panel (24 JXL pairs, blocks 8/16/32/64):
by_v2fy 86/96, all 24 block-8 cases pass; v2 + basic 87/96.

## Reading

* At moderate JPEG both sets steer 8×8 and 16×16 blocks well (every case passes, M3f ≈ 0.8–0.9) and far beat the SSE default.
* At KADID's heaviest JPEG both fail most block-8 cases on M3f (median ≈ 0.65) while M2 stays ≈ 1.0: the features' gradient
  predicts the per-block gain, but the map's finite-repair (`refinement_gain`) prediction ranks blocks less well when the
  repaired block sits among heavily distorted neighbours. It is still well above the SSE bar (0.18–0.34). The weakness is shared
  by both sets, so it is a map-side limit, not a cost of by_v2fy.
* Not measured: cf98-trained bakes, other JPEG encoders or quality settings, and closed-loop encoder RD gains.
