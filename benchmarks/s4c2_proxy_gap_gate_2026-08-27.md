# S4+C2 proxy-gap gate — PASS (2026-08-27)

Frozen in the plan doc's "S4+C2 DESIGN RULING (2026-08-27)" BEFORE measurement;
executed the same night. Question: is the zensimA ladder shape a valid proxy
for C-bake ladder targets (seed distance at t, local slope), so the C2a/C2b
regressors may train on stored bigcodec `score_zensim` without a 944 tbig
backfill?

## Method
- Probe: the 39-image dial-grid jxl DISTANCE ladders (~46 knots, 0.0→25.0),
  **decoded pixels** at `/mnt/v/output/zensim/dial-grid-pixels-2026-07-27`
  (no parquet regime hazard; the `~/tmp/dial924` originals had 3 corrupted
  files — mirror driver `drv_jxl_mnt.tsv` used, 1,846/1,846 clean PNG).
- Features: `v2_ab_extract` on the SAME pixels, `ZENSIM_AB_MODE=v1` (372)
  and `=foldapp2` (944); 1,807 rows each, input order preserved.
- Forward: canonical `predict_features_with_bake`.
  A = `zensim/weights/v47_strict_qat_native_2026-05-27.bin` (d0ef7a3054d1ed9e…) —
  the Profile-A family the 07-01 bigcodec sweep scored with (assumption
  stated: current in-repo A bytes = that scorer's model; A frozen since
  2026-05-27).
  C = `zensim/weights/c_sdr_mlp944_corrmix_2026-08-05.bin` (1a2c8d522fed8034…).
- Seed = first downward crossing of t in increasing d, linear interp;
  Δ measured as fractional INDEX distance in the ladder's own knot grid
  (log-free: d=0 knot needs no dodge); slope sign from the bracket.
- Generator: `scripts/canonical_corpus/s4c2_proxy_gap_gate.py`; outputs
  `/mnt/v/zen/zensim-training/s4c2-2026-08-27/proxy_gap_gate.json` +
  `proxy_gap_per_image.tsv`.

## Result — every frozen bar passes

| t | n measurable | median \|Δ\| (bar ≤2.0) | p90 \|Δ\| | slope-sign agree (bar ≥0.90) |
|---|---|---|---|---|
| 70 | 37/39 (2 never reach 70 by d=25) | **0.359** | 1.643 | 1.00 |
| 80 | 39/39 | **0.598** | 2.054 | 1.00 |
| 88 | 39/39 | **0.719** | 2.746 | 1.00 |

**VERDICT: PASS — option (b) validated; the C2a/C2b fits train on zensimA
targets (the derived `jxl_ladders_9pt` table); NO tbig 944 backfill.**

## Honesty
- Sign agreement 1.00 is a WEAK discriminator here (both models' ladders are
  monotone-decreasing at every measured crossing) — the load-bearing number is
  the seed-position delta.
- Photo/screen split PENDING a registered class mapping for the 39 refs
  (hex-hash synthetic-corpus tiles; no class column anywhere). The per-image
  TSV makes the split a relabel, not a rerun. p90 across ALL images already
  sits at/under the bar, bounding how bad any one class can be.
- Provenance rule from the ruling stands: any shipped table states
  "trained on zensimA-proxy targets; proxy gap: median ≤0.72 grid-steps
  (this gate)".
