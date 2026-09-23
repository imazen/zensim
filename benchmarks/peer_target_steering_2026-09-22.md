Devin-executed; Opus-reviewed 2026-09-23 (REVIEW_PAPER_MEASURE.md): PROMOTE WITH CORRECTIONS

Reviewer independently confirmed JXL D vs GMSD-dial Δ +0.1596 (D worse; 1 vs 4 families, n=6), WebP D vs SSIMULACRA2 Δ +0.2549 (D worse; n=11), and WebP PreviewV0_2 vs SSIMULACRA2 Δ −0.1807 (PreviewV0_2 better; n=9).

# Peer metrics as the loop target — September 22, 2026

Question (zensim paper, fairness): how well does the SAME scalar controller hit a
requested score when it steers on a peer metric instead of zensim? Registered in
[the steering protocol](../docs/TARGET_STEERING_PROTOCOL_2026-09-08.md)
("Peer metrics as the loop target — registered 2026-09-22") before any peer
number existed. Lane commits: `af48035f` (analyzer), `9871286c` (peer-scalar
adapter, `--peers`, protocol section, this record tooling).

## What was run

- Instrument: `zensim-target` `demo_matrix` bounds mode (owner of the Sept 8/14/15
  targeting records). `peer-scalar`/`peer-gmsd` are non-default features;
  `target_search_with_backend_and_scalar` is `#[doc(hidden)]`, calls the existing
  private `search` unchanged. A unit test requires probe-for-probe identity with
  the bake entry point and refusal before any encode on an invalid request.
- Same cases as `rev3_targeting_2026-09-14.md` (admitted manifests): seeds fitted on
  9 TRAIN families only; evaluation on the 8 validation families (8 images, one
  256-px rendition each); JXL, JPEG, WebP; 21-step native-knob ladders; fixed
  requests −10/30/70/90/99 plus five witnessed ladder quantiles per ladder;
  tolerance 1; budgets 1/2/3; midpoint and TRAIN-curve policies; every emitted
  bitstream re-encoded and re-scored; independent SSIMULACRA2 (fast-ssim2) and
  Butteraugli pnorm3 judges on every output.
- One process per stage scores **matched B, matched D, PreviewV0_2, C, SSIMULACRA2
  and GMSD** on identical reconstructions.
- Scales: the zensim dial is calibrated to the SSIMULACRA2 scale, so SSIMULACRA2
  requests and its ±1 band are literally the same numbers. GMSD's loop scalar is
  `100 − GMSD/g` (identity → 100) with `g` = median over all TRAIN ladder
  adjacent-probe pairs with |Δssim2| ≥ 1 of |ΔGMSD|/|Δssim2| — measured
  **g = 9.995e-4** (436 pairs; per-codec medians jpeg 1.039e-3 / jxl 9.456e-4 /
  webp 1.008e-3; quartiles 6.17e-4 / 9.99e-4 / 1.52e-3). Pre-registered, 4
  significant digits, before any GMSD value was seen; the provisional rate used
  by the stage-0 calibration was discarded. The ±1 band equals one SSIMULACRA2
  point only at the TRAIN median rate (disclosed).
- Reproduction gates all pass: 1,134 fit + 1,008 eval probes identical to the
  Sept-14 runs (max |Δ| = 0.0); calibration curves identical; 1,860/1,860 eval
  steering rows identical except timing; 504/504 probes identical to the
  Sept-15 R915 bounds.
- Environment: binary sha256 `11138d4c`, ZENSIM_FORMULA_REV=1, RAYON 4 threads;
  machine state per stage in the run's `MACHINE_STATE.jsonl`. Numbers are
  development validation, not qualification.

## Results — hits ±1 / errors at 3-shot train_curve (the operative policy)

Witnessed attainable targets only; fixed requests without a witness counted
separately (coverage below). n = requests.

| codec | model | n | median err | p95 err | worst | hits ±1 | hits ±2 |
|---|---|---:|---:|---:|---:|---:|---:|
| jpeg | B | 50 | 0.539 | 3.053 | 5.106 | 40/50 (80%) | 46 |
| jpeg | D | 55 | 0.417 | 2.951 | 7.224 | 42/55 (76%) | 48 |
| jpeg | PreviewV0_2 | 52 | 0.419 | 3.037 | 5.817 | 45/52 (87%) | 47 |
| jpeg | C | 52 | 0.635 | 5.474 | 11.087 | 38/52 (73%) | 43 |
| jpeg | **SSIMULACRA2** | 48 | 0.475 | 2.484 | 4.093 | 43/48 (90%) | 44 |
| jpeg | **GMSD-dial** | 59 | 0.242 | 3.154 | 4.602 | 49/59 (83%) | 52 |
| jxl | B | 50 | 0.355 | 3.024 | 8.938 | 45/50 (90%) | 47 |
| jxl | D | 52 | 0.254 | 1.653 | 8.014 | 49/52 (94%) | 51 |
| jxl | PreviewV0_2 | 59 | 0.356 | 2.842 | 9.070 | 55/59 (93%) | 56 |
| jxl | C | 60 | 0.436 | 1.431 | 4.105 | 52/60 (87%) | 57 |
| jxl | **SSIMULACRA2** | 55 | 0.466 | 0.989 | 8.931 | 53/55 (96%) | 54 |
| jxl | **GMSD-dial** | 53 | 0.153 | 3.971 | 10.626 | 47/53 (89%) | 49 |
| webp | B | 47 | 0.484 | 3.229 | 11.861 | 35/47 (74%) | 38 |
| webp | D | 56 | 0.375 | 1.690 | 10.125 | 50/56 (89%) | 55 |
| webp | PreviewV0_2 | 52 | 0.159 | 1.348 | 8.130 | 48/52 (92%) | 51 |
| webp | C | 53 | 0.496 | 5.831 | 7.648 | 38/53 (72%) | 46 |
| webp | **SSIMULACRA2** | 54 | 0.317 | 6.647 | 16.989 | 45/54 (83%) | 49 |
| webp | **GMSD-dial** | 59 | 0.518 | 5.442 | 40.202 | 46/59 (78%) | 51 |

Reference rows at the same 3-shot train_curve (earlier runs, same instrument):
R915 basic228 ens5 jpeg 39/50 · jxl 43/52 · webp 37/52 hits ±1; R915 y60 ens5
jpeg 39/51 · jxl 47/54 · webp 41/55; MT913 linear60 jpeg 37/47 · jxl 52/53 ·
webp 41/49. Full per-budget tables (shots 1/2/3, midpoint control, MT913 and
R915 ensembles) in `peer_target_steering_2026-09-22.json` (`rows`).

Registered production bar (median ≤0.5, p95 ≤1, worst ≤3, undershoots ≤1% at
3 shots): **no model meets it on any codec** — zensim or peer (all p95 >1 or
worst >3). At the lighter 1-shot bar (median ≤2, p95 ≤8) only D-on-JXL passes.
These are development-validation numbers, not qualification.

**The midpoint policy collapses for every model on JXL** (median |err| 12–66
pts at all budgets, worst ~25–100): it is a controller-quality control, not a
metric property — train_curve is the operative policy.

## What the independent judges say about the achieved encodes

Median |SSIMULACRA2 judge − request| at 3-shot train_curve (SSIMULACRA2's own
dial is the target, so its column is its hit error by construction):

- **SSIMULACRA2 target:** jpeg 0.475, jxl 0.466, webp 0.317 — the judge IS the
  target metric here, so this equals its own error. The Butteraugli-p3 judge on
  those encodes: jpeg 1.659, jxl 0.432, webp 1.427.
- **GMSD-dial target:** median |ssim2 judge − request| jpeg 15.4, jxl 5.4,
  webp 12.7 — hitting a GMSD-equivalent target lands far from the same number
  on the ssim2 scale. The exchange rate is a TRAIN population median, not a
  per-image equivalence; GMSD correlates only loosely with ssim2 per image.
- **zensim targets:** B jpeg 8.5/jxl 3.1/webp 7.9; D jpeg 2.6/jxl 1.5/webp 2.7;
  PreviewV0_2 jpeg 4.2/jxl 1.8/webp 3.2 — consistent with the dial being
  calibrated to the ssim2 scale only on the median.

## Cross-model pairing on identical fixed requests (3-shot train_curve)

Same image/codec/request witnessed by both models; family means, descriptive
(`mean_family_delta` = mean over families of |model err| − |versus err|, so a
positive value means the first model is WORSE; `families_model_better` counts
families where the delta is negative). Small n (3–12 common requests per
pair). Notable cells: on JXL, D vs GMSD-dial +0.160 — D's mean |err| is 0.16
points higher, GMSD-dial better in 4/6 families; on WebP, D vs SSIMULACRA2
+0.255 — D worse on the mean, SSIMULACRA2 better in 5/11 families; on WebP,
PreviewV0_2 vs GMSD-dial −0.046 and vs SSIMULACRA2 −0.181 (PreviewV0_2 better
on both means). Full matrix in the JSON `cross_model_fixed_requests`. Taken
as a set: in the zensim-vs-SSIMULACRA2 cells the delta is positive — the peer
lands closer to the request — in **11 of 12** codec × model comparisons (the
only exception is PreviewV0_2 on WebP, −0.181). Nothing here is a controlled
ranking; it shows the controller steers peers to the same order of precision
it steers zensim models, and where the dial's own scale is the peer's native
scale (SSIMULACRA2), the peer tends to hit it slightly closer.

## Fixed-request coverage (why n differs per model)

A request enters a model's error table only when a ladder reconstruction scored
within ±1 of it on that model (witnessed). Fixed-request coverage per codec:
jpeg B 10/D 15/C 12/V0_2 12/SSIM2 8/GMSD 19 of ~40; jxl B 10/D 12/C 20/V0_2
19/SSIM2 15/GMSD 13; webp B 7/D 16/C 13/V0_2 12/SSIM2 14/GMSD 19 — the rest
outside the measured envelope or unwitnessed inside it
(`fixed_request_coverage` in the JSON). GMSD has the most witnessed fixed requests on JPEG and WebP (19 each), but not JXL: C 20, PreviewV0_2 19, SSIMULACRA2 15, GMSD 13.

## Position / kind splits (3-shot train_curve)

- Interior vs endpoints: lower endpoints dominate p95 and worst error, not the medians. JXL D lower/upper medians are 0.033/0.137; JXL SSIMULACRA2 0.000/0.217. WebP lower medians are 0.000 for B, D, PreviewV0_2, GMSD and SSIMULACRA2 (each with a higher upper median); WebP C lower is 2.370.
- Fixed requests vs ladder quantiles: GMSD-dial has the most hits by count on JPEG (18/19) and WebP (17/19), but SSIMULACRA2 has the better JPEG rate (8/8) and PreviewV0_2 the better WebP rate (12/12). JXL GMSD-dial is 13/13 (100%, tied at that rate).
- Undershoots beyond tolerance (a real failure mode — encode lands below the
  request −1): 3-shot train_curve, fixed + quantile combined, jpeg
  B 3/D 5/C 6/V0_2 2/SSIM2 2/GMSD 2; jxl B 1/D 1/C 4/V0_2 1/SSIM2 0/GMSD 1;
  webp B 5/D 3/C 7/V0_2 2/SSIM2 2/GMSD 6.

## Fairness notes the paper must carry

- SSIMULACRA2's ±1 tolerance is literally the same band as zensim's (the dial is
  ssim2-calibrated); GMSD's ±1 is one ssim2 point only at the TRAIN median
  exchange rate — the record discloses this wherever the GMSD-dial appears.
- Each model is judged only on targets it witnessed; peers' witness sets differ
  from zensim's (coverage table above), so pooled comparisons across models are
  descriptive.
- GMSD is a gradient-domain distance not designed as a control scalar; the
  linear exchange rate is the minimal fair mapping, and the judge columns show
  what it costs in ssim2 terms.
- Peers scored in the same process on identical reconstructions; judges
  (SSIMULACRA2, Butteraugli p3) are independent of the loop for zensim targets —
  for the SSIMULACRA2 peer the judge is the target itself, so its judge column
  is trivially its error and must not be compared to other models' judge
  columns as if it were independent evidence.
- No threshold, recipe or model was changed; no peer was fitted. Timing columns
  (`median ms`) are **CONTENDED** (MACHINE_STATE.jsonl load1 8.2–19.6) and descriptive only.

## Inputs / binaries / reproduction

- Run: `/var/tmp/paper-target/runs/peer-20260923T055844Z/` (`COMMANDS.json`,
  `MACHINE_STATE.jsonl`, `COMPLETE.json`, `REPRODUCTION.json`, per-stage logs,
  `eval/{measurements,bounds}.jsonl`, `analysis_summary.{json,md}`).
- demo_matrix binary sha256 `11138d4c2c56…`; calibration `5117d3fc…`;
  formula revision 1; ZENSIM env cleared except `ZENSIM_FORMULA_REV=1`.
- Peer implementations: fast-ssim2 0.8.2 `compute_ssimulacra2` sRGB8 default;
  zenmetrics `gmsd` crate @`9ecfb9d4` `gmsd_rgb8` (libgmsd-parity, c=170).
- Reference rows: `reanalysis/{rev3,r915,rev1}` (the Sept-14/15 instruments
  re-analysed by the same analyzer; hashes in the JSON `runs`).
- Assembler: `benchmarks/peer_target_steering_2026-09-22.py` (copies numbers
  from the owner's `analysis_summary.json`; computes none itself).
- Driver scripts: `/var/tmp/paper-target/{build_and_run.sh,run_peer.py,reproduce_check.py}`;
  jxl-encoder ee80c785 archived to `/var/tmp/paper-target/src/` (sibling dirty). The build log also records zenjpeg `8f703a6e` dirty=2; the reproduction gate found unchanged JPEG output.
