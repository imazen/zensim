# Rev4 E2b — CVVDP accuracy by display model on TRAIN human legs

Lane: `cvvdp-safesyn`. Preregistration:
`benchmarks/cvvdp-safesyn_prereg_2026-09-23.md` (registered verbatim from the
E2a proposal `benchmarks/rev4_e2a_cvvdp_display_2026-09-23.md` Deliverable 4).
Machine-readable stats: `benchmarks/rev4_e2b_cvvdp_display_2026-09-23.json`.

**Question:** does any candidate display model make CVVDP a better match to
human judgement than `standard_4k` (75.4 px/deg — the geometry every CVVDP
teacher run has used)?

## Method (as preregistered)

- Binary: zenmetrics `19d6dd8ee378b64562c551358c1d935ca7ed58ee`
  (`quarantine/devin/cvvdp-safesyn` on `master@origin` `b02812ae`), glibc
  release sha256 `f8c352ee83e7424c5c709f9d1d3bf49795443dc809e3ceb31db3cba9f40425fe`.
  Contains the upstream V5 SIMD/rayon CVVDP (conformance-v2 vs pycvvdp 0.5.7)
  plus the lane's `cvvdp@<display>` jobexec strings; `batch --display-model`
  is upstream (cvvdpfix trio).
- 21 scoring runs: `zenmetrics batch --metric cvvdp [--display-model <d>]
  --group-by-ref --jobs 8`, one per (leg × display). Output column carries
  the display suffix; `standard_4k` keeps the plain
  `cvvdp_cpu_imazen_v0_1_0` column.
- Legs (TRAIN role only, per DATA_SPLITS): KADID-train 5,000 pairs / 40 refs;
  TID-train 1,440 pairs / 12 refs; KonFiG-train 327 pairs / 6 refs
  (10,782 within-reference stimulus triplets).
- Statistics via `zensim-validate::panel` only (no reimplemented stats):
  pooled SROCC per arm; paired Δ = arm − `standard_4k` on a
  reference-clustered bootstrap (B = 2000, `random.Random(20260923)` over
  reference names, same draws for every arm); KonFiG triplet ordering
  accuracy via `panel --pairwise --resample`. CI: two-sided percentile at
  1 − 0.05/6 = 0.991667.
- Selection rule (verbatim): a display beats `standard_4k` only if pooled
  Δ ≥ +0.010 SROCC **and** the 99.1667% CI of Δ excludes 0 on **both**
  KADID-train and TID-train, **and** KonFiG-train accuracy Δ is not
  significantly negative; among passers take the largest mean Δ over the two
  SROCC legs. If none passes, the `standard_4k` teacher stands.

## Results — KADID-train (5,000 pairs, 40 refs)

| display | pooled SROCC | per-ref SROCC | Δ vs 4k | 99.167% CI |
|---|---|---|---|---|
| standard_4k | 0.8390 | 0.8476 | — | — |
| sdr_4k_30 | 0.8543 | 0.8633 | +0.0153 | [+0.0124, +0.0182] |
| standard_fhd | 0.8625 | 0.8719 | +0.0235 | [+0.0120, +0.0344] |
| **sdr_fhd_24** | **0.8701** | **0.8792** | **+0.0311** | **[+0.0216, +0.0403]** |
| standard_phone | 0.7797 | 0.7886 | −0.0593 | [−0.0669, −0.0515] |
| iphone_14_pro | 0.7335 | 0.7427 | −0.1055 | [−0.1177, −0.0932] |
| modern_oled_phone_indoor | 0.7926 | 0.8016 | −0.0464 | [−0.0523, −0.0401] |

## Results — TID-train (1,440 pairs, 12 refs)

| display | pooled SROCC | per-ref SROCC | Δ vs 4k | 99.167% CI |
|---|---|---|---|---|
| standard_4k | 0.8637 | 0.8728 | — | — |
| sdr_4k_30 | 0.8738 | 0.8823 | +0.0101 | [+0.0048, +0.0144] |
| standard_fhd | 0.8775 | 0.8898 | +0.0138 | [−0.0067, +0.0339] |
| **sdr_fhd_24** | **0.8821** | **0.8928** | **+0.0185** | **[+0.0013, +0.0339]** |
| standard_phone | 0.8227 | 0.8334 | −0.0409 | [−0.0503, −0.0296] |
| iphone_14_pro | 0.7947 | 0.8055 | −0.0690 | [−0.0826, −0.0532] |
| modern_oled_phone_indoor | 0.8309 | 0.8411 | −0.0328 | [−0.0406, −0.0231] |

## Results — KonFiG-train (triplet ordering accuracy, 10,782 triplets)

| display | accuracy | Δ vs 4k | 99.167% CI |
|---|---|---|---|
| standard_4k | 0.8690 | — | — |
| sdr_4k_30 | 0.8648 | −0.0043 | [−0.0095, 0.0] |
| standard_fhd | 0.8390 | −0.0301 | [−0.0453, 0.0] |
| sdr_fhd_24 | 0.8452 | −0.0238 | [−0.0403, 0.0] |
| standard_phone | 0.8585 | −0.0106 | [−0.0248, +0.0109] |
| iphone_14_pro | 0.8406 | −0.0285 | [−0.0481, 0.0] |
| modern_oled_phone_indoor | 0.8641 | −0.0049 | [−0.0159, +0.0117] |

## Decision

| challenger | KADID Δ | TID Δ | KonFiG Δ | passes |
|---|---|---|---|---|
| sdr_4k_30 | +0.0153 ✓ | +0.0101 ✓ | −0.0043 (CI incl. 0) | **YES** |
| standard_fhd | +0.0235 ✓ | +0.0138 (CI incl. 0) | −0.0301 | no |
| **sdr_fhd_24** | **+0.0311 ✓** | **+0.0185 ✓** | −0.0238 (CI incl. 0) | **YES** |
| standard_phone | −0.0593 | −0.0409 | −0.0106 | no |
| iphone_14_pro | −0.1055 | −0.0690 | −0.0285 | no |
| modern_oled_phone_indoor | −0.0464 | −0.0328 | −0.0049 | no |

**Selected display: `sdr_fhd_24`** (37.84 px/deg geometry, 100-nit SDR peak)
— largest mean pooled Δ = **+0.0248 SROCC** over the two SROCC legs, passes
both legs' CI gates, KonFiG CI not entirely below zero. `sdr_4k_30` also
passes (mean Δ +0.0127) but is dominated.

Interpretation, per the lane's motivating question: CVVDP's historically
poor teacher showing is **substantially a display-geometry artifact** — the
same scorer at ~37.8 px/deg instead of 75.4 px/deg recovers +0.018…+0.031
SROCC on both human legs, with the dimmer 100-nit arm beating the 200-nit
arm at fixed geometry. Phone displays (much denser geometry) hurt.

`modern_oled_phone_indoor` parity status: CHECKED — max |Δ| = 0.0001 JOD vs
pycvvdp 0.5.7 on an 8-pair KADID probe (preset injected via `config_paths`).

## Effect on Part 2

Per the brief ("`standard_fhd` and `standard_4k`, plus the Part-1-selected
display if it differs"), the SafeSyn sidecar scores four metric arms:
`cvvdp` (=`standard_4k`), `cvvdp@standard_fhd`, `cvvdp@sdr_fhd_24`, `ssim2`.
Manifest: `safesyn_manifest4.json` (3,218 jobs) — declared only after
`FLEET_GO.md`.

## Provenance

- Scores: `/var/tmp/cvvdp-safesyn/e2b/scores/{leg}__{display}.tsv` (21 files,
  row counts verified == input pair counts).
- Stats driver: `benchmarks/cvvdp_safesyn/e2b_stats.py` →
  `scripts/lib/zen_stats.py::panel_batch_indexed` + `panel --pairwise`.
  Panel binary: `/var/tmp/rev4-e4/bin/panel`.
- Scoring driver: `benchmarks/cvvdp_safesyn/run_e2b_scores.sh` (944 s under
  the heavy lock, ~20 pairs/s @ 8 jobs).
- Exclusions honored: no CID22-B, no AIC label reads (Part 0 used stored
  score TSVs only), no holdout access, no fitting.
