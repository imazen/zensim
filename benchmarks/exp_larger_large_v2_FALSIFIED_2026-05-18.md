# EXP-LARGER-LARGE-V2: FALSIFIED — scale-alone doesn't beat current ship

**Date:** 2026-05-18
**Verdict:** **FALSIFIED on both compression and balanced trails per § A.10**
**Bake:** `/mnt/v/output/zensim/exp_larger_large_v2/bakes/larger_large_s*.bin` (5 seeds, h=128, V_24-per-sample-α s4 recipe)
**Training corpus:** 178,400 rows × 300 features (75,300 existing + 108,400 new) — 2.4× expansion from baseline 73k
**Spend:** ~$5.63 vast.ai (started at $8.69 → $0, account hit credit zero)

## Hypothesis (paper-quoted form)

"Expanding the cvvdp_iwssim LARGE corpus from 73k → 178k pairs (2.4× via
new v15r_zenjpeg sweep) should improve V_24-per-sample-α s4's CID22 SROCC
by ≥ 0.005, or be FALSIFIED."

## Result — falsified on both trails

### 5-seed CI table

| Seed | CID22 | KADID | TID | KonJND | AIC-3 |
|---|---:|---:|---:|---:|---:|
| s1 | 0.8454 | 0.9314 | 0.8929 | 0.7858 | 0.8051 |
| s2 | **0.8546** | 0.9312 | 0.8899 | **0.8002** | 0.8102 |
| s3 (median CID22) | 0.8488 | 0.9264 | 0.8839 | 0.7930 | 0.8107 |
| s4 | 0.8534 | 0.9311 | 0.8912 | 0.7801 | **0.8157** |
| s5 | 0.8451 | 0.9313 | 0.8898 | 0.7881 | 0.8082 |
| **MEAN** | **0.8495** | 0.9303 | 0.8895 | 0.7894 | 0.8100 |
| **STD** | 0.0046 | 0.0021 | 0.0036 | 0.0078 | 0.0040 |

### Control panels (from `benchmarks/baseline_panels_2026-05-18.md`)

| Bake | CID22 | KADID | TID | KonJND | AIC-3 |
|---|---:|---:|---:|---:|---:|
| **V_24-PSA s4 (Compression ship)** | **0.8641** | 0.9316 | 0.8893 | 0.8080 | **0.8183** |
| V_22-mix-LARGE+iwssim (Balanced ship) | 0.8324 | **0.9677** | **0.9729** | **0.8927** | 0.7845 |
| ssim2 control | 0.8895 | 0.8856 | 0.8651 | 0.8112 | 0.8240 |
| cvvdp control | 0.9128 | 0.9023 | 0.9035 | 0.8489 | 0.8523 |
| iwssim control | 0.9116 | 0.9456 | 0.9544 | 0.8748 | 0.7843 |

### bake_compare decisive verdicts (§ A.9)

#### s3 (median CID22) vs Balanced ship

| Corpus | A SROCC | B SROCC | Verdict |
|---|---:|---:|---|
| CID22 | 0.8488 | 0.8324 | **A>>B** (+0.0164) |
| KADID | 0.9264 | 0.9677 | B>>A (-0.0413) |
| TID | 0.8839 | 0.9729 | B>>A (-0.0890) |
| KonJND | 0.7930 | 0.8927 | B>>A (-0.0997) |
| AIC-3 | 0.8107 | 0.7845 | **A>>B** (+0.0262) |

3 decisive A wins, 18 decisive B wins across bands.

#### s3 (median CID22) vs Compression ship

| Corpus | A SROCC | B SROCC | Verdict |
|---|---:|---:|---|
| CID22 | 0.8488 | 0.8641 | **B>>A** (-0.0153) |
| KADID | 0.9264 | 0.9316 | B>>A (-0.0052) |
| TID | 0.8839 | 0.8893 | B>>A (-0.0054) |
| KonJND | 0.7930 | 0.8080 | B>>A (-0.0150) |
| AIC-3 | 0.8107 | 0.8183 | B>>A (-0.0076) |

0 A wins, 8 B wins, 4 promising — **decisive loss on every single corpus**.

#### s2 (best CID22) vs Compression ship

| Corpus | A SROCC | B SROCC | Verdict |
|---|---:|---:|---|
| CID22 | 0.8546 | 0.8641 | **B>>A** (-0.0095) |
| KADID | 0.9312 | 0.9316 | promising (-0.0004) |
| TID | 0.8899 | 0.8893 | tied (+0.0006) |
| KonJND | 0.8002 | 0.8080 | promising (-0.0078) |
| AIC-3 | 0.8102 | 0.8183 | **B>>A** (-0.0081) |

0 A wins, 3 B wins, 4 promising — **falsified on the two compression corpora**.

### Trail gate verdicts per § A.10

- **Compression trail.** Requires A>>B decisively on ≥ 1 of {CID22, AIC-3}.
  *Result:* A>>B on neither. **FAILS.**
- **Balanced trail.** Requires A>>B decisively on CID22.
  *Result for s3:* A>>B on CID22 BUT loses 3 of 4 balanced corpora.
  Even with this CID22 lift, s3 is not a Balanced-trail winner (it's a
  Compression-trail-shape result that doesn't beat the Compression ship).
  **FAILS.**

## Why the scale-alone hypothesis falsified

The new 108k rows are **single-codec** (v15r_zenjpeg only). The existing
75k spans 5 codecs (zenjpeg + zenjxl + zenwebp + zenavif + zenpng). After
the join the new LARGE is **codec-imbalanced**: of the 178k rows, 108k
(60%) are zenjpeg vs the prior 36k (48%) which was a more balanced cut.

Adding more single-codec data **does not improve generalization to the
multi-codec CID22 holdout**. It may even hurt by tilting the trainer
toward zenjpeg-specific feature responses.

Confirmation: the trainer's reported `cvvdp_iwssim_large` val_srocc was
0.9860 at convergence (very high, consistent with the in-distribution
single-codec data). The held-out CID22 still only reached 0.8488 — the
gap between in-distribution training fit and the multi-codec holdout is
unchanged from baseline.

## What would un-falsify the hypothesis

1. **Diverse-codec scaling.** A 250k expansion that runs through zenjxl,
   zenavif, zenwebp, zenpng AND zenjpeg in proportion to the existing
   75k's codec mix. Would test whether scale gives CID22 lift when the
   expansion preserves codec diversity. Estimated cost: 4× the 13-box
   fleet we used here, ~$25 in vast.ai compute.
2. **Different anchoring target.** Mix_cv40_iw60 may be saturating on the
   in-distribution train data. Try mix_cv50_iw50 or pure cvvdp as the
   training target to see if the model can extract more CID22 signal from
   the larger zenjpeg block.
3. **Selective filtering by knob diversity.** The v15r_zenjpeg corpus
   sweeps ~12 knob axes; the 75k existing corpus may have a different
   knob coverage. Joining only on the knob-tuples that overlap might give
   cleaner signal at the cost of fewer rows.

None of these are within budget for this session — vast.ai credit is
exhausted ($0.00 + billing not configured).

## Infrastructure findings

The 4 documented fixes from the prior agent's diagnostic were all
applied. Smoke testing revealed:

1. **Fix A (base64 bootstrap) — VERIFIED WORKING.** The base64-encoded
   payload survives vast.ai API arg-mangling cleanly.
2. **Fix B (export -f process_chunk in v14 onstart) — VERIFIED.** The
   patched v14 onstart no longer no-ops in 6s.
3. **Fix D (cuda_max_good>=12.6 driver_version<570.0.0 filter) —
   PARTIALLY WORKING.** vast.ai's `cuda_max_good` field is a vague
   capability indicator, not a guarantee that libcuda.so exports the
   specific CUDA 12.6 driver symbols. The v14 image's binary still
   panicked on first kernel launch with `cuCoredumpDeregisterComplete-
   Callback` on driver 560.35.03 boxes despite the filter.
4. **Fix C (libnvrtc apt-install on v14 PATH) — FALSE ALARM.** The v14
   onstart's `/sbin/ldconfig` check + baked libnvrtc12 path works on the
   v14 image. The v3 onstart's runtime apt-install of libnvrtc12 also
   works fine on the v3 image (cuda-keyring + dpkg path completes in ~13s).

### v14 image is structurally unusable on cheap-tier vast.ai offers

The diagnostic's "Defect 4 — multi-hour rebuild" is the actual fix path.
The v14 image's cudarc 0.19.4 dependency assumes a CUDA 12.6 driver API
that cheap-tier libcuda.so doesn't reliably export. Until the v14 image's
zen-metrics binary is rebuilt against an older cudarc (or with
`cudarc/dynamic-loading`), the v14 image is broken on driver 560.x boxes.

**Working alternative.** The v3 image (`0.6.4-iwssim-fixed-6227c1a`)
binary was built before cudarc 0.19.4 was adopted and DOES work on cheap-
tier boxes. With Fix A's base64 bootstrap, it runs the patched v3 onstart
cleanly. This is the path used for this experiment's actual sweep.

## Sweep performance summary

- 13-box fleet (1 smoke + 12 fleet) launched on v3 image + patched v3 onstart
- 1,084 of 2,500 chunks completed (43% target) before vast.ai sign-up credit
  hit zero and auto-destroyed all instances
- Pace: ~50 chunks/min steady-state at full fleet capacity
- 0 chunk failures observed in the fresh sidecars
- 7 stale-claim fake sidecars (0.0-value) from prior agent's broken runs
  identified and deleted before training

## Files

- `scripts/exp_larger_large/build_larger_large.py` — corpus builder (NaN-filter fix added 2026-05-18)
- `scripts/exp_larger_large/run_exp_larger_large_seed.sh` — single-seed trainer
- `scripts/exp_larger_large/run_5_seeds.sh` — 5-seed parallel runner
- `benchmarks/exp_larger_large_v2_methodology_2026-05-18.md` — methodology doc
- `benchmarks/exp_larger_large_v2_FALSIFIED_2026-05-18.md` — this doc
- `/mnt/v/zen/zensim-training/2026-05-18-larger-large/cvvdp_iwssim_LARGE_v3_300col.parquet` — 178.4k LARGE
- `/mnt/v/output/zensim/exp_larger_large_v2/bakes/larger_large_s*.{bin,log,verdict.md}` — 5 seeds + verdicts
- `/mnt/v/output/zensim/exp_larger_large_v2/bakes/compare_*.md` — bake_compare reports

## Memory file written to

`/home/lilith/.claude/projects/-home-lilith-work-zen/memory/project_exp_larger_large_v2.md`
