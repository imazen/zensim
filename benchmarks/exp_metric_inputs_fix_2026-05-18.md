# EXP-METRIC-INPUTS-FIX — FALSIFIED (2026-05-18)

**TL;DR**: With ALL four corpus-build defects from EXP-METRIC-INPUTS fixed, the
303 / 375-feature metric-inputs MLP still **fails both trail gates**. The
KonJND catastrophic regression PERSISTS unchanged (median 0.087 → 0.106), and
the CID22 +0.007 lift from the original run **disappears**: median CID22 SROCC
on the FIXED corpus is 0.8626–0.8638, **tied with the Compression ship**
(0.8641). The single decisive CID22 advantage of the original metric-inputs
direction was driven by the **broken** train/kadid + train/tid ssim2 signal
(constant-per-ref), which apparently helped CID22 by accident. Once that
"signal" is fixed, the lift evaporates.

## Hypothesis (FALSIFIED a second time)

Adding 3 strongly-aligned metric scores (ssim2 / cvvdp / iwssim) as MLP inputs
gives the network a "warm start" toward perceptual ranking. The first attempt
(EXP-METRIC-INPUTS) was falsified due to 4 corpus defects; the FIX targets
were:

1. **train/kadid + train/tid ssim2_gpu constant per image** (upstream join bug).
2. **konjnd-dense ALL-NaN metric scores** (no per-pair join key).
3. **val/konjnd no dist_path keying** at inference.
4. **Scale mismatch**: train=GPU SSIMULACRA2, val=fast-ssim2 CPU.

With all four fixed: hypothesis still falsified. The metric-inputs direction
does not produce a ship-grade improvement, even with clean training and
inference data.

## Defects diagnosed + fixed

### Defect 1 — KADID/TID train ssim2_gpu constant per image

**Diagnosis**: `canonical-2026-05-18/train/kadid.parquet` joined on
`(image_path='I01.png', codec='kadid', q=0, knob_tuple_json='{}')` — a key
shared by all 125 distorted variants of I01. The first-row-wins reducer
collapsed 125 pairs to a single ssim2 value. Confirmed:

```
ref=I01: n=125 ssim2 range [99.99, 99.99] std=0.00
ref=I02: n=125 ssim2 range [99.99, 99.99] std=0.00
... etc
```

**Source of correct data**:
`/mnt/v/zen/zensim-training/2026-05-18-ssim2/kadid_ssim2_local.parquet`
already has the correct per-pair scores
(99.99 → 83.83 → 55.37 → 17.52 → −48.31 for I01_01_01..I01_01_05). The local
parquet's join keys all collide to a single `image_path='I01.png'` row but the
**row order matches `dmos.csv` and the canonical train/kadid parquet
positionally**. Same for TID's `tid_ssim2_local.parquet`.

**Fix**: positional join — drop the canonical `ssim2_gpu` column, append the
local parquet's `ssim2_gpu` column at the same row index. Sanity checked
post-fix: per-ref ssim2 now varies properly (5+ distinct values per ref).

### Defect 2 — konjnd-dense all-NaN metric scores

**Diagnosis**: `canonical-2026-05-18/train/konjnd-dense.parquet` has
`ssim2_gpu = cvvdp_score = iwssim = null` for all 20,160 rows. The upstream
source (`konjnd_dense_features_mix_targets_372col.parquet`) carries no
`image_path / codec / q / knob_tuple_json` columns — the dense corpus has no
per-pair join key. The 20 pairs/SRC represent a synthetic densification
around the PJND threshold, but the exact distortions aren't recorded.

**Fix**: **drop konjnd-dense from training**. KonJND val signal at inference
comes from `val/konjnd.parquet` (the 1008 PJND-anchor pairs).

### Defect 3 — val/konjnd no dist_path keying

**Diagnosis**: `val/konjnd.parquet` rows are keyed only by `ref_basename =
SRC0001.png..SRC1008.png` with `human_score = mean PJND threshold`. No
dist_path → no way to look up per-pair metric scores.

**Fix**: `konjnd_pairs.tsv` (1008 rows, row-aligned to val/konjnd by SRC
ordering) carries the anchor `(ref_path, dist_path)` pairs. Use it to:
- Compute GPU SSIMULACRA2 via `zenmetrics batch --metric ssim2-gpu`
- Join CVVDP from `konjnd_{jpeg,bpg}_cvvdp_2026-05-17.tsv`
- Join IW-SSIM from `konjnd_{jpeg,bpg}_iwssim_scores.tsv`

Result: 1008/1008 join coverage for ssim2 + cvvdp + iwssim on val/konjnd.

### Defect 4 — CPU/GPU ssim2 scale mismatch

**Diagnosis**: training used `ssim2_gpu` (GPU SSIMULACRA2), the original
val-side fill used `fast_ssim2_score` (CPU implementation) from
`v0_22_iw_v3_seed1_2026-05-17_eval_per_pair.csv`. Even when both signals
exist, the implementations differ in scale.

**Fix**: computed GPU SSIMULACRA2 for ALL val corpora via
`zenmetrics batch --metric ssim2-gpu --gpu-runtime cuda`:
- val/kadid: 10125 pairs
- val/tid: 3000 pairs
- val/cid22: 4292 pairs
- val/aic3: 600 pairs
- val/konjnd: 1008 pairs

Outputs at `/tmp/exp_metric_fix/{corpus}_ssim2_gpu.tsv`.

Post-fix scale agreement: train/kadid ssim2 ∈ [-367, 100] std 49.68; val/kadid
ssim2 ∈ [-367, 100] std 49.68 — match. train/tid ∈ [-96, 90] std 45.50;
val/tid ∈ [-96, 90] std 45.49 — match.

## Pipeline

**Diagnosis + scoring** (logs at `/tmp/audit_*.log`, `/tmp/exp_metric_fix/`):
- Audited canonical scores parquets — only synthetic v12_sweep images
  covered; academic corpora not in scope.
- Audited train + val parquet schemas — confirmed defects #1–#3.
- Recovered KADID/TID per-pair ssim2 from local parquets (positional fix).
- Recomputed GPU ssim2 for all 5 val corpora.

**Augmented parquet builds**:
- `scripts/exp_metric_inputs/build_augmented_parquets_fixed.py` — 303-col
  variant (first 300 zenanalyze features + 3 metric inputs).
- `scripts/exp_metric_inputs/build_augmented_375col.py` — 375-col variant
  (full 372 zenanalyze features + 3 metric inputs at f372/f373/f374), built
  to enable apples-to-apples `bake_compare` against the 372-input ship bakes.

Output parquets at `/mnt/v/zen/zensim-training/2026-05-18-metric-inputs-fixed/`:
- `train/{safesyn,kadid,tid,cvvdp_iwssim_LARGE}.parquet` (303 col)
- `val/{cid22,kadid,tid,konjnd,aic3}.parquet` (303 col)
- `train_375/{safesyn,kadid,tid,cvvdp_iwssim_LARGE}.parquet` (375 col)
- `val_375/{cid22,kadid,tid,konjnd,aic3}.parquet` (375 col)
- `eval_features/` + `eval_features_375/` — symlinks for `bake_verdict`

**Training**: 5 seeds × 2 variants (303-col + 375-col), V_24 per-sample-α
recipe (same hyperparams as the original exp-metric-inputs and v_compression).
Bakes at `/mnt/v/zen/zensim-eval/exp_metric_inputs_fixed_2026-05-18/`.

## Results — both variants

### 303-col variant (5 seeds × held-out SROCC)

| seed | CID22 | KADID | TID | KonJND | AIC-3 |
|---:|---:|---:|---:|---:|---:|
| s1 | 0.8571 | 0.8496 | 0.8163 | 0.1060 | 0.8016 |
| s2 | 0.8626 | 0.8494 | 0.8168 | 0.1186 | 0.7979 |
| s3 | 0.8571 | 0.8515 | 0.8191 | 0.0844 | 0.7990 |
| s4 | 0.8665 | 0.8552 | 0.8205 | 0.0728 | 0.8013 |
| s5 | 0.8712 | 0.8547 | 0.8188 | 0.1313 | 0.7970 |
| **median** | **0.8626** | **0.8515** | **0.8188** | **0.1060** | **0.7990** |
| mean | 0.8629 | 0.8521 | 0.8183 | 0.1026 | 0.7994 |

### 375-col variant (5 seeds × held-out SROCC)

| seed | CID22 | KADID | TID | KonJND | AIC-3 |
|---:|---:|---:|---:|---:|---:|
| s1 | 0.8638 | 0.8539 | 0.8167 | 0.0867 | 0.7961 |
| s2 | 0.8618 | 0.8613 | 0.8169 | 0.0735 | 0.7956 |
| s3 | 0.8596 | 0.8697 | 0.8265 | 0.0974 | 0.7967 |
| s4 | 0.8670 | 0.8630 | 0.8241 | 0.0718 | 0.7946 |
| s5 | 0.8647 | 0.8513 | 0.8175 | 0.0815 | 0.7953 |
| **median** | **0.8638** | **0.8613** | **0.8175** | **0.0815** | **0.7956** |
| mean | 0.8634 | 0.8598 | 0.8203 | 0.0822 | 0.7957 |

Median CID22 seed in each variant chosen for `bake_compare` decisive run:
- 303-col: **s2** (median 0.8626; md5 `bbce8203977f468c5f404f8863018f4f`)
- 375-col: **s1** (median 0.8638; md5 `9b60ba76997a0c12696d723ed5a7546a`)
- Packed (i8+zerobias 0.005+lz4): `metric_inputs_fixed_375_s1_h128_packed.bin`
  52,113 bytes (md5 `8ed424f2d8504bc41cdbd4687275cb9a`). **Warning**: i8
  round-trip Δ=239.8 — quantization noise large; NOT a ship-ready pack
  (irrelevant given falsification).

## Trail-gate verdicts (bake_compare 1000-bootstrap on 375-col s1)

### vs Compression ship (V_24-per-sample-α s4 / `v_compression_persample_2026-05-18.bin`)

| Corpus | n | SROCC_A | SROCC_B | h_SROCC | DecScore | Verdict |
|---|--:|---:|---:|---:|---:|---|
| CID22 | 4292 | 0.8638 | 0.8641 | -0.48 | +0.000 | **tied** |
| KADIK10k | 10125 | 0.8539 | 0.9316 | -103.998 | -17.333 | **B>>A** |
| TID2013 | 3000 | 0.8167 | 0.8893 | -53.999 | -0.000 | **B>>A** |
| KonJND-1k | 1008 | 0.0867 | 0.8080 | -15.209 | -0.000 | **B>>A** |
| AIC-3 CTC | 600 | 0.7961 | 0.8183 | -26.073 | -0.000 | **B>>A** |

**Compression trail gate** (§ A.10): requires A>>B on ≥1 of {CID22, AIC-3}
decisive AND no decisive B>>A on the other AND no single corpus regression
worse than −0.10 on {KADID, TID, KonJND}.

- CID22: tied (not A>>B) ✗
- AIC-3: B>>A ✗
- KonJND regression: **−0.721** (way worse than −0.10) ✗
- KADID regression: **−0.078** ✓
- TID regression: **−0.073** ✓

**FAILS Compression gate.**

### vs Balanced ship (V_22-mix-LARGE+iwssim / `v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18.bin`)

| Corpus | n | SROCC_A | SROCC_B | h_SROCC | DecScore | Verdict |
|---|--:|---:|---:|---:|---:|---|
| CID22 | 4292 | 0.8638 | 0.8324 | +18.653 | **+15.545** | **A>>B** |
| KADIK10k | 10125 | 0.8539 | 0.9677 | -88.023 | -0.000 | **B>>A** |
| TID2013 | 3000 | 0.8167 | 0.9729 | -49.784 | -0.000 | **B>>A** |
| KonJND-1k | 1008 | 0.0867 | 0.8927 | -17.348 | -0.000 | **B>>A** |
| AIC-3 CTC | 600 | 0.7961 | 0.7845 | (small) | (small) | tie/A>B |

**Balanced trail gate** (§ A.10): requires A>>B decisive on CID22 AND not
decisive B>>A on any of {KADID, TID, KonJND, AIC-3}.

- CID22: A>>B ✓
- KADID, TID, KonJND: all decisive B>>A ✗

**FAILS Balanced gate.**

### vs Ensemble ship (`v05_ensemble_classifier_2026-05-18.bin`)

| Corpus | n | SROCC_A | SROCC_B | Verdict |
|---|--:|---:|---:|---|
| CID22 | 4292 | 0.8638 | 0.0103 | A>>B (Ensemble broken here?) |
| KADIK10k | 10125 | 0.8539 | 0.7104 | A>>B |
| TID2013 | 3000 | 0.8167 | 0.6632 | A>>B |
| KonJND-1k | 1008 | 0.0867 | 0.5623 | B>>A |
| AIC-3 CTC | 600 | 0.7961 | 0.0883 | A>>B |

Ensemble bake's CID22 SROCC of 0.0103 looks anomalously broken (it ships
PreviewV0_5Ensemble in the runtime but doesn't appear to work in
standalone bake_verdict eval — likely needs its multi-bake dispatch). Not a
useful comparison.

## Diagnosis — why does the fix not lift CID22?

1. **The original exp-metric-inputs +0.007 CID22 lift was an artifact**.
   Constant-per-ref ssim2 was the dominant pattern on KADID + TID training
   data, and the MLP must have learned that "ssim2 is meaningless on these
   corpora; ignore it" — which paradoxically helped on CID22 (where the
   metric inputs ARE varying per-pair). Fixing the train signal to be
   per-pair takes that artifact away.

2. **The KonJND catastrophic regression persists at the same magnitude**
   (~0.10 SROCC on fixed corpus vs 0.33 on broken corpus — actually WORSE
   on the fix). Whatever the per-sample-α head is doing with metric inputs
   produces a near-zero rank correlation on KonJND, regardless of training-
   side corpus health. Hypothesis: the per-sample-α head's pool reducer
   produces highly concentrated outputs on KonJND-domain images, then the
   3 metric input features (which DO vary per anchor pair) push the network
   into a different operating region.

3. **375-col vs 303-col has no meaningful difference**. KADID/TID lift
   slightly with more features (full f0..f371) but CID22 and KonJND are
   identical within noise. The metric inputs don't combine usefully with
   the extra 69 zenanalyze features.

## Falsification verdict

Both Compression and Balanced trail gates fail per § A.10, on both the
303-col and 375-col variants. The 3 metric inputs as additional MLP features
do not produce a ship-grade improvement, even when the training corpus is
clean and the val-side metric inputs are properly joined and scale-matched.

**Dead direction — confirmed twice.** The original SROCC win was a corpus-
defect artifact; cleaning the corpus removes the win entirely.

Future work using metric scores should NOT take this form. Alternative
directions worth trying:
- **Late-fusion** at inference: train a 372-feature MLP, then post-hoc
  blend with ssim2/cvvdp/iwssim via learned per-corpus weights.
- **Knowledge distillation**: train the MLP to mimic an ensemble of
  (ssim2, cvvdp, iwssim, current zensim) on safesyn — let the MLP learn
  the consensus internally rather than reading the scores at inference.
- **Per-corpus calibration**: don't use a single global standardizer for
  the metric inputs — build per-corpus scalers in training so each
  domain's distribution is normalized. Requires a corpus-tag input or a
  separate head per corpus, more complex than the current setup.
- **Drop konjnd from training while preserving its inference signal**:
  this experiment already does this. Still fails — the failure is at the
  forward pass, not the training data.

## Data caveats (post-fix)

- **konjnd-dense intentionally dropped from training** (no per-pair
  metric scores, no per-pair dist_path) — the KonJND val SROCC of 0.087
  is the network's prediction on val/konjnd's properly-keyed metric
  inputs, NOT a constant-fill artifact. This is honest catastrophic
  failure.
- **val/konjnd metric coverage 1008/1008** — every anchor pair has real
  ssim2/cvvdp/iwssim, all matched to the original training-side scale.
- **train/kadid + train/tid ssim2 now per-pair** — verified post-fix that
  per-ref ssim2 has 5+ distinct values, matching the per-pair scoring done
  via `zenmetrics batch`.

## Implementation notes

- Build scripts: `scripts/exp_metric_inputs/build_augmented_parquets_fixed.py`
  (303-col), `scripts/exp_metric_inputs/build_augmented_375col.py` (375-col).
- Train scripts: `scripts/exp_metric_inputs/run_metric_inputs_fixed_seed.sh`
  (303-col), `scripts/exp_metric_inputs/run_metric_inputs_fixed_375_seed.sh`
  (375-col).
- ssim2-gpu scoring: ran `/home/lilith/work/zen/zenmetrics/target/release/zen-metrics
  batch --metric ssim2-gpu --gpu-runtime cuda` on existing pairs TSVs at
  `/mnt/v/zen/zensim-eval/{corpus}_cvvdp_pairs_2026-05-17.tsv` plus
  `/mnt/v/zen/zensim-training/2026-05-18-extfeat/konjnd_pairs.tsv`.
- Verdict files: `verdicts/verdict_s{1..5}.md` (303-col),
  `verdicts_375/verdict_375_s{1..5}.md` (375-col),
  `verdicts_375/bake_compare_vs_{compression,balanced,ensemble}.md`.

Direct main commit: pushed via jj.

— claude-session-exp-metric-inputs-fix, 2026-05-18
