# EXP-IWSSIM-PERSAMPLE falsification (2026-05-18)

**Verdict:** Pure `iwssim_log_norm` target on per-sample-α head **FAILS BOTH trail gates** at median seed (s3, CID22 SROCC 0.8406, 5-seed mean CID22 0.8402).

The pure-iwssim target shifts the per-sample-α architecture from compression-trail (where its cousin V_24-per-sample-α s4 with `mix_cv40_iw60` ships) to a synthetic-distortion specialist that wins KADID + TID but loses **both** compression-band corpora (CID22 + AIC-3) decisively against the current `PreviewV0_5Compression` ship. The same bake also fails the balanced gate because KonJND is decisively worse than the V_22-mix-LARGE+iwssim Balanced ship.

## Hypothesis

Train per-sample-α head with `--target-column iwssim_log_norm` (pure iwssim, no cvvdp/ssim2 mix). Same data, same group weights, same arch as the current Compression ship (V_24-per-sample-α s4); only the target column changes from `mix_cv40_iw60` (40% cvvdp + 60% iwssim) to `iwssim_log_norm` (100% iwssim).

Goal: probe whether iwssim's known content-awareness (peak SROCC 0.85 on KADID per `benchmarks/baseline_panels_2026-05-18.md`) can deliver an AIC-3 lift versus s4 while keeping CID22 acceptable, opening a compression-trail Pareto rotation.

## Falsification criteria (defined ex ante per task brief)

Cell falsifies if EITHER:
1. CID22 5-seed mean < 0.84 (drops too much vs Compression ship's 0.864), AND
2. AIC-3 mean gain ≤ +0.01 (no AIC-3 lift to compensate).

5-seed result: CID22 mean **0.8402** (passes #1 by 0.0002), AIC-3 mean **0.7992** (− 0.0191 vs Compression ship's 0.8183, fails #2 by −0.0291). The CID22 criterion passes by margin smaller than seed variance (σ ≈ 0.0040), so it is *not robustly* above 0.84. AIC-3 regression is decisive in the opposite direction from the hypothesis.

Per § A.10 gate composition, this also falsifies as a **compression-trail Pareto candidate**: A.9 decisively returns B>>A on **both** compression corpora.

## Recipe (same as V_24-per-sample-α s4, only `--target-column` changed)

Trainer: `/home/lilith/work/zen/zensim--exp-iwssim-persample/target/release/zensim_mlp_train`
Data root: `/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/`
Bake output: `/mnt/v/zen/zensim-eval/exp_iwssim_persample_2026-05-18/iwssim_persample_s{1..5}_h128.bin`

Group weights (train_w:val_w):
- safesyn 1.0:0.0
- kadid 0.3:1.0
- tid 0.3:1.0
- konjnd 0.02:1.0
- cvvdp_iwssim_large 0.5:0.0

Hyperparams: h=128, max_features=300, epochs=300, pairs_per_epoch=50000, lr=0.001, l2=1e-5, leaky_alpha=0.01, minibatch=256, PWRC enabled (sensory threshold 5.0), NiN (w=0.1, p=1.0, q=2.0), per-sample-α head, val-policy=min, early-stop patience 60. Target column: **iwssim_log_norm** (only delta from V_24-per-sample-α s4).

Data note: `iwssim_log_norm` was added to `konjnd_mix_300col.parquet` as an alias for `human_score` (matching the parquet's existing `mix_cv*_iw*` convention where all mix targets are human_score copies for konjnd; konjnd contributes train_w=0.02 so this aliasing has negligible effect on the loss).

## 5-seed CI (aggregate Mohammadi SROCC per corpus, n_inputs=300 per bake)

bake_verdict eval feature root: `/mnt/v/zen/zensim-training/2026-05-15-full-features` (372-col features).

| Seed | CID22 | KADID | TID | KonJND | AIC-3 |
|---|---:|---:|---:|---:|---:|
| s1 | 0.8440 | 0.9674 | 0.9817 | 0.8039 | 0.7950 |
| s2 | 0.8363 | 0.9681 | 0.9815 | 0.7985 | 0.8075 |
| s3 *(median)* | **0.8406** | 0.9671 | 0.9814 | 0.8053 | 0.7929 |
| s4 | 0.8446 | 0.9631 | 0.9778 | 0.7958 | 0.7970 |
| s5 | 0.8357 | 0.9675 | 0.9814 | 0.8027 | 0.8035 |
| **5-seed mean** | **0.8402** | **0.9666** | **0.9808** | **0.8012** | **0.7992** |
| 5-seed σ | 0.0040 | 0.0021 | 0.0017 | 0.0040 | 0.0056 |

Median seed selected as s3 by CID22 SROCC; chosen for bake_compare.

### Controls (ssim2 / cvvdp / iwssim baselines from `benchmarks/baseline_panels_2026-05-18.md`)

| Metric | CID22 | KADID | TID | KonJND | AIC-3 (n=600 PTC superset) |
|---|---:|---:|---:|---:|---:|
| ssim2 (fast-ssim2)            | 0.8895 | 0.8133 | 0.8460 | n/a    | 0.7965 |
| cvvdp                         | 0.8214 | 0.8339 | 0.8531 | 0.0482 | 0.7918 |
| iwssim                        | 0.7836 | 0.8498 | 0.7794 | 0.1859 | 0.7735 |
| **iwssim_persample s3 (this)** | **0.8406** | **0.9671** | **0.9814** | **0.8053** | **0.7929** |
| **Compression ship (s4 cv40_iw60)** | 0.8641 | 0.9316 | 0.8893 | 0.8080 | 0.8183 |
| **Balanced ship (V_22-LARGE+iwssim)** | 0.8324 | 0.9677 | 0.9729 | 0.8927 | 0.7845 |

Read: iwssim_persample is essentially a **Balanced ship clone** on KADID + TID (matches +/- 0.01), but drops 0.027 CID22 vs Compression and 0.087 KonJND vs Balanced. It carves no Pareto frontier between the two existing ships.

## A.9 decisive verdicts vs current ships (1000-bootstrap, median seed s3)

### vs Compression ship (V_24-per-sample-α s4 `mix_cv40_iw60`)

| Corpus | n | SROCC_A (iwssim_persample) | SROCC_B (Compression) | h_SROCC | h_Z-RMSE | Aggregate verdict |
|---|--:|---:|---:|---:|---:|---|
| CID22       | 4292  | 0.8406 | 0.8641 | -52.86 | -164.16 | **B>>A** |
| KADID       | 10125 | 0.9671 | 0.9316 | +83.73 | +734.13 | **A>>B** |
| TID2013     | 3000  | 0.9814 | 0.8893 | +49.27 | +315.32 | **A>>B** |
| KonJND-1k   | 1008  | 0.8053 | 0.8080 |  -2.26 |  -43.50 | promising |
| AIC-3 CTC   | 600   | 0.7929 | 0.8183 | -36.11 |  -63.80 | **B>>A** |

**Compression-trail gate** (per § A.10): A.9 must be ADecisivelyBeatsB on ≥1 of {CID22, AIC-3} ∧ not B>>A on the other ∧ mean Δ on {KADID, TID, KonJND} ≥ −0.10. Both compression corpora are **B>>A**. → **FAILS compression gate decisively** (no compression-corpus A>>B, AND decisive B>>A on both).

Synthetic tolerance: ΔKADID = +0.0350, ΔTID = +0.0915, ΔKonJND = -0.0068 (all ≥ −0.10) — the synth wins are real but cannot rescue the compression-corpus regressions under the gate's logical structure.

Full report: `benchmarks/bake_compare_iwssim_persample_vs_compression_2026-05-18.md` (1000-resample bootstrap, 10-band per-corpus panels).

### vs Balanced ship (V_22-mix-LARGE+iwssim)

| Corpus | n | SROCC_A (iwssim_persample) | SROCC_B (Balanced) | h_SROCC | h_Z-RMSE | Aggregate verdict |
|---|--:|---:|---:|---:|---:|---|
| CID22       | 4292  | 0.8406 | 0.8324 | +6.16 | +15.55  | promising |
| KADID       | 10125 | 0.9671 | 0.9677 | -6.69 | -63.30  | promising |
| TID2013     | 3000  | 0.9814 | 0.9729 | +53.88| +1090.80| **A>>B** |
| KonJND-1k   | 1008  | 0.8053 | 0.8927 | -38.43| -127.45 | **B>>A** |
| AIC-3 CTC   | 600   | 0.7929 | 0.7845 | +4.84 | +13.59  | tied |

**Balanced-trail gate** (per § A.10): A must be decisive A>>B on ≥1 corpus AND not decisive B>>A on any. KonJND returns **B>>A**. → **FAILS balanced gate**.

Full report: `benchmarks/bake_compare_iwssim_persample_vs_balanced_2026-05-18.md`.

## Mechanism analysis

The mechanism finding from the prior EXP-PERSAMPLE-MIX3 (cv33_iw33_sm33) and EX-MIX3 experiments was that **target-shape determines corpus dominance**:
- cvvdp + iwssim → CID22 (the current Compression ship V_24-per-sample-α s4 result on `mix_cv40_iw60`).
- ssim2-mix → KonJND specialist.
- (Untested at the time of those experiments) **iwssim only → ?**

Today's result fills in the missing cell: **iwssim-only target → KADID + TID specialist** (matches the iwssim baseline-panel ranking — iwssim's strongest aggregate SROCC is on KADID 0.8498 / TID 0.7794, so a model trained to match it inherits that strength + amplifies it via the feature ensemble; meanwhile CVVDP's CID22 strength (0.8214 raw vs iwssim's 0.7836) is removed from the supervision signal, dragging CID22 toward iwssim's baseline level).

This makes the result a **near-clone of the Balanced ship** (which trains on the same five-group recipe with a different target) on the synthetic-distortion corpora, and a **regression on the compression-band corpora** (CID22 + AIC-3) where cvvdp's signal carries the load.

**No Pareto frontier carved.** iwssim-only fills neither the compression-trail slot (B>>A decisive on both compression corpora) nor the balanced-trail slot (B>>A decisive on KonJND).

## What this rules in / out

- **Compression-trail rotations require a target that includes the cvvdp signal.** Removing cvvdp from the target column drops CID22 by ~0.024 even when the per-sample-α architecture remains identical. cvvdp's CID22 advantage over iwssim (0.0378 raw) is load-bearing for compression-band content.
- **Balanced-trail rotations are unlikely from this experiment family.** The current Balanced ship's KonJND lead (0.8927) sits above what an iwssim-pure or mix-iwssim recipe can reach (~0.80–0.81 here, ~0.82–0.84 in EX-MIX3).
- **Target-shape mapping (updated):**
  - cvvdp + iwssim mix → CID22 + AIC-3 (compression trail).
  - iwssim only → KADID + TID specialist (no trail slot).
  - cvvdp + iwssim + ssim2 → KonJND-leaning (EX-MIX3 falsified previously).
  - ssim2-heavy → KonJND specialist (V_22 mix candlestick variants).

## Artifacts

- 5 bakes: `/mnt/v/zen/zensim-eval/exp_iwssim_persample_2026-05-18/iwssim_persample_s{1..5}_h128.bin`
- 5 verdict reports: `/tmp/iwssim_persample_s{1..5}_verdict.md`
- Training logs: `/tmp/exp_iwssim_persample_logs/seed{1..5}.log`
- bake_compare reports (this dir):
  - `bake_compare_iwssim_persample_vs_compression_2026-05-18.md`
  - `bake_compare_iwssim_persample_vs_balanced_2026-05-18.md`

## No ship action

Compression ship `PreviewV0_5Compression` unchanged (still V_24-per-sample-α s4 / `v_compression_persample_2026-05-18.bin`). Balanced ship `PreviewV0_5Balanced` unchanged (still `v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18.bin`). SOTA_TRAILS.md not updated.
