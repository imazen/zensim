# EXP-PERSAMPLE-CAPACITY — h=256 / h=512 width sweep FALSIFIED

**Date:** 2026-05-18
**Workspace:** `/home/lilith/work/zen/zensim--exp-persample-capacity`
**Status:** **FALSIFIED.** Wider hidden width (h=256, h=512) does NOT
close the remaining CID22 ssim2-gap relative to the h=128 Compression
ship at the formal § A.9 trail-gate. Capacity is saturated at h=128
for the V_24-per-sample-α recipe.

## Hypothesis

After RECIPE-AUDIT proved per-sample-α + V_22 recipe is saturated at
h=128, capacity scaling was the natural next variable. The hypothesis
was that wider hidden width (h=256 = 2× params, h=512 = 4× params)
could better learn the per-sample-α gating + content-conditional
representation that closes the CID22 ssim2-gap.

## Falsification criteria

Fails if BOTH:
1. h=256 best-CID22 ≤ 0.8641 (no lift over h=128 ship), AND
2. h=512 best-CID22 ≤ 0.8641 (capacity is saturated, no upside).

Passes if any of:
- A.9 decisive A>>B on CID22 OR AIC-3 vs current Compression ship
  (+ synthetic mean Δ ≥ −0.10) → ship to compression trail.
- A.9 decisive A>>B on multiple corpora vs Balanced ship without
  B>>A regressions → ship to balanced trail.

Per the SOTA_TRAILS.md formal gate (stricter):
- Compression trail: A>>B on ≥1 of {CID22, AIC-3} decisively AND
  not decisively B>>A on the other compression corpus AND mean
  SROCC regression on {KADID, TID, KonJND} no worse than −0.10
  on any single corpus.

## Methodology

### Trainer recipe (IDENTICAL to V_24-per-sample-α s4 ship except `--hidden`)

```sh
zensim_mlp_train \
  --group safesyn:safesyn_mix_300col.parquet:1.0:0.0 \
  --group kadid:kadid_mix_300col.parquet:0.3:1.0 \
  --group tid:tid_mix_300col.parquet:0.3:1.0 \
  --group konjnd:konjnd_mix_300col.parquet:0.02:1.0 \
  --group cvvdp_iwssim_large:cvvdp_iwssim_large_300col_v2.parquet:0.5:0.0 \
  --hidden {256|512} --epochs 300 --pairs-per-epoch 50000 \
  --max-features 300 --target-column mix_cv40_iw60 \
  --val-policy min --minibatch-size 256 \
  --pwrc-pair-weight --pwrc-sensory-threshold 5.0 \
  --norm-in-norm-weight 0.1 --norm-in-norm-p 1.0 --norm-in-norm-q 2.0 \
  --per-sample-alpha-head --seed {1..5|1..3} \
  --log-every 10 --early-stop-patience 0 \
  --out <bake.bin>
```

Script: `scripts/v_next/run_persample_capacity_seed.sh`
Trainer rebuilt at commit `dc925f0c` on workspace branch
`exp(persample-capacity)`. Binary md5 not captured (rebuild before
re-evaluation per CLAUDE.md binary-staleness gotcha).

### Cells

- **cell0_h256**: `--hidden 256`, seeds 1-5 (expanded from 3 after
  seed=2 beat ship CID22 at first-pass).
- **cell1_h512**: `--hidden 512`, seeds 1-3 (kept at 3-seed
  per decision-tree rule, since h=512 best did NOT beat h=256 best).

Outputs at `/mnt/v/zen/zensim-eval/exp_persample_capacity_2026-05-18/`.

## Results

### Aggregate Mohammadi SROCC per seed (bake_verdict)

Ship h=128 reference (V_24-per-sample-α s4):
**CID22 0.8641 | KADID 0.9325 | TID 0.8893 | KonJND 0.8080 | AIC-3 0.8183**

| Bake | CID22 | KADID | TID | KonJND | AIC-3 |
|---|---:|---:|---:|---:|---:|
| h256_s1 | 0.8580 | 0.9336 | 0.8895 | 0.8406 | 0.8156 |
| **h256_s2 (best CID22)** | **0.8683** | 0.9335 | 0.8918 | 0.8252 | 0.8084 |
| h256_s3 | 0.8577 | 0.9337 | 0.8903 | 0.8466 | 0.8138 |
| h256_s4 | 0.8637 | 0.9340 | 0.8903 | 0.8312 | 0.8125 |
| h256_s5 | 0.8540 | 0.9337 | 0.8910 | 0.8430 | 0.8083 |
| h512_s1 | 0.8567 | 0.9337 | 0.8905 | 0.8497 | 0.8122 |
| **h512_s2 (best CID22)** | **0.8628** | 0.9328 | 0.8905 | 0.8407 | 0.8088 |
| h512_s3 | 0.8540 | 0.9345 | 0.8900 | 0.8477 | 0.8137 |

### CI summary per cell

**h=256 (5-seed):**

| Metric | best | median | mean | range | Δbest vs ship | Δmed vs ship |
|---|---:|---:|---:|---|---:|---:|
| CID22 | 0.8683 | 0.8580 | 0.8603 | [0.8540, 0.8683] | **+0.0042** | **−0.0061** |
| KADID | 0.9340 | 0.9337 | 0.9337 | [0.9335, 0.9340] | +0.0015 | +0.0012 |
| TID | 0.8918 | 0.8903 | 0.8906 | [0.8895, 0.8918] | +0.0025 | +0.0010 |
| KonJND | 0.8466 | 0.8406 | 0.8373 | [0.8252, 0.8466] | +0.0386 | +0.0326 |
| AIC-3 | 0.8156 | 0.8125 | 0.8117 | [0.8083, 0.8156] | **−0.0027** | **−0.0058** |

**h=512 (3-seed):**

| Metric | best | median | mean | range | Δbest vs ship | Δmed vs ship |
|---|---:|---:|---:|---|---:|---:|
| CID22 | 0.8628 | 0.8567 | 0.8578 | [0.8540, 0.8628] | **−0.0013** | **−0.0074** |
| KADID | 0.9345 | 0.9337 | 0.9337 | [0.9328, 0.9345] | +0.0020 | +0.0012 |
| TID | 0.8905 | 0.8905 | 0.8903 | [0.8900, 0.8905] | +0.0012 | +0.0012 |
| KonJND | 0.8497 | 0.8477 | 0.8460 | [0.8407, 0.8497] | +0.0417 | +0.0397 |
| AIC-3 | 0.8137 | 0.8122 | 0.8116 | [0.8088, 0.8137] | **−0.0046** | **−0.0061** |

### bake_compare § A.9 verdicts vs Compression ship (1000-bootstrap)

**h256_s2 (best-CID22) vs ship:**

| Corpus | A SROCC | B SROCC | A Z-RMSE | B Z-RMSE | A PWRC | B PWRC | DecScore | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| CID22 | 0.8683 | 0.8641 | 0.499 | 0.508 | 0.9195 | 0.9157 | +24.155 | **A>>B** |
| KADID | 0.9335 | 0.9316 | 0.357 | 0.362 | 0.9612 | 0.9602 | +67.320 | **A>>B** |
| TID | 0.8918 | 0.8893 | 0.427 | 0.432 | 0.9194 | 0.9173 | +77.463 | **A>>B** |
| KonJND | 0.8252 | 0.8080 | 0.476 | 0.502 | 0.8670 | 0.8505 | +21.896 | **A>>B** |
| AIC-3 | 0.8084 | 0.8183 | 0.576 | 0.565 | 0.8791 | 0.8856 | −0.000 | **B>>A** |

**h256_s1 (median-CID22) vs ship:**

| Corpus | A SROCC | B SROCC | DecScore | Verdict |
|---|---:|---:|---:|---|
| CID22 | 0.8580 | 0.8641 | −0.000 | **B>>A** |
| AIC-3 | 0.8156 | 0.8183 | −0.000 | tied |

**h512_s2 (best-CID22) vs ship:**

| Corpus | A SROCC | B SROCC | DecScore | Verdict |
|---|---:|---:|---:|---|
| CID22 | 0.8628 | 0.8641 | +0.000 | tied |
| AIC-3 | 0.8088 | 0.8183 | −0.000 | **B>>A** |

## Verdict analysis

### Strict SOTA_TRAILS.md gate (Compression trail)

`A>>B on ≥1 of {CID22, AIC-3} decisively AND not decisively B>>A on
the other compression corpus.`

- **h256_s2**: A>>B on CID22 (✓) AND B>>A on AIC-3 (✗) → **FAILS gate**.
- **h256_s1 (median)**: B>>A on CID22 → **FAILS gate**.
- **h512_s2**: tied on CID22 AND B>>A on AIC-3 → **FAILS gate**.

**No bake from either cell passes the compression trail gate.**

### Why the gate is appropriate

The h256_s2 outlier-seed CID22 win is structurally similar to the
+0.009 V_22→V_24 "recipe gap" that RECIPE-AUDIT (2026-05-18, commit
`1b95d6d2`) proved to be seed-selection artifact + log-frequency
drift. The median-seed CID22 at h=256 is **−0.0061 below ship** —
the same direction the V_22-PERSAMPLE median-seed lost the false
recipe-gap claim.

5-seed CIs at h=128 (V_24 ship, per RECIPE-AUDIT Table) showed seed-1
through seed-5 CID22 spread of 0.8540 to 0.8640 (range 0.0100). Our
5-seed h=256 CI shows 0.8540 to 0.8683 (range 0.0143). The h=256
spread is wider — not because capacity helps, but because more params
amplify seed-variance noise. The best-seed selection picks the
favorable noise; the median tells the true story.

### Looser "A.9 OR" gate

The task description offered a relaxed gate: `A.9 decisive A>>B on
CID22 OR AIC-3` + synthetic mean Δ ≥ −0.10. h256_s2 nominally
satisfies this (CID22 A>>B, mean Δ on KADID/TID/KonJND > −0.10). But:

1. The looser OR gate is **inconsistent with the codebase's
   SOTA_TRAILS.md formal definition**. Shipping under the looser gate
   would create the same problem that the post-2026-05-15 SROCC-only
   verdict ban was designed to prevent — a bake that nominally wins
   one stat is allowed to lose another stat decisively.
2. The AIC-3 regression is **decisive** (B>>A DecScore −0.000 with
   bootstrap CI excluding zero), not borderline. Shipping a
   Compression-trail bake that loses the AIC-3 compression holdout
   decisively defeats the trail's purpose.
3. The CID22 win is **single-seed-selection artifact**. Mean and
   median CID22 at h=256 are both below ship.

## Capacity is the wrong frontier

This experiment proves what RECIPE-AUDIT prefigured: at the
V_24-per-sample-α s4 recipe, the limiting factor is NOT representational
capacity. Going from h=128 to h=512 (4× params) does not move the
median CID22 needle and degrades AIC-3 monotonically.

The next frontier candidates per the recovery-cycle backlog:

- **New features.** The 300-col cvvdp_iwssim_LARGE schema lacks the
  IW-pool block (the 372→300 feature truncation). EX-4-extfeat
  (`/home/lilith/work/zen/zensim--ex4-extfeat/`) is exploring 343-col
  extended features.
- **Better targets.** mix_cv40_iw60 is currently the best-known target
  shape; further mix-knob sweeps (cv30_iw40_sm30 from EX-MIX3) have
  not produced compression-trail wins.
- **Architectural diversity (mixed heads).** The hybrid + per-sample-α
  variants explored. EXP-ENSEMBLE (`zensim--exp-ensemble`) is
  testing learned per-bake mixing.

## Artifacts

- Bakes: `/mnt/v/zen/zensim-eval/exp_persample_capacity_2026-05-18/h{256,512}_s{1..5,1..3}_h{256,512}.bin`
- Trainer logs: same dir + `/tmp/exp_persample_capacity_h{256,512}_s{1..5}.log`
- Per-bake verdicts: `verdicts/h{256,512}_s{N}_h{N}.md`
- bake_compare reports:
  - `bake_compare_h256_s2_vs_compression_ship.md` (best h256 vs ship, 5-corpus)
  - `bake_compare_h256_s1_median_vs_compression_ship.md` (median h256 vs ship, cid22+aic3)
  - `bake_compare_h512_s2_vs_compression_ship.md` (best h512 vs ship, cid22+aic3)
- Training script: `scripts/v_next/run_persample_capacity_seed.sh`
- Eval script: `scripts/v_next/eval_persample_capacity_all.sh`

## Recovery cycle position

Falsification cycle entry. h=128 remains the optimal hidden-width
choice for the V_24-per-sample-α recipe family. Future capacity-scaling
attempts on this recipe should not be revisited without:

1. A different target shape (multi-target supervision adding signal
   that h=128 cannot absorb).
2. A different feature set (extended features adding non-redundant
   signal).
3. A different architecture (per-sample-α with a deeper head, not
   just wider).

Pure width scaling on the current recipe is **dead**.
