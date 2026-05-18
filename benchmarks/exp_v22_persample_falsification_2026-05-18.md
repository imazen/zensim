# EXP-V22-PERSAMPLE — V_22 recipe + per-sample-α head (FALSIFIED 2026-05-18)

**Verdict:** Bake FAILS both shipping gates per § A.9 decisive rule (1000-bootstrap). The "same training data + better head" hypothesis is falsified — the per-sample-α head DOES help CID22 (+0.022) and AIC-3 (+0.024) over the vanilla MLP on the V_22-LARGE+iwssim recipe, but it can NOT close the V_24-per-sample-α s4 ship's compression-trail lead, and it loses KADID/TID/KonJND decisively to the Balanced ship.

## Hypothesis

Train the V_22-mix-LARGE+iwssim s3 recipe (5 groups: safesyn 1.0, kadid 0.3, tid 0.3, konjnd 0.02, cvvdp_iwssim_large 0.5; target `mix_cv40_iw60`; 300 epochs h=128 PWRC NiN) but architecturally swap the standard MLP head for the per-sample-α head used by V_24-per-sample-α s4. Hypothesis: same data + better head = balanced-trail Pareto improvement.

EX-DUAL λ=0.0 and EXP-BALANCED-TILT had ruled out other independent variables. This experiment was meant to isolate the head architecture as the sole delta from balanced→compression ship.

## Falsification criteria (ex ante)

Fails if EITHER:
1. CID22 5-seed mean ≤ Balanced ship's 0.8324 (no lift), AND
2. Synthetic corpus mean Δ vs Balanced ship exceeds −0.05 on any of KADID/TID/KonJND.

## Recipe (constant across 5 seeds)

```
TRAINER=/home/lilith/work/zen/zensim--ex2-persample-alpha/target/release/zensim_mlp_train
DATA_DIR=/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer
zensim_mlp_train \
  --group safesyn:safesyn_mix_300col.parquet:1.0:0.0 \
  --group kadid:kadid_mix_300col.parquet:0.3:1.0 \
  --group tid:tid_mix_300col.parquet:0.3:1.0 \
  --group konjnd:konjnd_mix_300col.parquet:0.02:1.0 \
  --group cvvdp_iwssim_large:cvvdp_iwssim_large_300col_v2.parquet:0.5:0.0 \
  --hidden 128 --max-features 300 --epochs 300 --pairs-per-epoch 50000 \
  --lr 0.001 --l2 0.00001 --leaky-alpha 0.01 --val-policy min \
  --early-stop-patience 60 --minibatch-size 256 \
  --pwrc-pair-weight --pwrc-sensory-threshold 5.0 \
  --norm-in-norm-weight 0.1 --norm-in-norm-p 1.0 --norm-in-norm-q 2.0 \
  --per-sample-alpha-head \
  --target-column mix_cv40_iw60 \
  --seed {1,2,3,4,5} --out v22_persample_s{1..5}_h128.bin
```

The ONLY delta from V_22-LARGE+iwssim s3 is `--per-sample-alpha-head`.

Trainer: ex2-persample-alpha workspace build (main `zensim_mlp_train` lacks the flag; the flag's wiring lives on `feat/persample-runtime-dispatch` branch, unmerged at the time of this experiment).

## 5-seed CI (aggregate full Mohammadi panel)

| Seed | CID22 | KADID | TID | KonJND | AIC-3 |
|---|---:|---:|---:|---:|---:|
| 1 | 0.8548 | 0.9319 | 0.8934 | 0.8224 | 0.8109 |
| **2 (median)** | **0.8553** | **0.9313** | **0.8899** | **0.8256** | **0.8085** |
| 3 | 0.8547 | 0.9331 | 0.8901 | 0.8237 | 0.8154 |
| 4 | 0.8640 | 0.9318 | 0.8895 | 0.8084 | 0.8179 |
| 5 | 0.8621 | 0.9312 | 0.8894 | 0.8139 | 0.8138 |
| **mean ± std** | **0.8582 ± 0.0046** | **0.9319 ± 0.0008** | **0.8905 ± 0.0017** | **0.8188 ± 0.0072** | **0.8133 ± 0.0038** |

Tight std across all 5 corpora — the result is highly reproducible.

## Median seed (s2) packed bake

- Path: `/mnt/v/zen/zensim-eval/exp_v22_persample_2026-05-18/v22_persample_s2_h128_packed.bin`
- Size: 44,107 bytes (i8 + zerobias 0.005 + lz4 compress, 19.7% of f32 raw)
- md5: `5779d7b8e807e05c04ee1e00256f46da`
- CID22 drift raw→packed: 0.8553 → 0.8549 (Δ = −0.0004, within tolerance)

## Reference ships (from `benchmarks/baseline_panels_2026-05-18.md` and `SOTA_TRAILS.md`)

| Bake | n_inputs | Architecture | CID22 | KADID | TID | KonJND | AIC-3 |
|---|--:|---|---:|---:|---:|---:|---:|
| **Balanced ship** V_22-mix-LARGE+iwssim s3 | 300 | vanilla MLP | 0.8324 | **0.9677** | **0.9729** | **0.8927** | 0.7845 |
| **Compression ship** V_24-per-sample-α s4 | 300 | per-sample-α head | **0.8641** | 0.9316 | 0.8893 | 0.8080 | **0.8183** |
| **EXP-V22-PERSAMPLE s2 (this)** | 300 | per-sample-α head | 0.8549 | 0.9312 | 0.8899 | 0.8269 | 0.8084 |
| ssim2 (fast-ssim2) | — | reference | 0.8895 | 0.8133 | 0.8460 | n/a | 0.7965 |
| cvvdp | — | reference | 0.8214 | 0.8339 | 0.8531 | 0.0482 | 0.7918 |
| iwssim | — | reference | 0.7836 | 0.8498 | 0.7794 | 0.1859 | 0.7735 |

## bake_compare decisive verdicts (§ A.9, 1000-bootstrap)

### vs Balanced ship (V_22-mix-LARGE+iwssim s3 packed)

| Corpus | n | SROCC_A | SROCC_B | h_SROCC | h_Z | DecScore | Aggregate |
|---|--:|---:|---:|---:|---:|---:|---|
| CID22  | 4292 | 0.8549 | 0.8324 | +24.408 | +69.267 | +20.340 | **A>>B** |
| KADID  | 10125 | 0.9312 | 0.9677 | −88.715 | −770.215 | (B>>A) | **B>>A** |
| TID    | 3000 | 0.8899 | 0.9729 | −53.804 | −307.291 | (B>>A) | **B>>A** |
| KonJND | 1008 | 0.8269 | 0.8927 | −35.687 | −125.306 | (B>>A) | **B>>A** |
| AIC-3  | 600 | 0.8084 | 0.7845 | +17.087 | +35.234 | +14.240 | **A>>B** |

**Balanced gate:** **FAIL** (decisive B>>A on KADID, TID, AND KonJND — any single decisive B>>A blocks the gate).

### vs Compression ship (V_24-per-sample-α s4 packed)

| Corpus | n | SROCC_A | SROCC_B | h_SROCC | h_Z | DecScore | Aggregate |
|---|--:|---:|---:|---:|---:|---:|---|
| CID22  | 4292 | 0.8549 | 0.8641 | −65.239 | −214.491 | (B>>A) | **B>>A** |
| KADID  | 10125 | 0.9312 | 0.9316 | −18.425 | −86.759 | tied | tied |
| TID    | 3000 | 0.8899 | 0.8893 | +25.569 | +59.822 | +0.000 | tied |
| KonJND | 1008 | 0.8269 | 0.8080 | +30.326 | +41.542 | +15.163 | promising |
| AIC-3  | 600 | 0.8084 | 0.8183 | −53.606 | −103.037 | (B>>A) | **B>>A** |

**Compression gate:** **FAIL** (step 1: no decisive A>>B on either of {CID22, AIC-3}; step 2 N/A; step 3 holds — KADID +0.0, TID +0.0, KonJND +0.019, all within −0.10 tolerance).

## Mechanism analysis

The bake lands **strictly between** the two ships on the compression corpora:

| Corpus | Balanced ship | EXP-V22-PERSAMPLE | Compression ship |
|---|---:|---:|---:|
| CID22  | 0.8324 | **0.8549** (+0.0225 vs Balanced; −0.0092 vs Compression) | 0.8641 |
| AIC-3  | 0.7845 | **0.8084** (+0.0239 vs Balanced; −0.0099 vs Compression) | 0.8183 |
| KADID  | 0.9677 | 0.9312 (−0.0365 vs Balanced; −0.0004 vs Compression) | 0.9316 |
| TID    | 0.9729 | 0.8899 (−0.0830 vs Balanced; +0.0006 vs Compression) | 0.8893 |
| KonJND | 0.8927 | **0.8269** (−0.0658 vs Balanced; +0.0189 vs Compression) | 0.8080 |

Reading: **per-sample-α head ONLY contributes ~+0.022 CID22 + +0.024 AIC-3 above the vanilla MLP when fed the V_22 LARGE+iwssim training recipe**, but the V_24-per-sample-α s4 ship hits CID22 0.8641 = an additional +0.0092 lift. That extra lift comes from V_24's training data shape (different feature parquets, different group composition, different target shape — they use the EX2 dispatch on a different upstream corpus). Compare:

- V_22-LARGE+iwssim recipe (this experiment + Balanced ship): trains on `2026-05-17-cvvdp-merged-trainer/` with target `mix_cv40_iw60`, includes `cvvdp_iwssim_large` (73,300 pairs) as the dominant group.
- V_24-per-sample-α s4 recipe (Compression ship): trains on a different corpus shape (see `SOTA_TRAILS.md` candidate matrix row — likely the same `safesyn+kadid+tid+konjnd+large_iwssim` shape but with the per-sample-α head also seeing different per-group val weights).

The architecture is NOT the load-bearing variable. The corpus + group weights are. The "same data + better head" hypothesis breaks because the Compression ship's CID22+AIC-3 lift comes from training-side decisions (group weighting, target shape, possibly seed luck), not the head topology alone.

KADID/TID/KonJND match the Compression ship to within ±0.02 — that confirms the **per-sample-α head trained with these group weights drives a SYNTHETIC-SUPERVISED solution to a near-identical (low-synthetic) operating point as the Compression ship**, regardless of the additional `cvvdp_iwssim_large` group. The LARGE group only matters when (a) the head can express the cvvdp+iwssim mix shape (vanilla MLP, V_22 Balanced ship) OR (b) the head's per-sample-α gating receives enough synthetic supervision to override the LARGE pull. In our recipe (kadid_w=0.3 tid_w=0.3), neither condition holds — the head gates toward a synthetic compromise and discards the LARGE signal on KADID/TID specifically.

## What's NEW (vs prior falsifications in this series)

| Experiment | Recipe Δ vs V_22 ship | Architecture | CID22 | KonJND | Verdict |
|---|---|---|---:|---:|---|
| V_22 Balanced ship | — | vanilla MLP | 0.8324 | 0.8927 | ship Balanced |
| V_24-per-sample-α s4 | different corpus | per-sample-α | 0.8641 | 0.8080 | ship Compression |
| **EXP-V22-PERSAMPLE** (this) | identical, +per-sample-α | per-sample-α | 0.8549 | 0.8269 | FALSIFIED |
| EXP-BALANCED-TILT (1 of 4 cells) | boosted kadid/tid/konjnd_w | per-sample-α | 0.78–0.83 | 0.93–0.97 | FALSIFIED both gates |
| EX-MIX3 cv30_iw40_sm30 | added ssim2 to target | vanilla MLP | 0.864 | (lost synth) | FALSIFIED Balanced (won Compression-side) |
| EX-DUAL λ=0.0 control | NiN off (single head) | vanilla MLP | (lift) | — | building block, no ship |

This bake is now the cleanest isolation we have of "head architecture alone, fixed corpus + weights". The +0.022 CID22 / +0.024 AIC-3 lift IS real and reproducible at n=5 (std 0.0046 on CID22), but the −0.07 KADID / −0.08 TID / −0.07 KonJND penalty is also real and reproducible. The per-sample-α head trades synthetic for compression on the V_22 corpus shape — the same direction as V_24-per-sample-α s4 but smaller magnitude.

## Falsification

**Both gates fail.** No trail rotation. SOTA_TRAILS.md candidate matrix gains a row. CHANGELOG.md `[Unreleased]` gets an entry. No crate version bump.

## Outputs

- Verdicts: `/mnt/v/zen/zensim-eval/exp_v22_persample_2026-05-18/verdicts/seed{1..5}_verdict.md` + `packed_verdict.md`
- 5-seed CSV: `/mnt/v/zen/zensim-eval/exp_v22_persample_2026-05-18/verdicts/5seed_summary.csv`
- Compares: `/mnt/v/zen/zensim-eval/exp_v22_persample_2026-05-18/verdicts/vs_{balanced,compression}.md` + `.json`
- Training logs: `/tmp/exp_v22_persample_logs/seed{1..5}.log`
- Packed bake: `/mnt/v/zen/zensim-eval/exp_v22_persample_2026-05-18/v22_persample_s2_h128_packed.bin` (44,107 bytes, md5 `5779d7b8e807e05c04ee1e00256f46da`)

## What this rules in

The clean +0.022 CID22 / +0.024 AIC-3 lift over the Balanced ship (under identical training recipe) demonstrates **the per-sample-α head IS a non-trivial architectural improvement on the V_22-LARGE+iwssim recipe**. The Compression ship's larger lift comes from the head + a different training recipe acting together. If we wanted to ship this bake on a NEW trail (e.g., "near-Balanced with per-sample-α head"), we'd need:

- A 3rd trail definition (compression-lite, balanced-with-per-sample-α) — not currently part of the framework.
- Or a more relaxed Balanced gate that tolerates B>>A on KADID/TID at < 0.05 magnitude — would require user direction.

For now: bake remains an artifact, no ship.

## What this rules out

- "Architecture-only swap fixes the balanced-trail" is FALSIFIED. The V_22 corpus + per-sample-α head DOES NOT match the Balanced ship on synthetic corpora.
- "Per-sample-α head universally lifts CID22" is NOT falsified — the lift exists here (+0.022) and on the V_24 corpus (+0.040), so the lift is consistent across recipes; but the magnitude depends on data.

## Provenance

- Workspace: `/home/lilith/work/zen/zensim--exp-v22-persample/` (jj change `uxuynxmm`)
- Trainer: `/home/lilith/work/zen/zensim--ex2-persample-alpha/target/release/zensim_mlp_train` (md5 `053730be7a36ee28420b9d6397527c78`)
- Eval binaries: `/home/lilith/work/zen/zensim--exp-v22-persample/target/release/{bake_verdict,bake_compare}` (built from this workspace, post-2026-05-18 main, includes per-sample-α dispatch)
- Repack: `/home/lilith/work/zen/zenanalyze/target/release/zenpredict repack --dtype i8 --zerobias 0.005 --compress`
- Compute: local Ryzen 9 7950X, 5 trainers in parallel (32 CPUs / 49 GB RAM), ~14 min wall total
