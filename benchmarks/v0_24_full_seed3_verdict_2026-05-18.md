# V_24-full seed=3 result vs V_22-mix-LARGE+iwssim (EX-3 follow-up)

**Date:** 2026-05-18
**Branch:** `feat/v24-ex3-followup` (zensim) + `feat/v24-ex3-safesyn-discovery` (zenmetrics)
**Status:** seed=3 evaluated; NOT a Pareto win; 5-seed CI NOT run; NOT a ship candidate.

## TL;DR

Adding `ssim2_log_norm` as a third target (symmetric 3-way mix
`mix_target = (cv + iw + sm) / 3`) **decisively wins CID22**
(+0.038 SROCC, A>>B) but **decisively loses KADID + TID**
(−0.090 / −0.095 SROCC, B>>A). **AIC-3 SROCC is essentially
unchanged**: 0.7846 (V_24-full) vs 0.7831 (V_22-mix-LARGE) — the
0.85 target documented in EX-1 / EX-3 is NOT reached by adding the
ssim2 target.

Verdict: **V_24-full as currently designed is a CID22 specialist,
not a strict-Pareto ship candidate.** The cross-corpus tradeoff
(KADID + TID lose 9 SROCC points to gain 3.8 on CID22) is NOT what
EX-3 predicted (the prediction was "ssim2 closes the AIC-3 gap").

## Discovery: no ssim2 fleet needed

The prior EX-3 session flagged safesyn ssim2 backfill as a 57-hour
blocker requiring a vast.ai fleet (196,086 pairs × 6 pairs/sec).
This session's investigation found **safesyn already has ssim2** —
the parquet's `human_score` column equals `gpu_ssimulacra2 / 100`
(verified across 52,498 rows against the source CSV
`/mnt/v/output/zensim/synthetic-v2/training_safe_synthetic.csv`,
column `cpu_ssimulacra2`, zero mismatches).

This is consistent with `CLAUDE.md > Multi-target training corpus`:
"safesyn's `human_score` carries the ssim2-derived score per pair."
The fleet would have re-scored a corpus that already had its target.

**Cost of the discovery:** zero. **Cost avoided:** ~22-27 min × 26
vast.ai instances at $0.10/hr ≈ $0.50-$1.10, plus the engineering
effort to build/maintain pair-file uploads, chunk dispatch, and
sister destroyer.

## Corpus build

`scripts/v_next/build_v24_mix_target_corpus.py`:

1. **safesyn**: derive `ssim2_log_norm` from `human_score * 100` via
   the canonical `(x + 30) / 130 * 100` clip transform (matches
   EX-3 KADID/TID/LARGE exactly). Build
   `mix_target = (cvvdp_log_norm + iwssim_log_norm + ssim2_log_norm) / 3`.
   Output: `/mnt/v/zen/zensim-training/2026-05-18-v24/safesyn_4target_372col.parquet`
   (196,086 rows × 393 cols; MD5 `6402143260a920653debe034a63d0ed2`).

2. **kadid_4target / tid_4target**: rename `mix_cv33_iw33_sm33` →
   `mix_target` from the EX-3 outputs at
   `/mnt/v/zen/zensim-training/2026-05-18-ssim2/`.

3. **LARGE 3target**: rename `mix_cv33_iw33_sm33` → `mix_target`.
   Two infrastructure fixes vs EX-3 raw output:
   - Renamed `feat_0..feat_299` → `f0..f299` (the trainer's
     parquet_loader requires `fN` prefix).
   - Dropped 1100 NaN rows (2.0%; CVVDP coverage gap on rare
     codec-q combos).
   Output: 53,800 rows × 358 cols; MD5
   `32c5031ddb5ac71535ae252772fd5697`.

4. **konjnd**: rename `human_score` (PJND) → `mix_target`. PJND
   stays the target; group weight 0.02 handles scale mismatch with
   the codec-mix targets.

All five parquets share a unified `mix_target` column so the trainer
uses `--target-column mix_target --target-scale 1.0` globally.

## Training recipe (V_24-full seed=3)

Identical to V_22-mix-LARGE+iwssim except the target is the 3-way
`mix_target` and corpora come from `2026-05-18-v24/`:

```sh
./target/release/zensim_mlp_train \
  --group safesyn:safesyn_4target_372col.parquet:1.0:1.0 \
  --group kadid:kadid_4target_372col.parquet:0.3:1.0 \
  --group tid:tid_4target_372col.parquet:0.3:1.0 \
  --group konjnd:konjnd_features_mix_targets_372col.parquet:0.02:0.0 \
  --group large:large_3target_300feat_minus_ssim2.parquet:0.5:0.0 \
  --target-column mix_target --target-scale 1.0 \
  --hidden 128 --epochs 300 --pairs-per-epoch 50000 \
  --lr 1e-3 --l2 1e-5 --leaky-alpha 0.01 --val-policy min --seed 3 \
  --log-every 30 --max-features 300 \
  --minibatch-size 256 --pwrc-pair-weight --norm-in-norm-weight 0.1 \
  --early-stop-patience 0
```

Total: 263,994 rows across 5 groups. Wall time: **602.8 s** (10 min)
on a single CPU thread. Best validation mean SROCC at epoch 90:
0.9651.

Bake: `/mnt/v/zen/zensim-eval/v24_2026-05-18/v24_mix_4target_s3_h128.bin`
(MD5 `3fc6415df4b104be246237f30554a9aa`, 157,252 B).

## bake_compare result vs V_22-mix-LARGE+iwssim (seed=3)

`A = V_24-full seed=3`, `B = V_22-mix-LARGE+iwssim seed=3`. Decisive
rule per § A.9 of `PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md`:

| Corpus | n | SROCC_A | SROCC_B | ΔSROCC | Z_A | Z_B | PWRC_A | PWRC_B | Aggregate |
|---|--:|---:|---:|---:|---:|---:|---:|---:|---|
| **CID22** | 4292 | **0.8702** | 0.8323 | **+0.0379** | 0.506 | 0.559 | 0.9222 | 0.9005 | **A>>B** |
| KADID | 10125 | 0.8776 | **0.9677** | −0.0901 | 0.482 | 0.249 | 0.9263 | 0.9804 | B>>A |
| TID2013 | 3000 | 0.8782 | **0.9729** | −0.0947 | 0.461 | 0.236 | 0.9087 | 0.9833 | B>>A |
| KonJND-1k | 1008 | 0.8022 | 0.8928 | −0.0906 | 0.402 | 0.375 | 0.8474 | 0.9181 | promising B |
| **AIC-3** | 600 | 0.7846 | 0.7831 | +0.0015 | 0.604 | 0.608 | 0.8600 | 0.8619 | promising A |

**Decisive-cell totals:** 2 A wins, 16 B wins, 4 promising,
7 tied, 1 noisy.

**Overall winner across decisive cells: B.**

Report: `/mnt/v/zen/zensim-eval/v24_2026-05-18/v24_vs_v22_mix_LARGE_compare.md`.

## AIC-3 (the load-bearing gap)

EX-1 / EX-3 hypothesized that adding ssim2 as a target would close
the AIC-3 SROCC gap from 0.787 to ≥ 0.85. Measured:

- V_22-mix-LARGE+iwssim (no ssim2 target): **0.7831**
- V_24-full (3-way mix with ssim2): **0.7846**
- Target: ≥ 0.85
- Actual lift: **+0.0015** (essentially zero, p_SROCC = 0.0495 in
  MRR — borderline at the standard 0.05 cutoff).

**The ssim2-target hypothesis is FALSIFIED for AIC-3.** The 0.85
target is not crossable by changing supervision target on the
existing 5-group corpus; the gap is structural (different content
distribution, different distortion regime, or AIC-3's MOS shape
that none of cvvdp/iwssim/ssim2 captures well at this corpus
resolution).

## Why V_24 wins CID22 but loses KADID + TID

CID22 is **compression-focused human MOS**; ssim2 is its
strongest single-metric correlate per the 2023 paper. Adding
ssim2 as a target gives the trained MLP a CID22-shape that
emerges directly from the training target. **+3.8 SROCC on CID22
is the cleanest demonstration to date that the training target
shape is load-bearing for CID22 — not the architecture, not the
feature set.**

KADID and TID, by contrast, have **synthetic non-compression
distortions** (blur, noise, color shifts, geometric). The
cvvdp+iwssim mix in V_22 is well-suited for those distortion
types; adding ssim2 (which is compression-tuned) dilutes the
KADID/TID supervision and causes the regression.

## What this rules out

- **"Just add ssim2 as a target" does not close AIC-3.** The
  EX-3 prediction was wrong. AIC-3 requires a different lever
  (likely: AIC-3-specific feature engineering, MOS-anchored
  training, or richer-feature LARGE corpus).
- **3-way symmetric mix (0.33/0.33/0.34) is too aggressive.**
  Adding ssim2 with 1/3 weight rotates the training surface
  toward CID22 at the cost of KADID/TID. A weighted mix
  (e.g., 0.4 cv + 0.4 iw + 0.2 sm) might preserve more KADID/TID
  signal while keeping some CID22 lift.

## What this confirms

- **The corpus-build infrastructure works.** Building V_24-full
  from existing parquets via column rename takes ~10 s. Training
  takes 10 min. End-to-end iteration cost is well under 1 hour
  including bake_compare.
- **bake_compare is a 10-minute decisive verdict.** Mohammadi
  panel + bootstrap CI + MRR + § A.9 rule across 5 corpora × 10
  bands gives an unambiguous "A>>B / B>>A / promising / tied"
  call per (corpus, band).

## Honest gaps

- **Only seed=3 evaluated.** 5-seed CI not run because the seed=3
  result is NOT a Pareto win — the decisive rule already favors
  B by 16 vs 2 cells. A 5-seed CI on a known-mixed-result bake
  burns ~50 min of compute for marginal additional signal.
- **No AIC-3 per-band breakdown.** The bake_compare report skips
  per-band for AIC-3 (corpus uses a JND step grid that doesn't
  partition cleanly into 10-band cuts). Aggregate is the only
  read; promising-A by 0.0015 is within noise.
- **No safesyn fleet was launched.** The discovery that
  human_score = ssim2/100 already eliminated the need. The fleet
  scripts in `zenmetrics/scripts/sweep/ssim2_backfill/` remain
  ready for if-needed-later but were NOT executed this session.

## Open questions / next directions

If CID22 +0.038 is the desired direction, the per-corpus weight
needs tuning to keep KADID/TID near V_22 levels:

- **Ablation**: `mix = 0.4 cv + 0.4 iw + 0.2 sm` — softer ssim2
  weight may preserve KADID/TID while keeping some CID22 lift.
- **Per-group mix override**: keep safesyn/large on `mix_cv40_iw60`
  (2-way) and only flip kadid/tid/cid22 to 3-way mix where CID22
  benefits.
- **AIC-3-specific**: the AIC-3 corpus has its own MOS column the
  trainer doesn't currently see. Adding AIC-3 as a training group
  (NOT as a target) might be the better lever.

These are NOT done in this session; the V_24-full seed=3 result
above plus this discussion is the deliverable.

## Reproducibility

```sh
# Branch checkout
cd ~/work/zen/zensim
git fetch origin
git checkout feat/v24-ex3-followup

# Corpus build (assumes EX-3 outputs at 2026-05-18-ssim2/ exist)
python3 scripts/v_next/build_v24_mix_target_corpus.py

# Trainer + bake_compare (as commands documented above)
cargo build --release --bin zensim_mlp_train --bin bake_compare -p zensim-validate
```

The bake_compare JSON sidecar at
`/mnt/v/zen/zensim-eval/v24_2026-05-18/v24_vs_v22_mix_LARGE_compare.json`
is the machine-readable form of the verdict for follow-up analyses.
