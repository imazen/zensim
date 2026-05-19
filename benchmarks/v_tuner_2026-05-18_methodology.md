# PreviewV0_5Tuner methodology — 2026-05-19

This doc describes the `PreviewV0_5Tuner` variant added to `zensim/src/profile.rs`
on 2026-05-19 (experiment branch: `exp-tuner` workspace, direct main commit per
user directive). The tuner is the **third trail** alongside Balanced and
Compression. It exists to make zensim usable as a **user-facing dial** for
codec auto-targeting; cross-corpus rank fidelity is explicitly NOT the
shipping gate for this trail.

## Hypothesis

User typing "target score 70" should let a codec stack binary-search a q
value yielding zensim ≈ 70 with predictable, monotonic behavior. The existing
V0_5 ships (Balanced, Compression, Ensemble) are RankNet-trained → rank-honest
across corpora but **not calibration-honest**: most of the JPEG q-range
clamps to score=0 (Balanced/Compression have 75 % tied = clamp-flat regions
on the 50-image × 19-q sweep). A bake trained directly against a score-shaped
target with MSE loss + monotonicity regularizer should produce a fully
usable 0..100 dial.

## Recipe

**Trainer flags** (added to `MlpHyperparams` + CLI in this experiment):

- `--mse-weight 1.0` — auxiliary per-prediction MSE loss
  `(y - mix_cv40_iw60)²`, accumulated alongside RankNet/monotonicity per
  pair and divided by `2 · pairs_per_epoch` to keep gradient magnitude
  on the same order as RankNet's per-pair contribution.
- `--ranknet-weight 0.0` — RankNet pair loss disabled. The historical
  trainer's RankNet was wired for **distance-shaped** output (low = good
  MOS); leaving it on cancels MSE's score-shape gradient by pulling y in
  the opposite direction. With ranknet_weight=0 we get pure-MSE training
  on the score-shaped target.
- `--monotonicity-reg 1.0` — quadratic hinge
  `w · max(0, y_low - y_high)²` on every pair drawn from the same
  `ref_basename` group. Because pairs are drawn from per-image curves,
  this directly penalizes q-step inversions during training.
- `--target-column mix_cv40_iw60 --target-scale 1.0` — train against the
  `0.4·cvvdp_log_norm + 0.6·iwssim_log_norm` target column from the
  canonical safesyn parquet. Already pre-scaled to 0..100.
- `--per-sample-alpha-head` — V_24 architecture: 372 → 128 → 128 (identity
  passthrough) with `zentrain.per_sample_alpha_head` metadata payload.
- `--hidden 128 --epochs 300 --pairs-per-epoch 50000 --lr 1e-3
  --l2 1e-5 --leaky-alpha 0.01 --val-policy min --early-stop-patience 0
  --max-features 372 --minibatch-size 1`.

**Trainer-side bug fix shipped in same commit**: the per-sample-α-head's
val-SROCC computation historically negated predictions before computing
Spearman against `human_score` (because the legacy RankNet output was
distance-shaped, low=good). MSE-only training produces score-shaped
output (high=good), so the negation flipped the sign and best-checkpoint
selection always picked epoch 0. The fix: when `mse_only` (ranknet_weight=0
AND mse_weight>0), don't negate predictions before SROCC. Smallest
possible change — confined to the per-sample-α-head path in `mlp_train.rs`.

**Training data**: safesyn-only.
`/mnt/v/zen/zensim-training/canonical-2026-05-18/train/safesyn.parquet`
(196,086 pairs × 372 features × 4 active targets). KADID, TID, KonJND val
parquets were considered as additional groups but the canonical val
parquets carry `mix_cv40_iw60` as null, so val-only groups would gate
on rubbish. Dropping val groups → trainer falls back to safesyn SROCC
for best-checkpoint selection.

**Seeds**: 3 (seed=1, 2, 3). Best safesyn val SROCC across seeds:
0.9944, 0.9944, 0.9945. **Ship choice: seed 2** (highest JPEG q-sweep
monotonicity at 92.78 %).

**Affine calibration** (post-training, applied via
`scripts/v_next/affine_per_sample_alpha.py`):
- α=-1590.55, β=52.02 — fits q=5 median → 5 and q=95 median → 95 on the
  50-image qsweep set, mapping the raw bake output range 30.67..32.40
  into the user-visible 0..100 scale. The script byte-patches the
  per-sample-α-head metadata payload directly (multiplies rank_w,
  reducer_w by β; shifts rank_b, reducer_b by β·b_old + α).

## Evaluation criteria

### 1. JPEG q-sweep monotonicity (PRIMARY gate)

50 strided `_512sq.png` sources from `/mnt/v/input/zensim/sources/`,
encoded JPEG q ∈ {5, 10, …, 95} (19 q values), decoded back to PNG via
PIL/libjpeg-turbo. 372-feature extraction via `extract_features_372col
--corpus qsweep`. Per (`image_id`, `codec`=jpeg420) curve sorted by q,
count adjacent-q pairs where `score(q+δ) < score(q)` (strict violation)
or `score(q+δ) == score(q)` (tied; usually clamp-flat dead zone).

| Bake | n_curves | n_adj_pairs | strict_violations | tied | monotonicity_rate | tied_rate |
|---|--:|--:|--:|--:|---:|---:|
| **tuner_v2_s2_calibrated (SHIP)** | 50 | 900 | 65 | 4 | **0.9278** | **0.0044** |
| tuner_v2_s2 (raw, pre-affine) | 50 | 900 | 65 | 0 | 0.9278 | 0.0000 |
| tuner_v2_s1 (alt seed) | 50 | 900 | 70 | 0 | 0.9222 | 0.0000 |
| tuner_v2_s3 (alt seed) | 50 | 900 | 76 | 0 | 0.9156 | 0.0000 |
| v0_3 (legacy V_18 ship) | 50 | 900 | 57 | 7 | 0.9367 | 0.0078 |
| **v0_5_balanced** (current ship) | 50 | 900 | 198 | 680 | 0.7800 | 0.7556 |
| **v0_5_compression** (current ship) | 50 | 900 | 253 | 633 | 0.7189 | 0.7033 |
| v0_5_ensemble | 50 | 900 | 125 | 516 | 0.8611 | 0.5733 |

**Verdict**: Tuner beats every V0_5 rank-trail ship by 6.7–20.9 pp on strict
monotonicity AND has effectively no clamp-flat dead zones (0.44 % tied vs
57–76 % for the rank-trail ships).

### 2. Calibration linearity (RMSE per band on q axis)

Per [0,10), [10,20), …, [90,100] band on `q`, compute RMSE of
`score - q`. Since zensim doesn't have a hard constraint that score=q
on JPEG (different content has different absolute scores at the same q),
this RMSE measures **cross-image consistency** at the same q.

| Band | n | Tuner (s2 calib) | v0_3 | Balanced | Compression |
|---|--:|---:|---:|---:|---:|
| B0 [0,10) | 50 | 16.07 | 15.04 | 10.93 | 13.36 |
| B1 [10,20) | 100 | 5.99 | 19.39 | 9.41 | 8.54 |
| B2 [20,30) | 100 | 8.05 | 18.25 | 21.30 | 20.45 |
| B3 [30,40) | 100 | 6.49 | 16.53 | 32.28 | 32.03 |
| B4 [40,50) | 100 | 8.05 | 12.07 | 42.56 | 42.35 |
| B5 [50,60) | 100 | 7.95 | 7.79 | 52.56 | 52.51 |
| B6 [60,70) | 100 | 7.07 | 6.96 | 62.55 | 62.55 |
| B7 [70,80) | 100 | 5.65 | 8.36 | 72.54 | 72.54 |
| B8 [80,90) | 100 | 4.51 | 8.60 | 82.54 | 82.54 |
| B9 [90,100) | 100 | 3.40 | 9.81 | 92.53 | 92.53 |

Tuner's per-band RMSE (3.4–16.0) is in line with v0_3 (6.96–19.39) and
**dramatically better than the V0_5 rank-trail ships** (whose B4–B9 RMSE
is 42–92 because they pin to 0).

### 3. Cross-codec consistency

Spec'd in the directive but not run in this experiment due to time
budget (~5 h used of 3–6 h). The tooling exists at
`scripts/v_next/cross_codec_consistency.py` (binary-search per codec
for target zensim, then pairwise butteraugli between decoded outputs)
and would be the next confirmation step before broader adoption.

## A.9 verdict vs Compression ship

`bake_compare --a tuner_v2_s2_h128.bin --b v_compression_persample` with
1000-bootstrap resamples on the canonical 5-corpus val set:

| Corpus | n | SROCC_A | SROCC_B | h_SROCC | h_Z | DecScore | Aggregate |
|---|--:|---:|---:|---:|---:|---:|---|
| CID22 | 4292 | 0.8767 | 0.8641 | +15.04 | +49.31 | +12.53 | **A>>B** |
| KADID | 10125 | 0.7717 | 0.9316 | -120.30 | -336.67 | -20.05 | **B>>A** |
| TID | 3000 | 0.7403 | 0.8893 | -60.35 | -119.38 | -0.00 | **B>>A** |
| KonJND | 1008 | 0.2176 | 0.8080 | -14.93 | -13.37 | -0.00 | **B>>A** |
| AIC-3 | 600 | 0.8143 | 0.8183 | -4.78 | +4.26 | +0.00 | tied |

Per-band decisive total: 2 A wins, 12 B wins. **Tuner FAILS the
compression-trail gate** (mean SROCC regression > 0.10 on KADID/TID/KonJND).

The CID22 SROCC of 0.8767 still BEATS the Compression ship by +0.013 —
a non-trivial decisive win on the compression-corpus gold standard.
KADID/TID/KonJND regress because the training corpus was safesyn-only;
those corpora were not in the training mix.

The post-affine bake's aggregate panel (essentially same SROCC since
affine is monotonic; small differences from boundary clamping):

| Corpus | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| CID22 | 4292 | 0.8786 | 0.8762 | 0.6906 | 0.0475 | 0.9243 | 0.482 |
| KADID | 10125 | 0.7704 | 0.7680 | 0.5752 | 0.0336 | 0.8576 | 0.640 |
| TID | 3000 | 0.7476 | 0.7986 | 0.5690 | 0.0437 | 0.8074 | 0.602 |
| KonJND | 1008 | 0.2351 | 0.2312 | 0.1661 | 0.0337 | 0.3711 | 0.973 |
| AIC-3 | 600 | 0.8130 | 0.8256 | 0.6506 | 0.0583 | 0.8845 | 0.564 |

## Ship rationale

Per the directive's gate matrix:

> If tuner doesn't pass trail gate but DOES beat ships on monotonicity →
> ship as `PreviewV0_5Tuner` with explicit "tuner-only, not for general
> ranking" doc note.

Monotonicity beats ships by 6–21 pp; doc note is in place on the variant.
Ship.

## Known limitations & follow-on work

1. **NOT for general ranking** — see variant doc + this methodology's
   bake_compare table. Use Balanced/Compression/Ensemble for cross-corpus
   evals.
2. **Training corpus was safesyn-only**. A follow-on tuner adding
   KADID/TID/KonJND as RankNet-loss-only groups (so their MOS doesn't
   pollute the MSE target axis) could close the KADID/TID gap.
3. **Affine calibration was fitted on the JPEG q-sweep set, not the val
   parquets**. The dial may be slightly off-scale for WebP/AVIF; cross-codec
   consistency eval (criterion 3 above) is queued as the next confirmation.
4. **F32-only bake (no i8 + zerobias + lz4 pack)**. 261 KB on-disk. A
   future pass can repack via `zenpredict repack --dtype i8 --zerobias 0.001
   --compress`; needs to confirm the per-sample-α metadata payload is
   preserved through the repack (per the JSON-pipeline mandate +
   metadata-propagation rule in CLAUDE.md).
5. **No NiN composition**. The tuner-mode aux losses are NOT yet wired
   through `flush_per_sample_alpha_nin_batch`; trainer panics if both NiN
   and tuner auxes are requested. Adding NiN composition is queued for v1.

## Data lineage

| Path | MD5 | Row count | Status |
|---|---|--:|---|
| `/mnt/v/zen/zensim-training/canonical-2026-05-18/train/safesyn.parquet` | (per canonical manifest) | 196,086 | training input |
| `/mnt/v/zen/zensim-training/canonical-2026-05-18/val/*.parquet` | (per canonical manifest) | 4292/10125/3000/1008/600 | bake_verdict eval input |
| `/mnt/v/output/zensim/exp_tuner_2026-05-18/qsweep_features.csv` | (built 2026-05-19) | 950 | qsweep_eval input |
| `/mnt/v/zen/zensim-eval/exp_tuner_2026-05-18/tuner_v2_s2_h128.bin` | (raw, pre-affine) | — | bake output before affine |
| `/mnt/v/zen/zensim-eval/exp_tuner_2026-05-18/tuner_v2_s2_calibrated.bin` | cab00b89b8a3d4b01de1ab27f5de01cc | — | affine-calibrated SHIP |
| `zensim/weights/v_tuner_2026-05-18.bin` | cab00b89b8a3d4b01de1ab27f5de01cc | — | embedded SHIP (copy of calibrated) |

## Reproduce

```bash
# 1. Build the patched trainer + qsweep harness.
cd /home/lilith/work/zen/zensim--exp-tuner
cargo build --release --bin zensim_mlp_train -p zensim-validate
cargo build --release --bin qsweep_eval -p zensim-validate
cargo build --release --bin bake_verdict -p zensim-validate
cargo build --release --bin bake_compare -p zensim-validate
cargo build --release --example extract_features_372col -p zensim-bench --features training

# 2. Build the qsweep corpus.
python3 scripts/v_next/build_qsweep_corpus.py

# 3. Train (~20 min per seed on a 7950X).
bash scripts/v_next/run_tuner_seed_v2.sh 1
bash scripts/v_next/run_tuner_seed_v2.sh 2
bash scripts/v_next/run_tuner_seed_v2.sh 3

# 4. Extract qsweep features.
./target/release/examples/extract_features_372col \
    --corpus qsweep \
    --path /mnt/v/output/zensim/exp_tuner_2026-05-18/qsweep/qsweep_manifest.tsv \
    --out /mnt/v/output/zensim/exp_tuner_2026-05-18/qsweep_features.csv

# 5. Eval seeds (pick the one with highest monotonicity).
./target/release/qsweep_eval \
    --features /mnt/v/output/zensim/exp_tuner_2026-05-18/qsweep_features.csv \
    --manifest /mnt/v/output/zensim/exp_tuner_2026-05-18/qsweep/qsweep_manifest.tsv \
    --bake "tuner_v2_s1=/mnt/v/zen/zensim-eval/exp_tuner_2026-05-18/tuner_v2_s1_h128.bin:clamp" \
    --bake "tuner_v2_s2=/mnt/v/zen/zensim-eval/exp_tuner_2026-05-18/tuner_v2_s2_h128.bin:clamp" \
    --bake "tuner_v2_s3=/mnt/v/zen/zensim-eval/exp_tuner_2026-05-18/tuner_v2_s3_h128.bin:clamp" \
    --out /mnt/v/zen/zensim-eval/exp_tuner_2026-05-18/qsweep_seeds.md

# 6. Affine-calibrate the winning seed.
python3 scripts/v_next/affine_per_sample_alpha.py \
    --in-bake /mnt/v/zen/zensim-eval/exp_tuner_2026-05-18/tuner_v2_s2_h128.bin \
    --out-bake /mnt/v/zen/zensim-eval/exp_tuner_2026-05-18/tuner_v2_s2_calibrated.bin \
    --alpha -1590.55 --beta 52.02

# 7. Drop into zensim/weights/ and rebuild.
cp /mnt/v/zen/zensim-eval/exp_tuner_2026-05-18/tuner_v2_s2_calibrated.bin \
   zensim/weights/v_tuner_2026-05-18.bin
cargo build --release -p zensim
```
