# EXP-CROSS-CODEC-METRIC — methodology

**Date:** 2026-05-19
**Hypothesis:** Adding a cross-codec equivalence pair loss `(y_a - y_b)²` over
butteraugli-anchored cross-codec equivalence pairs should drive the
T=63 cross-codec mean pairwise butter (jpeg/webp/avif on the 6-img
feature subset) from 4.43 (Tuner +per-codec affine) toward < 2.5
(the structural floor is ~2 for genuinely different codec
distortions).

**Falsification:** If T=63 cross-codec butter stays ≥ 4.0 across 3
seeds AT ANY non-zero cross-codec-eq weight, the equivalence-pair
mechanism is falsified for this architecture (per-sample-α head with
the V_24 / Tuner-v2 recipe). Document the negative result; the
mechanism does not work without further changes.

**Cost ceiling:** 1 sweep build (~80 min wall), 1 trainer build
(~30 min wall), per-codec sweeps × 4 (~80 min wall in parallel),
equivalence parquet build (~15 min), 3 trainer seeds × 30 min each
(parallel ~30 min), 1 eval (~30 min). Total wall ~3.5 hr.

**Ship form:** if hypothesis succeeds, ship as `PreviewV0_5CrossCodec`
(a new variant), preserve `PreviewV0_5Tuner` for backward compat.
Update `SOTA_TRAILS.md` with the cross-codec trail and its gate.

## Reporting panel (decided upfront)

- **Primary metric (load-bearing for ship decision):** T=63 mean
  pairwise butter_max on the 6-img feature subset (the same 10-img
  subset used by `cross_codec_consistency.py`).
- **Secondary metrics (don't catastrophically regress):**
  - CID22 aggregate SROCC vs prior Tuner: tolerate −0.05.
  - AIC-3 aggregate SROCC vs prior Tuner: tolerate −0.05.
  - strict_mono on JPEG q-sweep: must stay ≥ 0.90 (per Tuner gate).
  - tied_rate: must stay < 0.10 (preserve dial-honesty).

## Data pipeline

1. **Sweep:** `cross_codec_butter_features` binary
   (`zensim-picker-prep/src/bin/cross_codec_butter_features.rs`)
   encodes 1000 source images at q ∈ [5, 95] step 5 per codec C ∈
   {zenjpeg, zenwebp, zenavif, zenjxl}, decodes back, scores
   butteraugli max + pnorm_3 AND extracts the 372-dim zensim
   feature vector. Outputs per-codec parquet
   `/mnt/v/zen/picker-training/2026-05-19/butter/<codec>.parquet`.

2. **Equivalence pair build:**
   `scripts/v_next/build_cross_codec_equivalence.py` reads the 4
   parquets and constructs equivalence pairs: at each of 20 butter
   levels per source, finds the q per codec whose butter_pnorm3 is
   nearest to the level, then emits C(K, 2) pairs where K is the
   number of codecs with valid features at that level. Rejects pairs
   whose butter gap > 0.5. Output:
   `/mnt/v/zen/picker-training/2026-05-19/cross_codec_equivalence.parquet`.

3. **Trainer extension:** `zensim-validate/src/mlp_train.rs` gains a
   new `EquivPairs` struct + a new entry point
   `train_mlp_with_tv_anchored_equiv` that plumbs equiv pairs into
   the per-sample-α head training loop. Each pair-step has
   probability `cross_codec_eq_step_p` of doing an extra equivalence
   step: forward (A, B) → loss `w · (y_a - y_b)²` → backprop both →
   single Adam update.

4. **CLI:** `zensim_mlp_train --cross-codec-eq-parquet PATH
   --cross-codec-eq-weight W --cross-codec-eq-step-p P`. Composes
   with the existing `--anchor-parquet` + `--anchor-loss-weight`.

## Recipe

Base recipe: today's Tuner v2 (`scripts/v_next/run_tuner_v2_seed.sh`)
PLUS the equivalence pair flag. Specifically:

```
zensim_mlp_train \
    --group safesyn:canonical/train/safesyn.parquet:1.0:0.0 \
    --hidden 128 --epochs 300 --pairs-per-epoch 50000 --lr 1e-3 --l2 1e-5 \
    --leaky-alpha 0.01 --val-policy min --early-stop-patience 0 \
    --max-features 372 --minibatch-size 1 \
    --target-column mix_cv40_iw60 --target-scale 1.0 --out-dtype f32 \
    --per-sample-alpha-head \
    --ranknet-weight 0.0 --mse-weight 1.0 \
    --monotonicity-reg 0.5 --monotonicity-margin 0.0 \
    --anchor-parquet /mnt/v/zen/zensim-training/2026-05-19-jnd-anchors/anchors_372col.parquet \
    --anchor-loss-weight 1.0 --anchor-target-score 63.0 --anchor-step-p 0.10 \
    --cross-codec-eq-parquet /mnt/v/zen/picker-training/2026-05-19/cross_codec_equivalence.parquet \
    --cross-codec-eq-weight <W> --cross-codec-eq-step-p 0.10 \
    --seed <SEED> --out <BAKE>
```

## Weight sweep

Per Step 3 (seed=1 first as cheap signal):

1. Seed=1 trial with W ∈ {0.1, 0.3, 1.0, 3.0}.
2. Pick the W with lowest T=63 cross-codec butter that also keeps
   CID22 SROCC within 0.01 of the prior Tuner.
3. Run that W at seeds 2, 3.
4. Median over 3 seeds is the ship candidate.

## Ship gate

- **Pass:** T=63 butter < 2.5 AND CID22 SROCC drop ≤ 0.05 AND
  strict_mono ≥ 0.90 AND tied_rate < 0.10.
- **Close:** T=63 butter in [2.5, 4.0] with other gates green → ship
  as opt-in variant `PreviewV0_5CrossCodec`, document the gap.
- **Far:** T=63 butter ≥ 4.0 → falsification, document mechanism
  failure, hypothesis dead.

## Lineage

- Base bake: `v_tuner_v2_s2_h128.bin` (current `PreviewV0_5Tuner`).
- Training corpus: safesyn (196k rows, 372-feat) + cross-codec equiv
  pairs (TBD k rows).
- Anchor: KonJND+safesyn JND mix (9373 rows, target=63).
