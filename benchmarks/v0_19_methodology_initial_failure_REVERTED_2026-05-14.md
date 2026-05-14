# V0_19 methodology — KADID/TID-purge experiment, NOT shipped

**Status**: built, validated, **rejected** for ship 2026-05-14.
**Reason**: CID22 SROCC dropped 0.0147 vs V0_18 baseline, falling
BELOW fast-ssim2's 0.8895 floor — the load-bearing ship gate
specified in zensim/CLAUDE.md.

## Build pipeline

Followed the canonical V0_18 3-way concat recipe exactly, but
with training inputs swapped to the 2026-05-14 clean corpus at
`/mnt/v/zen/zensim-training/2026-05-14-clean/` (V0_18 base
minus 149 KADID+TID perceptual-overlap basenames at d≤16).

Components (all 228→128→1 F32, h=128, epochs 300, seed varies):

| Component | Recipe | val_mean best | Bake |
|---|---|--:|---|
| base seed=1 | no TV regularizer | 0.9458 | benchmarks/v0_19_base_seed1_2026-05-14.bin |
| cycle-14 s1 | TV --tv-band-weights 10,30,10,30, seed=1 | 0.9344 | benchmarks/v0_19_cycle14_s1_2026-05-14.bin |
| cycle-14 s42 | same TV, seed=42 | 0.9259 | benchmarks/v0_19_cycle14_s42_2026-05-14.bin |

Concat at 0.65 / 0.30 / 0.05 mix via
`cargo run --bin concat_three_way`. Affine-calibrated with V0_16-
lineage α=28.0366, β=-5.0738. Resulting f32 ensemble bake:
`benchmarks/v0_19_calibrated_2026-05-14.bin` (355,332 B).

## 10-band validation (max-pairs 50000)

Full report at `benchmarks/v0_19_10band_2026-05-14.md`.

Aggregate SROCC:

| Corpus | n | V0_4 (V0_19) | fast-ssim2 | vs ssim2 | vs V0_18 |
|---|--:|--:|--:|--:|--:|
| CID22 | 4292 | 0.8786 | 0.8895 | **-0.0109** ⚠ FAIL | -0.0147 |
| KADID10k | 10125 | 0.9462 | 0.8133 | +0.1329 | +0.0035 |
| TID2013 | 3000 | 0.9553 | 0.8460 | +0.1093 | +0.0027 |

## Ship decision: NO

CID22 SROCC 0.8786 is below fast-ssim2's 0.8895 by 0.0109 — failing
the gold-standard ship gate (zensim/CLAUDE.md "match-or-exceed
fast-ssim2 across all bands" with CID22 as the primary anchor).

V0_18 stays as the live bake. The KADID/TID overlap inflation
documented in the 2026-05-14 V0_18 methodology addendum remains
the soft upper bound; CID22 0.8933 stays load-bearing because
the CID22 purge in 2026-05-12 made it honest.

## What happened — honest analysis

The 149 KADID+TID-overlap synth-v2 sources weren't ONLY inflating
KADID and TID — they were also providing useful training signal
for CID22 (their perceptual content overlaps not just with KADID
refs but, indirectly, with CID22-distribution images). Removing
them shifted the training distribution in a way that the V_X
arch + recipe couldn't compensate for.

**This is a real finding**, not a training bug:

1. The V0_19 base component reached val_mean 0.9458, BETTER than
   V0_18's 0.9403. So the model fit the (cleaner) training data fine.
2. But CID22 generalization regressed. The clean-set MLP became
   slightly less calibrated to CID22-shaped pairs.
3. KADID and TID numbers nudged UP slightly (+0.003), confirming
   the test pairs are no longer being scored against trained-from
   data — the small honest gain reflects fewer overfitted
   predictions on adjacent-content.

## Next steps

V0_19 going forward is treated as an experimental dead-end. The
3 component bakes + concat bake stay in `benchmarks/` for
reproducibility / future use, but are not shipped.

**For V_X work continuing past this point:**

- The contamination purge is RIGHT (per CLAUDE.md and audit).
  Future bakes must continue to use the canonical clean corpus.
- Recovering CID22 SROCC on the clean corpus likely requires:
  - More aggressive TV regularization on B1/B2 (the 50..70
    band where V0_19 regressed most vs V0_18)
  - Additional clean training data (perceptual-content rebalancing
    to replace the lost overlap signal)
  - Hyperparameter sweep around the 3-way mix coefficients
    (0.65/0.30/0.05 was tuned for V0_18's contaminated training
    distribution; the clean distribution may want different mix)
  - Increase --hidden to 192 or 256 (single-MLP capacity)

These are queued as V0_20 input-shaping research + a separate
"V0_22 — recipe sweep on clean corpus" experiment.

## Reproducibility

```sh
# 1. Component trains (all use the canonical clean corpus):
CLEAN=/mnt/v/zen/zensim-training/2026-05-14-clean
for seed in base_seed1 cycle14_s1 cycle14_s42; do
  case $seed in
    base_seed1)   TV="" ; S=1  ;;
    cycle14_s1)   TV="--tv-pairs-file $CLEAN/tv_pairs_bands.tsv --tv-weight 1.0 --tv-band-weights 10,30,10,30 --tv-apply-every 50 --tv-batch 32" ; S=1 ;;
    cycle14_s42)  TV="--tv-pairs-file $CLEAN/tv_pairs_bands.tsv --tv-weight 1.0 --tv-band-weights 10,30,10,30 --tv-apply-every 50 --tv-batch 32" ; S=42 ;;
  esac
  cargo run --release -p zensim-validate --bin zensim_mlp_train -- \
    --group safesyn:$CLEAN/safe_synth_v19_clean_features.csv:1.0:0.0 \
    --group kadid:$CLEAN/kadid_features.csv:0.3:1.0 \
    --group tid:$CLEAN/tid_features.csv:0.3:1.0 \
    --group konjnd:$CLEAN/konjnd_aligned_features.csv:0.5:1.0 \
    --hidden 128 --epochs 300 --seed $S $TV \
    --out benchmarks/v0_19_${seed}_2026-05-14.bin
done

# 2. Concat:
cargo run --release -p zensim-validate --bin concat_three_way -- \
  --base benchmarks/v0_19_base_seed1_2026-05-14.bin \
  --s1   benchmarks/v0_19_cycle14_s1_2026-05-14.bin \
  --s42  benchmarks/v0_19_cycle14_s42_2026-05-14.bin \
  --coeffs 0.65:0.30:0.05 \
  --out  benchmarks/v0_19_concat_3way_2026-05-14.bin

# 3. Calibrate:
python3 scripts/v_next/affine_calibrate_znpr_v2.py \
  --in-bake benchmarks/v0_19_concat_3way_2026-05-14.bin \
  --out-bake benchmarks/v0_19_calibrated_2026-05-14.bin \
  --alpha 28.0366 --beta -5.0738

# 4. Validate:
cargo run --release -p zensim-bench --example dataset_metric_baseline -- \
  --cid22 /mnt/v/dataset/cid22/CID22_validation_set \
  --kadid /mnt/v/dataset/kadid10k \
  --tid /mnt/v/dataset/tid2013 \
  --v04-bake benchmarks/v0_19_calibrated_2026-05-14.bin \
  --max-pairs 50000 > benchmarks/v0_19_10band_2026-05-14.md
```

