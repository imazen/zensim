# Context Handoff (2026-05-12)

## What just shipped

**V0_15 is the current runtime weight** (zensim/weights/v0_15_2026-05-12.bin,
md5 `73d5e418`). Honest replacement for the tainted V0_8.

| Metric | V0_15 (honest) | V0_8 (tainted, archived) | fast-ssim2 |
|---|--:|--:|--:|
| CID22 SROCC | **0.8914** (+0.0019) | 0.8948 (+0.0053, inflated) | 0.8895 |
| AIC-3 CTC | **0.8019** (+0.0054) | 0.8043 (+0.0078) | 0.7965 |
| Non-mono % | **2.51 %** (meets strict 4.86 % target) | 5.87 % | 5.08 % |
| Val mean | 0.9427 | 0.9416 | — |

## The purge (2026-05-12, user-directed)

Per the directive *"purge every copy of contaminated source files where
validation set variants slipped in, permanently, in every form"*, the
2026-05-12 audit at d≤16 perceptual-hash threshold identified **361 hex-hashed
source files** that were crops/resizes of 22 of the 49 CID22 held-out
references. The original 2026-05-11 cleanup ran at a looser threshold and
missed them, leaking **11,629 rows (7.43%)** into V0_8's training set.

**Deleted (~75 GiB freed)**:
- 361 source PNGs at `/mnt/v/input/zensim/sources/` + tower mirror
- 361 encoded variant dirs (30.6 GiB) at `/mnt/v/input/zensim/images/<stem>/`
- 27 .features.bin caches (9 GiB) + 5 on tower (3.2 GiB)
- 15 .pre-purge-bak intermediates (861 MiB)
- 6 tainted bakes (V0_9..V0_14) at `/tmp/zensim_loop/`
- 3 tainted derived CSVs

**Cleaned in place**: 15 training CSVs at `/mnt/v/output/zensim/synthetic-v2/`
stripped of 7-10% rows each.

**Audit trail**: `benchmarks/contaminated_sources_purged_2026-05-12.txt`
(361 absolute paths). V0_8 archived at
`zensim/weights/archive/v0_8_tainted_2026-05-11.bin`.

## V0_15 recipe (recoverable)

- Training CSV: `/tmp/zensim_loop/safe_synth_clean_features.csv` (144,791 rows
  after purge, was 156,420)
- TV pairs: `/tmp/zensim_loop/combined_purged_tv_pairs_bands.tsv` (205,654
  pairs after index remap)
- Hyperparams: h=128, flat TV=15, seed=1, 300 epochs (early-stop ep 190)
- Affine-calibrated: α=26.9332, β=-4.5520, R²=0.7447 (matches ssim2 truth
  distribution on synth corpus)
- Trainer: `target/release/zensim_mlp_train` (zensim-validate crate)

## What's running right now

**V0_16 trainer** (PID 3222018, launched tick 370): same as V0_15 but flat
TV=20 instead of 15. Hypothesis: does stronger TV on clean data help?
Output: `/tmp/zensim_loop/v0_16_purged_tv20_seed1.bin`. ETA ~12 min.

## Site state (Goal #6)

Live at <https://imazen.github.io/zensim/>. Deploy via
`.github/workflows/pages.yml` (note `enablement: true` fix).

Charts:
- Aggregate SROCC bar per dataset (CID22 selected by default)
- Per-band SROCC at the 4 CID22 paper bands (B0/B1/B2/B3)
- **Step-5 SROCC** (13 CID22 bins) — multi-bake comparison
  (V0_15 + V0_8_tainted + V0_2 + ssim2 + butter)
- Pareto: CID22 SROCC vs non-mono q-step rate
- Bake history table (8 bakes — V0_15 marked current; V0_8 archived)

Bake JSONs at `site/data/bakes/` (8 total).
Step-5 JSONs at `site/data/step5_bands/` (V0_15 + V0_8_tainted).

## Open work (queued, not started)

1. **KADID / TID step-5 panels** — need re-eval of V0_15 + V0_8 on those
   datasets with `--per-pair-output`. Currently only CID22 step-5 exists.
2. **Non-mono per-band** — extend `score_unified_with_bake.py` to bucket
   reversal rate by MCOS bin.
3. **V0_15 on KonJND-1k PJND anchor** — verify visual-threshold calibration
   (~63 ± 5 score at PJND).
4. **Coefficient repo blocklist** — add 361 hex stems from
   `contaminated_sources_purged_2026-05-12.txt` to
   `coefficient/examples/generate_zensim_training.rs CID22_VALIDATION_49`
   so the contamination can't be re-introduced if sources are regenerated.
   *Out-of-repo; needs user touch.*
5. **Image-type-aware MLP dispatch** (user's long-term direction) — multi-MLP
   architecture with a content classifier picking which MLP to use. Big
   project, not started.
6. **Butter-concordance training** — wrote audit + filter scripts; haven't
   yet trained a bake on butter-clean data. The 42% of curves with at
   least one ssim2/butter disagreement may be label noise that holds back
   B0/B1 SROCC.

## Useful pointers

- Champion log: `~/work/zen/zenanalyze/zensim_champion_log.md` (370+ ticks)
- AIC-3 dataset: `/mnt/v/dataset/aic3_ctc_epfl/` (600 pairs, 6 codecs)
- Unified parquets for non-mono: `/mnt/v/zen/zensim-training/2026-05-07/unified/`
- Eval binary: `target/release/examples/dataset_metric_baseline`
  (now supports `--aic3 <pairs.csv>` flag)
- Site Pareto chart: `site/js/app.js renderPareto()` + bake JSONs

## DO NOT

- Don't re-add CID22 holdout content to training (the 49 refs at
  `/mnt/v/dataset/cid22/CID22_validation_set/original/` are SACRED)
- Don't trust V0_8's CID22 0.8948 number anywhere public — it's inflated
- Don't `cargo publish` zensim (it's behind `__experimental_versions`
  feature and not yet ready for crates.io)
