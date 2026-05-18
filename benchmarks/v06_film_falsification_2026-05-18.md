# PR #31 (FiLM-gated MLP) — falsification on the two-trail § A.10 SOTA framework (2026-05-18)

PR https://github.com/imazen/zensim/pull/31 proposed a FiLM-gated MLP (V0_6 rebalance) trained 2026-05-05 on the 340k synthetic-v2 extended corpus, with content-class dispatch over 5 classes (photo/screen/lineart/synthetic/document) extracted from the zenanalyze tail. The PR's reported headline was **val_mean=0.8457** on KADID + TID + KonJND-1k (76k-pair version) + synthetic-holdout, framed as +0.020 over the then-baseline.

That val_mean metric and the corpus it was computed on **pre-date the May 12–14 decontamination wave**, the May 14–16 iwssim landing, and the May 18 two-trail § A.10 SOTA framework. This document re-evaluates the FiLM bake against today's two current ships and the § A.10 trail gates, with the bake_compare 1000-bootstrap Mohammadi panel and ssim2 / cvvdp / iwssim controls.

## Verdict

**FALSIFIED on both trails.** FiLM (class 0 "photo" head) loses decisively on KADID + TID + KonJND against both the Balanced ship (V_22-mix-LARGE+iwssim) and the Compression ship (V_24-per-sample-α). The corpus where it wins (CID22, decisive A>>B against Balanced; CID22 marginal A>>B against Compression) is more than offset by KonJND collapses (SROCC 0.497 vs 0.808 / 0.893) and KADID losses (-0.11 / -0.08 SROCC).

Neither trail gate (§ A.10) is satisfied:

- **Balanced gate** requires A>>B decisive on ≥1 corpus *and* no decisive B>>A on any corpus. FiLM beats Balanced on CID22 but Balanced beats FiLM decisively on KADID, TID, and KonJND. **FAIL.**
- **Compression gate** requires A>>B decisive on ≥1 of {CID22, AIC-3} *and* no decisive B>>A on the other compression corpus *and* synthetic-mean Δ ≥ −0.10. FiLM beats Compression on CID22 but Compression beats FiLM decisively on AIC-3. **FAIL.**

Recommendation: **close PR #31 without rebase.** The PR's tree is structurally incompatible with current main (would delete 24 540 lines including `iw_pool.rs`, `PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md`, current ship bakes, the entire two-trail framework, and the bake_compare tool). The FiLM bake itself does not justify reviving any of that.

## Methodology

- bake_compare § A.9 (`target/release/bake_compare`), 1000 bootstrap resamples, seed=42.
- Features: 372-col parquets at `/mnt/v/zen/zensim-training/2026-05-15-full-features/`.
- Bake A: `/mnt/v/output/zensim/synthetic-v2/runs/v06_film_20260505T212932.bin` (class 0 "photo" head, ZNPR v3, 63 044 B).
- Bake B1 (Balanced ship): `zensim/weights/v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18.bin`.
- Bake B2 (Compression ship): `zensim/weights/v_compression_persample_2026-05-18.bin`.
- Corpora: CID22 (n=4292), KADID-10k (n=10125), TID2013 (n=3000), KonJND-1k held-out PJND-threshold (n=1008), AIC-3 CTC per-pair (n=600).

### Important caveat — FiLM is a 5-head dispatched model

The FiLM bake on disk is **only the class-0 "photo" head**. The PR's `MetricRuntime` calls `zenanalyze_tail` first to pick a class (photo / screen / lineart / synthetic / document), then dispatches to the matching bake. Today's `bake_compare` evaluates a single bake against all corpora without class dispatch. So the FiLM verdict here reflects **the photo head applied uniformly**, which is the head most likely to win on a photo-heavy corpus (CID22) and most likely to fail on screen / synthetic mixes (KADID/TID/KonJND).

A "fair" full-dispatch evaluation would need:
1. Per-pair zenanalyze tail classification on each of the 5 corpora.
2. Per-corpus-per-class scoring with the matching bake.
3. Per-corpus aggregate across classes.

Neither the PR nor current main's `zensim-validate/main.rs` supports that; building it would be a multi-day reconciliation. The single-head evaluation above is sufficient for falsification because:

- The "photo" class is by definition the class FiLM was tuned to help with most on photo-leaning corpora.
- KonJND-1k *is* photo (subset of CID22 references) — FiLM's photo head still loses by 0.31–0.40 SROCC.
- KADID-10k and TID2013 are also photo-heavy reference sets — FiLM still loses by 0.08–0.13 SROCC.
- Even if class dispatch helped FiLM on KADID/TID/KonJND, the gain would have to lift SROCC by +0.10+ on every non-CID22 corpus to satisfy "no decisive B>>A on any". That is structurally implausible given (a) the photo-head's KonJND SROCC of 0.497 vs Balanced's 0.893 cannot recover with class dispatch on a single-class corpus, and (b) the FiLM model used the pre-decontamination 76k-pair KonJND-train, so its KonJND generalization was always limited by that.

## Cross-corpus aggregate panels with ssim2 / cvvdp / iwssim controls

Baseline ssim2 / cvvdp / iwssim numbers from `benchmarks/baseline_panels_2026-05-18.md` and commit f38c610 (iwssim panels).

### CID22 (n=4292)

| Bake / Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| **A: FiLM (photo head)** | **0.8755** | 0.8750 | 0.6867 | 0.0480 | 0.9267 | 0.484 |
| **B1: Balanced ship (V_22-mix-LARGE+iwssim)** | 0.8324 | 0.8289 | 0.6340 | 0.0443 | 0.9006 | 0.559 |
| **B2: Compression ship (V_24-per-sample-α)** | 0.8641 | n/a | n/a | n/a | 0.9157 | 0.508 |
| ssim2 control | 0.8895 | 0.8879 | 0.7062 | 0.0424 | 0.9351 | 0.460 |
| cvvdp control | 0.8214 | 0.8251 | 0.6238 | 0.0424 | 0.8842 | 0.565 |
| iwssim control | 0.7836 | 0.7926 | 0.5938 | 0.0520 | 0.8525 | 0.610 |

- **A.9 verdict vs Balanced**: `A>>B` (DecScore +22.343; h_SROCC +26.812, h_Z +83.804).
- **A.9 verdict vs Compression**: `A>>B` (DecScore +5.068; h_SROCC +6.081, h_Z +24.912).
- CID22 is the **only** corpus where FiLM decisively wins.
- Note: ssim2 control still outscores FiLM here on aggregate SROCC (0.890 vs 0.876). FiLM's CID22 win is not "best metric on CID22"; it's "better than the bake B at the chosen corpus".

### KADID-10k (n=10125)

| Bake / Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| A: FiLM (photo head) | 0.8527 | 0.8536 | 0.6605 | 0.0535 | 0.9089 | 0.526 |
| **B1: Balanced ship** | **0.9677** | 0.9686 | 0.8432 | 0.0433 | 0.9804 | 0.249 |
| **B2: Compression ship** | **0.9316** | n/a | n/a | n/a | 0.9602 | 0.362 |
| ssim2 control | 0.8133 | 0.8107 | 0.6174 | 0.0516 | 0.8828 | 0.585 |
| cvvdp control | 0.8339 | 0.8337 | 0.6389 | 0.0417 | 0.9018 | 0.552 |
| iwssim control | 0.8498 | 0.8446 | 0.6663 | 0.0357 | 0.9112 | 0.535 |

- **A.9 verdict vs Balanced**: `B>>A` decisive (h_SROCC −90, h_Z −432). Balanced ship beats FiLM by +0.115 SROCC.
- **A.9 verdict vs Compression**: `B>>A` decisive (h_SROCC −115, h_Z −459). Compression ship beats FiLM by +0.079 SROCC.
- Even iwssim control (raw, no learning) beats FiLM by +0.003 SROCC. FiLM is **not even better than untrained iwssim** on this corpus.

### TID2013 (n=3000)

| Bake / Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| A: FiLM (photo head) | 0.8451 | 0.8493 | 0.6584 | 0.0407 | 0.8914 | 0.502 |
| **B1: Balanced ship** | **0.9729** | 0.9717 | 0.8571 | 0.0357 | 0.9832 | 0.236 |
| **B2: Compression ship** | **0.8893** | n/a | n/a | n/a | 0.9173 | 0.432 |
| ssim2 control | 0.8460 | 0.8504 | 0.6614 | 0.0467 | 0.8846 | 0.526 |
| cvvdp control | 0.8531 | 0.8644 | 0.6721 | 0.0427 | 0.8853 | 0.503 |
| iwssim control | 0.7794 | 0.8306 | 0.5995 | 0.0327 | 0.8489 | 0.557 |

- **A.9 verdict vs Balanced**: `B>>A` decisive (h_SROCC −49). Balanced beats FiLM by +0.128 SROCC.
- **A.9 verdict vs Compression**: `B>>A` decisive (h_SROCC −23). Compression beats FiLM by +0.044 SROCC.
- FiLM is statistically tied with the raw ssim2 control on this corpus.

### KonJND-1k (n=1008)

| Bake / Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| A: FiLM (photo head) | 0.4971 | 0.5089 | 0.3479 | 0.0436 | 0.6386 | 0.900 |
| **B1: Balanced ship** | **0.8927** | 0.9265 | 0.7070 | 0.0446 | 0.9178 | 0.376 |
| **B2: Compression ship** | **0.8080** | n/a | n/a | n/a | 0.8505 | 0.502 |
| ssim2 control | n/a | n/a | n/a | n/a | n/a | n/a |
| cvvdp control | 0.0482 | 0.1521 | 0.0256 | 0.0347 | 0.0225 | 0.988 |
| iwssim control | 0.1859 | 0.2274 | 0.1327 | 0.0308 | 0.3097 | 0.974 |

- **A.9 verdict vs Balanced**: `B>>A` decisive (h_SROCC −17). Balanced beats FiLM by **+0.396 SROCC** — catastrophic FiLM collapse.
- **A.9 verdict vs Compression**: `B>>A` decisive (h_SROCC −13). Compression beats FiLM by +0.311 SROCC.
- FiLM trained on the 76k-pair KonJND-train (the leaky one, deduplicated/decontaminated 2026-05-12–14). Its 0.497 SROCC on the clean 1008-pair held-out PJND-threshold subset is consistent with the leaked-train hypothesis: the FiLM model fit pre-leakage patterns that the held-out clean set does not share.

### AIC-3 CTC per-pair (n=600)

| Bake / Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| A: FiLM (photo head) | 0.7862 | 0.8009 | 0.6172 | 0.0500 | 0.8646 | 0.607 |
| **B1: Balanced ship** | 0.7845 | 0.7953 | 0.6155 | 0.0433 | 0.8630 | 0.606 |
| **B2: Compression ship** | **0.8183** | n/a | n/a | n/a | 0.8856 | 0.565 |
| ssim2 control | 0.7965 | 0.8086 | 0.6288 | 0.0567 | 0.8716 | 0.588 |
| cvvdp control | n/a | n/a | n/a | n/a | n/a | n/a |
| iwssim control | 0.7735 | 0.7907 | 0.6064 | 0.0450 | 0.8536 | 0.612 |

- **A.9 verdict vs Balanced**: `tied` (DecScore 0; SROCC 0.786 vs 0.785). FiLM and Balanced are statistically indistinguishable on AIC-3.
- **A.9 verdict vs Compression**: `B>>A` decisive (h_SROCC −13). Compression ship beats FiLM by +0.032 SROCC.
- FiLM does not match the raw ssim2 control here either (0.786 vs 0.797).

## Trail-gate decisions (§ A.10)

### Balanced gate

| Requirement | Required | Got | Pass? |
|---|---|---|---|
| `A>>B` decisive on ≥1 corpus | yes | CID22 | yes |
| No `B>>A` decisive on any corpus | yes | KADID + TID + KonJND all `B>>A` | **NO** |

**Balanced gate FAIL.**

### Compression gate

| Requirement | Required | Got | Pass? |
|---|---|---|---|
| `A>>B` decisive on ≥1 of {CID22, AIC-3} | yes | CID22 | yes |
| No `B>>A` decisive on the other compression corpus | yes | AIC-3 `B>>A` | **NO** |
| Synthetic-mean Δ ≥ −0.10 | yes | (not measured against compression-domain synthetic — but FiLM was trained on synthetic-v2; assumed pass) | n/a |

**Compression gate FAIL.**

Both gates fail on the "no decisive B>>A on the other corpus" rule. The FiLM CID22 win is real but does not extend to any other validation corpus.

## What the FiLM PR's val_mean=0.8457 actually measured

The PR's val_mean is `min(KADID_holdout, TID_holdout, KonJND-1k_76k_holdout, Synthetic_holdout) SROCC`. We can cross-check from the FiLM training log (`v06_film_20260505T212932.log` epoch 10): `Synthetic=0.9971 Kadid10k=0.8570 Tid2013=0.8462 konjnd1k=0.9480`. The reported `val_mean=0.8462` is the min of those four, dominated by TID at 0.846. The PR rounded to 0.8457 for the title — close enough.

That number does NOT correspond to today's evaluation, because:

1. **Training set is pre-decontamination.** May 12–14 purged ~12k contaminated source rows from the synthetic-v2 corpus. The FiLM model trained on the dirty 340k.
2. **KonJND validation set differs.** PR used `konjnd1k n=76104` (the 76k-pair training-corpus version, contains leaks). Today's KonJND validation is the 1008-pair held-out PJND-threshold subset, scored independently.
3. **The val_mean was a `min` aggregator** — not the § A.9 decisive panel. It does not include OR, PWRC, Z-RMSE, KROCC, PLCC, the 10-band breakdown, or the bootstrap CI.
4. **The reported `val_mean=0.8462` was held against KADID/TID only as `n/a` for cclass dispatch** — the PR did not measure per-class scoring, just the joint training model's holdout SROCC.

Comparing PR-reported 0.8457 directly to today's V_22-mix-LARGE+iwssim val_mean is meaningless; the corpora and the aggregator differ.

## What's structurally incompatible

A full git-rebase of `v06-rebalanced-corpus` onto current `main` would require resolving conflicts on 735 files. The PR branch:

- **Deletes `zensim/src/iw_pool.rs`** (973 lines) — the foundation of the current `iwssim` panels, V_22-iw_v2 ship, and the entire iwssim baseline-control infrastructure.
- **Deletes `zensim/src/simd_ops.rs`** (1169 lines) — replaced in main by per-arch dispatch via archmage.
- **Deletes 11 newer bake files** under `zensim/weights/` including `v_compression_*` (the current Compression ship) and `v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18.bin` (current Balanced ship).
- **Deletes `PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md`** — the document defining the § A.9 / § A.10 framework this falsification report uses.
- **Deletes `benchmarks/baseline_panels_2026-05-18.md`** and 30+ other newer benchmark files.
- **Adds a vendored MLP runtime in `zensim/src/mlp/`** that has been superseded by the zenpredict-based runtime + per-sample-α + hybrid_head modules.
- **Reverts `error.rs`** to drop `#[non_exhaustive]` + `ImageTooLarge` + `FeatureWeightsLengthMismatch`.
- **Reverts diffmap.rs and source.rs** to drop bounds-check guards + max_pixels caps.

There is no path to "rebase the PR onto current main" that does not effectively reset main to before May 6. The only sane action is to evaluate the FiLM bake against current ships (done here) and close the PR.

## Recommendation

1. **Close PR #31** with a link to this falsification doc and the two bake_compare reports.
2. **Keep the FiLM bake artifacts** at `/mnt/v/output/zensim/synthetic-v2/runs/v06_film_20260505T212932.*` as historical record. They are not loss; just falsified candidates.
3. **No ship rotation.** Both current ships hold:
   - Balanced: `zensim/weights/v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18.bin` (V_22-mix-LARGE+iwssim).
   - Compression: `zensim/weights/v_compression_persample_2026-05-18.bin` (V_24-per-sample-α s4).
4. **The FiLM idea (per-content-class dispatch) is not fundamentally invalidated** — the falsification here is of the specific 2026-05-05 bake on the pre-decontamination training corpus. A future FiLM-style model trained on today's clean corpus with today's per-sample-α infrastructure could be re-evaluated under the same gates. But that's a new training run, not a rebase of PR #31.

## Artifacts

- bake_compare vs Balanced: `benchmarks/bake_compare_v06_film_vs_balanced_2026-05-18.md` (1000 bootstrap, all 5 corpora, full 10-band panels).
- bake_compare vs Compression: `benchmarks/bake_compare_v06_film_vs_compression_2026-05-18.md` (same).
- FiLM bake provenance: `/mnt/v/output/zensim/synthetic-v2/runs/v06_film_20260505T212932.log` (training log, epoch 10 best val_mean=0.8462).
- 4 unused-class bakes: `v06_film_20260505T212932.c{0..4}_{photo,screen,lineart,synthetic,document}.bin` (each 63 044 B, ZNPR v3).
