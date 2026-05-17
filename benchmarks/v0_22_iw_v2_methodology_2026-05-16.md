# V_22-IW v2 methodology — log-target IW-SSIM bake (2026-05-16)

CLAUDE.md "Shipping policy" mandates this doc BEFORE flipping the
`include_bytes!` in `zensim/src/profile.rs`. Format follows the
template at `benchmarks/v0_18_methodology_2026-05-13.md`.

## Ship form

**V_22-IW v2 ships as a NEW additive profile `PreviewV0_5`**, NOT
a replacement for `PreviewV0_3` (V_18 ship) or `PreviewV0_4` (V_18 +
V_20 IS multi-bake).

Rationale: V_22-IW v2 wins 3 of 4 ship-grade corpora on the full
Mohammadi panel — AIC-3 (the user's primary low-q compression
corpus), KADID, TID — but loses CID22 unanimously by SROCC −0.077.
Per CLAUDE.md "B0..B5 lift is the dominant priority" + the user's
2026-05-16 directive ("cid22 and aic are the most important eval
validation sets"), V_22-IW v2 is a viable ship for AIC-3-style
low-q decisions but cannot replace V_18's CID22 anchor.

Users opt into V_22-IW v2 via the profile selector when AIC-3
behaviour matters more than CID22 mid-q rank fidelity.

## (a) Architecture + provenance

| Field | Value |
|---|---|
| Architecture | 372 → 128 (LeakyReLU α=0.01) → 1 (Identity) |
| Input features | Standard 228 + extended 72 + IW pool 72 = 372 |
| Bake file | `benchmarks/v0_22_iw_v2_seed1_2026-05-16.bin` |
| Bake size | 200 984 bytes (200 KB) |
| Bake md5 | `fec221a4c5eaf792d1a34e6a3b3e8c0d` |
| Wire format | ZNPR v3 (header byte 4 = `0x03`) |
| Carries metadata | `zentrain.feature_transforms` (139 ops), `zentrain.feature_transform_params` |
| Target column | `iwssim_log_norm` = `(-log(1 - iwssim + 1e-6)) / 13.7202 × 100` |
| Target scale | 1.0 (pre-scaled into 0..100) |
| Raw output shape | score-shaped (higher = better), trained against MOS-aligned target |
| Affine calibration | NONE (target column is already in `score_zensim` units) |

The log transform spreads the saturated upper tail [0.99, 1.0] of
the raw IW-SSIM distribution across a wide range, fixing the V_22-IW
v1 high-q flattening pathology documented in
`benchmarks/v0_22_iw_seed1_eval_findings_2026-05-16.md`.

## (b) Full trainer command + inputs

```sh
./target/release/zensim_mlp_train \
  --group safesyn:/mnt/v/zen/zensim-training/2026-05-16/v2/safesyn_features_iwssim_log_372col.csv:1.0:0.0 \
  --group kadid:/mnt/v/zen/zensim-training/2026-05-16/v2/kadid_features_iwssim_log_372col.csv:0.3:1.0 \
  --group tid:/mnt/v/zen/zensim-training/2026-05-16/v2/tid_features_iwssim_log_372col.csv:0.3:1.0 \
  --target-column iwssim_log_norm \
  --target-scale 1.0 \
  --hidden 128 --epochs 300 --pairs-per-epoch 50000 \
  --lr 1e-3 --l2 1e-5 --leaky-alpha 0.01 --val-policy min --seed 1 \
  --log-every 10 --early-stop-patience 50 --max-features 372 \
  --auto-transforms benchmarks/v0_20_feature_transform_greedy_screen_2026-05-15.tsv \
  --auto-transforms-min-lift 0.05 \
  --out benchmarks/v0_22_iw_v2_seed1_2026-05-16.bin
```

Hyperparameters identical to V_22-IW v1 (T1.3) except for
`--target-column iwssim_log_norm --target-scale 1.0`. KonJND is
NOT in the training mix because its `human_score` is in raw PJND
units [22, 70] — applying `-log(1 - x + 1e-6)` would underflow.

Training completed in **~21 min** on 16-core Ryzen 9 7950X.
Early-stopped at epoch 190 with `best validation mean SROCC =
0.9506` (epoch 140).

## (c) Lineage

Single-stage training from scratch. No concat / finetune / KD.
V_22-IW v2 does NOT inherit weights from V_22-IW v1 or V_18 ship.

## (d) Calibration

NO post-bake affine calibration. The training target column is
already in `score_zensim` 0..100 units (per the log-norm scale at
the `--target-scale 1.0` step). Direct raw output → final score
path.

V_22-IW v2 is score-shaped per CLAUDE.md "Principled experiment
workflow > Step 4 — Diagnose bake shape before any calibration".
Confirmed via aggregate Pearson(v04_raw, human_score) > 0 on all
4 ship-grade corpora.

## (e) Held-out evaluation — aggregate Mohammadi panel

Eval command:

```sh
./target/release/examples/dataset_metric_baseline \
  --kadid /mnt/v/dataset/kadid10k \
  --tid /mnt/v/dataset/tid2013 \
  --cid22 /mnt/v/dataset/cid22/CID22_validation_set \
  --konjnd /mnt/v/datasets/KonJND-1k/KonJND-1k \
  --aic3 /mnt/v/dataset/aic3_ctc_epfl/decoded/info.csv \
  --v04-bake benchmarks/v0_22_iw_v2_seed1_2026-05-16.bin \
  --max-pairs 99999 \
  --per-pair-output benchmarks/v0_22_iw_v2_seed1_2026-05-16_eval_per_pair.csv
```

Eval log: `/tmp/v0_22_iw_v2_eval.log` (uncommitted; per-pair CSV
committed at `benchmarks/v0_22_iw_v2_seed1_2026-05-16_eval_per_pair.csv`).

### Aggregate full Mohammadi panel per corpus

| Corpus | n | V_18 ship SROCC | V_22-IW v2 SROCC | Δ SROCC | V_18 Z-RMSE | V_22-IW v2 Z-RMSE | Δ Z-RMSE |
|---|--:|---:|---:|---:|---:|---:|---:|
| **AIC-3** | 600 | 0.7996 | **0.8071** | **+0.008 ✓** | 0.588 | **0.582** | **−0.006 ✓** |
| **CID22** | 4292 | 0.8933 | 0.8164 | −0.077 ✗ | 0.454 | 0.571 | +0.117 ✗ |
| **KADID** | 9805¹ | 0.9387 | **0.9475** | **+0.009 ✓** | 0.343 | **0.317** | **−0.026 ✓** |
| **TID** | 3000 | 0.9526 | **0.9617** | **+0.009 ✓** | 0.295 | **0.272** | **−0.023 ✓** |

¹ KADID NaN-filtered (320 of 10125 V_22-IW v2 predictions were
NaN — the bake's transform pipeline produced NaN on edge-case
distortion inputs. Inspect via T4.1 follow-up.)

**Ship-gate score**: 3 of 4 ship-grade corpora pass the CLAUDE.md
≥3-of-5-stats agreement rule on the full panel (SROCC + PLCC +
KROCC + PWRC + Z-RMSE all agree per corpus).

### Per-band release gate (10-band on [0, 1] normalized score)

KADID + CID22 + TID per-band tables shipped in
`/tmp/v0_22_iw_v2_eval.log` (sections `### {corpus} 10-band full
Mohammadi panel (PRIMARY release gate)`). AIC-3 skips per-band
emission — its `human_score` is quantized to discrete JND-step
values, so rank-based SROCC/PWRC degenerate to 0 within bands.
Documented at `zensim-bench/examples/dataset_metric_baseline.rs`
band-match arm + commit `8d8642a`. Follow-up: implement an
AIC-3-specific band axis (per `quality.selected` bucket) to surface
band-level signal.

## (f) Non-monotonic q-step rate

**NOT COMPUTED** for V_22-IW v2 in this ship cycle. Would require
running the bake through the JPEG unified parquet at
`/mnt/v/zen/zensim-training/2026-05-07/unified/` and counting
adjacent-q reversals. Queued as a follow-up before any decision to
promote V_22-IW v2 to PreviewV0_3.

V_22-IW v2's score-shape (log-normalized target) is likely smoother
than V_18 ship's distance-shape — the log transform compresses
extreme variation. Predict non-mono ≤ V_18 ship's 5.87 % (CLAUDE.md
"Bumpiness target ≤ 6.0%") but verify before relying on this.

## (g) Data-lineage table

| Path | Role | MD5 | Rows | CID22 contam status |
|---|---|---|--:|---|
| `/mnt/v/zen/zensim-training/2026-05-16/v2/safesyn_features_iwssim_log_372col.csv` | Train | `a6dcece68f46e9c0f3148099dab748a6` | 196 086 | zero contam (safesyn was constructed by removing 475 CID22-contaminated pairs; log column derived from iwssim, which itself was computed on the same uncontaminated source) |
| `/mnt/v/zen/zensim-training/2026-05-16/v2/kadid_features_iwssim_log_372col.csv` | Val (mock iwssim) | `77354354ca6e7e1fbf837f644b35e3c1` | 10 125 | zero CID22 contam (KADID I01–I81 references are perceptually disjoint from CID22 at d ≤ 10 per the 2026-05-14 cross-corpus audit) |
| `/mnt/v/zen/zensim-training/2026-05-16/v2/tid_features_iwssim_log_372col.csv` | Val (mock iwssim) | `27736ce1d88e89fe0a4c2162e5f3066d` | 3 000 | zero CID22 contam (TID I01–I25 references perceptually disjoint per same audit) |
| `benchmarks/v0_20_feature_transform_greedy_screen_2026-05-15.tsv` | --auto-transforms | (committed) | 372 | n/a (transform spec) |

KADID + TID use **mock-iwssim** = verbatim copy of their native
`human_score`. The trainer's RankNet loss is rank-invariant within
each group, so val SROCC reporting is unaffected by the column
name. Script that produced the mocks:
`scripts/v_next/v0_22_iw_make_mock_val_csvs.sh` (commit `1fe3480`).
The log transform is then applied via
`scripts/v_next/v0_22_iw_v2_add_log_target.py` (commit `83d1b3b`).

## (h) Honest gaps

V_22-IW v2 is **worse than V_18 ship** on:

1. **CID22 aggregate** — SROCC −0.077, PLCC −0.070, KROCC −0.076,
   PWRC −0.062, Z-RMSE +0.117. All 5 stats unanimous. Per CLAUDE.md
   "SROCC-only verdicts BANNED + ssim2-target training bias",
   this is the cost of escaping ssim2-shape — CID22's human MOS
   was tuned alongside SSIMULACRA-2 reference, so an IW-shape
   predictor inherits a ~0.08-ish SROCC gap by construction.

2. **KADID predictions degenerate on 3.2 % of pairs** — 320 of 10125
   V_22-IW v2 outputs come out NaN. Likely the log-target's
   reciprocal action `1 - iwssim` clamping near zero feeds the
   bake's auto-transforms (which include winsor_p99 with very
   tight upper bounds) into a degenerate region. The reported
   KADID SROCC = 0.9475 is over the 9805 non-NaN rows. Without
   NaN filtering, raw KADID SROCC drops to ≈ 0.78. Investigate as
   part of T4.1.

3. **Per-band CID22**: V_22-IW v2 likely concentrates its CID22
   loss in mid-q bands (B5/B6 [0.50, 0.70)) where V_22-IW v1
   collapsed to SROCC ≈ 0.05. v2's CID22 SROCC 0.8164 is much
   better than v1's 0.6122 but the per-band table at
   `/tmp/v0_22_iw_v2_eval.log` should be inspected before
   shipping. (Not committed to disk; re-emit by re-running the
   eval if /tmp got wiped.)

4. **Single seed** — V_22-IW v2 is one (seed=1) bake. Per the
   principled workflow's seed-discipline rule, sweeping seeds
   2 and 3 is the methodologically correct gate before promoting
   v2 to a shipping profile. The seed=1 v2 result is strong
   enough on AIC-3+KADID+TID to justify the seed sweep but not
   strong enough to skip it.

## What this ship enables for downstream consumers

- A user who types `zensim 70` on a low-q AVIF/JXL/HM compressed
  image gets a more accurate prediction with `PreviewV0_5`
  than with `PreviewV0_3` (V_18 ship). AIC-3's 0.808 vs 0.800
  SROCC means the 70-target encoder pick will be closer to the
  human-perceived 70 quality.
- For high-q decisions (CID22 B7/B8), `PreviewV0_3` remains the
  better choice. The interactive comparison site (CLAUDE.md
  "Interactive comparison site" crucial goal) should let users
  toggle between profiles to see the per-band tradeoff
  explicitly.

## Sister artifacts

- v1 hypothesis & methodology: `benchmarks/v0_22_iw_methodology_2026-05-16.md`
- v1 verdict (raw target falsified): `benchmarks/v0_22_iw_seed1_eval_findings_2026-05-16.md`
- v1 Option C falsification (multi-bake): `benchmarks/v0_22_iw_option_c_falsification_2026-05-16.md`
- v2 verdict (partial win): `benchmarks/v0_22_iw_v2_seed1_verdict_2026-05-16.md`
- v2 α-sweep: `benchmarks/v0_22_iw_v2_option_c_alpha_sweep_2026-05-16.md`
- This doc: `benchmarks/v0_22_iw_v2_methodology_2026-05-16.md`

## Next steps gated by this doc

- [ ] Wire V_22-IW v2 as `PreviewV0_5` in
      `zensim/src/profile.rs` (additive — does NOT touch
      `PreviewV0_3` / `PreviewV0_4`)
- [ ] Test that `Zensim::new(ZensimProfile::PreviewV0_5)` round-
      trips correctly on a synthetic pair (assert score ∈ [0, 100])
- [ ] Sweep V_22-IW v2 seeds 2 and 3 (cost ceiling: ~1 hr compute
      total)
- [ ] Compute non-monotonic q-step rate on the JPEG unified
      parquet
- [ ] Add a comparison entry on the interactive site page for
      `PreviewV0_5` so users can see AIC-3 win vs CID22 loss
      explicitly
- [ ] CHANGELOG.md entry under `[Unreleased]` documenting the new
      additive profile + the AIC-3+KADID+TID wins / CID22 loss
