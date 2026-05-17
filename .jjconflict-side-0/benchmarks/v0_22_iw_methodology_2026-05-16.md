# V_22-IW methodology — IW-SSIM-target training (2026-05-16)

**Status**: hypothesis-stage. Feature extraction in flight at the
time of writing (PID 826668, started 2026-05-16T19:58Z, ETA
2026-05-16T20:35Z). Trainer kickoff queued behind this doc.

This is **step 1 of the principled experiment workflow** documented
at `CLAUDE.md > Principled experiment workflow for V_X bakes`. Write
the hypothesis BEFORE opening a trainer. If we can't write these four
sentences first, we're not ready to train.

## Hypothesis

The V_18 ship and every prior V_X bake was trained against
`cpu_ssimulacra2` (the SSIMULACRA-2-derived score per pair). Any
SROCC evaluation that uses an ssim2-aware MOS — like CID22 human
MOS, which was tuned alongside SSIMULACRA-2 reference — favors
ssim2-shaped predictions by construction. The IW-SSIM falsification
in `benchmarks/falsification_reeval_results_2026-05-15.md` is
**structurally rigged**: V_20a IW won TID on the full Mohammadi
panel (TID SROCC 0.9710, PWRC 0.9822, Z-RMSE 0.231 — best result of
any bake) but lost CID22 SROCC (0.4632) because the IW shape doesn't
match the CID22 MOS's ssim2-aware shape.

**Hypothesis**: a bake trained against **IW-SSIM** as the regression
target will produce an IW-SSIM-shaped output surface. Against the
held-out CID22 human MOS the bake will:

- Lose SROCC vs V_18 ship — the ssim2-shape advantage flips
- Win PWRC + Z-RMSE — the panel stats that capture calibrated
  error on important pairs
- Be neutral on TID + KADID across the full Mohammadi panel (those
  corpora are less ssim2-biased than CID22)

If at least 3 of 5 stats (SROCC, PLCC, KROCC, PWRC, Z-RMSE) agree
that V_22-IW wins CID22 — OR if the full panel agrees that V_22-IW
wins TID without losing CID22 on more than 1 stat — the
hypothesis is **confirmed**: the V_18 ship's ssim2-target training
bias is real, and the IW-target shape is the better default for
compression-product decisions.

**Headline claim threshold**: Δ ≥ +0.005 SROCC on TID full
Mohammadi panel vs V_18 ship, AND Δ ≤ -0.020 SROCC on CID22 (the
expected loss from shape change).

## Falsification

The hypothesis is **falsified** if:

1. The trainer produces a bake whose CID22 SROCC drops by > 0.030
   AND its PWRC + Z-RMSE on CID22 also drop. This would mean the IW
   target is worse across the entire panel, not just on the
   ssim2-shape gate.
2. The trainer produces a bake whose TID SROCC drops vs V_18 ship.
   The IW target should at minimum match V_18 on TID; if it loses
   there too, the training signal is broken.
3. Multiple seeds (1, 2, 3) all hit the falsification threshold.
   A single-seed loss is noise; three is signal.

## Cost ceiling

- 1× safesyn feature extraction (~30 min, **in flight**)
- 1× V_22-IW seed=1 fine-tune (V_18 architecture, 228 → 128 → 1,
  TV-regularized — expected ~60 min on this 16-core box)
- 1× full Mohammadi panel eval on KADID + TID + CID22 + AIC-3
  (~10 min)
- Budget: 2 hr total to seed=1 verdict. If hypothesis holds at
  seed=1, sweep seeds 2 and 3 (another 2 hr each).

If after seed=1 the falsification thresholds are hit on all three
gates, **stop**. Don't sweep more seeds hoping for noise to flip;
that's p-hacking.

## Ship form

If the hypothesis confirms (≥ 3 stats winning OR TID full-panel win
without CID22 collapse):

- **Option A** (lighter): keep V_18 as PreviewV0_3 ship; add V_22-IW
  as a new `PreviewV0_5` profile alongside. Users opt into the
  IW-target by selecting the profile. This is the conservative
  ship: no behavior change for existing consumers.

- **Option B** (replace): swap V_18 with V_22-IW as the new
  PreviewV0_3 ship. Requires affine calibration of V_22-IW raw
  output to MCOS 0-100 (per CLAUDE.md "Step 4 — Diagnose bake
  shape before any calibration"). More disruptive but lets the
  IW-target shape become the new default.

- **Option C** (multi-bake): replace PreviewV0_4's V_20 IS
  secondary with V_22-IW as the secondary; keep V_18 as primary.
  Maintains the V_18 SROCC anchor while pulling in IW-target
  shape via the multi-bake mix.

Decision deferred to verdict time. The methodology doc finalizes
which option ships AFTER eval — not before. This is per
CLAUDE.md step 2: "Decide the reporting panel upfront; decide the
ship form upfront." Done.

## Training corpus

**Source TSV**: `/mnt/v/zen/zensim-training/2026-05-16/safesyn_with_iwssim.csv`

- 196 086 pairs from the safesyn (safe-synthetic) corpus
- Columns: `source_path, decoded_path, codec, quality, width, height,
  gpu_ssimulacra2, gpu_butteraugli, cpu_ssimulacra2,
  cpu_butteraugli, size_bytes, run_id, dssim, iwssim`
- `iwssim` joined 1:1 from
  `/mnt/v/output/zensim/synthetic-v2/iwssim_targets_safesyn_2026-05-16.parquet`
  (computed via `piq.information_weighted_ssim` on Quadro RTX 8000,
  3.7 hr GPU, 0 errors)
- Lineage: zensim repo `scripts/v_next/merge_iwssim_into_safesyn.py`
  + `scripts/v_next/compute_iwssim_on_safesyn.py`
- CID22 contamination: **zero** (safesyn was constructed by
  removing the 475 CID22 validation pairs from
  `training_concordant.csv`). Confirmed via CLAUDE.md "Safe
  synthetic dataset" + manual schema inspection.

**Features CSV** (in flight at write time):
`/mnt/v/zen/zensim-training/2026-05-16/safesyn_features_iwssim_372col.csv`

- Generated via
  `cargo run --release -p zensim-bench --features training --example extract_features_372col -- --corpus safesyn --path … --out …`
- Schema: `ref_basename, human_score, iwssim, f0..f371`
- `human_score` = `cpu_ssimulacra2 / 100` with `gpu_ssimulacra2`
  fallback for the 95k rows that lack cpu ssim2 (zenavif / zenjxl
  codec families were scored GPU-only). The trainer's
  `--target-column human_score` invocation reproduces V_18-style
  training; `--target-column iwssim` is the V_22-IW path.
- 372-feature extraction = `extended_features = true` +
  `compute_iw_features = true` (4 scales × 3 channels × 31 features
  per scale).

**KADID / TID / KonJND**: the existing 2026-05-15-full-features
CSVs at `/mnt/v/zen/zensim-training/2026-05-15-full-features/`
have `human_score` only (no iwssim column). For V_22-IW training,
those corpora can either:

- Be trained against `human_score` alongside safesyn's `iwssim`
  (mixed-target training — each corpus gets its native target)
- Or be re-extracted with iwssim columns (more work, deferred)

Default plan: train ONLY on safesyn with `--target-column iwssim`,
no KADID/TID in the training mix. Reserves KADID/TID/CID22 for
held-out validation. This is the cleanest baseline; mixed-target
training can come later if seed=1 underperforms.

## Trainer command (seed=1, queued)

```sh
zensim_mlp_train \
  --group safesyn:/mnt/v/zen/zensim-training/2026-05-16/safesyn_features_iwssim_372col.csv:1.0:0.0 \
  --target-column iwssim \
  --target-scale 100.0 \
  --hidden 128 \
  --epochs 300 \
  --pairs-per-epoch 50000 \
  --lr 1e-3 \
  --l2 1e-5 \
  --leaky-alpha 0.01 \
  --val-policy min \
  --seed 1 \
  --log-every 10 \
  --early-stop-patience 50 \
  --max-features 372 \
  --auto-transforms benchmarks/v0_20_feature_transform_greedy_screen_2026-05-15.tsv \
  --auto-transforms-min-lift 0.05 \
  --out benchmarks/v0_22_iw_seed1_$(date -u +%Y-%m-%d).bin
```

Rationale per flag:

- `--max-features 372`: V_22-IW uses the full extended + IW feature
  set; runtime ProfileParams sets `extended_features = true` AND
  `compute_iw_features = true` to match.
- `--auto-transforms` with the same 98-transform set V_20 IS used:
  per CLAUDE.md V_20 learnings, input shaping is the
  least-controversial training-side win across the full Mohammadi
  panel. Keep it.
- `--val-policy min`: cosine annealing schedule + early stop, same
  shape as V_18 ship recipe.
- `--seed 1`: cheap signal per CLAUDE.md step 3 "Seed=1 first as
  cheap signal."

## Eval command (queued behind training)

```sh
dataset_metric_baseline \
  --kadid /mnt/v/dataset/kadid10k \
  --tid /mnt/v/dataset/tid2013 \
  --cid22 /mnt/v/dataset/cid22 \
  --aic3 /mnt/v/dataset/aic3_ctc_epfl/decoded/info.csv \
  --bake benchmarks/v0_22_iw_seed1_<date>.bin \
  --per-pair-output benchmarks/v0_22_iw_seed1_<date>_per_pair.csv \
  > benchmarks/v0_22_iw_seed1_<date>_eval.log 2>&1
```

The eval harness's **T3.1 upgrade** (commit `76360ae`, 2026-05-16)
now emits the **full Mohammadi panel per band** on both the 10-band
PRIMARY table and the legacy 4-band CID22 cut table. Each row reports
SROCC + PLCC + KROCC + OR + PWRC + Z-RMSE + MAE per (band, metric).
This is the load-bearing format for the V_22-IW verdict.

## Runtime forward path

V_22-IW will produce a ZNPR v3 bake. Runtime ABI:

- The bake has `n_inputs = 372` (or 376 with size-axes; the trainer
  inserts size-axes when `--mlp-size-axes` is passed; default is
  off so 372).
- `apply_mlp_scoring` dispatches to `predict_transformed` when the
  bake carries `zentrain.feature_transforms` metadata (the
  `--auto-transforms` invocation populates this). Verified by
  `model.has_nontrivial_feature_transforms()` check on the load
  path.
- Soft-clamp policy: V_22-IW raw output is MOS-shaped (target =
  iwssim ∈ [0, 1] × 100). After training the bake's raw output
  lands in approximately 0..100 and the soft-clamp can stay off
  (legacy hard clamp suffices for the single-bake forward path).
  If V_22-IW ships as a multi-bake secondary (Option C above),
  toggle `soft_clamp_score = true` on the wrapping ProfileParams
  to match PreviewV0_4's policy.

## What this experiment does NOT do

- It does NOT train on CID22 human MOS. CID22 is validation-only
  per `CLAUDE.md > CID22 is VALIDATION-ONLY`.
- It does NOT use SROCC alone as the verdict gate. Full Mohammadi
  panel per band, per `CLAUDE.md > SROCC-only verdicts BANNED`.
- It does NOT produce ZNPR v2 bytes. Output is v3 via
  `zenpredict-bake`, per `CLAUDE.md > ZNPR v2 PROHIBITED`.
- It does NOT inspect CID22 results mid-training. CID22 is opened
  ONCE at decision time, per `CLAUDE.md > Step 2` "CID22 is opened
  LAST".

## Pre-flight checklist (gating the trainer run)

- [ ] Feature extraction completes successfully (196 086 rows at
      372 features each)
- [ ] Spot-check the features CSV: `head -1` shows
      `ref_basename,human_score,iwssim,f0,f1,...,f371`
- [ ] Iwssim column range: p5..p95 in approximately [0.7, 0.999]
      (matches the parquet's distribution per
      `scripts/v_next/merge_iwssim_into_safesyn.py` smoke output)
- [ ] No row has empty iwssim (gpu_ssim2 fallback covers the 95k
      cpu-empty cases)
- [ ] Trainer invocation per "Trainer command" above; binary built
      from current HEAD
- [ ] Methodology doc committed BEFORE the trainer starts (this
      file)

When all six boxes are checked: kick the trainer. Output bake +
eval log go under `benchmarks/v0_22_iw_seed1_*`.
