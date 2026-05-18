# EXP-BALANCED-TILT falsification (2026-05-18)

**Verdict:** All 4 cells FAIL both shipping gates at seed=3 per § A.9 decisive rule (1000-bootstrap). No 5-seed CI follow-up justified. The "boost kadid_w / tid_w / konjnd_w on per-sample-α architecture" direction does not produce a shippable bake for either trail.

## Hypothesis

Train per-sample-α head (the architecture used by the current compression-trail ship V_24-per-sample-α s4) with kadid_w / tid_w / konjnd_w boosted above s4's defaults (kadid_w=0.3, tid_w=0.3, konjnd_w=0.02). Goal: match the balanced-trail KADID/TID/KonJND lead while keeping the per-sample-α CID22 + AIC-3 advantage.

## Falsification criteria (defined ex ante)

A cell falsifies if EITHER:
- It fails the balanced-trail gate (A>>B on ≥1 corpus and not B>>A on any of 5 per § A.9), AND
- It fails the compression-trail gate (decisive A>>B on ≥1 of {CID22, AIC-3}, not decisive B>>A on the other compression corpus, mean SROCC regression on {KADID, TID, KonJND} no worse than −0.10).

## Sweep grid

Recipe (constant across cells): h=128, 300 epochs, `--per-sample-alpha-head`, `--target-column mix_cv40_iw60`, NiN 0.1, mini-batch 256, PWRC 5.0, `--val-policy min`, seed=3.

Trainer: `/home/lilith/work/zen/zensim--ex2-persample-alpha/target/release/zensim_mlp_train`
Data root: `/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/`
Bake output: `/mnt/v/zen/zensim-eval/exp_balanced_tilt_2026-05-18/cell*_seed3.bin`

| Cell | safesyn_w | kadid_w | tid_w | konjnd_w | large_w |
|---|--:|--:|--:|--:|--:|
| cell0 small      | 1.0 | 0.5 | 0.5 | 0.05 | 0.5 |
| cell1 moderate   | 1.0 | 0.8 | 0.8 | 0.10 | 0.5 |
| cell2 heavy      | 1.0 | 1.0 | 1.0 | 0.10 | 0.3 |
| cell3 no_large   | 1.0 | 0.8 | 0.8 | 0.10 | 0.0 |

For reference, the per-sample-α s4 production recipe is (1.0, 0.3, 0.3, 0.02, 0.5).

## Aggregate Mohammadi panel (seed=3, eval bake_verdict)

Eval binary: `/home/lilith/work/zen/zensim--ex2-persample-alpha/target/release/bake_verdict` (the main `zensim/target/release/bake_verdict` LACKS per-sample-α dispatch — using it on a per-sample-α bake gives SROCC ~0.17 on CID22, which is the inverted-prediction failure mode).

| Bake | n_inputs | CID22 SROCC | KADID SROCC | TID SROCC | KonJND SROCC | AIC-3 SROCC |
|---|--:|---:|---:|---:|---:|---:|
| **Balanced ship** V_22-mix-LARGE+iwssim s3 (vanilla MLP) | 300 | 0.8324 | **0.9677** | **0.9729** | 0.8927 | 0.7845 |
| **Compression ship** V_24-per-sample-α s4 (per-sample-α dispatch) | 300 | **0.8641** | 0.9316 | 0.8893 | 0.8080 | **0.8183** |
| cell0 small      | 300 | 0.8249 | 0.9345 | 0.8898 | 0.9234 | 0.8144 |
| cell1 moderate   | 300 | 0.8159 | 0.9359 | 0.8896 | 0.9567 | 0.8051 |
| cell2 heavy      | 300 | 0.8112 | 0.9379 | 0.8901 | 0.9532 | 0.8070 |
| cell3 no_large   | 300 | 0.7686 | 0.9385 | 0.8906 | **0.9661** | 0.8056 |

Baseline ssim2 / cvvdp / iwssim controls (from `benchmarks/baseline_panels_2026-05-18.md`):

| Metric | CID22 | KADID | TID | KonJND | AIC-3 (n=600 PTC superset) |
|---|---:|---:|---:|---:|---:|
| ssim2 (fast-ssim2) | 0.8895 | 0.8133 | 0.8460 | n/a | 0.7965 |
| cvvdp              | 0.8214 | 0.8339 | 0.8531 | 0.0482 | 0.7918 |
| iwssim             | 0.7836 | 0.8498 | 0.7794 | 0.1859 | 0.7735 |

## bake_compare decisive verdicts vs Balanced ship (§ A.9, 1000-bootstrap)

| Cell | CID22 | KADID | TID | KonJND | AIC-3 | Balanced gate |
|---|---|---|---|---|---|---|
| cell0 small      | promising | **B>>A** | **B>>A** | A>>B | A>>B | **FAIL** (decisive B>>A on KADID + TID) |
| cell1 moderate   | **B>>A** | **B>>A** | **B>>A** | A>>B | A>>B | **FAIL** (decisive B>>A on CID22 + KADID + TID) |
| cell2 heavy      | **B>>A** | **B>>A** | **B>>A** | A>>B | A>>B | **FAIL** (decisive B>>A on CID22 + KADID + TID) |
| cell3 no_large   | **B>>A** | **B>>A** | **B>>A** | A>>B | A>>B | **FAIL** (decisive B>>A on CID22 + KADID + TID) |

## bake_compare decisive verdicts vs Compression ship (§ A.9, 1000-bootstrap)

| Cell | CID22 | KADID | TID | KonJND | AIC-3 | Compression gate |
|---|---|---|---|---|---|---|
| cell0 small      | **B>>A** | A>>B | tied | A>>B | tied | **FAIL** (B>>A on CID22 with no decisive AIC-3 win) |
| cell1 moderate   | **B>>A** | A>>B | tied | A>>B | **B>>A** | **FAIL** (decisive B>>A on BOTH CID22 + AIC-3) |
| cell2 heavy      | **B>>A** | A>>B | tied | A>>B | **B>>A** | **FAIL** (decisive B>>A on BOTH CID22 + AIC-3) |
| cell3 no_large   | **B>>A** | A>>B | A>>B | A>>B | **B>>A** | **FAIL** (decisive B>>A on BOTH CID22 + AIC-3) |

## Falsification

**All 4 cells FAIL both gates.** The systematic failure mode across all 4 cells:

1. **Wins KonJND decisively** (h_SROCC 26–43, ΔSROCC +0.03 to +0.07 vs Balanced; ΔSROCC +0.11 to +0.16 vs Compression). Driver: konjnd_w (0.05 → 0.9234 KonJND; 0.10 → 0.95-0.97 KonJND vs Balanced 0.8927).
2. **Wins AIC-3 decisively vs Balanced** (h_SROCC 14–24, ΔSROCC +0.02 to +0.03). But **LOSES AIC-3 decisively vs Compression** (h_SROCC −24 to −45, ΔSROCC −0.01 to −0.03).
3. **Loses KADID + TID decisively vs Balanced** (h_SROCC −52 to −85 on KADID; −52 to −53 on TID; ΔSROCC −0.03 on KADID and −0.083 on TID — consistent across all cells).
4. **Loses CID22 decisively vs Compression** (h_SROCC −80 to −106, ΔSROCC −0.04 to −0.10).
5. **Cell3 (no LARGE) tanks CID22 by an extra 0.04** vs the other cells — confirms `cvvdp_iwssim_large` (73,300 pairs) is load-bearing for CID22 generalization.

The pattern is structural, not a seed-luck issue. 5-seed CI on any single cell would not flip the gate decision.

## Why the direction fails

- **Training-target shape mismatch.** `mix_cv40_iw60` is a cvvdp+iwssim-derived target (calibration optimized for the LARGE group's distribution). Boosting groups with `val_w=1.0` carrying native DMOS / MOS / PJND shapes (kadid_w, tid_w, konjnd_w) pulls the model toward heterogeneous calibrations that the per-sample-α head's mix-gate can't reconcile across all 5 corpora simultaneously.
- **The current Balanced ship (V_22-mix-LARGE+iwssim s3)** is a vanilla 300→128→1 MLP trained WITHOUT KADID/TID/KonJND groups (synth + LARGE only) — and it dominates KADID + TID by 0.03-0.08 SROCC over anything we trained here. The per-sample-α architecture is not the bottleneck for KADID/TID; the bottleneck is the training-group composition.
- **The compression-trail ship (V_24-per-sample-α s4)** uses kw=0.02 (not kw=0.05+) — that smaller weight is what lets the per-sample-α head focus on the compression-anchored CID22 + AIC-3 regions without contaminating its KADID/TID calibration. Boosting kw past 0.02 trades CID22 + AIC-3 SROCC for KonJND SROCC linearly, which doesn't pass either gate.

## What the data does NOT rule out

- **KonJND-densify + balanced-trail base recipe** (no per-sample-α boost of KADID/TID; just add the 20,160-pair densified KonJND group to the Balanced training recipe at small weight). The +0.06 KonJND lift from per-sample-α here is suggestive that KonJND data is informative; the architecture being explored is what failed, not the data.
- **Different target column.** `mix_cv25_iw75` or `mix_cv75_iw25` or a synth-DMOS-blended target may behave differently. Not explored here.
- **The per-sample-α architecture itself** still defends the compression-trail ship at the s4 configuration — only the weight-tilt direction is falsified.

## Outputs (all under `/mnt/v/zen/zensim-eval/exp_balanced_tilt_2026-05-18/`)

- `cell{0,1,2,3}_*_seed3.bin` — trained bakes (~224 KB each, ZNPR v3, per-sample-α metadata)
- `cell{0,1,2,3}_*_seed3.log` — full training logs
- `verdicts/cell{0,1,2,3}_*_seed3_verdict.md` — bake_verdict full Mohammadi panel per cell
- `compares/cell{0,1,2,3}_*_seed3_vs_{balanced,compression}.md` — bake_compare § A.9 reports (1000-bootstrap)

## Lineage

- Workspace: `/home/lilith/work/zen/zensim--exp-balanced-tilt/` (jj change `qkuwxzkk` / commit `f7a0acc`)
- Launcher scripts: `scripts/v_next/run_balanced_tilt_seed.sh` + `scripts/v_next/run_balanced_tilt_compare.sh`
- Trainer build: `/home/lilith/work/zen/zensim--ex2-persample-alpha/target/release/zensim_mlp_train` (commit unknown — pre-existing)
- Eval binaries: `/home/lilith/work/zen/zensim--ex2-persample-alpha/target/release/bake_{verdict,compare}` (same workspace, same date)

## Status of trails after this falsification

- **Balanced trail**: SHIP unchanged — V_22-mix-LARGE+iwssim s3 packed (CID22 0.8324, KADID 0.9677, TID 0.9729, KonJND 0.8927, AIC-3 0.7845).
- **Compression trail**: SHIP unchanged — V_24-per-sample-α s4 packed (CID22 0.8641, KADID 0.9316, TID 0.8893, KonJND 0.8080, AIC-3 0.8183).
