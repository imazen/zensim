# SROCC-gated falsification re-eval triage (2026-05-15)

Catalogue of every SROCC-gated falsification / no-ship from the
recovery cycle (2026-05-11 to 2026-05-13) and V_20 cycle (2026-05-15).
Each entry is classified by **re-eval priority** against the
Mohammadi 2025 full-stat panel (SROCC + PLCC + KROCC + OR + PWRC +
Z-RMSE) AND whether retraining with zenpredict feature_transforms
might recover the falsification.

## Hypothesis

SROCC-gated falsifications may hide PWRC / Z-RMSE / per-band / OR
wins that the full panel reveals. Additionally, pre-V_20
falsifications might be re-rankable if retrained with feature
transforms (zenpredict input shaping, available 2026-05-15+).

**Falsification of this meta-hypothesis**: If every falsified bake is
ALSO losing on PWRC / Z-RMSE / OR AND per-band — AND no retrain-
with-transforms recovers ship-level performance — the original
SROCC-only gating was correct and our diagnostic discipline holds.

## What's already in the full-stat panel

Per `benchmarks/v0_20_all_bakes_stat_comparison_2026-05-15.md`:

| Bake | Already in full panel? |
|---|---:|
| V_18 ship (3-way concat) | yes |
| V_18 base seed=1 (single MLP) | yes |
| V_20 IS (98 transforms) | yes |
| V_20b distortion manifold | yes |
| D1 3-way concat with transforms | yes |
| D3 tighter transforms (lift≥0.10) | yes |
| fast-ssim2 / V0_2 / butteraugli (static) | yes |

**Findings from existing full panel**:

- **V_20b**: confirmed falsified on CID22 (SROCC 0.866 + PLCC 0.853 +
  PWRC 0.913 + Z-RMSE 0.484 ALL worse than V_18 ship). But wins
  every stat on KADID + TID — confirmed training-side win that
  doesn't transfer to CID22 (FRIQUEE caveat). **No verdict flip.**
- **V_20 IS**: confirmed falsified on CID22 aggregate (PWRC + Z-RMSE
  agree with SROCC). B3 lift IS real per per-band. **No verdict
  flip.**
- **D3 lift≥0.10**: confirmed falsified vs V_20 IS. **No verdict
  flip.**

## Re-eval gaps (highest leverage first)

### Tier A — ship-decision-grade falsifications (committed bakes)

These have durable bakes in `benchmarks/` AND were gated on CID22
aggregate SROCC. High leverage for re-eval.

| Bake | Falsification reason | Δ CID22 SROCC | Re-eval needed? |
|---|---|---:|---|
| `v0_19_calibrated_2026-05-14.bin` | KADID/TID-purge retrain, CID22 dropped | −0.0149 | **yes** |
| `v0_18_1_calibrated_2026-05-14.bin` | full 218k retrain | −0.0109 | **yes** |
| `v0_20a_sweep/iw_k1_s1.bin` (single MLP) | V_20a IW-SSIM path A | −0.0276 | **yes** |
| `v0_20a_sweep/iw_k4_s1.bin` | V_20a IW k=4 variant | (in falsification log) | **yes** |
| `v0_20a_sweep/iw_k8_s1.bin` | V_20a IW k=8 variant | (in falsification log) | **yes** |

### Tier B — cheap-knob falsifications (bakes in /tmp/zensim_loop/bakes/)

128 bakes from recovery cycle 7–13 falsifications. Most live ONLY
in /tmp/ — they survive until next reboot. The most consequential:

| Bake | Cycle | Falsification reason | Bake path |
|---|---:|---|---|
| v0_24_v2_dssim03 | 7 | dssim_w=0.3 co-train, −0.025 CID22 | `/tmp/zensim_loop/bakes/v0_24_v2_dssim03_2026-05-13.bin` |
| v0_27_konjnd_dssim01 | 7 | dssim_w=0.1 co-train | `/tmp/zensim_loop/bakes/v0_27_konjnd_dssim01_2026-05-13.bin` |
| v0_28_konjnd_cosine | 7 | cosine LR, −0.0089 CID22 | `/tmp/zensim_loop/bakes/v0_28_konjnd_cosine_2026-05-13.bin` |
| v0_29_konjnd_smallLR | 7 | small LR catastrophic | `/tmp/zensim_loop/bakes/v0_29_konjnd_smallLR_2026-05-13.bin` |
| v0_kadid_tid_midq15_seed1 | 12 | mid-q-boost, **first positive** but multi-seed falsified | `/tmp/zensim_loop/bakes/v0_kadid_tid_midq15_seed1_2026-05-13.bin` |
| v0_kadid_tid_lowmidq15_seed1 | 12 | combined low+mid-q, −0.0093 AIC-4 | `/tmp/zensim_loop/bakes/v0_kadid_tid_lowmidq15_seed1_2026-05-13.bin` |
| v0_kadid_tid_h64_seed1 | 11 | h=64 architecture | `/tmp/zensim_loop/bakes/v0_kadid_tid_h64_seed1_2026-05-13.bin` |
| v0_kadid_tid_h256_seed3 | 11 | h=256 architecture (−0.014 CID22) | `/tmp/zensim_loop/bakes/v0_kadid_tid_h256_seed3_2026-05-13.bin` |
| v0_kadid_tid_ep200_seed3 | recovery | underconverged | `/tmp/zensim_loop/bakes/v0_kadid_tid_ep200_seed3_2026-05-13.bin` |
| v0_kadid_tid_ep600_seed3 | recovery | overfit | `/tmp/zensim_loop/bakes/v0_kadid_tid_ep600_seed3_2026-05-13.bin` |
| v0_pairboost2p0_seed1..7 | 9b | pair-resampling boost (6 seeds, mean +0.0026 p=0.47) | `/tmp/zensim_loop/bakes/v0_pairboost2p0_seed*_2026-05-13.bin` |

These bakes are at risk (`/tmp` is wiped on reboot). Persist them to
`benchmarks/cycle_7_to_13_bakes/` before re-eval where leverage is high.

### Tier C — genuinely-failed (do NOT re-eval)

These failed on multiple stats simultaneously or for orthogonal
bugs. Re-eval won't recover them.

- **V0_5 multi-codec**: KADIK10k 0.37, TID2013 0.63 — cross-codec
  synthetic ≠ cross-distortion-type. Architectural issue.
- **V0_29 small LR**: −0.142 CID22 — catastrophic underconvergence.
- **Multi-seed weight averaging**: 75 % relative diff — different
  PyTorch seeds land in different loss basins; averaging is
  fundamentally broken for different-init runs.
- **4-group KonJND retrain**: val-policy=Min latched onto broken
  KonJND SROCC (~0.01) — orthogonal failure mode (alignment bug),
  not a falsification of the underlying hypothesis.
- **Phase 4 silent failures**: pipeline bugs, not metric-relevant.

### Tier D — pre-V_20 bakes that could benefit from feature_transforms

These were trained BEFORE zenpredict 0.2.0 added feature_transforms
(commit `8cae13b`, 2026-05-15). Retraining with the V_20 transform
set MIGHT recover ship-worthy performance. High leverage if the
mechanism transfers.

| Falsified experiment | Why feature_transforms might help |
|---|---|
| V0_24 v2 dssim co-train | dssim signal is concentrated in low-q; winsor_p99 on low-q features amplifies signal |
| V_20a IW-SSIM k=1 | IW signal is multiplicative; signed_log1p on IW columns could expose linearity |
| Cycle-9 / 9b low-q boost | low-q regime has high feature variance; quantile_bins normalizes |
| V0_19 KADID/TID-purge | clean corpus has different feature distribution; per-feature transforms adapt |

## Execution plan

### Phase 1 — Full-panel re-eval of Tier A (~3 hours)

5 bakes × 30 min each = 2.5 hours wall. Run sequentially in
background per dataset_metric_baseline (CPU contention if parallel).

Bakes:
1. v0_19_calibrated_2026-05-14.bin
2. v0_18_1_calibrated_2026-05-14.bin
3. v0_20a_sweep/iw_k1_s1.bin
4. v0_20a_sweep/iw_k4_s1.bin
5. v0_20a_sweep/iw_k8_s1.bin

Output: extend `v0_20_all_bakes_stat_comparison_2026-05-15.md` with
these 5 rows + verdict per CID22 / KADID / TID.

### Phase 2 — Persist + re-eval Tier B candidates (~5 hours)

Move /tmp/zensim_loop/bakes/ to `benchmarks/cycle_7_to_13_bakes/`
(at least the 10 highest-leverage; full 128 is 30 GB and probably
not worth it). Run full panel on:

- 3 dssim co-train variants (v0_24_v2, v0_27, v0_25 control)
- 2 LR variants (v0_28 cosine, v0_29 small)
- 3 mid-q-boost variants (midq15 seed=1, lowmidq15 seed=1, midq2 seed=1)
- 2 architecture variants (h64, h256)
- 5 pair-boost variants (seed=1, 2, 3, 7, 42) for multi-seed re-eval

≈ 15 bakes × 30 min = 7.5 hours. Run overnight in cron-mode.

### Phase 3 — Retrain Tier D candidates with feature_transforms (~3 hours)

3 retrains × 17 min train + 30 min eval = ~2.5 hours. Top candidates:

1. **V0_24 v2 dssim co-train + V_20 transforms**: hypothesis dssim
   signal × transforms may recover the −0.025 CID22 gap.
2. **V_20a IW-SSIM k=1 + V_20 transforms**: hypothesis IW signal ×
   transforms may recover the falsification.
3. **V0_19 KADID/TID-purge + V_20 transforms**: hypothesis clean
   corpus + transforms preserves CID22 SROCC vs V_18.

### Phase 4 — Synthesis + CLAUDE.md update (~1 hour)

Update `falsification_reeval_2026-05-15.md` with:

- Verdict table: every falsification, with full-panel result and
  whether the original SROCC-only gating was correct.
- Updated CLAUDE.md "V_20 input-shaping + multi-bake runtime —
  learnings" section if any verdict flips.
- New task list entries for any recovered ship candidates.

## Re-eval acceptance gate

A falsification's verdict FLIPS if:

- The full panel shows PWRC or Z-RMSE wins on CID22 ≥ V_18 floor
  AND SROCC within −0.01 of V_18 ship (full panel re-calls the
  ship decision).
- OR per-band CID22 shows ≥ +0.05 on B0/B1/B2 (priority bands)
  with aggregate within −0.005 of V_18 ship.
- OR retrain-with-transforms recovers the SROCC gap AND maintains
  KADID/TID parity.

A "no verdict flip" finding is ALSO data — it confirms the original
diagnostic was right and that SROCC-only gating was correctable by
the full-panel discipline going forward.

## Provenance

- Falsification source: this session's transcript at
  `/home/lilith/.claude/projects/-home-lilith-work-zen/679264b1-8280-4131-a871-34b51559c43e.jsonl`
  (89 MB; 127 falsification mentions across recovery cycles 7–14).
- Existing full panel: `v0_20_all_bakes_stat_comparison_2026-05-15.md`.
- Methodology source: `CLAUDE.md` "Statistical rigor" + "Per-band
  reporting rule" + "Principled experiment workflow for V_X bakes".
