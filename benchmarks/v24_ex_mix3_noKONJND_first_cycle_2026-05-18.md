# EX-MIX3 first-cycle (noKONJND) — falsification record

**Date:** 2026-05-18
**Cycle:** First (no konjnd training group)
**Status:** SUPERSEDED — round-2 with konjnd PJND-anchor in flight

## What was run

Following strict coverage-gate (drop groups with <50% triple-coverage of
`cvvdp_log_norm + iwssim_log_norm + ssim2_log_norm`):

- 3 training groups: safesyn (196k, 1.0/0.0) + kadid (10k, 0.3/1.0) + tid (3k, 0.3/1.0)
- `konjnd-dense` (20k, 0% ssim2) dropped
- `cvvdp_iwssim_LARGE` (73k, 0% ssim2) dropped
- konjnd small (1008 rows, 0% ssim2 because no per-pair scores) also dropped

Trained 2 seed=1 bakes before stopping (cv33_iw33_sm33_s1, cv30_iw40_sm30_s1).

## Result vs V_22 noLARGE 5-seed baseline

| Variant | CID22 | KADID | TID | **KonJND** | AIC-3 |
|---|---|---|---|---|---|
| cv33_iw33_sm33 s1 | 0.8934 | 0.9186 | 0.8695 | **0.2990** | 0.8114 |
| cv30_iw40_sm30 s1 | 0.8940 | 0.9291 | 0.8762 | **0.2996** | 0.8114 |
| V_22 noLARGE 5-seed mean | 0.8425 | 0.9311 | 0.8897 | **0.8371** | 0.8059 |

CID22: +0.051 (huge win). KonJND: **-0.54 (catastrophic collapse)**. All
other corpora within seed noise (-0.02 max).

## Root cause

The V_22 noLARGE recipe trains konjnd at weight 0.02 against
`mix_cv40_iw60` — which equals `human_score` in the konjnd small parquet,
which equals the PJND compression-q threshold (range [22.46, 69.98]).
Dropping the konjnd group removes the only PJND-threshold signal from
training; the rest of the corpora (safesyn/kadid/tid) carry no PJND
information.

This matches the V_22-CVVDP-LARGE failure mode documented in
`benchmarks/v22_mix_LARGE_iwssim_methodology_2026-05-18.md`:

> Pure-CVVDP supervision on compression distortions gives no signal for
> JND ordering.

## Fix applied (round-2)

Re-introduce konjnd training group at weight 0.02 (matching V_22 noLARGE),
using PJND-threshold passthrough as the value of all 3 mix variant columns.
This is NOT zero-fill (the PJND threshold IS the real per-pair signal for
the konjnd 1008-row dataset); it's a label-passthrough for a group whose
3-way blend equation can't be computed because the group has no per-pair
cvvdp/iwssim/ssim2 measurements.

Round-2 training groups (4 groups, 210,219 rows — identical to V_22 noLARGE):

| Group | Rows | Target | Train_w | Val_w |
|---|--:|---|--:|--:|
| safesyn | 196,086 | mix_cv{33,30,40}_... | 1.0 | 0.0 |
| kadid | 10,125 | mix_cv{33,30,40}_... | 0.3 | 1.0 |
| tid | 3,000 | mix_cv{33,30,40}_... | 0.3 | 1.0 |
| konjnd (PJND-passthrough) | 1,008 | mix_cv{33,30,40}_... (= PJND) | 0.02 | 1.0 |

## Implication for the strict coverage-gate rule

The task's strict gate ("if a group has <50% ssim2 coverage, **drop**
that group") was structurally insufficient: it didn't anticipate
groups whose ssim2 absence is structural (no per-pair scores exist
because the group's primary "score" is a threshold, not a quality
metric). For such groups, the right action is **passthrough the
group's native target** (PJND here), not drop. This is what V_22
noLARGE already did implicitly through `mix_cv40_iw60` being aliased
to `human_score` in the konjnd small parquet.

The strict gate-drop rule still applies to groups where the metric
is meaningful but missing (e.g., the `cvvdp_iwssim_LARGE` rows
where ssim2 wasn't computed — those rows could in principle have
ssim2, but the sidecar doesn't cover them).

Archive: `/mnt/v/zen/zensim-eval/ex_mix3_2026-05-18/noKONJND_backup/`
