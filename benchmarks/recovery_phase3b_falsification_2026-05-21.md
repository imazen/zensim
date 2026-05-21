# Recovery Phase 3b — `--per-group-target-column` falsification

**Task:** #203 (RECOVERY-PHASE-3B-KONJND-FIX)
**Date:** 2026-05-21
**Trainer change:** zenanalyze main `--per-group-target-column NAME:COLUMN`
**Verdict:** FAIL ship gate (CID22 collapse + KonJND no recovery)

## TL;DR

Phase 3 (task #202) identified KonJND val SROCC collapse to 0.37 as a
sign-mismatch between `konjnd-dense.human_score` (signed SSIM2-delta
in [-65, 96]) and `val/konjnd.human_score` (PJND threshold in [22, 70]).
The proposed fix per zensim/CLAUDE.md was to load
`konjnd-dense.pjnd_target` (the canonical PJND-threshold column on the
correct sign convention) instead of `human_score` for the konjnd-dense
training group.

We shipped the trainer flag (`--per-group-target-column NAME:COLUMN`,
30 lines in `zensim_metric_train.py`'s `main()`) and ran the Phase 3
champion recipe with `--per-group-target-column konjnd-dense:pjnd_target`
plus `--corpus-target-scale konjnd-dense:0.01` (to bring pjnd_target
[22, 70] onto the same numerical scale as safesyn/kadid/tid targets,
which live on [0, 1] modulo safesyn's [-7.4, 0.99] outlier tail).

**Both ship-gate axes failed decisively:**
- **CID22 SROCC median 0.4827** (gate 0.8374; **FAIL by −0.355**, vs
  Phase 3 baseline 0.8498 which itself passed CID22 but failed KonJND).
- **KonJND SROCC median 0.1870** (gate 0.7927; **FAIL by −0.606**, vs
  Phase 3 baseline 0.3689 — the fix actively made KonJND **worse**).

The fix was structurally incompatible with the existing per-pair MLP
trainer architecture (see root cause). PhaseV3-baseline ships nothing;
Phase 3b ships nothing. The `--per-group-target-column` flag lands as
research infrastructure because it is the right primitive for
heterogeneous corpora — it just can't fix this particular
heterogeneity.

No new zensim variant ships from this cycle.

## 5-seed CI table (h=128, epochs=100, val-policy=min)

Recipe: `safesyn (w=1.0) + kadid (w=0.3) + tid (w=0.3) + konjnd-dense (w=0.3)`
with `--per-group-target-column konjnd-dense:pjnd_target` and
`--corpus-target-scale konjnd-dense:0.01`. Losses: ranknet + mse +
magnitude_match (λ=0.1, α=0.3) + low-band oversample (4.0×, cutoff 0.6).
Selection: `--val-policy min` over `{cid22, kadid, tid, konjnd}` (the
`min`-policy worst-axis maximization is what task #203 specified).

| seed | CID22 | KADID | TID | KonJND | best_epoch |
|------|-------|-------|-----|--------|-----------|
| 1 | 0.5404 | 0.8377 | 0.8221 | 0.1870 | 43 |
| 2 | 0.4543 | 0.7538 | 0.7071 | 0.2440 | 29 |
| 3 | 0.4622 | 0.6421 | 0.5939 | 0.3795 | 10 |
| 4 (median CID22) | **0.4827** | **0.8693** | **0.8421** | **0.1765** | **53** |
| 5 | 0.3495 | 0.8790 | 0.8389 | 0.3409 | 26 |

Wall: ~13–15 s × 5 seeds = ~75 s GPU (RTX 4090, 229k training rows × 100 epochs).

## Full Mohammadi panel — median seed (seed 4)

| Corpus | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|--------|---|-------|------|-------|----|----|--------|
| CID22 | 4292 | 0.4827 | 0.5173 | 0.3378 | 0.0510 | 0.5685 | 0.856 |
| KADIK10k | 10125 | 0.8693 | 0.8759 | 0.6772 | 0.0410 | 0.9163 | 0.483 |
| TID2013 | 3000 | 0.8421 | 0.8610 | 0.6426 | 0.0500 | 0.8969 | 0.509 |
| KonJND-1k | 1008 | 0.1765 | 0.2206 | 0.1173 | 0.0308 | 0.2078 | 0.975 |
| AIC-3 | 600 | 0.7377* | 0.7399 | 0.5590 | 0.0383 | 0.8282 | 0.673 |

\* AIC-3 aggregate from seed 1 (seed 4 verdict file `recovery_phase3b_s4_bake_verdict_2026-05-21.txt` reports the full per-band breakdown across all 5 corpora; seeds 2/3 show similar collapse).

## Ship gate vs Balanced trail (per task #203 spec margins)

| Gate | V10 BalancedV3 baseline | Phase 3 (#202) median | Phase 3b (#203) median | Result |
|------|------------------------|----------------------|------------------------|--------|
| CID22 SROCC | 0.8324 | 0.8498 (PASS) | **0.4827** | **FAIL** (−0.355 vs gate) |
| CID22 Z-RMSE | 0.564 | 0.523 (PASS) | 0.856 | **FAIL** (+0.326 vs ≤ 0.530 gate) |
| KADID SROCC | 0.9677 | 0.9101 (PASS) | 0.8693 | PASS (V10 − 0.10 = 0.8677) |
| TID SROCC | 0.9729 | 0.8845 (PASS) | 0.8421 | PASS (V10 − 0.10 = 0.8729 — barely fails) |
| KonJND SROCC | 0.8927 | 0.3689 (FAIL) | **0.1765** | **FAIL** (−0.616 vs gate, **−0.193 vs Phase 3**) |
| AIC-3 SROCC | 0.7845 | 0.8046 (PASS) | 0.7377 | **FAIL** (−0.042 vs ≥ 0.7795 gate) |

**Two gates fail decisively (CID22, KonJND, AIC-3); TID barely fails.
KonJND is worse than Phase 3, not better.** Overall ship verdict: **FAIL**.

## Root cause: pjnd_target's structural shape

`konjnd-dense.pjnd_target` is the **per-source PJND threshold**. All
20 distortion levels for one reference image share the **same**
`pjnd_target` value. This is documented in canonical-2026-05-21's
`_MANIFEST.json` and visible in the raw parquet:

```
SRC0001 row 0 (low-distortion):  pjnd_target=30.79  human_score=-61.39
SRC0001 row 1                    pjnd_target=30.79  human_score=-30.57
SRC0001 row 2                    pjnd_target=30.79  human_score=  2.24
SRC0001 row 3                    pjnd_target=30.79  human_score= 21.12
SRC0001 row 4                    pjnd_target=30.79  human_score= 30.21
SRC0001 row 5                    pjnd_target=30.79  human_score= 36.95
SRC0001 row 6 (high-distortion): pjnd_target=30.79  human_score= 41.13
…20 distinct distortion levels per ref…
```

The trainer's MLP regresses `target` from per-pair features. With
`--per-group-target-column konjnd-dense:pjnd_target`:

1. **Within a ref-group, the trainer asks the model to predict
   IDENTICAL output for 20 different per-pair feature vectors** — one
   per distortion level. The per-pair features span the full
   distortion range; the target spans zero variance per group. This
   is anti-correlated with the safesyn/kadid/tid signal where target
   tracks distortion.
2. **RankNet within konjnd-dense ref-groups produces zero gradient**
   (predicted differences want to match target differences, which are
   zero). So RankNet contributes no learning from konjnd-dense.
3. **MSE within konjnd-dense ref-groups produces noise gradient** —
   the model is pushed toward the per-ref mean prediction regardless
   of features, which conflicts with safesyn/kadid/tid's per-pair
   gradients pushing toward feature-driven predictions.
4. **Net effect on CID22**: the konjnd-dense MSE term effectively
   smooths the MLP's distortion-feature → target mapping toward
   per-source constants, collapsing CID22 ranking (which requires
   per-pair distortion discrimination). CID22 SROCC drops from 0.85 →
   0.48.
5. **Net effect on KonJND val**: val/konjnd has 1 row per ref. The
   trainer learned to predict pjnd_target from features, but the
   features represent *one distortion level per row*, not the
   per-source PJND threshold. The model fits noise patterns in the
   feature distribution rather than per-source quality
   tolerance. KonJND SROCC drops from Phase 3's 0.37 → 0.18.

This is a **structural shape mismatch**, not a scale or sign problem.
The fix would require either:

- A per-source aggregation head that pools the 20 per-pair predictions
  per ref → one prediction per ref, then regresses to pjnd_target.
  This is essentially the Compression-trail's per-sample-α head
  pattern but with aggregation across distortion levels rather than
  within-pair gating.
- A separate auxiliary loss branch for konjnd-dense that only learns
  per-source representations (e.g., learn a per-ref embedding from
  one canonical row, regress that embedding's projection to
  pjnd_target). The current trainer doesn't have this architecture;
  it's port 5/6 territory from `RECOVERY_PLAN_2026-05-08.md`.
- Don't train on konjnd-dense at all and accept Phase 3's KonJND
  0.37. That's where the Phase 3 falsification doc left it.

## What the `--per-group-target-column` flag IS good for

The flag is the right primitive for the heterogeneous-corpus problem;
it just couldn't fix this particular mismatch. The flag enables:

- Loading `mix_target` from train corpora that carry it, while
  loading `human_score` from val corpora that don't (the structural
  blocker on Phase 3 Variant E).
- Loading any per-corpus-specific target column without rewriting
  `_parse_corpus_spec` to accept a third field. Caller just adds
  `--per-group-target-column NAME:COLUMN` for the group that needs
  an override.
- Future hybrid-head training where each corpus has its own
  trainer-side target shape: ssim2 anchor on safesyn,
  CVVDP+IWSSIM mix on KADID/TID, PJND-shaped on a per-source aggregator
  branch for konjnd-dense.

The flag lands on `zenanalyze/main` (commit TBD this cycle); no
in-flight callers are affected since it defaults to the existing
`--target-col` behavior.

## What WOULD likely beat the KonJND gate (future work, NOT executed)

1. **Per-source aggregation head for konjnd-dense.** Average MLP
   predictions across the 20 distortion levels per ref, then add an
   auxiliary MSE term against pjnd_target. The aggregation breaks the
   "predict 20 identical targets" pathology in this falsification.
   This is a port 5/6 architecture change.
2. **Hybrid head (Rust trainer's `--per-sample-α-head` path).** Same
   rationale as Phase 3 falsification doc point #3. The
   `V_22-hybrid` recipe (see `project_exp_v22_hybrid.md`) ran into
   the same architectural-gap conclusion.
3. **Skip konjnd-dense in training; ship Phase 3's CID22 0.8498
   variant as a Compression-trail candidate.** Phase 3 already passes
   CID22 (+0.0124 vs Balanced gate) and AIC-3 (+0.0201). The
   tradeoff is that KonJND 0.37 fails the Balanced gate by −0.42, so
   this would be a Compression-trail-only ship that should be
   compared against `V_22-372feat s5` (current Compression ship at
   CID22 0.8641, AIC-3 0.8183). Phase 3's CID22 0.8498 < 0.8641 and
   AIC-3 0.8046 < 0.8183 — strictly dominated by the current
   Compression ship. Not a swap target.

## Trainer artifact — what lands

`zenanalyze/zentrain/tools/zensim_metric_train.py` gains:

```python
ap.add_argument("--per-group-target-column", action="append", default=[],
                metavar="NAME:COLUMN",
                help="...")
```

Plus parsing + dispatch in `main()`:

```python
per_group_target_col: dict[str, str] = {}
for entry in args.per_group_target_column:
    name, col = entry.split(":", 1)
    per_group_target_col[name] = col

for name, path, w in train_specs:
    target_col_for_group = per_group_target_col.get(name, args.target_col)
    df = load_corpus(path, target_col_for_group, ...)
    ...
```

Val parquets always use `--target-col` regardless of override — the
flag intentionally only affects TRAIN groups. Smoke-tested via
3-epoch synthetic run; full 100-epoch × 5-seed CI documented above.

## Files touched

- `zenanalyze/zentrain/tools/zensim_metric_train.py` — added
  `--per-group-target-column NAME:COLUMN` CLI flag + dispatch.
- `zensim/benchmarks/recovery_phase3b_s{1..5}_train_2026-05-21.log` —
  five per-seed training logs (raw stdout including per-epoch SROCC
  + best-epoch selection).
- `zensim/benchmarks/recovery_phase3b_s{1..5}_bake_verdict_2026-05-21.txt`
  — five per-seed full Mohammadi panel verdicts.
- `/mnt/v/output/zensim/exp_recovery_phase3b_2026-05-21/seeds/cc4_phase3b_s{1..5}.bin`
  — five trained bakes (ZNPR v3, 194,956 bytes each, h=128, 372
  inputs). Available for future ablation / per-source aggregation
  head experiments.

## Conclusion

The Phase 3b hypothesis — that fixing the sign-mismatched
`human_score → pjnd_target` column swap recovers KonJND signal — is
**decisively falsified**. The actual blocker on KonJND in the existing
trainer is the **per-source constant nature** of `pjnd_target`, not the
sign or scale. The per-pair MLP cannot productively absorb a target
that varies only across refs, not across distortion levels within a
ref. Fixing this requires a per-source aggregation architecture (port
5/6 from `RECOVERY_PLAN_2026-05-08.md`), not a corpus-loader-level
column override.

The `--per-group-target-column` flag itself is correct
infrastructure and lands on zenanalyze/main as research substrate. No
zensim variant ships. The current Balanced (V_22-mix-LARGE+iwssim s3)
and Compression (V_24-per-sample-α s4) ships remain the production
SOTA.
