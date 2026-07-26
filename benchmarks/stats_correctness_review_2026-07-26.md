# Stats correctness + appropriateness review (2026-07-26)

Scope: the statistics the zensim eval/dashboard uses to compare perceptual-metric
bakes — the Mohammadi-2025 6-stat panel (`zenstats::panel`), per-group SROCC, the
bespoke **M3** diffmap-coherence, and the **dial** + **corruption** stats. Triggered
by "evaluate the correctness and appropriateness of the stats we use."

Method: read the canonical implementation line by line
(`zenmetrics/crates/zenstats/src/panel.rs`, 2296 lines; `bake_verdict.rs`;
`eval_report.rs`; `diffmap_block_coherence.rs`; `run_full_eval.sh`; `gauntlet.py`),
then re-derived discrimination + significance from the 15-model summer-gauntlet
fulleval JSONs (`/mnt/v/output/zensim/reports/fulleval/`).

**Provenance:** zensim pins `zenstats` at git rev `82e7fa93`; verified that rev is an
ancestor of zenmetrics HEAD and `panel.rs` is **unchanged since it** → this review is
of the actually-compiled code.

## TL;DR

- **Core 6-stat panel correctness: SOLID.** Every estimator matches its canonical
  (Mohammadi 2025 / VQEG / ITU-T P.1401) definition, including the two things usually
  gotten wrong: tie-corrected Spearman (midrank + Pearson-on-ranks) and
  logistic-rescale-BEFORE PLCC/OR/PWRC/Z-RMSE.
- **`per_group_srocc` is best-in-class** for the pooled-vs-per-ladder confound — but
  it is **not surfaced** in the fulleval JSON or the dashboard.
- **The appropriateness problems are about what's *surfaced* and how *redundant* the
  panel is, not about wrong formulas:**
  1. **OR is near-degenerate** on every corpus; per-sample σ is available nowhere, so
     OR + Z-RMSE never use their correct per-stimulus forms — both run the lenient
     corpus-σ fallback. OR is a catastrophe floor, not a ranking signal.
  2. **The panel collapses to ~1 discriminating axis** for model ranking: SROCC ≈
     PLCC ≈ KROCC ≈ PWRC ≈ −Z-RMSE (|rank-agreement| 0.83–1.00 across all 10 corpora).
     OR adds nothing. The "≥4-of-6 decisive rule" is not 6 independent votes.
  3. **No confidence intervals in the single-bake artifact.** Bare 3-decimal SROCCs
     invite reading ties as orderings. Measured: winner (0.894) is a *real* #1 (paired
     bootstrap P=1.000 vs all), but #2–4 (0.882/0.880/0.879, span 0.003 inside a 0.014
     CI) is a statistical tie the ranking hides.
  4. **`|SROCC|` is applied to *every* corpus**, so a globally-inverted metric reads
     as a high positive headline; the per-ref `frac_negative` backstop (a) isn't
     surfaced and (b) via `Orientation::Auto` can't catch a *global* inversion anyway.
  5. **M3 is the weakest stat**: 3 fixtures × 1 quality level, a *bespoke* Spearman,
     and **structurally blind to f156-371** — it only spatializes basic-156 (+v2 if
     built `--features feature-regime-v2`). It conflates "incoherent diffmap" with
     "model uses pooled features." Do not rank coherence by M3 as-is.
  6. **KADID/TID (train==val memorization) are not flagged** in the summary table or
     the dashboard heatmap — indistinguishable from held-out corpora.
  7. **Two ship composites disagree**: `bake_verdict`'s weighted-goal barely uses CID22
     (0.5 weight, no KADID/TID/CSIQ/LIVE/nonphoto/imazen26); the dashboard composite
     centers CID22. They can rank bakes differently.

## Part 1 — Core 6-stat panel (VERIFIED correct)

`compute_panel(scores, humans)` (`panel.rs:888`): SROCC + KROCC on raw ranks
(rescale-invariant), PLCC + OR + PWRC + Z-RMSE on the **logistic-rescaled** prediction
(`:901-905`) — the Mohammadi §IV-A / VQEG convention.

| Stat | impl | Verdict | Notes |
|---|---|---|---|
| SROCC | `spearman` :53 | ✅ | midrank ties (`ranks` :33) + Pearson-on-ranks — tie-correct. `.abs()` in panel. |
| PLCC | `pearson` :75 on `rescale_logistic` :690 | ✅ | rescale-then-Pearson = VQEG. 13 multi-start seeds, affine fallback, polarity-aware. |
| KROCC | `kendall_tau` :96 | ✅ | proper τ-b. O(n²), fine for eval-n. |
| OR | `outlier_ratio` :147 | ✅ correct / ⚠ lenient | τ = 1.96·σ on rescaled. Per-sample form (`:187`, true P.1401) exists but **σ never joined** → always corpus-σ. |
| PWRC | `pwrc_sa_st_auc` :243 | ✅ | SA-ST sorting-accuracy AUC, normalized [0,1]; O(n²)-time/O(n_pts)-mem rewrite has a bit-parity test (`:1992`). |
| Z-RMSE | `z_rmse` :422 / `_per_sample` :463 | ✅ correct / ⚠ | per-sample = Mohammadi Eq 6 (log-lik-optimal) but **σ never joined** → always corpus-σ global form. |

`per_group_srocc` (`:1632`): within-reference SROCC summarized across ladders,
exposing `frac_negative` (fraction of ladders ranked *backwards*) — the one signal
with no pooled equivalent. Polarity resolved once via `Orientation` (never per-group
`.abs()`), degenerate groups dropped, `None` when unmeasurable. **Correct + rigorous;
under-used** (only for corpora carrying `ref_ids`, and not emitted to the fulleval JSON).

## Part 2 — Appropriateness of the panel (MEASURED, all 10 corpora × 15 models)

### 2a. OR is near-degenerate; per-sample σ available nowhere
OR std per corpus: aic3 0.000, cid22 0.007, imazen26 0.007, konjnd 0.014, live 0.018 —
max OR anywhere 0.063; ≈0 for every real model, only the broken ebothg_m504 lifts it.
`per_pair` carries `{mos|jnd, pred}` — **no σ column on any corpus** — so both OR and
Z-RMSE always fall back to corpus-σ; the correct per-stimulus P.1401 / Eq-6 forms are
**dead code on this eval**. OR is a catastrophe floor, not a discriminator.

### 2b. The panel is ~1 discriminating axis for model ranking
Rank-agreement of SROCC vs each stat across the 15 models, per corpus (Spearman of the
15 model-values): SROCC~PLCC 0.93–1.00, SROCC~KROCC 0.97–1.00, SROCC~PWRC 0.83–1.00,
SROCC~Z-RMSE **−0.93 to −1.00** (Z-RMSE tracks rank inversely). So for *model ranking*
the six stats resolve to one rank axis + a weak PWRC nudge + a dead OR. This isn't a
bug — each non-rank stat is meant to catch a distinct *failure mode* (outliers,
calibration, important-pair weighting) — but the good models here have none of those
defects, so only rank discriminates. The "≥4-of-6" agreement rule therefore mostly
re-counts rank.

### 2c. No CI in the single-bake artifact; the ranking over-states precision
The fulleval JSON is bare point estimates (`{srocc,plcc,krocc,or,pwrc,z_rmse,n}`, |·|,
no CI). Measured (paired bootstrap, 2000 draws, MOS order verified identical n=4292
across all 15 models):

| model | CID22 SROCC | marginal 95% CI | paired vs winner |
|---|---|---|---|
| winner_dial | 0.894 | [0.888, 0.900] | — |
| B_shipped | 0.882 | [0.875, 0.890] | P=1.000 |
| cl_tfm | 0.880 | [0.873, 0.887] | 1.000 |
| Ebothg_scr0.5 | 0.879 | [0.872, 0.887] | 1.000 |
| E-K5 | 0.871 | [0.864, 0.878] | 1.000 |

Winner is a real #1 (paired power resolves CI overlap), but B/cl_tfm/Ebothg
(0.882/0.880/0.879) sit within 0.003 — inside the ~0.014 CI — a statistical tie the
3-decimal ranking obscures. The CI/bootstrap machinery **exists** (`bootstrap_ci_delta`
+ `decisive`, `panel.rs:1350-1520`) but only in the separate **`bake_compare`** (A-vs-B)
tool; the single-bake `bake_verdict`/dashboard never calls it.

### 2d. `|SROCC|` hides polarity uniformly
`compute_panel` abs's SROCC + KROCC for *all* corpora (`:893-894`), so an
anticorrelated metric reads as a high positive headline. The backstop —
`per_group_srocc.frac_negative` — is (a) absent from the fulleval JSON and (b) via
`Orientation::Auto` resolves one sign from the pooled correlation, so it flags only
*within-image* ladder disagreements, never a *globally* inverted metric.

## Part 3 — Bespoke stats

### M3 diffmap coherence (`diffmap_block_coherence.rs`)
Per 32px block, **signed Spearman** of: X = the deployable diffmap summed over the
block, vs Y = ΔS, the *rescore after pasting reference pixels into that block* (a
leave-one-block-out-toward-reference perturbation — not a gradient, not a mask).
Aggregation: sum per block, then **plain mean over 3 fixtures (city/dog/girl) at q50**
(`run_full_eval.sh:63-96`) → `m3_coherence`.

**Structural feature-regime bias — and it resolves the v47 anomaly.** The per-pixel map
spatializes only the basic f0-155 gradient (`s_basic = s[..156]`); f156-371
(masked/iw/peak) are dropped from the map while ΔS uses the *full* vector. So M3
penalizes iw/masked reliance **by construction**, and a bespoke Spearman (not
`zenstats`) is used. The surprising result — v47 (v1-372, uses iw/masked) posting the
**highest** M3 (+0.74), above every foldable model built for coherence (E-K5 +0.65) —
means v47 is **basic-156-dominated** (its iw/masked weights carry little
spatially-incoherent mass, so basic-156 predicts its ΔS well), *not* that v47 has a
better deployable diffmap. M3 as a scalar conflates "incoherent map" with "model uses
pooled features," rests on 3 images × 1 q, and uses a duplicate stat impl. **It is a
weak diagnostic, not a coherence ranking. Do not cite M3 to rank coherence until it is
widened (size×q×content sweep), moved to `zenstats::spearman`, and reported alongside
the dropped-f156-371-mass %.**

### Dial (`bake_verdict.rs::dial_panel`)
Adjacent-q pairs (sorted by q within each `(image, codec)` curve) bucket into 5
outcomes with `MATERIAL_INV = 0.5` score-pts:
- **`mono_pct` = 1 − (material inversions / all pairs)**, pooled across all codecs+images.
  "Monotone" = *not a >0.5pt backward step* — **sub-0.5pt backward wiggles don't count**,
  so `mono=0.93` tolerates unlimited small reversals. Gate ≥0.93.
- **`tied_pct`** = fraction with `|Δ| ≤ 1e-9` **and** distinct feature vectors (a real
  dead-zone; the "codec emitted an identical image" case is excluded). Absolute 1e-9.
  Gate ≤0.05.
- **`reach`** = pooled max − min (outliers in); **`dynamic_range`** = p95 − p5 (robust).
  Gate G1: p5 ≤ 25 ∧ p95 ≥ 85.
- Grid: densified 720 grid, 4 codec families (q0 + step-1 q90→100 + fractional
  near-lossless + JND zone + jxl-in-butteraugli-distance). ⚠ the *372-col* default is the
  `_quarantined_v2` grid (9/115 corrupt ladders + JXL-NL OOD) — dial numbers are
  grid-sensitive, not cross-grid comparable.
- ⚠ pooling mono/flat across codecs lets a high-curve-count family mask a single broken
  dial; per-codec monotonicity is in a side table but not the headline.

### Corruption (`eval_report.rs:663-738`)
A "triple" = `<ref>__<fam>__<region>__<sev>` with up to `corruption`/`q20`/`q10` rows.
**Pass = `score(corruption) < score(q20)`, strict, no margin** (equality fails).
`pass_q20 = pass/n_triples`; `pass_q10 = pass/n10` (**different denominator** — only
groups with a q10 row). Gate `pass_q20 ≥ 0.95`. `per_family` breaks out q20 pass rate
only, worst-first. Definitionally sound; the strict `<` with no epsilon is defensible
but will fail a triple on an exact tie.

## Part 4 — Aggregation + corpus handling

- **No single headline SROCC; two composites that disagree on inputs.**
  (i) `bake_verdict` weighted-goal (`:1897`): `3·g1(dial) + 2.5·g8(AIC3 Z-RMSE) +
  1.5·g5(KonJND+AIC3) + 1·g9(AIC3 DS-AUC) + 0.5·g7(CID22)` — **CID22 only 0.5;
  KADID/TID/CSIQ/LIVE/nonphoto/imazen26 don't feed it.**
  (ii) dashboard (`gauntlet.py:72`): `cid22 + 0.30·nonphoto + 0.20·konjnd + 0.10·aic3 +
  0.05·aic4`, reject if `cid22<0.84 or nonphoto<0.80`.
  These can order bakes differently; neither is labeled "the" ship metric.
- **KADID/TID train==val: not flagged** in the summary table or the dashboard heatmap
  (only omitted from the dashboard *composite*). A reader can't tell them from held-out.
- **`|SROCC|` uniform** (see 2d) — a global inversion is invisible in the headline.
- **Single-bake = bare point estimates**; the decisive `≥4-of-6` significance rule is
  paired **A-vs-B only** (`bake_compare`), not a single-bake ship gate. Single-bake
  shipping uses the fixed-threshold soft-gate scorecard.

## Recommendations (priority order)

1. **Wire `bake_compare`'s paired-bootstrap CI into the single-bake fulleval/dashboard**
   (the code exists) and render a **tie band**, so mid-cluster ties read as ties.
2. **Emit `per_group_srocc.frac_negative` per model** into the fulleval JSON + dashboard
   — it's computed, it's the one non-pooled signal, and a dial product lives on it.
3. **Demote OR to a pass/fail gate**, not a ranking column (near-degenerate; never fed σ).
4. **Join per-stimulus σ** where a corpus has observer std (CID22/KADID/TID do), so
   OR + Z-RMSE use their correct P.1401 / Eq-6 forms — otherwise delete the per-sample
   paths as unreachable.
5. **Report signed SROCC** for quality corpora (reserve `.abs()` for KonJND's
   dual-column target); the uniform abs hides global inversion.
6. **Flag KADID/TID as train==val** (badge or exclude) in the table + heatmap.
7. **Reconcile the two composites** into one documented ship metric, or clearly mark
   which is authoritative and why.
8. **Rebuild M3** before using it to rank coherence: widen sampling to a size×q×content
   sweep (not 3×q50), switch to `zenstats::spearman`, and report the dropped-f156-371
   mass alongside so "incoherent map" ≠ "model uses pooled features."
9. **Weight the aggregate toward product-relevant corpora** (imazen26 real-codec, CID22
   MOS) rather than a flat mean that over-weights synthetic KADID/TID.

None of these are correctness bugs in the estimators — the math is right and matches the
compiled code. They are about **surfacing uncertainty, retiring redundant/degenerate
columns, flagging memorization corpora, and rebuilding the one thin bespoke stat (M3).**
