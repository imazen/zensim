# joint-core-v1 — 52,963-pair provenance-complete TRAIN view (2026-09-20)

**Verdict: built and measured; one admission gate measured FAIL by its own
rule.** The set exists, is reproducible, versioned, kernel-clean and within
the one-hour repeat budget — but the permuted-column control says a
~53k-pair core **cannot detect 30 destroyed inputs above seed noise**
(mean cost +0.0021 vs the ~0.003 floor). Per the plan's own rule
("the core is too small and the target grows before any feature is
screened"), joint-core-v1 is a working TRAIN view but is **not yet large
enough to screen features**. Growing the target is the measured
recommendation, not a defect to hide.

Evidence root: `/mnt/v/output/zensim/joint-core-v1/` (11 GB; `_MANIFEST.json`
at the root). Build commits: `ab845295` (zensim-bench examples),
`6f056876` (tools/joint_core), `0516365b` (prereg amendment + clippy fix).
Plan: `docs/PLAN_JOINT_CORE_SET_2026-09-19.md` (cifix worktree); guards:
`docs/FITTED_CONSTANT_GUARDS_2026-09-19.md`.

## MISSING / caveats first

1. **Feature-screening power is unproven — measured too small.** The
   30-column permutation cost is indistinguishable from seed noise
   (below). Any feature-level screen on this core would be unearned.
2. **All "human" labels in the MLP arms are each leg's own target
   scale** (CID22 MOS×100 complement, KADID/TID raw MOS×100, KonFiG JND
   fraction, SafeSyn/codec ssim2-teacher). The R915 recipe mixes them
   per-leg with its own loss modes — this is the leaders' contract, not
   a normalised human panel.
3. **The DVIFM constants fit ran on a seeded uniform subset** (3,154 of
   50,463 SDR rows, rng 20260920). A bare positional stride aliases the
   cell-blocked TSV order (measured: stride-16 sample mean score 19.1 vs
   domain 51.9 — discarded before this run). The full-domain fit is
   measured too slow for the ≤10-min budget (≈10 min for the *first*
   (plane,level) cell alone at 50k rows).
4. **β hits the grid ceiling on most (plane,level) cells** — boundary
   pile-up is recorded per cell in the fit artefact (`grid_edge` flags),
   per the guards doc this is a finding (masking curve weakly identified
   at high β), not a silent clamp.
5. **KonFiG (327 pairs) had to be re-extracted**: its source table is the
   `ext944` era, not column-compatible with `ceiling_rev3`. Re-extracted
   at `ZENSIM_FORMULA_REV=3`; all other legs are `ceiling_rev3#b782e349`.
   The HDR leg is PQ-regime (`foldapp2hdrpq`) — kept in the dataset but
   out of every SDR feature table and SDR DVIFM cache.
6. **160 fresh cells are rung-≥-source passthroughs** (no resample,
   `kernel=native,no-resample(rung>=src)`); a 101.7-MP png-v3 source
   needed the zenpng pixel cap raised to the CLI's own 1e9 stance.
7. **No eval-corpus reads, no holdout fits, no CID22-B**, no production
   qualification. The dHash audit flagged 21 (core-ref, holdout-ref)
   pairs; every one was adjudicated false-positive by pixel RMSE/NCC
   (`flagged_confirm.tsv`), including the documented AIC2026-S14
   degenerate-flat-page hash mode.

## What it is

52,963 train pairs across 6 legs:

| leg | pairs | share | kernel | target |
|---|---|---|---|---|
| fresh_imazen26 | 32,928 | 62.2% | Mitchell sharpen=0 | ssim2 (signed) |
| cid22 | 7,500 | 14.2% | native (no resample) | MOS complement ×100 |
| human (KADID/TID train) | 5,100 | 9.6% | native | raw MOS ×100 |
| fresh_safesyn | 4,608 | 8.7% | Mitchell sharpen=0 (from CLIC .jpg) | ssim2 |
| hdr | 2,500 | 4.7% | Mitchell sharpen=10, linear PQ | PQ-era score 0–100 |
| konfig | 327 | 0.6% | native | JND fraction |

Coverage (measured, `coverage_report.json`): mid 384–1024 px **55.9%**
(≥55 ✓), small 192–256 **24.4%**, tiny 64–128 **19.7%**; photography
**81.6%** (≥75 ✓); screen/doc/lineart **5.62% each** (≥5 ✓); AI
**1.51%** (<5 ✓). Per-codec, per-rung, per-quality-decile and per-leg
breakdowns are in `coverage_report.json`; kernel per pair in
`pairs/pairs_core.tsv`.

## The four gates (all measured)

**1. Leader reproduction** — `basic228/h128`, leaders' seeds, R915 recipe
verbatim vs the frozen R915 result on the leaders' own tables:

| seed | R915 frozen | core | gap |
|---|---|---|---|
| 17101 | 0.9788 | 0.9696 | −0.0092 |
| 17103 | 0.9783 | 0.9715 | −0.0068 |
| 17107 | 0.9775 | 0.9711 | −0.0064 |
| 17111 | 0.9784 | 0.9729 | −0.0055 |
| 17113 | 0.9778 | 0.9718 | −0.0060 |

Mean gap **−0.0068** (best-val geomean3). Above the ~0.003 seed-noise
yardstick but consistent with a ~3.3× smaller train pool; all five seeds
healthy (≥0.9696). **Reported, not excused.**

**2. Convergence** — dev val(geomean3), epoch 50 → 99 (same LR phase):

| seed | ep50 | ep99 | Δ |
|---|---|---|---|
| 17101 | 0.9619 | 0.9637 | +0.0018 |
| 17103 | 0.9669 | 0.9651 | −0.0018 |
| 17107 | 0.9435 | 0.9651 | +0.0216 |
| 17111 | 0.9682 | 0.9684 | +0.0002 |
| 17113 | 0.9628 | 0.9644 | +0.0016 |

Mean **+0.0047** — flat-to-rising. Epoch-100 point values dip on some
seeds (17111: 0.9417) because the recipe's LR schedule restarts at
epoch 100 (lr 0→0.001); the identical transient exists in the frozen
R915 logs (e.g. 17101: 0.9743→0.9710). 17111 recovers to its best
0.9729 by epoch 110. **PASS** — no sustained decline; the 8.3k-estate
failure mode (systematic fall) is absent.

**3. Permuted-column control** — same recipe, same seeds, columns
f64..f93 (30 used columns inside the kept 0..227 prefix) row-permuted in
all 4 train legs AND the 4 dev copies (dev seed 6620):

| seed | real | perm30 | cost (real−perm) |
|---|---|---|---|
| 17101 | 0.9696 | 0.9701 | −0.0005 |
| 17103 | 0.9715 | 0.9705 | +0.0010 |
| 17107 | 0.9711 | 0.9687 | +0.0024 |
| 17111 | 0.9729 | 0.9700 | +0.0029 |
| 17113 | 0.9718 | 0.9670 | +0.0048 |

Mean cost **+0.0021**, per-seed range −0.0005…+0.0048 — at the ~0.003
seed-noise floor. The permuted columns are all non-degenerate
(std 7.8e-5…0.23). **FAIL by the plan's own rule — the core is too
small for feature screening; grow the target before any feature is
screened.** This is the same signature as the 2c codec panel
(permuted-twin Δ −0.0021, |Δ| ≤ noise) at 373 fit rows — at 53k rows
the measurement exists but the sensitivity does not yet.

**4. Coverage** — all floors pass (numbers above). Per-class and
per-codec splits in `coverage_report.json`. **PASS.**

## DVIFM constants fit (native3, Y′CbCr 3-plane)

Standalone fitter (`dvifm-standalone-fit-v2`), K1 init
(`dvifm-local-fitted-final.json`), per-cell map refit, accepted
candidates rescored on all kept blocks, ≤512-block Adam subsamples,
15 (plane,level) cells × ≤3 sweeps + head.

Fit domain: seeded uniform 3,154-row subset (rng 20260920; mean score
51.4 ≈ domain 51.9). Block caches: 50,463 rows × 3 planes, cap 1024
records/row, f16, with full-population 256×256 (C̃,m) histograms.

Histogram/per-block equivalence (400-pair seeded subset, uncapped f32):
hist vs bin-snapped blocks max_rel ≈ **4e-16** (exact); hist vs true
5.4e-4…3.3e-3 by level (256-bin quantisation). Results in
`tools/joint_core/dvifm_equiv.py` output `equiv_f32.json`.

Result @3,154 rows: init refit-MSE 530.14 → **final 335.28** (var(y)
924.39, R²≈0.64); SROCC(−E,y) 0.7296 → **0.8401**, KROCC 0.6531;
3 sweeps, wall 1448.5 s. Output map A=124.4, B=−26.6, λ=133.4.
Channel weights Y/C≈0.394/0.303. The refined β runs far past the
literature band (up to ~77) with C₀ pinned at the grid floor on most
levels — recorded `grid_edge`/`sharpness` per cell in the artefact; per
the guards doc this is the masking-as-gate regime (a finding; the
constants are inputs to study, not shipped values).

Budget variant: `--row-stride 64` (788 rows, mean(y) 51.1 ≈ domain
51.9): **206.1 s wall — inside the ≤10-min gate**; SROCC 0.8190,
MSE 353.3/var 918.5 on its subset. The 3,154-row fit (24.1 min) misses
the gate; the row count is the sanctioned lever and 788 rows is the
measured fitting point.

## One-hour repeat budget (measured)

| stage | measured | budget |
|---|---|---|
| fresh generation (1,173 renditions, 37,536 cells) | ~21 min | — |
| 944-feature extraction (37,536 rows) | 246.6 s | — |
| DVIFM block caches (3 planes × 50,463 rows) | ~195 s/plane (~9.8 min) | — |
| 5-seed basic228/h128 (concurrent) | ~490 s/seed → ~8.7 min wall | ≤40 min ✓ |
| native3 constants fit @3,154 rows | 1448.5 s | ≤10 min ✗ |
| native3 constants fit @788 rows | **206.1 s** | ≤10 min ✓ |

Repeat cycle (gen + extraction + caches + both fit families): measured
≈47 min end-to-end — inside the 1-hour budget with the 788-row constants
fit. At 3,154 rows the fit alone misses its gate; that is a measured
capacity limit of this core size, consistent with the gate-3 finding.

## Reproduce

```bash
# plan + pairs
python3 tools/joint_core/select_core.py
python3 tools/joint_core/assemble_core.py
# features
ZENSIM_FORMULA_REV=3 zensim-bench --example extract_features_372col \
  --corpus pairs-tsv --path pairs_extract_fresh.tsv --full-944
python3 tools/joint_core/assemble_features.py
# dvifm caches + equivalence
extract_features_372col ... --dvifm-block-stats <bin> --dvifm-cap 1024 \
  --dvifm-quant f16 --dvifm-hist <hist>
python3 tools/joint_core/dvifm_equiv.py <spec> <bin> <hist> <out.json>
# fits
tools/joint_core/fit_repro.sh            # real arm
FEATDIR=.../perm30 DEVDIR=.../perm30/dev tools/joint_core/fit_repro.sh
python3 tools/joint_core/fit_core.py <init> <out.json> <surfaces/> \
  ycbcr_y=<y.bin> ycbcr_cb=<cb.bin> ycbcr_cr=<cr.bin> \
  --pairs-tsv pairs_dvifm_sdr.tsv --row-stride 16
```
