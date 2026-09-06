# Pre-registration — corruption-head theory tests (2026-09-06)

**Lane:** `claude-corrtheories`, jj sibling workspace `~/work/zen/zensim--corrtheories`.
**Artifacts:** `/mnt/v/output/zensim/corruption-head-2026-09-05/theories/`.
**Record (to be written):** `benchmarks/corruption_head_theories_2026-09-06.md`.

This document is written and pushed BEFORE any result is computed. Every gate,
threshold, metric definition and falsification criterion below is fixed here.
Anything the results turn out to need that is not written here gets added as a
clearly-labelled POST-HOC section in the record, never silently.

---

## 0. The questions (user, verbatim, 2026-09-06)

> "so is masking out the corruption head based on d ranges, or doing some kind of
> interpolation a scientific path? should the corruption head be an mlp instead?
> what kind of corruptions get missed in general, and if we subtract one type of
> corruption from training do false positives go down? maybe test some theories.
> do this fast"

Four questions → T1 (gating/interpolation), T2 (MLP), T3 (what gets missed +
the FP mechanism), T4 (leave-one-family-out). T5 (family-grouped mixture) runs
only if T1–T4 land inside budget.

---

## 1. Era caveat, carried on every number

Everything runs at **rev1** — the post-option-C (`56bbcda2`) extraction era that
the 2026-09-05 lane built, and the era the current runtime reads. **Revision 2
(landing) changes 12 basic slots this head reads**, so every number in this study
is a rev1 number and is not comparable to a rev2 re-extraction. No re-extraction
is performed here; nothing is wired into the runtime; no bake in
`zensim/weights/` is replaced.

## 2. Frozen inputs (identical for every arm, T1–T5)

| role | path | rows |
|---|---|--:|
| positives + matched anchors | `corruption-head-2026-09-05/im26_corruption_372_postC.parquet` | 116,928 + 348 |
| severe-honest negatives | `corruption-head-2026-09-05/negrich_372_postC.parquet` | 60,000 |
| broad-honest negatives (ladder) | `ladder-2026-09-05/instruments/dial_grid_372col_ladder.parquet` | 9,593 |
| gate grid (never trained on) | `corruption-head-2026-09-05/corruption_grid_372col_postC_2026-09-05.parquet` | 2,016 |
| Profile D bake (dial) | `zensim/weights/d_sdr_add156_id100_negrich_dial_2026-09-05.bin` | — |
| incumbent head | `corruption-head-2026-09-05/d228/corruption_head_d228.bin` | — |

**The split is FROZEN and REUSED VERBATIM**, not re-derived: the incumbent's own
`d228/split.tsv` (48,279 sources → train/val/test), which is the source-held-out
60/20/20 split `train_corruption_head.py` wrote. Source keys are the owner's:
corpus `ref_id`, negrich `severe/{source_id}`, ladder `ladder/{image_id}`.
Reusing the file rather than re-running `split_ids` removes any chance of an
RNG-consumption-order difference silently changing folds between arms.

**Parity gate (must pass before any arm is reported):** the reconstructed dataset
must reproduce the incumbent's `metrics.json` exactly — subclass counts
`{corruption 116928, matched_anchor 348, severe_honest 60000, broad_honest 9593}`
and fold sizes `train 112033 / val 41245 / test 33591`. A mismatch aborts.

**Feature slice: `f0..f227` (d228)** for every arm, because that slice is free at
D's runtime (`V1PoolsMode::Peaks`) and `f228..371` is not. T2/T5 do not change the
slice; only T1(d) adds a non-feature input (the dial), stated explicitly.

**Owner discipline.** `scripts/v_next/train_corruption_head.py` is THE owner and
is EXTENDED, not forked: the dataset assembly and the classifier factory are
lifted into importable functions that `main()` itself calls, so the study and the
owner run the same code. No statistic is re-implemented — sklearn for
classification metrics, `scripts/lib/zen_stats.py` if any rank statistic is
needed. The extension is byte-parity-gated against the incumbent recipe.

## 3. Metric definitions (fixed)

All measured on the **test fold only**, sources held out.

- **detection(T)** = fraction of test-fold corruption rows with `P > T`.
- **FP_severe(T)** = fraction of test-fold `severe_honest` rows with `P > T`.
- **FP_honest(T)** = fraction of test-fold `broad_honest` (ladder) rows with `P > T`.
- **FP_anchor(T)** = fraction of test-fold `matched_anchor` rows with `P > T`.
- **q-bands** (ladder only, the only rows with a `q`): `q<50`, `50<=q<85`,
  `85<=q<95`, `q>=95`. Reported per band and per codec
  (`jpeg / webp / avif-svt / avif-rav1e / jxl`).
- **gate pass rate** = `pass_q20` / `pass_q10` on the 2,016-row gate grid, via the
  registered `min(perceptual, gate)` composition, `bake_verdict --corruption-head`
  where a bake exists; where an arm has no ZNPR bake (MLP, HGB, dial-feature), the
  gate is computed in Python from the same probability vector using the same
  composition rule, and that is stated on the row.

**Matched operating points.** Arms are never compared at a shared `T` — a
threshold is not a comparable quantity across model forms. Every cross-arm
comparison sweeps `T` per arm to hit a target **FP_honest** and reports detection
there. Target FP_honest levels, fixed here: **0.25 %, 0.5 %, 1 %** (plus 5 % and
the arm's own T=0.9/0.95 rows for continuity with the incumbent record). If an arm
cannot reach a target level, the row reads NOT REACHABLE, never an extrapolation.

## 4. T1 — is dial-gating scientific?

Five policies, one probability model unless stated, same split, same slice:

- **(a) no gate** — head alone, the incumbent.
- **(b) hard mask** — fire only if `P > T AND dial < G`, for `G in {80, 90}`.
- **(c) soft interpolation** — fire score `P' = P * sigma((G - dial)/s)`, with
  `(G, s)` fit on the **train** fold by maximizing likelihood of the label under
  `P'` (2 parameters, no test-fold information).
- **(d) dial as an INPUT FEATURE** — refit the head on `[f0..227, dial]` (229
  inputs), so the boundary is quality-conditional rather than hand-set. Where an
  honest q-anchor-relative position is available (ladder rows carry `codec`,`q`),
  a second variant adds it; corruption/negrich rows have no q, so the variant is
  reported only if a well-defined value exists for every row, else it is dropped
  with the reason stated.
- **(e) per-band heads** — separate head per dial band, boundaries from train-fold
  dial quantiles, fixed before fitting.

**Decision rule, fixed:** the answer is "(d)/(e) dominate" iff at **every** target
FP_honest level in §3 and in the `q>=85` band, (d) or (e) has detection **>=**
(b) and (c), with at least one level strictly greater by more than the paired
bootstrap noise (1,000 resamples over test-fold sources, 95 % interval). If (b)
matches (d) inside noise, the honest conclusion is "the hand-set prior is as good
as the calibrated version here" and it will be stated that way.

## 5. T2 — MLP vs logistic

Same slice, same split, same isotonic calibration on the val fold, same
`class_weight="balanced"`:

- `logistic` — `LogisticRegression(C=0.05, max_iter=3000)` (incumbent).
- `mlp32` — `MLPClassifier((32,), early_stopping=True)`.
- `mlp64_32` — `MLPClassifier((64,32), early_stopping=True)`.
- `hgb` — `HistGradientBoostingClassifier` (early stopping), **only if it fits in
  minutes**; otherwise reported as NOT RUN with the reason.

Reported: detection at matched FP_honest (0.25/0.5/1 %), per q-band, per codec,
plus **train-fold vs test-fold detection** for each arm. The registered finding to
watch for: an MLP that wins in-sample and loses on the 173-source held-out split
is the result, and will be reported as such rather than as a win.

## 6. T3 — what gets missed, and the FP mechanism

1. **Per-family recall at T=0.9** for the incumbent, all 44 families, test fold.
2. **Characterize the misses**: for the worst families, their `region` / `kind` /
   `severity` distribution (the corpus carries all three) and their position in
   the head's own margin (decision-function distribution) against the honest
   populations.
3. **Nearest-honest analysis**: for missed corruption rows, the nearest honest
   test-fold rows in standardized feature space (cosine / L2 on the head's own
   standardized inputs), reporting which `codec`,`q` they are.
4. **The FP side**: which honest ladder cells (codec × q) are flagged, and which
   positive families sit nearest to those flagged cells — the measured mechanism
   for the `q>=95` false positives, not a guess.

No hypothesis is pre-registered as true here; §4.3 of the 2026-09-05 record
proposes "tiny local break looks like near-lossless", and this test either
supports it with the nearest-neighbour evidence or does not.

## 7. T4 — subtract one family

Leave-one-family-out over all **44** families: for each family `F`, refit on
train with every `F` row removed (positives only; negatives untouched), then
report on the **same frozen test fold**:

- ΔFP_honest overall and in `q>=85` (arm minus incumbent, at T=0.9 **and** at the
  matched-0.5 % operating point);
- recall of `F` itself when `F` is absent from training (generalization to an
  unseen corruption type);
- Δ recall of every other family (the label-noise / near-duplicate direction).

Then the **greedy top-k removal**: remove the k families with the largest measured
FP reduction, k = 1..8, refit each time, and report the trade curve
(FP_honest vs total detection). Greedy, not exhaustive; stated as greedy.

Fixed sanity requirement: a family whose removal changes FP_honest by less than
the arm-to-arm reproducibility of the fit (measured by refitting the incumbent
recipe with the identical inputs and comparing) is reported as NO EFFECT, not as
a small effect.

## 8. T5 — family-grouped heads (only if T1–T4 finish inside budget)

Groups fixed here from the family names, before seeing any result:

- **structural**: `block_*`, `edge_*`, `overlay_*`, `chroma_boundary`, `aliasing`
- **photometric**: `channel_*`, `tone_*`, `composite_*`, `noise_*`
- **geometric**: `geometric_*`

One head per group (positives = that group only, negatives shared), combined by
`P = max_g P_g`, against the single head at matched FP_honest.

## 9. Deliverables

- `benchmarks/corruption_head_theories_2026-09-06.md` — one table per T, a
  one-paragraph verdict per user question, repro commands.
- `benchmarks/INDEX.md` row; `docs/DATASET_HISTORY.md` ROUND row.
- artifacts + `_MANIFEST.json` under
  `/mnt/v/output/zensim/corruption-head-2026-09-05/theories/`.
- **Nothing wired into the runtime. No bakes replaced. No public API changed.**

## 10. What would falsify each claim

| claim | falsified by |
|---|---|
| "dial-gating is unscientific / a conditional head is better" | (d)/(e) failing to beat (b)/(c) at every matched FP level in `q>=85` |
| "the head should be an MLP" | mlp arms not beating logistic at matched FP on the **held-out** fold |
| "the q>=95 FP is a local-vs-near-lossless confusion" | flagged near-lossless cells' nearest positives NOT being the small-region families |
| "subtracting a family reduces FP" | LOO ΔFP within fit reproducibility for every family |
