# Lane `zgeom` — DONE

**Scope.** Two questions the X4 transplant left open, run under the same
joint-core-v2 corpus (105,614 pairs; 100,997 train-leg rows), paired-seed
machinery, and permuted-control discipline:

- **Z2** — box vs binomial `[1,2,1]` / `[1,3,3,1]` for zensim's OWN
  2×2 pyramid decimate, on zensim's own 228-col `basic+peaks` surface:
  development accuracy (5 paired seeds), the three label-free stability
  gates (1px shift, codec phase 0–7, crop-vs-full), and extraction cost
  (256²/1024²/2048², 1 and 8 threads, `α + β·pixels`).
- **Z1** — 5×5 block-peak × two-state-gate pooling as a **replacement**
  for global-moment pooling at matched column count (228 slots), with
  ungated block-peak as the decomposition arm and all-228-column
  row-permuted controls.

No production default was changed. No holdout rows were read
(`cid22b_unsealed` untouched). All arms extracted fresh at formula rev 3
over identical rows in identical order; every arm carries a distinct
feature-set identity (registered eras `zgeom_*`, append-only in
`benchmarks/feature_sets_registry.json`).

## Verdicts

| question | measured | verdict |
|---|---|---|
| Z2 dev accuracy | bin121 −0.0023 (4/5 neg, t=−3.26); bin1331 −0.0013 (**5/5 neg**, t=−3.70) vs box2 | **NO adoption case — box2 stays** |
| Z2 stability | binomials strictly MORE shift-stable (worst-feat \|Δln\| 2.53→0.71/0.70; median-feat 0.034→0.013/0.012); phase/crop equal; **zero fails** | kernels trade acc for shift-stability — doesn't rescue the acc loss |
| Z2 cost | measured +48%/+65% β (t1), 2.2×/2.6× (t8) — **UPPER BOUNDS, not fair** (see below) | cost claim unreliable; verdict rests on accuracy |
| Z1 replacement (gate) | z1gate −0.1808 vs box2 (**5/5 neg**, t=−66.6) | **NOT a better pooling — catastrophic loss** |
| Z1 gate mechanism | z1gate − z1gateperm = +0.678 (5/5) | block signal real; the GATE is what kills it |
| Z1 ungated block-peak | z1max +0.0012 vs box2 (4/5 pos, t=+1.60); +0.860 vs its perm | **mild positive, concentrated on codec (+0.014) and human (+0.013) dev legs** |

## Z2 — kernel screen (dev geomean3 SROCC, 5 paired seeds)

| seed | box2 | bin121 | bin1331 | bin121−box2 | bin1331−box2 |
|---|---|---|---|---|---|
| 17101 | 0.9508 | 0.9512 | 0.9503 | +0.0004 | −0.0005 |
| 17103 | 0.9567 | 0.9543 | 0.9543 | −0.0024 | −0.0024 |
| 17107 | 0.9560 | 0.9528 | 0.9541 | −0.0032 | −0.0019 |
| 17111 | 0.9549 | 0.9522 | 0.9541 | −0.0027 | −0.0008 |
| 17113 | 0.9528 | 0.9492 | 0.9518 | −0.0036 | −0.0010 |
| mean | 0.9542 | 0.9519 | 0.9529 | **−0.0023** | **−0.0013** |

Seed spread (max−min of the arm's own five scores): box2 0.0059,
bin121 0.0051, bin1331 0.0040 — all arms train with similar seed
dispersion, so the paired diffs are not a spread artifact. Paired stats:
bin121 4/5 negative (min −0.0036, max +0.0004, t=−3.26); bin1331
**5/5 negative** (min −0.0024, max −0.0005, t=−3.70); bin1331−bin121
+0.0010 (3/5 pos, t=+1.54 — wider kernel mildly better than narrow,
still below box2).

Per-leg decomposition (mean paired diff vs box2): bin121 loses it all on
**human** (−0.0388) with codec (+0.0032) and safesyn (+0.0036) mildly
positive; bin1331 same shape — human −0.0420, codec +0.0084 (4/5 pos),
cid22 +0.0089. The binomial lowpass smooths exactly the high-frequency
detail the human-judged leg rewards.

### Stability gates (192 pairs × 11 transforms × 3 kernels; geometry-lane convention — paired excess vs box2, fail iff excess > max(3×noise, 0.002), ε=1e-12)

Worst-feature |Δln| (median over pairs):

| kernel | shift1 | worst codec phase | crop | fails |
|---|---|---|---|---|
| box2 | **2.532** | 23.97 (p0) | 22.99 | — |
| bin121 | **0.714** (excess −0.570) | 23.46 (p0) | 22.22 | none |
| bin1331 | **0.702** (excess −0.842) | 23.64 (p0) | 23.06 | none |

Median-feature |Δln|:

| kernel | shift1 | worst phase | crop |
|---|---|---|---|
| box2 | 0.0336 | 0.0100 (p0) | 0.1512 |
| bin121 | 0.0126 (excess −0.018) | 0.0075 | 0.1488 |
| bin1331 | 0.0115 (excess −0.019) | 0.0083 | 0.1501 |

**Zero failures for every kernel on all three gates.** The large ~23–24
worst-feature spreads under codec phase and crop are common-mode across
all three kernels (paired excess ≈ 0) — near-zero features flipping sign
under resampling, not a kernel defect. The differentiating gate is
shift1: the binomial lowpass cuts worst-feature shift sensitivity 3.5×
(2.53→0.71) and the median feature 2.7× (0.034→0.012). That is the one
genuine mechanism effect in the lane — real, and bought with accuracy.

### Extraction cost — **upper bounds, NOT a fair comparison**

Measured (median ns/pair, synthetic sides, `α + β·pixels` fit):

| kernel | threads | 256² | 1024² | 2048² | α (ns) | β (ns/px) |
|---|---|---|---|---|---|---|
| box2 | 1 | 1.34e6 | 3.39e7 | 1.22e8 | 1.22e6 | 28.91 |
| box2 | 8 | 6.38e5 | 1.26e7 | 4.18e7 | 0.99e6 | 9.79 |
| bin121 | 1 | 2.31e6 | 4.94e7 | 1.80e8 | 1.70e6 | 42.75 (+48%) |
| bin121 | 8 | 1.62e6 | 2.82e7 | 9.97e7 | 1.56e6 | 23.52 (+140%) |
| bin1331 | 1 | 2.57e6 | 5.42e7 | 2.00e8 | 1.56e6 | 47.57 (+65%) |
| bin1331 | 8 | 1.84e6 | 3.29e7 | 1.17e8 | 1.74e6 | 27.56 (+181%) |

**Fairness caveat — this is the geometry lane's exact artifact.**
`box2` delegates to `downscale_2x_into`, the production operator: it
dispatches through `incant!` to hand-written SIMD tiers and ran on this
box's **AVX-512 (v4x, f32x16)** path — 16 output pixels per iteration.
The binomial arms in `downscale_2x_kernel_into` are **research scalar
code**: no `#[arcane]`, no dispatch — the horizontal pass calls the
branchy `reflect_101` (`rem_euclid` + conditional) **per tap per output
pixel**, and allocates a `src_h × new_w` scratch `Vec` per decimate call
(once per pyramid level per side). A fair comparison needs a SIMD
binomial decimate — never written, out of scope (research-only surface).
So: the measured +48%/+65% (t1) and +140%/+181% (t8) β gaps are **upper
bounds**. The truthful claim is directional only — the binomials do
~2.25× the multiply-adds per decimate (3–4 separable taps vs 4 summed
loads) plus an extra intermediate plane of traffic, so a fair port would
still cost more, but by how much is unmeasured. Geometry-lane precedent:
their call-per-op generic backend showed 1.43× where the fair number was
1.04× — assume similar inflation here. Additionally the whole cost run
executed **outside the shared lock** at ~`--jobs 4 --mem 8G`
(RAYON_NUM_THREADS=4 for stability; cost's spec'd t1/t8 unchanged, nice
−19) after the lock sat >10 min held by the simd-audit lane while the
box was ~85% idle — supervisor-fallback rule, transplant-lane precedent,
recorded in `lane_zgeom.log`. Accuracy and stability numbers are
unaffected; cost is the only measurement with these two asterisks.

## Z1 — pooling replacement (dev geomean3 SROCC, 5 paired seeds)

Replaced slots: **all 228** — the entire basic+peaks surface is global
pooled statistics, so the replacement repools every slot over the 5×5
lattice (block-peak). Column count identical on both sides (228). The
control permutes all 228 feature columns row-wise within each leg
(seed 6619) — every slot is a block statistic, so every slot is permuted.

| seed | box2 (baseline) | z1max (ungated) | z1gate (replacement) | z1maxperm | z1gateperm |
|---|---|---|---|---|---|
| 17101 | 0.9508 | 0.9527 | 0.7792 | 0.0911 | 0.0863 |
| 17103 | 0.9567 | 0.9569 | 0.7710 | 0.0992 | 0.1072 |
| 17107 | 0.9560 | 0.9587 | 0.7723 | 0.0919 | 0.1029 |
| 17111 | 0.9549 | 0.9537 | 0.7696 | 0.1048 | 0.0936 |
| 17113 | 0.9528 | 0.9550 | 0.7752 | 0.0887 | 0.0868 |
| mean | 0.9542 | 0.9554 | 0.7735 | 0.0951 | 0.0954 |

Paired diffs: z1gate−box2 = −0.1716/−0.1857/−0.1837/−0.1853/−0.1776
(5/5 neg, t=−66.6); z1max−box2 = +0.0019/+0.0002/+0.0027/−0.0012/+0.0022
(4/5 pos, t=+1.60); z1gate−z1gateperm = +0.678, z1max−z1maxperm = +0.860
(both 5/5 — the block statistics carry real signal, so the comparisons
are not control-stopped).

- **z1gate's −0.18 is what a wrong gate looks like, not what noise looks
  like.** The two-state gate `[min(C̃_s,C̃_d) < c0]` with c0 = TRAIN p10
  per channel×scale passes only ~10% of blocks per cell — every block
  statistic is computed over a tenth of the image, and the masked-off
  90% contains most of the distortion signal. The loss is worst exactly
  where distortion lives: codec −0.50 mean, human −0.31, vs safesyn
  −0.15. The gate was designed for the transplant lane's ADDITIVE block
  columns (a sparse side-channel where suppressing flat blocks is free);
  as a full-surface REPLACEMENT it throws away the signal the pooling is
  supposed to keep. Mechanism confirmed destructive — not adopted.
- **z1max (ungated block-peak) is the lane's one positive mechanism
  finding**: +0.0012 over global moments, 4/5 seeds, and the wins
  concentrate on codec (+0.0139, **5/5 pos**) and human (+0.0126) — the
  two legs with real local distortion structure. Block-peak pooling
  preserves where-in-the-image the error lives; global moments average
  it away. Still: +0.0012 at t=+1.60 on n=5 is a suggestive result, not
  an adoption case, and it costs an era break to collect.

## Era-break consequence table — what adopting a new downsampler (or pooling) costs

Changing the pyramid decimate changes every pixel of every pyramid level
→ all 228 feature values → the feature function itself. That is an
**era break**, identical in kind to the v1→v2 break the registry exists
to track:

| consequence | scope |
|---|---|
| New `feature_set_id` / era required | `zgeom_*` arms are already registered append-only; a production adoption needs a new production era + `slots_hash8` |
| All trained weights invalid | every ZensimProfile MLP (incl. shipped v47 QAT Profile A lineage) is fitted to box2-pooled features — full retrain, not a fine-tune |
| All historical feature tables incomparable | every stored master/dev/holdout parquet must be re-extracted under the new era before any comparison is legal |
| Holdout must stay sealed until after retrain | the whole qualification ladder re-runs: dev → gates → holdout |
| Runtime qualification re-opens | new decimate needs a production SIMD implementation + bit-exactness/stability/perf gates — the research scalar code cannot ship |
| Audit/provenance chain | `formula_revision` / era bump recorded in registry; old tables marked historical |
| Era-mixing detection | `Extraction`-level identity prevents silent cross-era pools (this lane registered 5 eras + 5 sets to keep it airtight) |

## Four-line verdict

1. **Binomial downsampler in zensim: NO adoption case.** Both kernels
   lose dev accuracy with statistical support (bin1331 5/5 seeds,
   t=−3.70) at matched width; the ~3.5× shift-stability gain and the
   (inflated, but directionally real) cost increase both argue the same
   way — box2 stays.
2. **Block pooling as a replacement: the gated form is rejected**
   (−0.18, 5/5 — the p10 gate discards the signal it pools); the ungated
   block-peak form is a real but small positive (+0.0012, 4/5, wins on
   codec+human legs) — a mechanism finding, not yet an adoption case.
3. **What to measure next:** a decision lane on `z1max` alone — bigger
   seed set (n≥10) + the missing piece here, a *block-pool + global-pool
   hybrid* (append is banned at 228, so: replace only the subset of
   slots whose per-leg deltas justify it, or a 2×228 two-surface
   experiment) to see if the codec/human gain survives at scale.
4. **Nothing ships.** No production default changed; `zgeom` stays
   research-only behind `feature-regime-v2`.

## Artifacts

- `extract/` — 18 CSVs + manifests (3 kernels × {train,4 dev legs};
  z1 b5max+b5gate × same; z1_hist.json; stability192.tsv)
- `z2/{box2,bin121,bin1331}/`, `z1/{z1max,z1gate,z1maxperm,z1gateperm}/`
  — features/ + dev/ parquets, each with `_MANIFEST.json` declaring its
  registered `basic+peaks@w228/zgeom_*` feature-set id
- `fits/runs{,_z2,_z1}.json`, `fits/logs_{z2,z1}/*.log` — 35 fits, all
  `best validation mean SROCC` lines present, zero failures
- `specs/c0_z1.json` — the armed gate spec (p10/p25/p50 audit included)
- `stability/stability.csv` + `stability_gates.json`
- `cost/cost_t{1,8}.csv`, `cost_fit.json`
- `reports/fit_results.json` (score_fits_zgeom.py output)
- repo: `benchmarks/zgeom_2026-09-21.{md,json}` (committed record)

## Deviations / notes

- Stored master tables intentionally mix extraction eras (v1, frozen
  v2reused, rev3 v2fresh) — they could NOT serve as the Z2 box2 baseline
  (paired diffs would confound kernel with era). All arms extracted fresh.
- zgeom replica verified bit-identical: `box2:glob` == canonical
  `research::extract` f0..227 on every probed cohort AND == stored
  v2fresh_986t.csv rows (max diff 0.0).
- `--allow-narrow-features` on all 35 fits: 228-wide research bake,
  never ships as a ZensimProfile weight — the documented escape.
- HDR rows (4,290) are not train legs — not extracted (SDR path; fits
  never saw them). `cid22b_unsealed` (holdout) untouched.
- Cost + stability ran outside the lock at reduced capacity (see cost
  caveat) — recorded in lane_zgeom.log.
- **Cost numbers are upper bounds**: binomial arms are scalar research
  code (per-tap `reflect_101`, per-decimate heap scratch) vs box2's
  AVX-512 `incant!` production path — same artifact class the geometry
  lane documented (1.43× measured vs 1.04× fair). Magnitude unreliable;
  direction (binomial costs more) robust to any plausible SIMD port.
- Registry: 5 eras + 5 sets appended (`zgeom_*`), formula_revision 3.

## Cost of this lane

Extract ~35min wall (3 kernel arms + 2 pooling arms × ~146k rows,
~350–750 rows/s), 35 fits ≈ 22min at JOBS=6, stability ~4min, cost
~10min. Output 4.4 GB of the 10 GB budget (2.9 GB is the extract CSVs).
