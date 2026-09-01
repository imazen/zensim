# Failure profiles — what each model gets wrong, and where that bites

**The complaint this answers (user, 2026-08-31):** the gauntlet board shows
*minimal information about a model's flaws and where those flaws will hurt*. It
is a leaderboard — SROCC columns, gates, badges — and a reader cannot see what a
given model gets WRONG or in which production situation that failure bites.

**The principle everything below is built to:** translate every statistic into a
production consequence. "nonphoto SROCC 0.845" is not a flaw statement. "On the
near-lossless band it ranks 16.3 % of reference ladders backwards" is. Every row
names (a) the failure, (b) its size, (c) the situation where a user meets it,
and (d) the evidence — and says NOT MEASURED plainly where that applies, never a
blank and never a zero.

**What shipped**

1. An inventory of the failure evidence the 379 board cells *already* carried
   (§2), with the statistic → situation mapping (§3).
2. One new board-wide measurement — **ladder inversions split by codec ×
   quality zone and content class × zone**, plus the worst ladders *by reference
   image name* (§4). 322 of 379 cells; the other 57 each carry a reason.
3. The **Failure profile** panel on the board (§6), which reads all of it and
   recomputes none of it.
4. Two findings that are corrections to how existing board numbers must be read
   (§5), both registered in `benchmarks/eval_annotations.json`.

Artifacts + shas: [`failure_profiles_2026-08-31.pointer.md`](failure_profiles_2026-08-31.pointer.md).
Board: <http://localhost:3300/zensim/reports/summer_gauntlet.html>.

---

## 1. Why "ladder inversions" is the measurement that was missing

zensim's product is a **consistent dial**: a user types a target score, the
codec tunes to hit it, using the diffmap to close the loop. What a codec loop
consumes is not a pooled SROCC over a corpus — it is *"of these two encodes of
THIS image, which is better?"*. The board had exactly one number of that shape,
`dial.mono_pct`, pooled over every codec and every quality at once.

That pooling is where the information was. A model can read `mono_pct 0.985` —
a pass — while inverting 6.2 % of avif rungs in the near-lossless band and 0 %
of jxl rungs anywhere. The loop that meets the 6.2 % is a real loop, tuning a
real codec, at the quality most web pipelines actually operate at.

So the split is on the two axes a reader meets the failure on — **which codec**
and **which quality band** — plus **which content**, and it is reported at two
granularities: per adjacent rung pair (does the next step up score higher?) and
per whole ladder (does the ladder's best-quality end score above its worst?).

## 2. Inventory — what failure evidence already existed

Measured by scanning all 379 `*.fulleval.json` (no re-measurement):

| evidence | cells carrying it | what it can say about failure |
|---|--:|---|
| `rank.<corpus>.srocc_signed` | 379 | a **negative** on a quality-oriented corpus = ranks it backwards outright |
| `rank.<corpus>.or` | 379 | outlier ratio — catastrophic single-pair errors |
| `rank.<corpus>.z_rmse` | 379 | calibration error in human-rating units |
| `rank.<corpus>.srocc_ci` | 375 | is a gap between two models inside noise |
| `rank.<corpus>.train_eq_val` | 375 | which corpora reward memorisation (KADID, TID) |
| `rank.<corpus>.per_ref_mean` | 374 | within-image ranking — the loop's criterion |
| **`rank.<corpus>.frac_negative`** | **374** | **share of whole reference ladders ranked BACKWARDS** |
| `rank.<corpus>.bands[]` (usable) | 215 | where in the quality range the ranking fails |
| `dial.mono_pct` / `tied_pct` | 378 | pooled backwards rate; dead-zone rate |
| `dial.p5` / `p95` / `reach` | 374 | which targets are unreachable |
| `dial.per_codec[]` | 374 | which codec family the pooled number hides |
| `corruption.pass_q20` + `per_family` | 367 | can a corrupted image score as good |
| `corruption_head` | 9 | the shipping design's corruption owner |
| `m3_coherence` / `m3a_coherence` | 211 | is the steering map usable at all |
| `block_profile` | 183 | which feature blocks are live → wrong-root exposure |
| `repro` | 346 | can the model be rebuilt |
| `dominated_by` | 17 | strictly beaten by a same-class sibling |
| `per_pair` predictions | 212 | raw scatter (curated cells only, by the size rule) |

**`frac_negative` was the buried lede.** It is the share of *references* whose
within-image ranking is inverted — the ADD156 audit's "shipped `B` ranks 21 % of
HF near-lossless reference ladders backwards while ADD156 ranks 0 %" is exactly
this field. It was on every cell and on no panel. It is now a scoreboard-adjacent
column and a first-class failure row.

**Two evidence gaps found, and NOT closed here:**

* **Per-reference SROCC exists only as an aggregate.** `zenstats::PerGroupSrocc`
  returns `n_groups / mean / median / frac_negative / frac_perfect` — the
  per-group values are not returned, so *naming* a model's worst reference on a
  rank corpus is not possible from stored evidence. `bake_verdict
  --per-pair-output --per-pair-refs` emits an *interned group id*, not a name.
  Closing this means extending `zenstats` (a sibling repo) or the loader; it is
  flagged, not done. §4's named worst ladders close the equivalent gap on the
  **dial** side, where `image_id` is carried.
* **KonJND carries no per-reference statistic at all** (0 of 379 cells) — the
  JND corpora are scored per pair, not per reference group.
* **G-RANGE (extrapolation exposure)** — the share of rows whose raw prediction
  leaves the output spline's knot domain — is owned by `bake_dial_refit gate`
  and is not run per board cell. It has been run for exactly one model
  (`benchmarks/add156_ship_audit_2026-08-31.md` §1.4, which found 8 of 14
  corpora failing and **100 % of HF near-lossless rows above the top knot**).
  The panel says NOT MEASURED with that reason rather than inventing a proxy.

## 3. Statistic → production situation

The mapping the panel encodes. Left column is what the verdict stores; right
column is what it predicts about a user.

| statistic | the production situation it predicts |
|---|---|
| `srocc_signed < 0` on a quality-oriented corpus | the metric ranks that whole content type backwards — anything automated on it optimises the wrong way |
| `frac_negative` (per corpus) | **share of images where a per-image tuning loop walks the wrong way**; on `hfnlproxy` that is a q90+ web pipeline |
| `per_ref_mean` ≪ pooled `srocc` | the model separates *images* well and *encodes of one image* badly — pooled looks fine, the loop does not work |
| `bands[].srocc_signed` (with `n`, `span`) | where in the quality range it fails; the top band is "two encodes already close in quality", the near-lossless decision |
| `dial.zones` codec × zone `inv_rate` | raising that codec's quality in that band can LOWER the reported score |
| `dial.zones` `ladders_ends_backwards` | the whole ladder runs backwards — a loop pointed there converges to the worst encode |
| `dial.zones` `inv_mag_max` | how large a single wrong step can be, in dial points a user targets in |
| `dial.zones` class × zone | which *content* it mis-orders: text/line-art, rendered/synthetic, photographic |
| `dial.tied_pct` (dead zone) | distinct encodes score identically — the loop cannot discriminate and stops converging |
| `dial.p5 > 25` / `p95 < 85` | those targets are unreachable; the loop saturates and returns the wrong bytes |
| `corruption.pass_q20`, `per_family` at 0 | a truncated / mis-decoded / occluded image can be reported as good and shipped |
| `m3a_coherence < 0.85` (or `m3` < 0.70) | per-block steering points at the wrong blocks — bits spent where the metric will not reward them |
| `or` (outlier ratio) | rate of catastrophic single-pair errors — the visible-to-a-human mistakes |
| `z_rmse` | calibration error: the number is right in rank and wrong in value |
| `train_eq_val` | that corpus number is an integrity guard, not evidence of generalisation |
| `block_profile.uses_f156_371` + regime | scored at a folded root this model reads structural zeros and still returns a plausible number |
| `repro` absent | it cannot be rebuilt; a regression cannot be bisected |
| `srocc_ci` | whether a gap to another model is inside noise |

## 4. The measurement: ladder inversions by codec × zone × content

### 4.1 Method

Implemented **in the existing owner** — `bake_verdict`'s DIAL panel already
classifies every adjacent-rung pair of every (image, codec) ladder into five
mutually-exclusive outcomes (forward / material inversion / codec-saturated /
flat-clamp dead zone / sub-resolution). The change buckets those SAME events by:

* **quality zone**, cut on the grid's normalised `q` axis at **[50, 85]** —
  `q<50` aggressive web compression, `q50-85` ordinary web quality, `q>=85`
  high-fidelity / near-lossless. A rung PAIR is keyed on its q **midpoint**
  (the decision "is the next step up better" is made between the two rungs).
* **codec** (avif / jpeg / jxl / webp).
* **content class** — a recorded **hand review** of the 39 dial-grid reference
  images (`benchmarks/dial_grid_content_classes_2026-08-31.tsv`): every source
  PNG was rendered into contact sheets and labelled by eye as `photo` (27),
  `text_lineart` (9 — sheet music, handwriting, engraved lettering) or
  `nonphoto` (3 — a flat-shaded illustration, a CGI render, a display capture
  whose dither grid dominates). Deliberately **not** a feature-threshold
  classifier, which would produce a plausible split nobody can audit. An image
  the table does not name goes to an explicit `unclassified` bucket.

Two ladder-level statements are added, because they are what a loop meets:
**`ladders_with_inv`** (share carrying ≥1 material backwards rung in the zone)
and **`ladders_ends_backwards`** (share whose best-quality rung in the zone
scores materially BELOW its worst-quality rung). Plus the worst ladders **by
reference image name**.

Material threshold is the gate's own 0.5 dial points, lifted to one module-level
constant so gate and split cannot drift apart. **The `all` rows re-derive the
pooled counters and the run asserts it** — verified on ADD156: 4,318 pairs / 65
inversions → `1 − 65/4318 = 0.9849467345993516` = the stored `mono_pct` to the
last digit.

### 4.2 Running it board-wide without guessing the regime

The board's dial cells were **not all measured on one grid**, and a mismatched
grid does not always fail loudly (the `--regime 944` hazard). So
`scripts/v_next/measure_dial_zones.py` does not choose: for each cell it re-runs
`bake_verdict` under every dial grid on disk and **accepts only the run whose
pooled dial block (`mono_pct`/`tied_pct`/`p5`/`p95`/`reach`/`per_codec`/`curves`)
is byte-identical to the value already on the board.** Same bake + same pooled
dial numbers = same grid, same regime, same code path.
`promote_fulleval.py --graft-dial-zones` writes under the same gate.

Result: **322 of 379 cells measured, 0 graft failures.** Grids that reproduced
the board: 944 (296), 372-quarantined-v2 (13), **un-quarantined 2026-05-29 372
(7 — see §5.1)**, 720 (5), wlin7b carriers POOLS with `--cross-regime` (1).

The 57 not measured, each with its reason on the board:

| n | reason |
|--:|---|
| 37 | no dial grid on disk reproduces the cell's own dial block (34 are the `sota944_A_shaped_*_w` family, 3 the `foldcanon`/`kbase` 720 family) — the grid they were cut on is not on this machine |
| 14 | ensemble — the dial panel scores one ZNPR |
| 4 | peer reference metric (ssim2 / butteraugli / cvvdp / iwssim) — `bake_verdict` does not run on a reference metric |
| 1 | `coherent924_selected` — the board cell carries no dial block |
| 1 | `ebothg_m504` — no dial grid exists at 504 features |

### 4.3 Board-wide results (322 cells)

**The near-lossless band is where the board fails.**

| zone | median inv rate | p90 | max | cells with ≥1 ladder ending backwards |
|---|--:|--:|--:|--:|
| `q<50` | 0.76 % | 2.39 % | 17.45 % | 48 / 322 |
| `q50-85` | 0.79 % | 1.50 % | 7.23 % | 58 / 322 |
| **`q>=85`** | **2.83 %** | **8.56 %** | **17.62 %** | **189 / 322** |

**59 % of measured models carry at least one whole ladder that runs backwards
in the near-lossless band**, against 15 % and 18 % in the two lower bands.

**AVIF is the worst codec; WebP is nearly clean** (q≥85, across all 322 cells):

| codec | median inv | p90 | max | cells with a backwards ladder |
|---|--:|--:|--:|--:|
| avif | 3.64 % | 14.03 % | 31.30 % | 185 |
| jxl | 3.07 % | 11.88 % | 21.46 % | 108 |
| jpeg | 0.40 % | 1.58 % | 14.03 % | 72 |
| webp | 0.00 % | 0.76 % | 6.53 % | 8 |

At `q<50` the ordering changes: avif 1.43 % median, jpeg 1.09 %, jxl and webp
0.00 %. So "which codec is risky" is a function of the band, which is precisely
what the pooled number could not say.

**Content class** (q≥85 medians): `nonphoto` 3.62 %, `photo` 2.75 %,
`text_lineart` 1.99 %. **Read the n:** `nonphoto` is 3 reference images (12
ladders) — its rate is coarse and its ordering against `photo` (27 images) is
not a reliable claim. The honest reading is that no class is clean at q≥85, not
that synthetic content is worst.

### 4.4 The discussion set (`2026-08-31-era2-fast-profile`) + incumbents

`all` rows, from `ladder_inversions_2026-08-31.tsv`. Cells on different grids
are not directly comparable — the grid is named for that reason.

| model | grid | zone | rung pairs | inv | inv % | ladders | ≥1 inv | ends backwards | worst step |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|
| `ADD156…@cur372` | 372-quar-v2 | q<50 | 722 | 2 | 0.28 % | 96 | 2 | 0 | 3.6 |
| | | q50-85 | 802 | 10 | 1.25 % | 106 | 6 | 0 | 6.8 |
| | | **q>=85** | 2794 | 53 | **1.90 %** | 106 | 15 | **2** | 12.8 |
| `b_sdr_…dense_dial` (shipped **B**) | 372-quar-v2 | q<50 | 722 | 11 | 1.52 % | 96 | 11 | 0 | **30.2** |
| | | q50-85 | 802 | 17 | 2.12 % | 106 | 13 | 0 | 12.2 |
| | | **q>=85** | 2794 | 62 | **2.22 %** | 106 | 18 | **2** | 19.2 |
| `W10L9P_s4005_packed` | 944 | q<50 | 794 | 6 | 0.76 % | 105 | 5 | 0 | 5.2 |
| | | q50-85 | 883 | 8 | 0.91 % | 115 | 5 | 0 | 3.1 |
| | | **q>=85** | 3025 | 11 | **0.36 %** | 115 | 7 | **0** | 15.7 |
| `W10L9_s4003_packed` | 944 | q<50 | 794 | 8 | 1.01 % | 105 | 7 | 0 | 9.3 |
| | | q50-85 | 883 | 6 | 0.68 % | 115 | 4 | 0 | 4.3 |
| | | **q>=85** | 3025 | 18 | **0.60 %** | 115 | 11 | **0** | 18.3 |
| `Q7b_pools_g0.2_a0.2_b0.97` | 944-POOLS | q<50 | 794 | 6 | 0.76 % | 105 | 6 | 0 | 30.6 |
| | | q50-85 | 883 | 10 | 1.13 % | 115 | 5 | 0 | 35.7 |
| | | **q>=85** | 3025 | 16 | **0.53 %** | 115 | 11 | **0** | **91.3** |
| `ADD156…` (era row) | 372-**UNQUAR** | q>=85 | 3025 | 76 | 2.51 % | 115 | 38 | 16 | 108.2 |
| `v47_strict_QAT_native` | 372-**UNQUAR** | q>=85 | 3025 | 94 | 3.11 % | 115 | 53 | 16 | 56.8 |

Read in product terms:

* **ADD156 vs shipped B, same grid, near-lossless band:** ADD156 inverts 1.90 %
  of rungs, B inverts 2.22 %; both end 2 of 106 ladders backwards. At `q<50` the
  gap is larger and in ADD156's favour — B inverts 1.52 % of aggressive-compression
  rungs against ADD156's 0.28 %, and B's worst single reversal there is **30.2
  dial points**: at low quality, B can report a materially better encode as 30
  points worse. That is a new statement about the shipped default that the
  pooled `mono_pct` (0.9792) does not contain.
* **The 944 models are materially cleaner on ladders than either 372-era
  model** — `W10L9P_s4005_packed` inverts 0.36 % at q≥85 with no backwards
  ladders in any band.
* **`Q7b_pools` has no backwards ladders but the deepest single step on the
  set — 91.3 dial points at q≥85.** Low frequency, high amplitude: a loop that
  hits it once takes a very large wrong step. That is a different risk shape
  from B's, and the two are indistinguishable in a `mono_pct` column.

### 4.5 The worst ladders, by name

`worst_ladders_2026-08-31.tsv` names them. ADD156 on the quarantined grid, the
only two `q>=85` ladders that end backwards:

| reference image | codec | content | end delta | worst step |
|---|---|---|--:|--:|
| `00b13be94a4867dd_1022x818` | avif | photo | **−5.19 pts** | 12.83 pts |
| `f65a24b7e176eb47_1022x818` | avif | text_lineart | −2.47 pts | 6.66 pts |

Both are avif near-lossless; the sources are at
`/mnt/v/input/zensim/sources/<image_id>.png` (a waterfront cityscape and a page
of printed sheet music). This is the form a reader can act on — go and look at
the image.

## 5. Two corrections to how existing board numbers must be read

Both registered in `benchmarks/eval_annotations.json` (append-only), so
`freeze_check --annotations` and the board's ⚠ badges carry them.

### 5.1 Seven cells' dial numbers are inflated by a known-defective grid

`dial-cells-on-unquarantined-2026-05-29-grid`. The byte-identity gate found that
**7 board cells** — `ADD156_safesyn_only_raw_lasso`, `v47_strict_QAT_native`,
`Ebothg_scr0_5_dial`, `bhdr_linear_shaped_cvvdpmix`,
`cl_tfm_corruption_LQ_MLP_s13`, `v02_bvls_NO_shaping`,
`winner_dial_Ebothg_hfgain_winsor_dial` — have their entire dial block cut on
`dial_grid_372col_2026-05-29.parquet`, the **un-quarantined** grid, which carries
both documented defects (the 9 w11-corrupt masked/IW ladders and the 33 JXL cells
at butteraugli distance 0.025 encoded before jxl-encoder `eeb52735`).

MEASURED consequence: on that grid **66.7 % of the 372 features GROW by 5–8
orders of magnitude** from the `q99.8` rung to the `q99.9` rung; on the 944 grid,
whose `q99.9` rung is healthy, **0.6 % grow and by ≤1.55×**. Every model scored
on the bad grid therefore drops ~19 dial points at that rung on 23 of 33 JXL
ladders — and **that drop is correct**: it is the model scoring a broken encode
down. So those cells' `mono_pct`, inversion rates and backwards-ladder counts are
inflated **by the grid, not by the model**: ADD156 reads 16 of 115 backwards
ladders at q≥85 there against **2 of 106** on `_quarantined_v2`.

*This started life as this pass's headline finding — "the default grid hides a
near-lossless collapse in the 372-era models" — and was falsified by measuring
the features at the rung before publishing it.* The probe that settled it is
kept at `failure-profiles-2026-08-31/nearlossless/`:

| model | grid | median jxl score q99.7 → q99.8 → q99.9 | Δ(99.8→99.9) | ladders dropping |
|---|---|---|--:|--:|
| ADD156 | 372-unquar | 95.86 → 96.21 → 78.76 | −17.45 | 23 / 33 |
| shipped B | 372-unquar | 95.66 → 95.95 → 76.63 | −19.31 | 23 / 33 |
| v47 QAT | 372-unquar | 94.80 → 95.18 → 74.87 | −19.09 | 23 / 33 |
| `W10L9P_s4005` | 944 | 96.18 → 96.17 → 96.24 | +0.02 | 1 / 33 |
| `W10L9_s4003` | 944 | 96.84 → 96.77 → 96.57 | +0.05 | 6 / 33 |

The right conclusion is the boring one: the quarantine is correct, the seven
cells need re-scoring on `_quarantined_v2` (or reading via their `@cur372`
siblings), and no near-lossless model failure is demonstrated by this rung.

### 5.2 AIC-4's per-reference inversion is a corpus property

`aic4-corpus-wide-per-ref-inversion`. The ADD156 audit noted (§3.4) that AIC-4
inverts for two unrelated bakes and that no registry entry covered it. Measured
board-wide over **373 cells**: median **60.0 %** of AIC-4's 5 references
backwards, median per-reference SROCC **−0.018**, median pooled signed SROCC
**−0.903**, 187 of 373 cells above 10 % backwards. `sdr25` (a subset) reads
median 20.0 % over 356 cells. These are corpus-level facts; the panel reports
them but never as the model's defect.

### 5.3 The board-wide near-lossless picture that IS real

Independently of §5.1, near-lossless is where the board is weakest — and it is
the zone that describes a q90+ web pipeline. **Two distinct corpora carry it and
they must not be conflated.**

**`hf_nearlossless`** — the real 48-reference HF corpus, the one the ADD156 audit
used. It is scored on only **13 of 379 board cells**, and it is where models
separate hardest:

| board cell | pooled SROCC | per-reference mean | **% of 48 refs backwards** |
|---|--:|--:|--:|
| `ADD156_safesyn_only_raw_lasso@cur372` | 0.458 | 0.949 | **0.0 %** |
| `v47_strict_QAT_native@cur372` | 0.622 | 0.948 | 0.0 % |
| `v02_bvls_shaped@cur372` | 0.152 | 0.946 | 0.0 % |
| `T_appT_b372_lam1e-3` | 0.465 | 0.946 | 0.0 % |
| `Ebothg_scr0_5_dial@cur372` | 0.712 | 0.939 | 0.0 % |
| `cl_tfm_corruption_LQ_MLP_s13@cur372` | 0.063 | 0.818 | 0.0 % (36 refs) |
| `v02_bvls_NO_shaping@cur372` | 0.052 | 0.810 | 2.1 % |
| `bhdr_linear_shaped_cvvdpmix@cur372` | 0.764 | 0.729 | 6.3 % |
| **`b_sdr_linear_cid80_inclwinsor_dense_dial` (shipped B)** | 0.614 | **0.488** | **20.8 %** |
| `winner_dial_Ebothg_hfgain_winsor_dial@cur372` | 0.587 | **−0.639** | **85.4 %** |
| `mlp_2L_diverse_H128@cur372` | 0.299 | **−0.666** | **87.5 %** |

This reproduces the audit exactly — shipped **B** ranks **20.8 %** of HF
near-lossless reference ladders backwards while ADD156 ranks **0 %** — and adds
two cells the audit did not look at that are far worse: `mlp_2L_diverse_H128`
and `winner_dial_Ebothg_hfgain_winsor_dial` invert **~86 %** of references
there, with a *negative per-reference mean*, while both still publish a positive
pooled SROCC (0.299 and 0.587). **A healthy pooled number sitting on top of an
inverted per-reference mean is the single most misleading shape on the board**,
and it is now a `blocker` row on the panel.

**`hfnlproxy`** is a different, larger population — 757 references derived by
`derive_hfnlproxy_372.py` from the imazen26 reslice — scored on 372 cells. It is
a proxy, not the HF corpus, and its numbers are not comparable with the table
above:

| corpus | cells | refs | median % refs backwards | p90 | max | cells > 10 % | median per-ref | median pooled |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **`hfnlproxy`** | 372 | 757 | **14.9 %** | 52.8 % | 80.2 % | **208** | 0.498 | 0.204 |
| `kadid` | 374 | 81 | 0.0 % | 7.4 % | 53.1 % | 23 | 0.773 | 0.607 |
| `csiq` | 373 | 30 | 0.0 % | 0.0 % | 53.3 % | 6 | 0.907 | 0.902 |
| `live` | 373 | 29 | 0.0 % | 0.0 % | 48.3 % | 5 | 0.931 | 0.931 |
| `cid22` | 374 | 49 | 0.0 % | 0.0 % | 16.3 % | 1 | 0.953 | 0.877 |
| `imazen26` | 374 | 632 | 0.0 % | 0.4 % | 42.6 % | 2 | 0.919 | 0.908 |
| `nonphoto` | 374 | 1115 | 0.0 % | 0.2 % | 47.5 % | 2 | 0.925 | 0.911 |
| `tid` | 374 | 25 | 0.0 % | 0.0 % | 4.0 % | 0 | 0.892 | 0.891 |
| `aic3` | 373 | 10 | 0.0 % | 0.0 % | 30.0 % | 2 | 0.941 | 0.788 |

**On the proxy, 208 of 372 cells rank more than 10 % of references backwards and
99 are anti-correlated pooled** — so the failure the audit found on B is not an
outlier, it is the shape of the board's weakest axis at the population level.
Every other corpus has a median of 0.0 % backwards. Other quality-oriented
anti-correlations across the board: `kadid` 77 cells (the ext-lineage inversion,
already registered), `live` 2, `csiq` 1, `tid` 1.

Note the `hfnlproxy` medians are far milder than the `hf_nearlossless` table
(B reads 1.3 % backwards on the proxy against 20.8 % on the real corpus). The
proxy is easier; **do not substitute it for the HF corpus when the question is
near-lossless ranking.** Only 13 cells carry the real one — closing that
coverage gap is the highest-value follow-up this pass identifies.

## 6. What the board panel shows

New section **"Failure profile — what breaks, how big, where you meet it"**,
mounted directly under the scoreboard so it is the second thing a reader sees.

* **A side-by-side comparison table** over the visible set (the discussion-set
  dropdown drives it): blockers, serious, ladders backwards at q≥85, ladder-inv
  at q≥85 and at q<50, references backwards on HF-NL and on CID22, dial
  dead-zone, corruption pass rate, M3a, and the worst single reversal in dial
  points. Sortable. This is the "choosing between these two models" view.
* **A per-model card** listing findings **ranked by product impact**, each in the
  four-part form — *what breaks* / *how big* / *where you meet it* / *evidence
  (the exact fulleval path)*. Severity is `blocker` (a codec loop is led the
  wrong way) / `serious` (a gate the shipping design depends on) / `watch`
  (qualified or thin evidence).
* **The named worst ladders**, per model, as a sortable sub-table (reference
  image, codec, content class, zone, endpoint delta, worst step) with the source
  PNG path.
* **The honest inverse** — a "reliably good at" list: corpora ranked well *and*
  consistently per image, codecs never ordered backwards in any band, a monotone
  dead-zone-free dial, a coherent attribution map.
* **An explicit NOT MEASURED list with reasons** on every card, including the
  two that apply to every cell (G-RANGE extrapolation exposure; G-RD / G-TARGET
  bytes-at-equal-quality).

**Nothing on the page is recomputed.** Every number is read from the verdict;
the page only thresholds into severity and names the situation. Payload is the
~1.7 KB compact zone projection rather than the 9.4 KB verdict block (which
would have added 3.0 MB), and `dial` is embedded with `zones` stripped.

**Size:** 19,946,713 → 20,696,199 bytes (**+749,486, +3.76 %**), 19.02 → 19.74
MiB. The documented cap is 12 MB and the board passed it long before this pass;
if it must come down, the lever is the registered size rule — per-pair scatter
is already curated-only, and dropping it for the *dominated* curated cells, or
capping `SCATTER_MAX`, is worth ~1 MB per 8 curated cells. That is a board-owner
decision, not one this pass took.

**Gates:** `scripts/v_next/gauntlet_gates.sh` **PASS** — `node --check` on both
script blocks; DOM-shim render (362 bakes, 13 sections, 29 tables, 598 rows,
sort clicks reorder attached tables, ECharts mounts + per-panel SSR, 974
registry-annotated cells carry ⚠). The harness gained a **failure-panel test**
that fails on a blank panel: heading present, one comparison row per visible
bake, severity-tagged findings actually rendered, and — the honest-reporting
half — the NOT MEASURED cells must match the payload exactly (a cell without
zones must read NOT MEASURED, never 0 %). It caught a real bug before ship:
`corruption.per_family` ships as a list on bake cells and as an object on peer
rows, and the list-only read threw during render.

## 7. What is NOT measured, and why

| not measured | why |
|---|---|
| ladder inversions for 57 board cells | §4.2 — 37 grids not on this machine, 14 ensembles, 4 peers, 1 no dial block, 1 no 504 grid |
| ladder inversions for the 4 **peer** metrics (ssim2, butteraugli, cvvdp, iwssim) | `bake_verdict` does not score a reference metric on the dial grid — so "does ssim2 invert ladders?", the natural comparator, is open |
| a model's worst reference **by name** on a rank corpus | `zenstats::PerGroupSrocc` returns aggregates only; `--per-pair-refs` gives an interned id. Named worst ladders exist on the dial side only |
| per-reference statistics on KonJND | 0 of 379 cells carry one |
| G-RANGE extrapolation exposure | owned by `bake_dial_refit gate`, not run per board cell; measured for one model in the ADD156 audit |
| G-RD / G-TARGET (bytes at equal judged quality, target hitting) | owned by the codec probe matrix; only mapped bakes carry the JXL loop panel |
| M3 / M3a for 168 cells, `block_profile` for 196, `corruption_head` for 370 | the instruments were not run for those cells; absence is not a fail |

## 8. Reproduce

```sh
cargo build --release -p zensim-validate --bin bake_verdict --bin panel
# board-wide measurement + graft (193 s, 4 workers)
ZL_BV=target/release/bake_verdict python3 scripts/v_next/measure_dial_zones.py --jobs 4
# evidence tables
python3 scripts/v_next/failure_profile_report.py
# board + the mandatory gates
ZEN_PANEL_BIN=target/release/panel python3 scripts/v_next/gauntlet.py \
  --out /mnt/v/output/zensim/reports/summer_gauntlet.html
scripts/v_next/gauntlet_gates.sh /mnt/v/output/zensim/reports/summer_gauntlet.html
```
