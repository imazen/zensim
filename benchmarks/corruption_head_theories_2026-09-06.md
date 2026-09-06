# Corruption-head theories — gating, model form, what gets missed, family subtraction (2026-09-06)

**Pre-registration:** [`../docs/PLAN_CORRHEAD_THEORIES_2026-09-06.md`](../docs/PLAN_CORRHEAD_THEORIES_2026-09-06.md)
(written and pushed at `bdb46b75`, before any result was computed).
**Lane:** `claude-corrtheories`, jj sibling workspace `~/work/zen/zensim--corrtheories`.
**Artifacts:** `/mnt/v/output/zensim/corruption-head-2026-09-05/theories/` (+ `_MANIFEST.json`).
**Predecessor:** [`corruption_head_d_2026-09-05.md`](corruption_head_d_2026-09-05.md) — this
document **corrects two of its conclusions** (§3, §4 below) and leaves the rest standing.

**ERA, on every number:** everything runs at **rev1** — the post-option-C (`56bbcda2`)
extraction era, `ssim_form::SHIPPED_REVISION = Rev1`. **Revision 2 changes 12 basic slots
this head reads**, so nothing here is comparable to a rev2 re-extraction. Nothing was
re-extracted, nothing is wired into the runtime, no bake in `zensim/weights/` was
replaced, and no public API changed.

**Split:** the incumbent `d228/split.tsv`, **read verbatim, never re-derived** — 48,279
sources, source-held-out 60/20/20, so an RNG-consumption-order difference cannot move a
source between folds between arms. The reconstruction passes the pre-registered parity
gate against the incumbent's own `metrics.json`: subclass counts
`{corruption 116928, matched_anchor 348, severe_honest 60000, broad_honest 9593}` and
fold sizes `train 112033 / val 41245 / test 33591`, exact.

**Slice:** `f0..f227` (basic + peaks) for every arm — the slice that is FREE at D's
`V1PoolsMode::Peaks` walk. No arm reads `f228..371`.

**Model form is the SHIPPED one, not the reported one.** Each arm is the plain estimator
refit on train plus an `IsotonicRegression` fit on val — the model the bake carries — not
the `CalibratedClassifierCV` the trainer prints. Reproduction of the incumbent: **86.01 %
detection / 0.31 % severe FP / 11.37 % ladder FP / 0.00 % anchor FP** at T = 0.9, against
the record's baked `d228` row of 85.9 / 0.31 / 11.22 / 0.00.

**One measurement rule does all the cross-arm work.** Arms are never compared at a shared
`T`; every comparison sweeps `T` per arm to a target ladder-FP and reports detection there,
plus **pAUC₅** — the partial area under detection-vs-ladder-FP over FP ∈ [0, 5 %],
normalized to 1 and reported ×100. pAUC₅ is invariant to any monotone calibration, so it
measures the model's ordering and nothing else.

> **A measurement defect found and fixed mid-run, stated because it would have produced a
> wrong answer.** Isotonic regression is a step function, so many rows share an identical
> calibrated probability. The first T1 pass reported the per-band arm at **FP exactly
> 0.00 %** for three different targets — its top plateau held more rows than the 1 % budget —
> so "matched operating point" was not matched, and the soft-gate arm gained ~6 points of
> apparent detection purely by breaking those ties. Fixed by adding
> `eps · normalized_rank(p_raw)`, which is strictly monotone inside a plateau and cannot
> reorder across plateaus. After the fix every arm lands on the identical achieved FP
> (0.20 / 0.46 / 0.97 / 4.97 %) at every target. All numbers below are post-fix.

---

## 1. T1 — is dial-gating scientific? (test-fold, matched operating points)

| policy | pAUC₅ | det@0.25 % | det@0.5 % | det@1 % | det@5 % | det@T .9 | FP@T .9 | FP q≥95@T .9 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **(a)** no gate (incumbent) | 54.20 | 41.96 | 43.48 | 46.73 | 70.65 | 86.01 | 11.37 | 50.00 |
| **(b)** hard mask `dial < 80` | 45.96 | 43.76 | 44.36 | 45.09 | 47.09 | 39.19 | **0.00** | **0.00** |
| **(b)** hard mask `dial < 90` | 56.08 | **49.50** | **51.68** | **53.46** | 59.81 | 53.16 | 0.92 | 0.30 |
| **(c)** soft logistic gate | 54.20 | 41.96 | 43.48 | 46.73 | 70.65 | 86.01 | 11.37 | 50.00 |
| **(d)** dial as an input feature | 57.68 | 40.28 | 43.61 | 46.13 | 70.34 | 86.38 | 12.09 | 50.89 |
| **(e)** per-dial-band heads | **73.27** | 47.09 | 48.12 | 51.66 | **95.84** | 93.68 | 4.15 | 7.74 |

Detection at the matched 0.5 % point with a clustered bootstrap over test-fold sources
(1,000 resamples): (a) 43.5 [40.1, 46.4] · (b80) 44.4 [42.9, 45.7] · **(b90) 51.7
[49.6, 53.6]** · (c) 43.5 [40.1, 46.4] · (d) 43.6 [40.1, 46.8] · (e) 48.1 [46.3, 49.9].

**The pre-registered dominance criterion FAILS.** It required (d) or (e) to be ≥ (b) and (c)
at *every* target FP level; at 0.25 / 0.5 / 1 % the hard `dial < 90` mask beats the
conditional head (51.7 vs 48.1 at 0.5 %, non-overlapping CIs), and only at 5 % does the
conditional head win, by 36 points. Reported as it fell.

**(c) is a measured no-op, not a tuning failure.** The two-parameter logistic gate
`P' = P · σ((G − dial)/s)` fit by train-fold maximum likelihood converged to
**G = 110, s = 0.25** on a grid running to G = 160 — an *interior* optimum whose weight is
1.0 for every row, because D's dial never exceeds ~100. Its numbers are identical to (a) to
six significant figures. **A likelihood fit of a soft dial gate chooses not to gate.**

**(d) ≈ nothing, (e) ≫ (d): the dial matters as a CONDITION, not as a term.** Adding the
dial as a 229th input to the same linear head moves pAUC₅ 54.20 → 57.68. Splitting the head
by dial tercile — the same information, expressed as an interaction — moves it to **73.27**.
The gap between +3.5 and +19.1 points is the whole answer to "interpolation vs conditioning".

**But §5 supersedes all of this.** Every row above is a *linear* head; T8 re-runs the same
policies on a nonlinear one, where the dial gate is a large net loss.

## 2. T2 — should the head be an MLP? Yes, and a tree is better still

Identical slice, split, calibration and class weighting where the estimator supports it.

| arm | pAUC₅ | det@0.5 % | det@1 % | det@T .9 | FP@T .9 | FP q≥95@T .9 | fit |
|---|--:|--:|--:|--:|--:|--:|--:|
| `logistic` (incumbent) | 54.20 | 43.48 | 46.73 | 86.01 | 11.37 | 50.00 | 2 s |
| `mlp32` | 84.27 | 67.31 | 70.83 | 98.69 | 4.05 | 8.63 | 5 s |
| `mlp32` balanced-resampled | 89.02 | 71.97 | 76.70 | 98.54 | 3.28 | 7.74 | 5 s |
| `mlp64_32` | 97.73 | 91.02 | 97.31 | 98.68 | 2.10 | 7.14 | 5 s |
| **`hgb`** (`HistGradientBoosting`) | **98.11** | **98.52** | **98.71** | **98.90** | **1.23** | **2.38** | 22 s |

**It is not overfitting.** At each arm's own matched-0.5 % threshold, train-fold vs
held-out-source detection: logistic 46.0 / 43.5 · mlp32 68.2 / 67.3 · mlp64_32 90.4 / **91.0**
· hgb 99.2 / **98.5**. No arm has a generalization gap; the pre-registered
"wins in-sample, loses held-out" outcome did not occur.

**It is not a class-prior artifact.** `MLPClassifier` has no `class_weight` while the other
two arms use `balanced`; the balanced-resampled control moves `mlp32` in the *same*
direction (84.27 → 89.02), so the weighting is not what separates the forms.

**It is not a content confound (T6).** The positives come from imazen-26 sources and the
broad-honest negatives from the ladder's own images, so a high-capacity model could in
principle separate the two by *content*. The 2,016-row gate grid removes that freedom
entirely — 672 (corruption, q10, q20) triples from **one** reference (`gb82_dog`) that
appears in no training fold:

| arm | pAUC₅ | det@0.5 % | det@T .9 | FP same-source anchors | head `pass_q20` | DEPLOY `pass_q20` |
|---|--:|--:|--:|--:|--:|--:|
| `logistic` | 98.96 | 99.70 | 84.08 | 0.00 | 99.85 | 91.37 |
| `mlp32` | 99.26 | **100.00** | 98.36 | 0.00 | **100.00** | 99.40 |
| `mlp64_32` | 99.26 | **100.00** | 98.07 | 0.00 | **100.00** | 98.96 |
| `hgb` | 99.26 | **100.00** | 98.51 | 0.00 | **100.00** | **99.85** |
| D dial alone | — | — | — | — | — | 26.79 |

The logistic scores 98.96 pAUC₅ here too, so content identification is not what the
nonlinear arms are doing — on a single source they are all near-perfect, and the T2 gap is
cross-source generalization plus the near-lossless ladder negatives the gate grid does not
contain. `DEPLOY` is the registered `min(perceptual, gate)` composition with T = 0.9.

Second, independent argument against the confound: the incumbent's ladder FP is **0.0 % below
q50 and 50.0 % at q ≥ 95 on the same held-out images**. A content discriminant cannot be
q-dependent within an image.

## 3. T3 — what gets missed, and the real FP mechanism

**Corrects `corruption_head_d_2026-09-05.md` §4.3.** That record explains the near-lossless
false positives as *"a small-region corruption (`sq8`, `sq16`) is also almost identical to its
reference"* and concludes the confusion is *"a separability limit of the feature set, not a
coverage gap"*. Both halves are falsified by the pre-registered tests.

**(i) Small regions are the EASIEST, not the hardest** (incumbent, T = 0.9, held-out sources):

| region | `sq8` | `sq64` | `frac4` | `sq16` | `frac2` | `whole` |
|---|--:|--:|--:|--:|--:|--:|
| recall | **94.99** | 92.05 | 93.07 | 88.22 | 85.06 | **66.93** |

The misses are **whole-image, low-amplitude** edits. Worst families:
`edge_duplicate_top_row` **17.2 %** (n=87), `edge_shift_interior1px` **34.5 %**,
`channel_zero_b` **38.7 %** (its dial median is **91.0** — the dial says "fine" too),
`block_garbage` 65.1 %, `channel_zero_g` 65.1 %. Best: `chroma_boundary`,
`noise_bit_flip_n1`, `edge_border_top_k1` at 100 %. Severity runs *backwards* —
op20 90.4 % > op50 84.7 % > op100 82.9 %.

**(ii) The FP mechanism is a CHROMA confusion, measured by nearest neighbour** in the head's
own standardized input space. Of the 222 flagged honest ladder cells, the nearest positive
is `chroma_boundary` for **143 (64.4 %)**; restricted to the q ≥ 95 band it is **111 of 168
(66.1 %)**. The `sq8`/`sq16` families the record names are not in the list at all. Next:
`tone_brightness_d40neg` 11.7 %, `geometric_shift1px` 5.9 %, `aliasing` 5.4 %. So the head's
single most confident positive family is a chroma-plane boundary error, and near-lossless
AVIF/JXL output lands in that same region:

| codec | q<50 | 50–85 | 85–95 | q≥95 |
|---|--:|--:|--:|--:|
| avif-rav1e | 0.0 | 6.2 | 66.1 | **100.0** |
| jxl | 0.0 | 0.0 | 0.0 | 47.6 |
| jpeg | 0.0 | 0.0 | 1.8 | 20.3 |
| webp | 0.0 | 0.0 | 7.1 | 12.8 |
| avif-svt | 0.0 | 1.0 | 12.5 | 12.5 |

**(iii) It is not a separability limit.** `hgb` on the *identical* 228 features reads
**2.38 % q ≥ 95 FP at 98.90 % detection**, against the logistic's 50.00 % / 86.01 %. The
populations are separable; the linear boundary could not do it.

**(iv) The misses are a model-form artifact too.** Per-family recall at matched 0.5 % ladder
FP, incumbent → `hgb`: `edge_duplicate_top_row` **2.3 → 82.8**, `edge_shift_interior1px`
**8.0 → 88.5**, `channel_zero_g` 25.1 → 95.4, `channel_swap_gb` 14.0 → 96.9,
`tone_contrast_boost` 10.7 → 96.9. **`hgb`'s worst of 44 families is 82.8 %**, and by region
`whole` 39.9 → **94.5**, `sq8` 37.5 → **100.0**.

## 4. T4 — does subtracting a corruption family reduce false positives?

Leave-one-family-out over all 44 (44 refits, positives only removed, negatives untouched,
same frozen test fold). Two readings, and they disagree — the fixed-T reading is confounded.

**At a fixed T = 0.9, removing a family mostly makes things WORSE and BETTER at once**: the
class prior shifts, so both FP and detection rise together (`noise_bit_flip_n256`: FP
+2.97 pt, detection +4.89 pt). Only 8 of 44 removals lower FP at all, and against the
unpaired bootstrap noise band (baseline 11.37 %, CI [8.37, 15.14]) not one of them clears it.

**At a matched operating point the effect is real and large.** Paired bootstrap (2,000
clustered resamples of test-fold sources, difference taken on each resample):

| removed | ΔFP @ T = 0.9 | 95 % CI | Δdetection @ 0.5 % FP | 95 % CI |
|---|--:|---|--:|---|
| **`channel_zero_b`** | **−1.69 pt** | **[−2.88, −0.61]** | **+6.36 pt** | **[+5.58, +7.16]** |
| `block_garbage` | −0.26 | [−1.26, +0.59] | +1.43 | [+0.83, +2.04] |
| `channel_swap_rb` | −0.26 | [−0.77, 0.00] | +2.32 | [+1.94, +2.73] |
| `edge_border_top_k4` | −0.26 | [−0.67, 0.00] | +0.21 | [+0.11, +0.31] |
| `channel_swap_gb` | −0.15 | [−0.37, 0.00] | +2.26 | [+1.95, +2.61] |
| **greedy top-8 together** | **−4.30 pt** | **[−6.45, −2.42]** | **+12.88 pt** | **[+11.47, +14.29]** |

**Exactly one family of 44 — `channel_zero_b` — reduces honest FP on its own with a CI
excluding zero.** Every other single removal is NO EFFECT on FP. But all five top removals
raise detection at matched FP with CIs excluding zero, and the greedy cull compounds:

| k | families removed | FP @ T .9 | FP q≥85 | det @ T .9 | **det @ 0.5 % FP** |
|--:|---|--:|--:|--:|--:|
| 0 | — | 11.37 | 36.20 | 86.01 | 43.48 |
| 1 | `channel_zero_b` | 9.68 | 30.98 | 84.62 | 49.84 |
| 3 | + `block_garbage`, `channel_swap_rb` | 6.97 | 22.90 | 81.90 | 54.16 |
| 5 | + `edge_border_top_k4`, `channel_swap_gb` | **6.71** | **22.05** | 82.62 | 56.18 |
| 8 | + `noise_salt_pepper_n16`, `edge_duplicate_top_row`, `noise_bit_flip_n1` | 7.07 | 23.23 | 83.03 | **56.36** |

The detection column is measured over **all** test families including the removed ones —
`channel_zero_b`'s own recall collapses 38.7 → 10.5 % and the net is still +6.4 points, so
the family was costing more at the boundary than it contributed.

**The inverse direction exists and is chromatic.** Removing `channel_zero_b` drops other
families' recall: `channel_swap_gb` −9.4 pt, `channel_zero_g` −8.2 pt, `channel_swap_rb`
−7.7 pt; `edge_duplicate_top_row` ↔ `edge_shift_interior1px` trade −6.9 / −4.6 pt. 274
ordered pairs move by ≥ 2 pt. The channel families teach each other, which is consistent
with §3's finding that the head's confusable axis is chroma.

## 5. T8 (post-hoc, and it supersedes T1) — the dial gate was a crutch for the linear head

Not in the pre-registration; added because T1 and T2 interact. Same policies as T1, with
the T2 winners as the base head:

| base | no gate | `dial < 90` | `dial < 80` | dial as feature |
|---|--:|--:|--:|--:|
| `hgb` pAUC₅ | **98.11** | 64.93 | 49.19 | 97.86 |
| `hgb` det @ 0.5 % FP | **98.52** | 64.12 | 48.36 | 98.60 |
| `mlp64_32` pAUC₅ | **97.73** | 64.58 | 48.98 | 98.15 |
| `mlp64_32` det @ 0.5 % FP | 91.02 | 63.52 | 48.05 | **91.43** |

On a linear head the `dial < 90` mask is worth **+1.9 pAUC₅**; on a nonlinear head it costs
**−33.2**, because the mask hard-caps detection at the fraction of corruptions whose dial is
below 90 no matter how good the head is. The dial as an *input* is neutral either way
(within the arm-to-arm spread). **The gate was compensating for the model form.**

## 6. T5 — family-grouped heads

Three heads (structural / photometric / geometric, groups fixed in the pre-registration),
combined by `max`: pAUC₅ **73.65** vs the single linear head's 54.20; per-group recall at
matched 0.5 % FP rises structural 54.5 → 77.4, photometric 39.7 → 56.3, geometric
33.0 → 65.6. That is the same magnitude as T1(e)'s per-dial-band split (73.27) and both are
far below `hgb` (98.11) — **added capacity is the lever; which axis you condition on barely
matters.** A mixture is not worth its bookkeeping when one nonlinear head does better.

---

## 7. Verdicts, one per question

**"Is masking out the corruption head based on d ranges, or interpolation, a scientific
path?"** — Masking is a real, measurable effect and interpolation is not: a two-parameter
soft gate fit by maximum likelihood converges to *no gate* (G = 110 on a grid to 160, weight
1.0 everywhere, output identical to no gate to six figures). For the **linear** head the
hard `dial < 90` mask is the best low-FP policy there is (51.7 % detection at 0.5 % FP vs
43.5 % ungated, CIs disjoint) and it is the honest answer to T1 as pre-registered. But it is
a *hand-set prior that caps what the head can ever detect*, and on a nonlinear head it
destroys 33 points of pAUC₅ (98.11 → 64.93). The dial's information is worth ~3.5 pAUC₅ as
an additive input and ~19 as a conditioning variable — and ~0 once the head can bend its own
boundary. **The scientific path is to fix the model, not to gate it.**

**"Should the corruption head be an MLP instead?"** — Yes, and a gradient-boosted tree is
better still. On identical features, split and calibration: pAUC₅ 54.20 (logistic) → 97.73
(`mlp64_32`) → **98.11 (`hgb`)**; at T = 0.9, **98.90 % detection at 1.23 % honest FP and
2.38 % near-lossless FP**, against 86.01 / 11.37 / 50.00. Train ≈ test for every arm, and
the same ordering holds on the single-source gate grid (`DEPLOY pass_q20` 99.85 % vs the
logistic's 91.37 % and D's dial alone at 26.79 %). **`hgb` needs no dial guard at all** — the
guard's whole purpose was 0.74 % honest FP at 64 % gate pass; `hgb` reads 1.23 % honest FP at
99.85 % gate pass. It cannot be baked as ZNPR today (`emit_znpr` writes one identity layer
from `coef_`), so shipping one is a wire-format question, not a modelling one.

**"What kind of corruptions get missed in general?"** — For the shipped linear head:
**whole-image, low-amplitude** edits, not small localized ones. `whole` region recall 66.9 %
vs `sq8` 95.0 %; the two worst families are a duplicated top row (17.2 %) and a 1-pixel
interior shift (34.5 %), then a zeroed blue channel (38.7 %, at a dial of 91 — the dial
misses it too). Severity runs backwards: the *strongest* corruptions are the least detected.
For `hgb` the answer changes: worst family 82.8 %, worst region 94.5 %. **The miss profile
was a property of the linear boundary, not of the corruption catalogue.** Separately, the
false positives are **not** the "tiny local break looks near-lossless" mechanism the
2026-09-05 record proposes — 64.4 % of flagged honest cells (66.1 % of the q ≥ 95 ones) have
`chroma_boundary` as their nearest positive, i.e. the head confuses near-lossless chroma
with a chroma-boundary break.

**"If we subtract one type of corruption from training do false positives go down?"** — For
**one** family of 44, yes and significantly: removing `channel_zero_b` cuts honest FP by
1.69 pt (CI [−2.88, −0.61]) *and* raises matched-FP detection by 6.36 pt (CI [+5.58, +7.16]).
For the other 43, single removal is NO EFFECT on FP against a paired bootstrap. Removing the
greedy top-8 together cuts FP by 4.30 pt (CI [−6.45, −2.42]) and lifts matched-FP detection
by 12.88 pt (CI [+11.47, +14.29]). **Top-5 FP drivers:** `channel_zero_b`, `block_garbage`,
`channel_swap_rb`, `edge_border_top_k4`, `channel_swap_gb` — four of five chromatic, matching
the chroma mechanism above. Two cautions: at a fixed threshold the effect is invisible
(removing positives shifts the class prior, raising FP *and* detection together — always
compare at matched FP), and family removal buys ~13 points where switching model form buys
~55, so **it is the second lever, not the first.**

---

## 8. What was NOT done

- **Nothing wired in.** No runtime change, no bake replaced, no ZNPR emitted. A nonlinear
  head has no wire format here: `emit_znpr` writes a single identity layer from `coef_`, and
  `make_classifier`'s new forms are refused for `--bake-out` (loudly) rather than baked wrong.
- **rev2 is unmeasured.** Every number is rev1; the 12 basic slots revision 2 changes are
  slots this head reads.
- Post-hoc, labelled as such: **T8** (policies on a nonlinear base) and the isotonic-plateau
  tie-break in the measurement machinery.
- The q-anchor-relative T1(d) variant was **dropped**, as the pre-registration allows:
  corruption and negrich rows carry no `q`, so it is undefined for 177,276 of 186,869 rows.
- The T4 noise reference is the **clustered bootstrap CI**, not "fit reproducibility" as
  pre-registered — the fit is deterministic, so reproducibility is exactly zero and would
  have called every delta significant.
- `hgb`/MLP have not been through `bake_verdict --corruption-head`; their gate numbers here
  are computed in Python from the same probability vectors using the registered
  `min(perceptual, gate)` rule, as the pre-registration requires that row to state.
- Two wall-time cells (T8, T9) were measured while another lane's job held the box at load
  ~80. No statistic depends on them; every fit is deterministic.

## 9. Parity gate on the owner extension — and a reproducibility defect it exposed

The extension's claim ("`logistic` reproduces the historical estimator exactly, so the
default path is unchanged") is **gated, not asserted**. Running the incumbent `d228` recipe
through the patched owner and through `main`'s untouched copy, same box, same environment:

| comparison | `corruption_head_d228.bin` | `..._w944.bin` | `split.tsv` |
|---|---|---|---|
| patched owner vs `main` owner | **IDENTICAL** | **IDENTICAL** | IDENTICAL |
| patched owner @ 28 threads vs the SHIPPED 2026-09-05 bake | **IDENTICAL** `da411c8c…` | **IDENTICAL** `a7ad4e85…` | IDENTICAL |

**⚠ Found on the way, and it is not mine: `train_corruption_head.py`'s bake is a function of
the BLAS THREAD COUNT.** The same recipe, same data, same deterministic split, same commit,
differing only in `OMP_NUM_THREADS`, produces four different bakes:

| threads | 1 | 4 | 8 | 28 |
|---|---|---|---|---|
| `corruption_head_d228.bin` sha256 | `6f97b653…` | `1229842d…` | `23ad9c5b…` | **`da411c8c…`** |

`da411c8c…` is the shipped artifact, so the 2026-09-05 head was baked at the box's default
thread count; a `run-heavy --jobs 8` re-run of the identical command does **not** reproduce
it. The mechanism is the lbfgs solve: the split is byte-identical, the estimator is
deterministic given its input, and the residual moves only with the BLAS reduction order,
which the f16 pack then quantizes differently. The published `metrics.json` moves with it —
detection at T = 0.9 reads 0.89465 at 8 threads against the shipped 0.89527, and per-family
recalls shift by up to 0.4 pt. This is the same *class* of defect as the v1 extractor's
`RAYON_NUM_THREADS` dependence (CLAUDE.md Known Bugs), at a much smaller amplitude, and it
is **not fixed here** — fixing it means pinning the thread count (or the solver) inside the
owner, which re-dates every published head number. **Quote a head bake's sha with the thread
count that produced it, and do not treat a re-run at a different `--jobs` as a
reproduction.**

Nothing in this study is affected: every arm was fit in one process at one thread count and
every comparison is against this lane's own reproduction of the incumbent (86.01 % detection
/ 11.37 % ladder FP, against the record's baked 85.9 / 11.22).

## 10. Repro

```sh
cd ~/work/zen/zensim/scripts/v_next
run-heavy --mem 24G --jobs 8 -- python3 corrhead_theories.py prep          #  9 s
run-heavy --mem 24G --jobs 8 -- python3 corrhead_theories.py t1 t2 t3 t5   # ~2 min
run-heavy --mem 24G --jobs 8 -- python3 corrhead_theories.py t4 t7         # ~3 min
run-heavy --mem 24G --jobs 8 -- python3 corrhead_theories.py t6 t8 t9      # ~4 min
```

`prep` re-asserts the parity gate and aborts on a mismatch. Outputs land in
`/mnt/v/output/zensim/corruption-head-2026-09-05/theories/` (`t1_policies.tsv`,
`t1_detection_ci.tsv`, `t2_models.tsv`, `t2_generalization.tsv`, `t2_per_codec.tsv`,
`t3_per_family.tsv`, `t3_miss_axes.tsv`, `t3_fp_cells.tsv`, `t3_flagged_nearest_family.tsv`,
`t3_flagged_q95_nearest_family.tsv`, `t3_missed_nearest_honest.tsv`, `t4_loo.tsv`,
`t4_cross_family.tsv`, `t4_greedy.tsv`, `t5_grouped.tsv`, `t5_per_group.tsv`,
`t6_gate_samesource.tsv`, `t6_gate_pass.tsv`, `t7_paired_bootstrap.tsv`,
`t8_policy_on_nonlinear.tsv`, `t9_per_family_best.tsv`, `t9_axes_best.tsv`).

The owner `scripts/v_next/train_corruption_head.py` gained `make_classifier(name, seed)` +
`--model {logistic|mlp32|mlp64_32|hgb}` (additive; `logistic` reproduces the historical
hard-coded estimator exactly, so the default path is unchanged, and `--bake-out` is refused
for the non-linear forms). The study driver consumes that factory rather than building its
own estimators.

## 11. Addendum 2026-09-06 — the thread-count bug (§9) is fixed at the owner

§9 found, and left unfixed, that `train_corruption_head.py`'s bake was a function of
the ambient BLAS/OpenMP thread count. This addendum closes it.

**The fix.** `train_corruption_head.py` now force-sets `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`,
`NUMEXPR_NUM_THREADS`, `BLIS_NUM_THREADS` to `"1"` — unconditionally, not
`setdefault`, since the whole point is to stop depending on whatever a caller
(a shell, `run-heavy --jobs N`, a fleet worker) already exported — before `numpy`
is imported, plus a `threadpoolctl.threadpool_limits(1)` call right after import
as a belt-and-suspenders for whatever's already loaded at that point. The pin
lives at module level (before `NFEAT = 372`), so every caller of `make_classifier`
gets it, including `corrhead_theories.py`/`corrhead_tests.py`, which import it as
a module rather than subprocess it.

**Why env vars, not only `threadpoolctl`.** `threadpoolctl.threadpool_limits(1)`
only clamps libraries already loaded into the process at call time. `--model hgb`
lazily imports `HistGradientBoostingClassifier` (and therefore lazily loads
`libgomp`) *inside* `make_classifier`, after the module-level `threadpool_limits`
call has already run. Forcing the env vars before `numpy` import instead relies on
the standard, well-documented behavior that OpenMP/OpenBLAS runtimes read their
thread-count env var at first use, not at `dlopen` time — so a library loaded
later in the same process still picks up the pin.

**Test 1 — invariance across ambient thread counts (the requested proof).**
`scripts/v_next/corrhead_determinism_gate.py` runs the shipped `d228` recipe
(`corrhead_arms.sh`'s exact argv) with `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/
`MKL_NUM_THREADS` forced to 1, 4, 8 and 28 in the AMBIENT environment (simulating
a caller that has already set its own opinion) and asserts the resulting bakes are
byte-identical. MEASURED, this box, `main@478bc28e` + the fix:

| ambient threads | `corruption_head_d228.bin` sha256 | `..._w944.bin` sha256 | `metrics.json` | weights `.json` |
|---|---|---|---|---|
| 1  | `6f97b653ba5fea2d…` | `c7cc9c874921d24e…` | `65171a8270935e64…` | `73fcd75b95439975…` |
| 4  | `6f97b653ba5fea2d…` | `c7cc9c874921d24e…` | `65171a8270935e64…` | `73fcd75b95439975…` |
| 8  | `6f97b653ba5fea2d…` | `c7cc9c874921d24e…` | `65171a8270935e64…` | `73fcd75b95439975…` |
| 28 | `6f97b653ba5fea2d…` | `c7cc9c874921d24e…` | `65171a8270935e64…` | `73fcd75b95439975…` |

All four columns are byte-identical across all four thread counts (`diff` on the
raw files, not just the truncated sha above) — the fix holds on the FULL output
surface (bake bytes, the `_w944` sibling, the trainer's printed `metrics.json`,
*and* the persisted weights JSON), not only on the artifact the bug report named.
As a cross-check, `23ad9c5b…` — the UNPATCHED script at ambient 8 threads,
re-measured on this box before touching the file — reproduced §9's recorded 8T
value exactly, confirming this addendum's repro setup matches §9's byte for byte.

A synthetic (20,000×50, seed 0) smoke test of `make_classifier("hgb", seed=0)`
— the `HistGradientBoostingClassifier` candidate the task asked to check
separately, since `libgomp`'s histogram-reduction threading is a different
mechanism from BLAS's — was bit-identical across the same four ambient thread
counts **both before and after this fix** at that data scale; no thread-order
sensitivity was observed to fix. It is not shipped (`can_bake` refuses
`--bake-out` for it) and this addendum does not claim its histogram building is
provably order-invariant in general, only that a moderate-scale check found no
divergence, and that the fix's env-var mechanism defends it the same way it
defends `LogisticRegression` regardless.

**Test 2 — reproduction gate against the shipped 28-thread artifact.**
`corrhead_determinism_gate.py --skip-gate-comparison` false (the default) also
diffs the new deterministic build's sha256 against `d228/corruption_head_d228.bin`
(`da411c8c9cd6a6e216c81515714fecf76b7e3d0dcf38c9be2e11dc2f390fd8b2`, unchanged on
disk, timestamp verified untouched):

```
shipped d228.bin sha256:            da411c8c9cd6a6e216c81515714fecf76b7e3d0dcf38c9be2e11dc2f390fd8b2
new deterministic d228.bin sha256:  6f97b653ba5fea2d33f2d49b36916e53d327749eca7e2c94a76c2e37280ef2aa
```

**They differ — this is the documented, expected outcome, not a gate failure.**
Pinning to 1 thread reproduces the natural *1-thread* reduction order, which is
not the same reduction order as the historical *28-thread* ambient (unpinned) run
that produced the shipped artifact — `6f97b653…` is in fact exactly the "1T"
value §9's own table already recorded. There is no thread count you can pin to
that is simultaneously (a) independent of the ambient environment on every box
and (b) equal to a historical run's ambient-dependent value, so reproducing
`da411c8c…` exactly was never on the table once the fix forces a *fixed* count.
**The shipped bake was NOT replaced** — `d228/corruption_head_d228.bin` and its
`_w944` sibling are byte-identical to before this addendum, and no file under
`/mnt/v/output/zensim/corruption-head-2026-09-05/d228*/` was touched. The new
runs live at `/mnt/v/output/zensim/corruption-head-2026-09-05/detfix/`.

**Registered delta and its effect on detection/FP, through the existing
evaluation path.** Two independent measurements, both small:

1. *The actual baked bytes*, scored via `predict_features_with_bake` (the tool
   CLAUDE.md names for "quote the bake, not the training log") on the canonical
   `gb82_dog` held-out gate grid (672 triples × `corruption`/`q10`/`q20`, 372-wide,
   held out from training by construction — the SAME grid the D-companion's own
   `_MANIFEST.json` names as `gate_grid`), shipped vs. new-deterministic, at T=0.9
   (score < 10):

   | | detection | FP (q10 anchor) | FP (q20 anchor) |
   |---|---|---|---|
   | shipped (`da411c8c…`) | 83.929 % | 0.000 % | 0.000 % |
   | new deterministic (`6f97b653…`) | 84.077 % | 0.000 % | 0.000 % |
   | Δ (new − shipped) | **+0.149 pt** | 0.000 pt | 0.000 pt |

2. *The trainer's own held-out test-fold curve* (`metrics.json`, which — per the
   already-known, separate discrepancy this addendum does not touch — reports the
   `CalibratedClassifierCV` ensemble, not the `LogisticRegression`+isotonic model
   that actually gets baked; included here only because it is what §9's own
   magnitude claim was quoted against): T=0.9 detection **89.527 % → 89.424 %**
   (Δ −0.103 pt), `fp_broad_honest` 15.830 % → 15.881 % (Δ +0.051 pt),
   `fp_severe_honest` 0.223 % → unchanged to 3 dp; per-family recall moves up to
   **0.38 pt** (`channel_swap_rg` 84.48 % → 84.10 %) — consistent with, and inside,
   §9's own "up to 0.4 pt" characterization of the 8-thread case. Split (`split.tsv`,
   905,503 bytes) is BYTE-IDENTICAL between the shipped run and every new run — the
   PRNG-based train/val/test partition does not depend on BLAS at all, only the
   fitted weights do.

**Status: FIXED, not narrowed.** The mechanism (ambient-thread-dependent bakes) is
closed — any future `train_corruption_head.py` run, at any ambient thread count, on
any box, produces the same bytes. The *specific historical artifact*
`corruption_head_d228.bin` remains, by design, an unreproduced 28-thread-ambient
snapshot; whether to retrain and ship the now-deterministic bake (a <0.15 pt
detection change, 0 pt FP change, on the canonical gate) is a separate, ungated
product decision this addendum does not make.

**Repro**: `python3 scripts/v_next/corrhead_determinism_gate.py` (no args needed;
defaults point at the canonical `d228` recipe and the shipped bake). ~35 s total
(4 fits × ~7 s + one gate-grid scoring pass). Exit 0 = determinism holds.
