# R6 — the F4 arm, decided by measurement

**2026-09-05.** Runs the gate
[`../docs/PLAN_FEATURE_REV2_2026-09-05.md`](../docs/PLAN_FEATURE_REV2_2026-09-05.md)
§7, pre-registered and pushed (`090d55d7`) before a single table was extracted.
It is the one gate
[`feature_rev2_2026-09-05.md`](feature_rev2_2026-09-05.md) §1.6 could not run:
the cheap ladder proxy gives **identical** results for all four arms, so the
choice needed a real monotone-linear fit on real corpora.

Artifacts: `/mnt/v/output/zensim/rev2-2026-09-05/r6/`.
Drivers: `scripts/r6_extract_arms.sh`, `r6_pack_arm.py`, `r6_fit_arms.sh`,
`r6_eval_arms.sh`, `r6_dial_arms.sh`, `r6_build_dial_instruments.py`,
`r6_arm_delta.py`, `r6_decide.py`, `r6_safesyn_subset.py`.

---

## 0. How the four arms were produced

**One binary, four arms, no rebuild between them.** `ZENSIM_SSIM_LUMA` ∈
`{ssim2, c1, lorentz, clamp}` selects `ssim_form::SsimLumaForm` at runtime;
`ssim2` **is** the shipped revision-1 form. Built once from `main@origin`
`ceb86c2d`. A rebuild alone has been measured in this repo to move a 2304²
timing ~10 %; it is not allowed near the FEATURES.

| leg | rows | pixels |
|---|--:|---|
| safesyn (training) | 196,086 | bitstreams — `.jpg` 111,068 / `.avif` 34,001 / `.jxl` 26,362 / `.webp` 24,655, pre-scanned **0 missing** |
| cid22val / kadid / tid / konjnd / aic3 / csiq / live | 4,292 / 10,125 / 3,000 / 1,008 / 600 / 866 / 779 | the datasets `build_eval372_root.sh` already owns |
| ladder dial grid | 9,593 | the 2026-09-05 floor-dense instrument's own pairs list |
| negative-tail probe | ≤2,000 | the registered rule (`ssim2 < 0`) applied to the arm's OWN safesyn table |
| identity | 400 | self-pairs (`ref == dist`) over 3,277 distinct references |

**Decoder era is an input, not a footnote** (§3.34 measured decoder era at 73 %
of the extractor-era shift): every leg decodes through
`zensim-bench/examples/shared/zen_decode.rs` — `zencodec` magic-byte detect,
then zenjpeg / zenpng / zenwebp / zenavif / zenjxl — at that one build commit,
recorded per format in every `_MANIFEST.json`.

**Not produced, and NOT copied in:** sdr25, aic4, nonphoto, imazen26,
hfnlproxy, hf_nearlossless (byte-copies in the postC root, not re-extractable on
this box) and pipal (outside the decision set). `pack_eval372_root.py` gained an
`EVAL372_NO_STORED` mode for exactly this: copying a stored table is right for a
root that changes only the EXTRACTOR era, and wrong for one that changes the
FORMULA, because the copy is then a different arithmetic revision sitting inside
a directory that claims to be one arm.

---

## 1. ★ C1 — the pipeline control PASSES, on all seven corpora

At arm `ssim2`, the extraction is **byte-identical** to the registered postC 372
root's own source CSVs — `cid22 kadid tid konjnd aic3 csiq live`, `cmp` clean on
every one. Two things follow, and neither was assumed:

* the arms are comparable, because the `ssim2` arm IS the published era; and
* the rev2 lane's R1 byte-identity claim reproduces **end to end on real
  corpora**, at a later commit (`ceb86c2d` vs the root's `4fbd8ff8`), not merely
  on the 22,397-row invariant dump.

---

## 2. ⛔ A CORRECTION TO THIS LANE'S OWN PRE-REGISTRATION — "944 sees only the basic 36" is FALSE for one of the 944 feature sets

§7.2 justified scoping R6 to width 372 by asserting that the folded 944 regimes
zero `f156..371`, so a 944 read would see only F4's 36 basic slots. **Measured,
and it depends on the feature set, not on the width** — which is precisely what
`CLAUDE.md`'s "FEATURE SETS ARE NAMED, NOT COUNTED" says and what a count-based
argument cannot see:

| 944/924-class table | `f156..371` nonzero cells | max \|·\| | F4 reaches |
|---|--:|--:|--:|
| `ext944-canonical-2026-08-01/ext_cid22val` | **0** / 927,072 | 0 | 36 slots |
| `ext924-canonical-2026-07-27/ext_cid22val` | **0** / 927,072 | 0 | 36 slots |
| `ladder-2026-09-05/instruments/dial_grid_944col_ladder` (`foldapp2pools`) | **2,044,645** / 2,072,088 (98.7 %) | 1.668 | **132 slots** |

So the campaign's 944 roots ARE the weaker instrument the pre-registration
described, and the 2026-09-05 pools-live 944 instrument is **exactly as
sensitive as 372**. The conclusion — run R6 at 372 — is unchanged and is now
supported by a measurement instead of a blanket claim; the claim as written was
wrong and is corrected here rather than quietly dropped.

**Consequence for the recalculation:** a rev2 wave must key each 944 table's F4
blast radius on its POOL STATE (`feature_set_id`), not on its width. A wave that
assumes "944 ⇒ 36 slots" will under-declare the moved slots of every
pools-live table by 96.

---

## 3. ★ C2 — F4's pathology DOES NOT OCCUR in any corpus this box has pixels for

`clamp` is `max(0, 1 − D²)`: bit-identical to the shipped form wherever
`D² ≤ 1`, different only above it. So a row where clamp moves is a row holding a
pathological pixel, and clamp is a *detector*, not merely a candidate.

**It fires on nothing.** Across **217,756 rows** — the seven human corpora plus
the full 196,086-row safesyn training leg — `clamp` moves **0 cells, 0 rows,
0 slots**:

| corpus | rows | clamp moved rows | rev1 max \|f\| over the 132 F4 slots |
|---|--:|--:|--:|
| cid22val | 4,292 | **0** | 1.266 |
| kadid | 10,125 | **0** | 1.971 |
| tid | 3,000 | **0** | 1.900 |
| konjnd | 1,008 | **0** | 1.051 |
| aic3 | 600 | **0** | 1.085 |
| csiq | 866 | **0** | 1.763 |
| live | 779 | **0** | 1.754 |
| **safesyn** | **196,086** | **0** | **1.913** |

Not one row anywhere reaches even `|f| > 2`, against the 5,814,302 on record.
That number came from `bigcodec_hqdedup_traindigits_2026-07-02.parquet`, a
2.3 M-row sweep with **no local pixels** (recalculation manifest §3f); the only
other sighting is 3 rows in 10,000 of `nonphoto` val (max ~72), which is a
byte-copy with no pixels either.

**So the F4 outlier is a property of the bigcodec population and is not
reproducible on this box.** Three consequences, all load-bearing:

1. **G3 cannot be OBSERVED here.** It is instead established where it belongs —
   at the owner, by test: `ssim_form::tests::only_the_legacy_arm_is_unbounded`
   (F4 as a *failing* property of the shipped form) and
   `bounded_arms_keep_d_in_zero_to_two_everywhere`, over a sweep that does reach
   the pathology. The empirical half of G3 is this table: the shipped population
   never enters the regime.
2. **`clamp`'s features are bit-identical to rev1 on every row R6 fits or
   scores.** Its gram, its fit and its rank are therefore identical too — it
   cannot win G1, and it cannot lose it. That is a property of the arm, not an
   inconclusive measurement.
3. **The rank contest is `c1` vs `lorentz` vs rev1**, and it is a contest about
   the HEALTHY regime — which is exactly what §1.4's design tension
   (`lib.rs:204`'s deliberate no-`C1` choice vs `bounded_sim`'s Weber
   normalisation) is about.

---

## 4. The fits — one owner, one recipe, one thing varied

`bake_dial_refit gram` → `bake_dial_refit fit-lasso`. No script computes a fit,
a spline or a statistic.

Flags are the `did100 ctl` recipe — the one that reproduces shipped Profile D
BYTE-IDENTICALLY — with only the input tables swapped for the arm's own:

```
--space raw --target human_score --lam 2e-3 --tau 0 --n-sweeps 400 --tol 1e-10
```

* **slices** `0..155` (the ADD156 / Profile-D lineage) and `0..227`
  (basic + peaks). F4 reaches **36** of the first and **60** of the second, so
  both carry the signal; the 372-wide slice would add the masked+IW 72 but is
  not the shipped monotone-linear class.
* **solvers** `lasso` (the shipped recipe) and `bvls` — the sign-constrained
  monotone-linear class the user's directive names, `--solver bvls --bounds-tsv
  benchmarks/feature_sign_mask_2026-05-26.tsv`. The bounded-variable CD solver
  already existed at the owner (`gram_lasso::box_cd_slice`, with an
  active-bound fixture test), so no solver was written for this. **Note for
  readers of the numbers:** `box_cd_slice` takes no `λ` — the `bvls` arm is
  *bounded least squares*, not "lasso with bounds", and `--lam` is accepted and
  ignored there.
* **the sign mask is held FIXED across arms** on purpose: it encodes the
  structural direction of an error feature, and re-deriving it per arm would
  vary two things at once.
* **anchor** — each arm's own 2,000-row safesyn subset (stratified 6 codecs × 16
  quality points, fixed seed, the SAME rows for every arm), in-era, because a
  spline fit on revision-1 pixels is not a spline for a revision-2 dial. SROCC
  is invariant under the monotone spline, so G1 is unaffected by this choice;
  the dial gates are not, which is why it is in-era. The anchor keeps the raw
  (unclamped) ssim2 target — the shipped anchor's `max(ssim2, 0)` clamp is the
  measured cause of shipped B's missing negative tail (G-ADDR §10).

4 arms × 2 slices × 2 solvers = **16 fits**, each differing from the others only
in its input table and its declared slice/solver.

**The reference point.** Shipped Profile D (`d_sdr_add156_dense_dial_2026-08-31`)
on the same postC root, scored through the same `bake_verdict`: CID22
**0.86333**, KonJND **0.53670** (the registered JPEG-504 ruler, which is what
`bake_verdict` reads for `konjnd`), AIC-3 **0.77700**, CSIQ 0.90167, LIVE
0.96029, TID 0.82368, KADID 0.80806.
