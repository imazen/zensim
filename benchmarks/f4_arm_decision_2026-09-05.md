# R6 — the F4 arm, decided by measurement

**2026-09-05.** Runs the gate
[`../docs/PLAN_FEATURE_REV2_2026-09-05.md`](../docs/PLAN_FEATURE_REV2_2026-09-05.md)
§7, pre-registered and pushed (`090d55d7`) before a single table was extracted.
It is the one gate
[`feature_rev2_2026-09-05.md`](feature_rev2_2026-09-05.md) §1.6 could not run:
the cheap ladder proxy gives **identical** results for all four arms, so the
choice needed a real monotone-linear fit on real corpora.

Artifacts: `/mnt/v/output/zensim/rev2-2026-09-05/r6/` (9.1 GB, `_MANIFEST.json` at
the root carries `build_commit`, the arm tokens, the decoder era per format, every
input sha256 and all 24 bake shas; each `evalroot/<arm>/` carries its own).
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

**The registry's 132 is independently confirmed, on real corpora.** The
moved-slot set here is MEASURED (`r6_arm_delta.py` never reads the registry),
and it matches the registered blast radius family for family — **basic 36,
peaks 24, masked 36, IW 36** — with the peaks slots landing on the
every-third-slot comb (`f156, f159, f162, …`) that §1.4 predicted and that no
per-block reading of the symptom would have produced. The 2026-09-05 widening
from 72 to 132 was right.

**So the F4 outlier is a property of the bigcodec population and is not
reproducible on this box.** Three consequences, all load-bearing:

1. **G3 cannot be OBSERVED here.** It is instead established where it belongs —
   at the owner, by test: `ssim_form::tests::only_the_legacy_arm_is_unbounded`
   (F4 as a *failing* property of the shipped form) and
   `bounded_arms_keep_d_in_zero_to_two_everywhere`, over a sweep that does reach
   the pathology. **RUN, not merely cited**, at this lane's tree
   (`cargo test --release -p zensim --features training,custom-profiles,feature-regime-v2
   --lib ssim_form`): 7 passed, 0 failed — including
   `clamp_arm_is_bit_identical_to_legacy_below_the_knee` and
   `c1_is_derived_from_the_constants_already_present`. The empirical half of G3
   is this table: the shipped population never enters the regime.
2. **`clamp`'s features are bit-identical to rev1 on every row R6 fits or
   scores.** Its gram, its fit and its rank are therefore identical too — it
   cannot win G1, and it cannot lose it. That is a property of the arm, not an
   inconclusive measurement.
3. **The rank contest is `c1` vs `lorentz` vs rev1**, and it is a contest about
   the HEALTHY regime — which is exactly what §1.4's design tension
   (`lib.rs:204`'s deliberate no-`C1` choice vs `bounded_sim`'s Weber
   normalisation) is about.

**Where the two live arms actually move, per family** (CID22, 4,292 rows, max
|Δ| against rev1; the rev1 feature scale is on the last row for reference):

| arm | basic | peaks | masked | IW |
|---|--:|--:|--:|--:|
| `c1` | 1.313e-2 | **4.069e-2** | 1.316e-2 | 1.309e-2 |
| `lorentz` | 5.05e-8 | 3.58e-7 | 3.06e-8 | 8.26e-8 |
| `clamp` | 0 | 0 | 0 | 0 |
| *rev1 max \|f\|* | *3.724* | *1.266* | *0.317* | *0.388* |

**And on the SDR default's ACTUAL 28 inputs** (CID22; the other 24 are not F4
slots and are bit-identical in every arm by construction):

| arm | max \|Δ\| over all 28 | f14 | f26 | f91 | f93 | worst relative |
|---|--:|--:|--:|--:|--:|--:|
| `c1` | 4.77e-3 | 1.76e-3 | 1.01e-5 | 1.29e-3 | 4.77e-3 | **16.9 %** (f93) |
| `lorentz` | 5.05e-8 | 5.05e-8 | 1.79e-8 | 1.84e-8 | 1.35e-8 | 0.00 % |
| `clamp` | **0** | 0 | 0 | 0 | 0 | 0.00 % |

On the gold holdout `lorentz` is a **near-no-op** — 1e-7 against feature scales
of 0.3–3.7, i.e. f32 rounding — while `c1` is a ~1–4 % relative perturbation,
biggest on the peaks family. On KADID / TID / CSIQ / LIVE `lorentz` grows to
9.0e-2 / 1.2e-2 / 1.2e-2 / 4.4e-3, so it is not a no-op everywhere; those are
the corpora with the strongest local mean differences.

**A fourth control, unplanned and free.** The `ssim2` arm's re-extracted LADDER
grid comes out at sha256 `4c3874a78c469e15…` — **the registered ladder grid's
own sha256**, byte for byte (`ladder-2026-09-05/instruments/
_MANIFEST_ladder.json`). So the rev1 arm reproduces a second independently-built
instrument exactly, on top of the seven eval CSVs of §1.

**And the join that produced it caught a real bug.** The first version of the
grid rebuild joined the extraction to the registered grid POSITIONALLY, on the
reasonable belief that the pairs list the ladder was built from IS the grid's row
order. It is not: the pairs list is grouped by IMAGE and the grid by CODEC, so
they agree for 51 rows and then diverge, and **9,287 of 9,593 cells would have
been labelled with the wrong codec and quality**. The gate — `image_id` equality,
row for row — failed loudly instead. The join is now BY `row_id` (which the pairs
TSV carries, and which is the grid's row index) and the gate is strictly
stronger: every row must agree with `grid[row_id]` on `image_id` AND
`codec_param`, so a permuted, truncated or mis-keyed list all fail. *Keeping a
real gate on a join you believe is trivially correct is what made this a
five-minute fix instead of a silently wrong dial panel.*

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

**Side-finding, from a smoke test of the `bvls` path on the REGISTERED (rev1-era)
gram** — recorded because it is about the solver, not about F4, and it is large:
sign-constraining the same 156-slice fit trades **−0.029 CID22** for **+0.081
KonJND and +0.016 AIC-3** against shipped D (0.83466 / 0.61778 / 0.79298 vs
0.86333 / 0.53670 / 0.77700). The monotone-constrained class is not a small
perturbation of the lasso class, which is why R6 carries both solvers rather
than assuming the arm ranking transfers between them.

---

## 5. ★ The clamp prediction, tested end to end and CONFIRMED

§3 predicted, before the training leg finished extracting, that `clamp` would
reproduce revision 1 exactly on every row R6 touches. It does, at three
independent levels:

* **Features.** `clamp`'s 196,086-row safesyn table is `cmp`-clean against the
  `ssim2` arm's — byte-identical, not merely close.
* **Fits.** All **six** `clamp` bakes are **byte-identical** to their `ssim2`
  siblings, sha for sha:

  | variant | `ssim2` = `clamp` sha256 (16) | `lorentz` | `c1` |
  |---|---|---|---|
  | `s156_lasso` | `badb848d56f7a9f8` | `17db43011d163e70` | `84d43133a62d637f` |
  | `s156_bvls`  | `2209be8faec73ed1` | `f69f24daf7113497` | `f50a77afca58e976` |
  | `s228_lasso` | `7e0348d473dad5dd` | `4f812895cf9f6a4c` | `c6f06ed892a01f1e` |
  | `s228_bvls`  | `9639920ebf535eae` | `1f05e94c903cdf7d` | `304bb826824c2dc1` |
  | `s372_lasso` | `e567dc1670ace071` | `a02845f9da9a9648` | `c664545ce25ba348` |
  | `s372_bvls`  | `68bca6a6b054a582` | `4269fb684e195565` | `bb08afc0bbb2ccf9` |

  The whole chain — features → Gram → lasso/BVLS solve → output spline → f16
  pack → ZNPR bytes — is reproduced exactly, through four independent stages
  that each had the opportunity to diverge.
* **Rank.** Byte-identical bakes on byte-identical features give byte-identical
  predictions, so `clamp`'s G1 delta against revision 1 is **exactly 0** with a
  degenerate CI. It cannot win G1 and it cannot lose it — a property of the
  arm, not an inconclusive measurement.

---

## 6. G1/G2 — RANK, and the paired bootstrap

Pooled |SROCC|, each arm's bake on that arm's own eval root. `clamp` reads
**identical to rev1 in every cell of this table**, which is the §5 byte-identity
showing through rather than a coincidence.

| variant | arm | CID22 | KonJND | AIC-3 | CSIQ | LIVE | TID | KADID |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| `s156_lasso` | rev1 / `clamp` | 0.84134 | 0.58361 | 0.79575 | 0.88563 | 0.73566 | 0.76061 | 0.71670 |
| | `c1` | **0.84384** | 0.58091 | 0.79601 | 0.85667 | 0.74198 | 0.75648 | 0.69239 |
| | `lorentz` | 0.84135 | 0.58371 | 0.79575 | 0.88543 | 0.73450 | 0.76055 | 0.71642 |
| `s156_bvls` | rev1 / `clamp` | 0.83603 | 0.59840 | 0.79132 | 0.73804 | 0.51440 | 0.46062 | 0.41081 |
| | `c1` | 0.83603 | 0.59806 | 0.79154 | 0.73824 | 0.51435 | 0.46076 | 0.41094 |
| | `lorentz` | 0.83604 | 0.59843 | 0.79133 | 0.73805 | 0.51440 | 0.46071 | 0.41090 |
| `s228_lasso` | rev1 / `clamp` | 0.83934 | 0.54799 | 0.78214 | 0.88918 | 0.79603 | 0.74541 | 0.71919 |
| | `c1` | 0.84011 | 0.54440 | 0.78209 | 0.86460 | 0.79000 | 0.74170 | 0.69757 |
| | `lorentz` | 0.83934 | 0.54808 | 0.78203 | 0.88906 | 0.79562 | 0.74544 | 0.71906 |
| `s228_bvls` | rev1 / `clamp` | 0.83071 | 0.58904 | 0.78313 | 0.74135 | 0.51282 | 0.46072 | 0.41120 |
| | `c1` | 0.83074 | 0.58860 | 0.78318 | 0.74164 | 0.51283 | 0.46138 | 0.41143 |
| | `lorentz` | 0.83071 | 0.58891 | 0.78316 | 0.74136 | 0.51282 | 0.46062 | 0.41114 |
| `s372_lasso` | rev1 / `clamp` | **0.85196** | 0.56034 | 0.79327 | **0.90890** | 0.61054 | 0.72611 | 0.72005 |
| | `c1` | 0.85221 | 0.55796 | 0.79396 | 0.87792 | 0.61427 | 0.72180 | 0.70161 |
| | `lorentz` | 0.85195 | 0.56040 | 0.79326 | 0.90890 | 0.61047 | 0.72621 | 0.71996 |
| `s372_bvls` | rev1 / `clamp` | 0.84091 | **0.60536** | 0.78346 | 0.75120 | 0.51158 | 0.52623 | 0.46409 |
| | `c1` | 0.84107 | 0.60629 | 0.78353 | 0.75146 | 0.51143 | 0.52678 | 0.46435 |
| | `lorentz` | 0.84098 | 0.60553 | 0.78337 | 0.75119 | 0.51137 | 0.52647 | 0.46426 |

**G1, the paired bootstrap** (B = 2,000, seed 20260905, same resampled index sets
on both sides, through `wave6_paired_bootstrap.py` → `panel --batch`). A win
counts only when the 95 % CI on the delta excludes 0. Across all 18 (variant ×
primary-corpus) cells:

* **`clamp`: 0 wins, 0 losses, every delta exactly `+0.00000` with CI
  `[+0.00000, +0.00000]`.** Structural, not marginal.
* **`lorentz`: 1 CI-excluding win** in 18 — CID22 at `s372_bvls`, `+0.00007`
  `[+0.00005, +0.00010]`. Never a majority in any variant.
* **`c1`: 4 CI-excluding wins** in 18 — CID22 at `s156_lasso` (`+0.00250`
  `[+0.00196, +0.00307]`), `s228_lasso` (`+0.00076`), `s372_bvls` (`+0.00016`),
  and KonJND at `s372_bvls` (`+0.00093` `[+0.00026, +0.00168]`). **In exactly
  one variant (`s372_bvls`) that is a strict 2-of-3 majority** — the
  pre-registered §7.6 rule-2 condition, met once out of six.

**G2, reported not weighted:** `c1`'s CID22 gains come with consistent losses on
the lasso variants — CSIQ **−0.029 / −0.025 / −0.031** and KADID **−0.024 /
−0.022 / −0.018** at `s156` / `s228` / `s372`. `lorentz` and `clamp` move nothing
outside the fourth decimal. (KADID is train==val for this class and is an
integrity guard, never ranking signal.)

---

## 7. G3 / G4 / G5

**G3 — outlier removal: all three arms PASS**, and so does rev1 on this
population, because §3's population never enters the regime. Max |f| over the
132 F4 slots across all ten legs: rev1 **1.972**, `c1` 1.971, `lorentz` 1.972,
`clamp` 0 (no slot moves). The structural half is the owner's tests, run at this
tree (§3).

**G4 — healthy-cell perturbation (bar 1e-4), on the population where *every* row
is healthy:**

| arm | moved cells on C2-unflagged rows | worst \|Δ\| | verdict |
|---|--:|--:|---|
| `c1` | 29,403,444 | 0.771001 (TID) | **FAIL** |
| `lorentz` | 24,023,289 | 0.0900643 (KADID) | **FAIL** |
| `clamp` | **0** | **0** | **PASS** |

`lorentz` passes per-corpus on CID22 (3.6e-7), KonJND (1.8e-7), AIC-3 (2.4e-7)
and the ladder (2.0e-5) and fails on KADID, TID, CSIQ, LIVE and safesyn. `c1`
fails on all eight.

**G5 — dial, on each arm's own in-era ladder + probes.** `clamp` is identical to
rev1 on every row of the panel (mono 0.9848 / tied 0.0000 / reach 223.98 / min
−126.42 / max 97.563 / p5 −19.324 at `s156_lasso`); `c1` and `lorentz` move
monotonicity by ≤0.0014 and reach by ≤4.0 in either direction. No arm regresses.

**C3 — identity ⇒ zero: PASS at every arm.** The 400 self-pairs give the
**all-zero 372-vector at all four arms** (the identity parquets are byte-identical
across arms, sha `9a9789aaea1c0a43…`), which is the registered 372 identity fact
reproduced here. The dial *value* at identity is then `dial(0⃗)` — a scalar
property of each bake, e.g. 96.7490 for `ssim2_s156_lasso` — and the G-ADDR C5
band `[97.5, 100]` fails for it, which is a property of these R6 probe bakes and
not of any arm.

---

## 8. ★ THE DECISION — `Clamp`, by the pre-registered rule, applied step by step

§7.6's rule was written and pushed at `090d55d7` before a table existed. Applied
to the numbers above, in its own order:

1. **An arm that fails G3 is out.** None do. Nothing is eliminated here.
2. **One surviving arm wins a strict 2-of-3 majority of {CID22, KonJND, AIC-3}
   with CI-excluding deltas AND does not regress G4/G5?** `c1` meets the
   majority condition in exactly one of six variants (`s372_bvls`) — and
   **fails G4** (29.4 M healthy cells moved, worst 0.771 against a 1e-4 bar).
   `lorentz` never reaches a majority and also fails G4. `clamp` cannot win a
   rank contest it is byte-identical in. **Rule 2 selects nothing.**
3. Not reached.
4. **No arm wins a rank majority, so the arithmetic fix lands anyway, on the
   smallest healthy-cell perturbation.** `clamp` 0 cells / 0 max; `lorentz`
   24.0 M / 0.0901; `c1` 29.4 M / 0.7710. **`clamp` wins by an unbeatable
   margin — it is the only arm that changes nothing outside the regime F4 is
   about.**
5. **The prediction stated in advance is CONFIRMED.** §7.6 rule 5 said "rule 4
   is expected to select `clamp`, because `clamp` is bit-identical to rev1
   wherever `D² ≤ 1`. If the measurements land that way, that is a prediction
   confirmed, not a rule bent to fit." They did.

### ⇒ `SsimLumaForm::REV2_LUMA = Clamp`. `SHIPPED_REVISION` stays `Rev1`.

**Why this is the right answer and not merely the rule's answer.** `clamp` is
the unique arm that is *only* a fix: it changes the metric exactly where the
metric was unbounded and nowhere else, so a rev2 flip re-extracts nothing whose
content resembles these 217,756 rows, and every table, bake and verdict built on
such content stays valid across the era boundary. `c1` and `lorentz` are
redesigns wearing a fix's clothes — they move 24–29 million healthy cells to buy,
at most, `+0.0025` CID22 in one variant.

**What the tie-break prior cost, stated honestly.** §1.4 registered *"if the
probe cannot separate them, arm A (`c1`) ships"*, because `c1` reuses the crate's
own `bounded_sim` owner and preserves severity ORDER among pathological pixels
where `clamp` flattens it. The probe DID separate them, so the prior never
fires — but the property it was protecting is real and is now a known cost:
**above `D² = 1`, `Clamp` gives every pixel the same `num_m = 0`.** Three things
bound that cost: no corpus with local pixels reaches there at all (§3); the one
population that does (bigcodec) is a *training* sweep; and `d` stays bounded and
correct in `[0, 2]` regardless. If a future population makes tail ORDER
load-bearing, `Lorentz` — bounded, no Weber term, monotone in `D²`, and 4-6
orders of magnitude closer to rev1 than `c1` on healthy content — is the
registered successor, not `c1`.

**A note the decision does not depend on:** the ONE variant where `c1` wins a
majority, `s372_bvls`, wins it by `+0.00016` CID22 and `+0.00093` KonJND. Those
are real (CI-excluding, B = 2,000) and they are also 1–2 orders below the
seed-to-seed spread this repo has measured for any trained head. A rule that
shipped a 24-million-cell feature-definition change for them would be a bad rule.

---

## 9. ⛔ AN UNBOUNDED FEATURE THAT *DOES* FIRE HERE — and it is NOT F4

Chasing G3's "rev1 max over ALL slots" column turned up something the F4 audit
does not cover. Over the same ten legs, the largest rev1 feature values are:

| corpus | max \|f\| over the 132 **F4** slots | max \|f\| over **all 372** | which slot |
|---|--:|--:|---|
| safesyn | 1.913 | **36,465.7** | `f129` |
| LIVE | 1.754 | **3,598.2** | `f12` |
| TID | 1.900 | **927.9** | `f38` |
| KADID | 1.971 | **618.3** | `f38` |
| CSIQ | 1.763 | **163.1** | `f12` |
| CID22 | 1.266 | 3.724 | `f12` |

**Every one of those slots is local index 12 of its basic group** —
`contrast_inc`, i.e. `hf_energy_gain = max(0, hf_dst_L2 / hf_src_L2 − 1)`
(`metric.rs:75-92`). It is **unbounded above by exactly F4's mechanism**: a flat
source region drives `hf_src_L2 → 0` while the distorted image has HF, and the
ratio runs away. Its two siblings in the same family are bounded by
construction — `var_loss` = `max(0, 1 − ratio)` and `tex_loss` = the L1 form,
both measured at max exactly 1.0 — and `mse` maxes at 0.169. So the family has
the same shape as F4's: **one member is unbounded and the rest are not.**

Unlike F4 it is not a rare tail. Cells above 100 on some `contrast_inc` slot:
**LIVE 122 of 779 rows (15.7 %)**, KADID 113, TID 91, safesyn 183, CSIQ 5;
p99.9 is **1,088 on LIVE** and 167.7 on TID.

**This is REPORTED, not fixed** — R6's scope is the F4 arm, and `contrast_inc`
has no registered defect entry, no arm and no pre-registered gate. It is
recorded here because the F4 audit's own framing ("the one live arithmetic
defect") is measurably incomplete: **on every corpus this box has pixels for, the
unbounded value that actually occurs belongs to `contrast_inc`, and F4's belongs
to a population with no local pixels.** Registering `contrast_inc` as a defect,
choosing its bounded form, and pricing its blast radius is the obvious next
lane, and it should reuse this one's shape — `ssim_form`'s revision selector
already generalises.

---

## 10. Handoff to the recalculation lane

What a rev2 fleet wave needs from R6, in the form it needs it:

* **Arm token.** The chosen arm is named ONCE, at
  `zensim::ssim_form::SsimLumaForm::REV2_LUMA`, and `FormulaRevision::Rev2`
  resolves through it. A worker selects it with `ZENSIM_FORMULA_REV=2`; the
  per-arm override `ZENSIM_SSIM_LUMA` is a MEASUREMENT knob and must not appear
  in a production wave's environment.
* **Blast radius keys on the FEATURE-SET ID, not the width** (§2). At 372 and at
  a pools-live 944 (`foldapp2pools`) F4 moves **132** slots; at the campaign's
  zeroed `ext944` / `ext924` roots it moves **36**. A wave that declares "944 ⇒
  36" under-declares every pools-live table by 96 slots.
* **F4's exposure per SHIPPED bake, measured** (`bake_block_profile` read set ∩
  the 132 measured F4 slots) — so the wave prices the re-verdict rather than
  guessing at it:

  | shipped bake | reads | F4 slots read | share |
  |---|--:|--:|--:|
  | `v47_strict_qat_native` (Profile A) | 285 | **125** | 43.9 % |
  | `bhdr_linear_shaped_anchored2` | 50 | 20 | 40.0 % |
  | `bhdr_linear_shaped_cvvdpmix` | 133 | 49 | 36.8 % |
  | `d_sdr_add156_*` (Profile D, the SDR default) | 28 | **4** | 14.3 % |
  | `b_sdr_linear_cid80_*` (Profile B) | 95 | 12 | 12.6 % |
  | `c_sdr_mlp944_*` / `c_sdr_purity944` / `c_hdr_l1t1944` (Profile C) | 667–697 | 36 | 5.2–5.4 % |

  The SDR default is the LEAST exposed of the single-block profiles — 4 slots
  (`f14, f26, f91, f93`) of its 28 inputs — and Profile A the most. The 944
  Profile-C family reads exactly the 36 basic slots, which is the zeroed-pool
  count of §2 and a second, independent confirmation of it.

* **⛔ The "winsor already clamps it" mitigation does NOT cover the SDR default.**
  `ssim_moment_explosion_2026-07-16.md` §7b argued F4 was already neutralised
  because "B applies `winsor_p99` to all 372 features", so a 5.8e6 never reaches
  the linear head. MEASURED with `zenpredict inspect`: that is true of
  **Profile B only** (`zentrain.feature_transforms` = 372 × `winsor_p99`, with
  per-feature params). **Profile D — today's SDR default — carries NO
  `feature_transforms` and NO `feature_bounds` at all**, and neither do
  `v47_strict_qat_native` (Profile A) or the 944 Profile-C family. On every
  shipped bake except B, an unbounded `d` would go into the head unclamped. So
  the F4 fix is load-bearing for the default, not merely tidy — and arm C's
  registered downside ("flattens rank among the worst pixels") is NOT already
  being paid downstream the way §7b's argument implied.

* **F5 is NOT free** (`feature_rev2_2026-09-05.md` §2.6): flipping revision 2
  moves 22 of the 33 `GLOBAL_*` slots each of three shipped 944 bakes reads, so
  Profiles C and CHdr must be re-verdicted in the same wave.
* **Decoder era is an input.** Every table in this lane decoded through
  `shared/zen_decode.rs` at `ceb86c2d`; a wave that decodes at a different era is
  measuring two changes at once, and §3.34 priced decoder era at 73 % of an
  extractor era.
* **The `ssim2` arm is the reproduction control.** Any rev2 worker image should
  be able to reproduce the postC 372 root byte-for-byte with
  `ZENSIM_FORMULA_REV=1` before it is trusted to produce rev2 tables.
* **Cost, measured here:** the full 196,086-row safesyn leg extracts in ~12–13
  min at 254–290 pairs/s on one 26-thread box, including in-process decode of
  111,068 `.jpg` / 34,001 `.avif` / 26,362 `.jxl` / 24,655 `.webp` bitstreams;
  the seven eval corpora together take ~30 s.
