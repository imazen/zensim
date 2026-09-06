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
