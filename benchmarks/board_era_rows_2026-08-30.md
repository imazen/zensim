# The current-extractor 372 rows on the summer-gauntlet board (2026-08-30)

**What this executes.** Follow-up 2 of `benchmarks/eval372_current_root_2026-08-30.md` §6
("the board is NOT regenerated — registered for the board owner"): the 11 ready-to-promote
current-era verdicts at `/mnt/v/output/zensim/eval372-roster-2026-08-30/json/<label>_new.json`
are now board rows, and the board is regenerated through the canonical pipeline. **Nothing
here re-derives a statistic** — `promote_fulleval.py` relabels and gates, the dashboard reads.

**Board HTML:** `/mnt/v/output/zensim/reports/summer_gauntlet.html` (18.87 MB, 378 fulleval
files → 361 rendered rows after the 17 dominated exclusions), browseable at
<http://localhost:3300/zensim/reports/summer_gauntlet.html>. The pre-change file is kept at
`summer_gauntlet_pre_era372_2026-08-30.html` (18.30 MB).
**Gates:** `scripts/v_next/gauntlet_gates.sh` PASS — `node --check` on both script blocks,
DOM-shim render (361 bakes, 12 sections, 18 tables, 511 rows, sort clicks reorder attached
tables, ECharts mounts + SSR per panel kind), 718 registry-annotated cells carry ⚠.

---

## 1. Naming convention: `<stored-era board name>@cur372`

A board name ending in **`@cur372`** is the SAME BAKE as its unsuffixed sibling, read on the
dated current-extractor 372 root (`/mnt/v/zen/zensim-training/2026-08-30-full-features-372`,
`_MANIFEST.json build_commit ea16c7ee`). Chosen over the `__r372cur` spelling *suggested* in
`eval372_current_root_2026-08-30.md` §6.2 (nothing had been promoted under it) because:

* the stem is unchanged, so the two halves of a pair sort adjacently, share every
  `family_of` prefix rule, and read as one model on two rulers;
* `@` occurs in no other board name, so `name.endswith("@cur372")` is an unambiguous test —
  a stem could plausibly contain `__r372cur`;
* it survives a filename (`<name>.fulleval.json`) and a JS object key without escaping.

Owner: `gauntlet.ERA372_CUR_SUFFIX`. The promoter is `scripts/promote_era372_board.py` — a
caller of `scripts/promote_fulleval.py`, carrying the frozen roster map (established by
`bake_sha256` identity against the board, **not** by name similarity: the verdicts' own
`name` field is the bake stem, which differs from the board name on 8 of 11).

**The stored-era rows were not touched.** A never-overwrite gate re-hashes all 9 paired
stored-era files after the run and refuses on any change — it PASSED.

### The 11 rows

| @cur372 board name | curated | scatter | CID22 | KonJND | composite | M3a | uses f156-371 |
|---|---|---|---|---|---|---|---|
| `b_sdr_linear_cid80_inclwinsor_dense_dial@cur372` | **yes** | kept | 0.88212 | −0.6497 | 0.8407 | 0.597 | yes |
| `mlp_2L_diverse_H128@cur372` | **yes** | kept | 0.88191 | −0.5048 | 0.8688 | — (not measured) | yes |
| `cl_tfm_corruption_LQ_MLP_s13@cur372` | **yes** | kept | 0.88052 | −0.2720 | 0.8408 | 0.587 | yes |
| `v02_bvls_NO_shaping@cur372` | **yes** | kept | 0.83929 | −0.7275 | 0.8160 | 0.199 | yes |
| `v02_bvls_shaped@cur372` | family toggle | stripped | 0.83647 | −0.1703 | 0.7633 | — (not measured) | yes |
| `v47_strict_QAT_native@cur372` | family toggle | stripped | 0.86606 | −0.3996 | 0.8180 | 0.633 | yes |
| `T_appT_b372_lam1e-3@cur372` | family toggle | stripped | 0.86927 | −0.5411 | 0.8229 | 0.626 | yes |
| `bhdr_linear_shaped_cvvdpmix@cur372` | family toggle | stripped | 0.84316 | −0.4804 | 0.7949 | 0.772 | yes |
| `ADD156_safesyn_only_raw_lasso@cur372` | family toggle | stripped | 0.86338 | −0.4462 | 0.8161 | 0.954 | no |
| `Ebothg_scr0_5_dial@cur372` | family toggle | stripped | 0.87929 | −0.2707 | 0.8286 | 0.912 | no |
| `winner_dial_Ebothg_hfgain_winsor_dial@cur372` | family toggle | stripped | 0.89390 | −0.3352 | 0.8170 | 0.923 | no |

Every CID22 / KonJND value reproduces the round-4b era table exactly.

**Curation (registered size rule).** Default-visible = the four decision-relevant cells of
the 41 ordering flips, promoted as PAIRS: shipped B (4th → 1st on CID22), the 2-layer blend
(new composite leader — its published "+0.004 over B" is the era artifact round-4b
overturned), `cl_tfm` (1st → LAST on KonJND *and* AIC-3, composite −0.049) and the BVLS
no-shaping arm (current-era KonJND leader). `cl_tfm_corruption_LQ_MLP_s13`'s **stored** half
joined `CURATED_BOARD` so the pair reads together; it is the only pre-existing row whose
curation changed. Those four keep embedded per-pair scatter; the other seven are
grid-interior — every scalar stat present, `--strip-per-pair`, full data still in
`source_verdict`. Cost of the 11 rows: **+0.57 MB** (18.30 → 18.87 MB).

**Family toggle.** `family_of()` returns `"@cur372 (current extractor)"` for every suffixed
name, checked FIRST so a pair's two halves never split across families.

**Carried, not recomputed.** M3/M3a came from the stored-era row via
`--carry-coherence-from` (sha-gated, provenance in `coherence_source`) for the 9 paired
rows — coherence is a property of the BAKE, measured by `diffmap_block_coherence` on
images, not a function of the 372 eval root. The two unpaired rows render an em-dash =
NOT MEASURED, never a zero. `block_profile` was recomputed from the bake bytes on all 11
(`bake_block_profile`, sha-gated) so the "uses f156-371" chip — the exact discriminator
between the era-sensitive and era-immune halves of this roster — is populated.

**Two rows have no stored-era sibling** (`mlp_2L_diverse_H128`, `v02_bvls_shaped`): no board
fulleval existed for either bake. No stored-era counterpart was created — that would add
fresh un-annotated extinct-era numbers to the board.

---

## 2. ⚠ MEASURED CORRECTION: 7 of the 9 "stored-era" board rows were never read on the stored root

Pairing the rows exposed this. Round-4b's registry entries assume the nine 372-class board
cells are reads of the 2026-05-15 root. **They are not.** Each board fulleval carries its
own `regime` stamp, and a fresh `bake_verdict --regime 720 --corpora cid22` reproduces the
regime-720 rows **bit-exactly**:

| board row | `regime` stamp | root the row actually used | per-pair evidence (CID22, 4,292 pairs) |
|---|---|---|---|
| `b_sdr_linear_cid80_inclwinsor_dense_dial` | 372 | **2026-05-15 372 root** | max\|Δ\| **0.0** vs round-4b `_old.json` |
| `T_appT_b372_lam1e-3` | 372 | **2026-05-15 372 root** | max\|Δ\| **0.0** vs `_old.json` |
| `cl_tfm_corruption_LQ_MLP_s13` | 720 | **ext720-canonical-2026-07-22** | max\|Δ\| **0.0** vs a fresh `--regime 720` run |
| `v02_bvls_NO_shaping` | 720 | ext720 | max\|Δ\| **0.0** |
| `v47_strict_QAT_native` | 720 | ext720 | max\|Δ\| **0.0** |
| `ADD156_safesyn_only_raw_lasso` | 720 | ext720 | max\|Δ\| **0.0** |
| `Ebothg_scr0_5_dial` | 720 | ext720 | max\|Δ\| **0.0** |
| `winner_dial_Ebothg_hfgain_winsor_dial` | 720 | ext720 | max\|Δ\| **0.0** |
| `bhdr_linear_shaped_cvvdpmix` | 720 | ext720 | max\|Δ\| 4.9e-6 (not bit-exact) |

`--regime 720` swaps the root to `/mnt/v/zen/zensim-training/ext720-canonical-2026-07-22/
ext_<corpus>.parquet`, and **that root's masked/IW block is POST-FIX**: its cid22
`f156 / f200 / f300 / f371` columns are element-identical to the 2026-08-30 372 root's
(same nonzero counts and min/max over all 4,292 rows). So those seven rows already agree
with the current-extractor 372 read to **≤2e-4** on CID22 — `v02_bvls_NO_shaping` 3e-5,
`cl_tfm` 7e-5, `v47` 9e-5, `bhdr` 1.9e-4, the three basic-only controls 6e-5..1e-4 — while
they differ from the true stored-root read by up to **0.0153 SROCC / 96 score units**
(`cl_tfm` per-pair max\|Δ\| 96.4, `v02_bvls_NO_shaping` 59.5).

**Consequence.** The ⚠ era-stale badge that `eval372-stored-root-thread-dependent-2026-08-30`
puts on `cl_tfm`, `v02_bvls_NO_shaping`, `v47_strict_QAT_native` and
`bhdr_linear_shaped_cvvdpmix` is **misdirected** — those numbers are not from the stale
root. The badge is CORRECT on `b_sdr_linear_cid80_inclwinsor_dense_dial` and
`T_appT_b372_lam1e-3`, the two genuine stored-root reads.

Registered append-only as **`board372-row-read-on-ext720-root-2026-08-30`** (kind
`annotated`, fields `rank` + `composite`, scope = the seven), per the registry's
supersede-don't-delete rule. It renders alongside the original badge with its own hover
reason.

**Round-4b's science is unaffected.** Its era table compares `_old` vs `_new` produced by
ONE instrument on the two 372 roots and never used a board row; only the *board-row
attribution* in its §7 was wrong. That paragraph is flagged for the round-4b lane rather
than edited here (docs-correction rule: batch and ask).

**Consequence for pair reading.** Only the B pair is a clean era A/B on the board today
(`b_sdr_...` 0.87638 stored-root vs `@cur372` 0.88212). For the other pairs the visible
delta is *ext720-vs-current-372*, which is ≈0 — read the era delta from round-4b's table,
not by subtracting two board cells.

---

## 3. Accuracy annotations added for the new rows

Per the standing "keep the gauntlet updated AND accurate" directive, an `@cur372` row is not
clean on every axis, so two append-only entries flag what is not a current-extractor read:

* **`eval372-current-root-copied-corpora-2026-08-30`** (annotated) — fields `rank.aic4`,
  `rank.nonphoto`, `rank.imazen26`, `rank.sdr25`, `rank.hfnlproxy`, `rank.hf_nearlossless`,
  `composite`. Only 8 of the 14 corpora on the new root are current-extractor reads
  (cid22/kadid/tid/konjnd/aic3 re-extracted; csiq/live/pipal measured bit-identical). The
  other six are byte-copies of the 2026-05-15 root; `aic4` is PRE-FIX and unrefreshable.
  `product_composite` inherits the mix: imazen26 0.50 + nonphoto 0.30 + aic4 0.05 of its
  2.15 denominator = **39.5 % of the composite LEVEL rides on stored-era tables** (verified
  by recomputing B's 0.8407364996 from the rank block against the emitted
  0.8407364995733521). The between-era *delta* is clean — those terms cancel.
* **`dial372-grid-thread-dependent-era-current-rows-2026-08-30`** (annotated) — fields
  `dial`, `gates.g1_dynamic_range`. Extends the existing dial entry (whose `names` scope
  listed only the stored halves) to the `@cur372` halves: the dial grid is its own file
  outside `--features-root`, so an `@cur372` row's dial is bit-equal to its sibling's by
  construction and is *not* a current-extractor measurement.

Verified on the built board: the `@cur372` rows match **neither**
`eval372-stored-root-thread-dependent-2026-08-30` **nor**
`eval372-basic-only-bakes-era-independent-2026-08-30`; the stored rows match exactly one of
those two (six invalidated, three era-independent); a 944 row (`W10L9_s4003_packed`) matches
neither.

---

## 4. One code defect found and fixed while wiring the family

`family_of()` is not only a toggle label — `build_html`'s knob-end gate scopes its
peers/HDR exemption on it. With the era rule winning over the prefix rules,
`bhdr_linear_shaped_cvvdpmix@cur372` was judged by the SDR knob-end rule and failed all four
codecs, while its identical-dial stored sibling (family `HDR`) was exempt. Fixed by
`gauntlet.era_base_name()` — any rule that judges the MODEL rather than the ruler now scopes
on the era-stripped stem. Confirmed on the rebuilt board: `knob_end_fail` is `[]` for both
halves of the HDR pair.

---

## 5. Open

1. **`LOOP_BAKE_MAP` is untouched.** Its `blend2L_base` key still has no board target even
   though `mlp_2L_diverse_H128@cur372` now exists. Mapping it would put loop columns on a
   *current-era* row while every other loop mapping points at a *stored-era* row — an
   inconsistency a reader comparing down that column would not see. Left for a lane that can
   decide the whole column's era.
2. **Board size.** 18.87 MB against the 12 MB cap recorded in `zensim/CLAUDE.md`; the cap
   was already exceeded (18.30 MB) before this change, which contributed +0.57 MB. Not
   addressed here — re-stripping the ~29 curated cells is a curation decision, not a
   promotion one.
3. **A verdict still does not RECORD its `--features-root`.** The root had to be *inferred*
   from the `regime` stamp and then *proved* by bit-exact reproduction (§2). Partly closed
   the same day: `bake_verdict` now PRINTS its ruler
   (`bake_verdict: features-root era — … :: …`, `zensim_validate::eval_roots::era_of`) as
   part of the default-root flip (`docs/DATASET_HISTORY.md` §3.29) — but the line goes to
   stderr, not into `--full-json`, so a *stored* verdict still cannot be attributed without
   re-running it. A `features_root` + era field in the JSON would make §2's archaeology a
   grep, and would have made this section unnecessary.
4. The round-4b doc's §7 scope claim (§2 above) is for that lane to correct.
