# The dated current-extractor 372 eval root, and what the era shift does to the roster (2026-08-30)

**What this executes.** The two registered-not-executed follow-ups of
`benchmarks/v1_extractor_drift_2026-08-30.md` §4c: **(a)** a NEW dated 372 eval
root built with today's extractor, and **(b)** the 372-class B-lineage
re-verdicted on BOTH eras with the same instrument. The 2026-05-15 root is
untouched. Nothing here re-derives a statistic: every SROCC / Z-RMSE / dial
number was produced by `bake_verdict` → `zensim_validate::panel` → `zenstats`.

**Root:** `/mnt/v/zen/zensim-training/2026-08-30-full-features-372/`
(`_MANIFEST.json`: `build_commit` `ea16c7ee`, per-file sha256, row accounting,
per-corpus era, per-slot drift-vs-stored).
**Verdicts + tables:** `/mnt/v/output/zensim/eval372-roster-2026-08-30/`
(`roster_era_table.tsv` / `.md`, `json/<label>_{old,new}.json`,
`kon504/<label>_{old,new}.json`, `old/`+`new/` markdown verdicts).
**Reproduce:** `scripts/canonical_corpus/build_eval372_root.sh` →
`pack_eval372_root.py` → `eval372_roster.sh` → `eval372_roster_table.py`.

---

## 1. Headline

1. **The era shift is MODEL-SPECIFIC, not a constant.** Across 11 bakes it runs
   from **exactly 0.00000** (three basic-block-only bakes, on all 15 corpora) to
   **|Δ| 0.489** (`cl_tfm_LQ_MLP` on KonJND). **There is no correction factor**
   for a published 372-era number; the cell has to be re-verdicted.
2. **Ordering flips: 41**, including the ones that decide things —
   shipped **B goes from 4th to 1st on CID22** among {B, `blend_2L_H128`,
   `cl_tfm_LQ_MLP`, `Ebothg_scr05`}, `cl_tfm_LQ_MLP` goes from **1st to last on
   KonJND** (0.761 → 0.272) and from 1st to last on AIC-3, and the composite
   leader changes from `cl_tfm_LQ_MLP` to `blend_2L_H128`.
3. **The 2026-07-18 FR tables (csiq / live / pipal) are BIT-IDENTICAL to a HEAD
   re-extraction** — 0 slots differing, max_abs exactly 0 on 866 / 779 / 21,800
   rows. The drift doc's commit-level era map ("`2026-07-18` and HEAD are the
   same post-fix era") is now a direct measurement on three corpora, and it
   holds through six weeks of extractor work including the blur rewrite.
4. **The drift doc's own KADID / TID / AIC-3 rows in its §3b are wrong** (§5) —
   its fresh tables were key-aligned, and those corpora's `(ref_basename,
   human_score)` key repeats. Its CID22 and KonJND rows are unaffected and
   reproduce here exactly.
5. **TID's silently-dropped 120 rows are recovered** (`2d94890c`): the loader
   forced the reference stem upper-case and TID ships `i25.bmp` lowercase. The
   new root's TID is **3,000/3,000**.

---

## 2. The root

Built by re-extracting through the SAME tool the stored table used
(`zensim-validate --extract-only` for cid22/kadid/tid/pipal;
`extract_features_372col` for konjnd/aic3 and the pairs-TSV FR corpora), at
`build_commit ea16c7ee`. **File names are deliberately the OLD ones** —
`bake_verdict` hardcodes each corpus's filename, so a drop-in root must reuse
them; the ROOT carries the date.

| corpus | rows | era | new sha256 | masked max_abs | IW max_abs | cells over golden tol |
|---|---:|---|---|---:|---:|---:|
| cid22 | 4,292 | **current** | `b97b5a1638438b01…` | 0.03744 | 0.1235 | 582,499 |
| kadid | 10,125 | **current** | `36b3d2ff291c18fb…` | 0.1550 | 0.6153 | 1,403,831 |
| tid | 3,000 | **current** | `29fb9300f9329b3c…` | 0.1211 | 0.3776 | 418,263 |
| konjnd | 1,008 | **current** | `8258021e50f4459e…` | 0.0302 | 0.1200 | 139,211 |
| aic3 | 600 | **current** | `3ea81880939932b5…` | 0.01368 | 0.04906 | 79,995 |
| csiq | 866 | **current** | `56cccd7f6900e147…` | **0** | **0** | **0** |
| live | 779 | **current** | `ca456952a4a4720c…` | **0** | **0** | **0** |
| pipal | 21,800 | **current** | `c21da88e422ef54a…` | **0** | **0** | **0** |
| aic4 | 300 | COPIED (pre-fix) | `393846556b9d7e80…` | — | — | — |
| nonphoto | 8,241 | COPIED (post-fix build date) | `99dceb0f42ad5b81…` | — | — | — |
| imazen26 | 7,844 | COPIED (post-fix build date) | `a41eadc06383783a…` | — | — | — |
| sdr25 | 50 | COPIED | `4f567646dcc629a6…` | — | — | — |
| hfnlproxy | 11,356 | COPIED | `ae60be7c9aaf4db4…` | — | — | — |
| hf_nearlossless | 300 | COPIED | `6fc953f6159cb22f…` | — | — | — |
| **kon504** (+ `kon504/` side root) | 504 | **current** | `e6a2662dac9ccd1c…` | 0.0302 | 0.1200 | 69,779 |

`masked`/`IW max_abs` and `cells over tol` are stored-vs-new, **positionally
aligned**, at the repo's golden policy `|Δ| ≤ max(1e-6, 1e-5·scale)`.
`basic f0..155` and `peaks f156..227` are **bit-identical on every corpus**
(0 cells over tolerance, max_abs 0) — the blast radius is exactly masked+IW, as
§3.27 says.

**COPIED ≠ current.** Six corpora cannot be re-extracted on this box (their
distorted material is bigcodec/R2 encodes or, for aic4, a source CSV that no
longer exists — drift doc §4c.5). They are byte-copies of the old root, so any
era delta reported for them is **exactly zero by construction, not evidence of
era-cleanliness**. `aic4` in particular is pre-fix and unrefreshable. The
2026-07-15/16 nonphoto + imazen26 tables post-date `6af83b60`, so they are
believed post-fix — that is a build-date inference, not the measurement the FR
trio got.

### 2a. Determinism + cross-lane checks that came free

- **Re-pack is byte-stable**: a second `pack_eval372_root.py` run reproduced
  every parquet sha256.
- **A HEAD re-extraction of konjnd is md5-identical** to the first
  (`ade2919c5c9f…`), on a tree that had gained two commits.
- **The blur perf commits are bit-exact on real corpora.** My HEAD extraction
  vs the drift lane's `f9fac41e` extraction: **0 cells over tolerance and 0
  slots differing on cid22 (4,292 rows) and konjnd (1,008)** — an independent
  confirmation of the `BIT-EXACT` claim in `714da506` + `8a98a286`, on corpus
  data rather than fixtures.

---

## 3. The roster, both eras, same instrument

11 bakes × 2 roots × the DEFAULT `bake_verdict` corpus list (14 corpora) + a
kon504 side-root run, `--full-json` throughout. **SIGNED** SROCC as the panel
reports it (the JND corpora are negative by construction).

**Excluded, named:** `wlin4_a0.5` and the C flagship `C_co3a` are **944-input** —
their root (ext944) is not this drift's, and a 372-root read of them is the
wrong-root class in `zensim/CLAUDE.md` Known Bugs. `aic4` cannot be refreshed.

| bake | n_in | CID22 | KonJND (full 1008) | kon504 | KADID (t=v) | TID (t=v) | AIC-3 |
|---|---:|---|---|---|---|---|---|
| **B_shipped** | 372 | 0.87638 → **0.88212** (+0.00574) | −0.54665 → **−0.64967** (\|+0.103\|) | −0.59349 → **−0.51938** (\|−0.074\|) | 0.82008 → 0.80847 (−0.0116) | 0.78683 → 0.77852 (−0.0083) | 0.77743 → 0.76501 (−0.0124) |
| cl_tfm_LQ_MLP | 372 | 0.88288 → 0.88052 (−0.0024) | −0.76071 → **−0.27195** (\|−0.489\|) | −0.55832 → −0.42977 (\|−0.129\|) | 0.92932 → 0.87257 (−0.0568) | 0.94547 → 0.89906 (−0.0464) | 0.81155 → 0.75582 (−0.0557) |
| v02_bvls_NO_shaping | 372 | 0.82401 → 0.83929 (+0.0153) | −0.59406 → **−0.72747** (\|+0.133\|) | −0.58870 → **−0.32961** (\|−0.259\|) | 0.75671 → 0.74233 (−0.0144) | 0.73425 → 0.72043 (−0.0138) | 0.74721 → 0.69492 (−0.0523) |
| v02_bvls_shaped | 372 | 0.82807 → 0.83647 (+0.0084) | −0.07189 → −0.17025 (\|+0.098\|) | −0.25942 → −0.39084 (\|+0.131\|) | 0.66712 → 0.64572 (−0.0214) | 0.70872 → 0.69452 (−0.0142) | 0.84640 → 0.83384 (−0.0126) |
| blend_2L_H128 | 372 | 0.88067 → 0.88191 (+0.0012) | −0.50865 → −0.50479 (\|−0.004\|) | −0.48945 → −0.54017 (\|+0.051\|) | 0.81694 → 0.80758 (−0.0094) | 0.84304 → 0.84045 (−0.0026) | 0.78646 → 0.78527 (−0.0012) |
| v47A_strict_QAT | 372 | 0.86570 → 0.86606 (+0.0004) | −0.41847 → −0.39961 (\|−0.019\|) | −0.46124 → −0.44313 (\|−0.018\|) | 0.79334 → 0.79378 (+0.0004) | 0.79273 → 0.79264 (−0.0001) | 0.76800 → 0.77039 (+0.0024) |
| T_b_lam1e-3 | 372 | 0.86951 → 0.86927 (−0.0002) | −0.53065 → −0.54112 (\|+0.010\|) | −0.56744 → −0.55095 (\|−0.016\|) | 0.80275 → 0.80372 (+0.0010) | 0.80288 → 0.80312 (+0.0002) | 0.77478 → 0.77495 (+0.0002) |
| BHdr_sdr_route | 372 | 0.84399 → 0.84316 (−0.0008) | −0.48232 → −0.48036 (\|−0.002\|) | −0.64014 → −0.63797 (\|−0.002\|) | 0.73568 → 0.73370 (−0.0020) | 0.75703 → 0.75566 (−0.0014) | 0.79640 → 0.79489 (−0.0015) |
| **ADD156** (control) | 372 | 0.86338 (=) | −0.44622 (=) | −0.53319 (=) | 0.80822 (=) | 0.82348 (=) | 0.77734 (=) |
| **winner_dial** (control) | 156 | 0.89390 (=) | −0.33521 (=) | −0.43049 (=) | 0.94639 (=) | 0.95773 (=) | 0.80406 (=) |
| **Ebothg_scr05** (control) | 156 | 0.87929 (=) | −0.27068 (=) | −0.41034 (=) | 0.93900 (=) | 0.95456 (=) | 0.80743 (=) |

`(=)` is **exactly** `+0.00000` on all 15 corpora, not a rounding. The three
controls read only `f0..155`, the block that is bit-identical across the eras —
so they are the harness's structural negative control, and they pass it. (The
table script also asserts a zero delta on every byte-identical corpus; it does.)

Z-RMSE, per-ref SROCC and every other panel column are in
`roster_era_table.tsv`. B's Z-RMSE improves on CID22 (0.4823 → 0.4717) and
KonJND (0.8596 → 0.7887) and worsens on kon504 (0.7743 → 0.8444).

### 3a. Composite and the dial

| bake | composite stored | composite current | Δ |
|---|---:|---:|---:|
| cl_tfm_LQ_MLP | **0.88996** | 0.84081 | **−0.04916** |
| blend_2L_H128 | 0.86865 | **0.86882** | +0.00016 |
| B_shipped | 0.82906 | 0.84074 | **+0.01168** |
| Ebothg_scr05 | 0.82857 | 0.82857 | 0 |
| T_b_lam1e-3 | 0.82203 | 0.82290 | +0.00087 |
| v47A_strict_QAT | 0.81947 | 0.81800 | −0.00147 |
| winner_dial | 0.81703 | 0.81703 | 0 |
| ADD156 | 0.81606 | 0.81606 | 0 |
| v02_bvls_NO_shaping | 0.79888 | 0.81596 | +0.01708 |
| BHdr_sdr_route | 0.79554 | 0.79491 | −0.00064 |
| v02_bvls_shaped | 0.75081 | 0.76328 | +0.01247 |

**The composite leader changes**: `cl_tfm_LQ_MLP` (0.88996) → `blend_2L_H128`
(0.86882). B moves from 3rd to 3rd but closes to within 0.00007 of 2nd.

**The dial panel is IDENTICAL across the two eras for every bake** — mono, tied,
dynamic range, reach, all bit-equal. That is a **structural identity, not a
finding**: the dial grid is its own file (`--dial-grid`), outside
`--features-root`, so both runs read the same features. It is also **not
era-clean**: `dial_grid_372col_2026-05-29*.parquet` was extracted 2026-05-29,
i.e. after `2dab8f30` but BEFORE `6af83b60`, so its own masked/IW block comes
from the thread-count-dependent era. It is not re-extractable today — its pairs
TSV names the `q<X>.png` decode cache deleted 2026-06-22 (**2,560 of 2,560 dist
paths missing**), so a rebuild needs a decode pass first. Registered, not
executed; registry id `dial372-grid-thread-dependent-era-2026-08-30`.

### 3b. Ordering flips (41; full list in `roster_era_table.md`)

The decision-relevant ones:

| corpus | stored order | current order |
|---|---|---|
| CID22 | cl_tfm 0.8829 > blend_2L 0.8807 > Ebothg_scr05 0.8793 > **B 0.8764** | **B 0.8821** > blend_2L 0.8819 > cl_tfm 0.8805 > Ebothg_scr05 0.8793 |
| KonJND | **cl_tfm 0.761** > v02_NO 0.594 > B 0.547 > T_b 0.531 > blend_2L 0.509 … | v02_NO 0.727 > **B 0.650** > T_b 0.541 > blend_2L 0.505 … > **cl_tfm 0.272 (last)** |
| kon504 | **B 0.594** ≈ v02_NO 0.589 > T_b 0.567 > cl_tfm 0.558 > ADD156 0.533 | T_b 0.551 > blend_2L 0.540 > ADD156 0.533 > **B 0.519** > cl_tfm 0.430 > v02_NO 0.330 |
| AIC-3 | **cl_tfm 0.812** > Ebothg 0.807 > BHdr 0.796 > blend_2L 0.787 > B 0.777 | Ebothg 0.807 > BHdr 0.795 > blend_2L 0.785 > T_b 0.775 > **cl_tfm 0.756 (last)** |

**A published claim this overturns:** the 2-layer blend's headline
"CID22 0.8807 vs shipped B 0.8764, **+0.004**"
(`benchmarks/mlp_2layer_diverse_2026-07-15.bin.pointer.md`,
`benchmarks/blend_2layer_methodology_2026-07-15.md`) is an **era artifact**. Both
bakes read the same two roots here; on the current extractor the gap is
**−0.0002 in B's favour** (0.88191 vs 0.88212). The blend's other wins survive
(TID +0.062, nonphoto +0.088 — nonphoto is byte-identical across the eras, so it
was never at risk); its **KonJND loss deepens** (−0.038 → −0.145) and its
kon504 position **improves** (from behind B to ahead of it). The blend-vs-B
decision is still open, but not on the CID22 number it was argued from.

---

## 4. kon504: two files, two eras, and why two lanes are BOTH right

The W-LIN r7 lane (`df931814`) reports "kon504 has NO era difference … a fresh
HEAD extraction of the 504 keyed pairs is BIT-IDENTICAL to R1b's". This lane
reports B's kon504 SROCC moving |0.5935| → |0.5194|. **Both are correct** —
they are different files:

- **R1b's keyed 504 rebuild** is a 2026-08-30 (post-fix) extraction. HEAD
  reproduces it bit-for-bit. ✔
- **`konjnd_jpeg504_372_2026-08-29.parquet` in the 372 root** is a byte-exact
  row subset of the **pre-fix** `konjnd_features_372col_2026-05-15.parquet`
  (established by the drift doc). MEASURED here against a fresh subset of the
  same 504 rows: **basic+peaks bit-identical, masked 34,525 cells / IW 35,254
  cells over tolerance on 100 % of rows (max_abs 0.0302 / 0.1200)** — the §3.27
  signature exactly. ✔

So "the kon504 ruler" is ambiguous today and a reader can pick up either era
under nearly the same name. The new root ships **`konjnd_jpeg504_372_2026-08-30.parquet`**
plus a `kon504/` one-file side root, both current-era, both derived from this
root's konjnd by the stored 504's own keys. Note the drift doc's §4c.6
`SRC0437` pair-list defect applies to anything keyed off
`konjnd_jpeg_val_pairs.tsv`; the tables here go through `load_konjnd`, which
picks `SRC0437_JPEG_059.jpg`.

---

## 5. CORRECTION to `benchmarks/v1_extractor_drift_2026-08-30.md` §3b

Its §3b table's **KADID, TID and AIC-3 rows are not valid** (CID22 and KonJND
are). Cause: `mkroots.py` aligned stored↔fresh on
`(ref_basename, round(human_score, 9))` with a first-occurrence-wins index. That
key is **not unique** on those corpora — stored-side rows sitting in a repeated
key group: **KADID 6,557 of 10,125 (64.8 %), AIC-3 600 of 600 (100 %), TID 726
of 3,000 (24.2 %)** — so every row of a group was matched to the SAME fresh row.
MEASURED on the drift lane's own `freshroot/` tables (distinct feature vectors
vs rows):

| corpus | rows | distinct | duplicated |
|---|---:|---:|---:|
| cid22 | 4,292 | 4,292 | **0** |
| konjnd | 1,008 | 1,008 | **0** |
| kadid | 10,125 | 6,227 | 3,898 (38.5 %) |
| tid | 2,880 | 2,505 | 375 (13.0 %) |
| **aic3** | **600** | **100** | **500 (83.3 %)** |

AIC-3's fresh column therefore scored 100 distinct images repeated six times
against 600 distinct targets. Positionally aligned (both tables come from the
same loader over the same label file, and `human_score` is elementwise equal —
guarded by `drift_cmp.py --positional`):

| corpus | drift doc §3b (key-aligned) | corrected (positional) | note |
|---|---|---|---|
| KADID | 0.82008 → 0.80426 | 0.82008 → **0.80847** | Δ −0.0116, not −0.0158 |
| TID | 0.78866 → 0.79691 | **0.78683 → 0.77852** | **sign flips**; and on 3,000 rows, not 2,880 |
| AIC-3 | 0.77743 → 0.79410 | 0.77743 → **0.76501** | **sign flips** |
| CID22 | 0.87638 → 0.88212 | identical | unique key |
| KonJND | 0.54665 → 0.64967 | identical | unique key |

**Consequence for the drift doc's conclusion.** "On every corpus that is a
genuine holdout, the runtime B is **better** than the evaluated B" is
FALSIFIED: **AIC-3 is a genuine holdout and it goes DOWN** (−0.0124). The
honest statement is: the runtime B is better on CID22 and KonJND(full) and
worse on kon504, KADID, TID and AIC-3 — the era shift is not a uniform
improvement, it is a re-weighting. Nothing about the *mechanism* (§1, §2) or the
decision (§4: keep the extractor, the stored tables are stale) changes.

This is the correction the drift lane's own §4c.7 was one step away from: the
row-drop it flagged and the alignment defect share a cause — a corpus's key not
being the thing that identifies a row.

---

## 6. What is registered, not executed

1. **`bake_verdict`'s default `--features-root` is NOT flipped** to the new
   root. Flipping it silently changes every future 372-regime number for every
   lane mid-campaign; that is a governance call for the campaign owner, not this
   lane. Until then, a current-era 372 verdict needs an explicit
   `--features-root /mnt/v/zen/zensim-training/2026-08-30-full-features-372`.
2. **The board is NOT regenerated** (registered for the board owner). The
   ready-to-promote current-era verdicts are
   `/mnt/v/output/zensim/eval372-roster-2026-08-30/json/<label>_new.json`;
   promote with
   `python3 scripts/promote_fulleval.py --verdict <that file> --name <board name>`.
   Suggested names keep the era explicit, e.g.
   `b_sdr_linear_cid80_inclwinsor_dense_dial__r372cur`.
3. **The 372 dial + corruption grids** are 2026-05-29 / 2026-05-28 artifacts
   from inside the thread-dependent window and cannot be re-extracted without a
   decode pass (their `q<X>.png` cache was deleted 2026-06-22). §3a.
4. **aic4 stays pre-fix** — source CSV gone (drift doc §4c.5).
5. **B's training legs** (safesyn 196,086 + cid22_train 17,611 + kadid 10,125 +
   tid 3,000 + `hdr_v3mix`) are still pre-fix; the retrain remains a fleet wave
   (drift doc §4c.3). This lane changes nothing about B's weights.
6. **BHdr on its own PU-linear HDR route** is still unmeasured. The
   `BHdr_sdr_route` row here is BHdr read on the SDR 372 tables — it bounds the
   *SDR-route* era sensitivity at ≈0.002 SROCC and says nothing about the HDR
   route (drift doc §4c.4).

## 7. Registry entries added (`benchmarks/eval_annotations.json`)

- `eval372-stored-root-thread-dependent-2026-08-30` — **invalidated**, the 6
  measured f156-371-using 372-class board cells, fields
  `rank.{cid22,kadid,tid,konjnd,aic3}` + `composite`.
- `eval372-basic-only-bakes-era-independent-2026-08-30` — **annotated**, the 3
  measured basic-only cells: the invalidation does NOT apply, Δ is measured at
  0.00000, do not re-verdict them expecting a change.
- `dial372-grid-thread-dependent-era-2026-08-30` — **annotated**, all 9, fields
  `dial` + `gates.g1_dynamic_range`.

Rendering verified with
`freeze_check --profile balanced-2026-08-04 --annotations benchmarks/eval_annotations.json`:
the invalidation shows on `b_sdr_linear_cid80_inclwinsor_dense_dial`, the
era-independent note (and NOT the invalidation) on
`ADD156_safesyn_only_raw_lasso`, and neither on a 944 cell.
