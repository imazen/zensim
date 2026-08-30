# v1's 372-feature vector width — root cause, fix, and blast radius (2026-08-30)

**Resolves** `docs/DATASET_HISTORY.md` §3.26 and
`benchmarks/r1b_keyed_rebuild_2026-08-30.md` §8.5(d).
**Fix commit:** `f9fac41e` (`fix(v1-372 width): reflect-pad at EVERY pyramid entry`), on `main@origin`.
**Gate:** `zensim/tests/v1_feature_width_pure_function.rs` (8 tests).
**Artifacts:** `/mnt/v/output/zensim/v1width-fix-recheck-2026-08-30/` (+ `_MANIFEST.json`
with `build_commit`, sha256s, and the exact short-pair lists per slice).

---

## 1. What §3.26 recorded, and what is actually true

§3.26 recorded that ~6.5 % of R1b eval-slice rows came out **279** wide instead of 372
(453/6,953 imazen26, 422/6,142 nonphoto, 493/7,717 hfnlproxy), that the rows carried real
values, and — after explicitly retracting a first "size-dependent" explanation — that
**"a v1-372 feature vector is not a pure function of its pair"**, being a function of the
input BATCH instead. The evidence given was: the same 453 pairs re-run as their own batch
give only **33** short, and 5 of them run alone give **0**.

**That is wrong. The width is a pure function of `(W, H)`.** Two independent measurements:

**(a) The stored rows classify exactly.** With
`n_scales(W,H)` = the scale walk's own recurrence (`w ← simd_padded_width(W)`, `h ← H`,
halve, stop at `< 8`), the predictor `2 + n_scales·3·31` reproduces the **field count of
all 20,812 stored rows with ZERO errors** across the three slices. Every short row's min
side is in `[36, 55]`; the short and full size sets are **disjoint** (13 short size
classes vs 241 full for imazen26; overlap 0).

**(b) The contrast does not reproduce.** A binary built from the pre-fix tree
(`6d0a393a`, saved as `v2_ab_extract_PREFIX`) run on the exact lists:

| set | rows | short (279) | full (372) |
|---|---:|---:|---:|
| 5 of the short pairs, alone | 5 | **5** | 0 |
| the 453 short pairs, as their own batch | 453 | **453** | 0 |
| the full imazen26 slice | 6,953 | **453** | 6,500 |

§3.26 expected 0 / 33 / 453. The measured pre-fix answer is 5 / 453 / 453 — the width
never moved. Values too: for both the pre-fix and the fixed binary, the short pairs'
rows are **byte-identical** whether run 5-alone, 453-alone, or inside the 6,953-row
batch (453/453 and 5/5). Whatever the earlier 33/0 probe measured, it was not this
extractor on these pairs — the plausible candidate is a re-run routed through a
`Zensim::compute*` entry (which reflect-pads, hence 0 short) and/or a pair list rebuilt
from `ref_basename`, which §8.5(a) itself records as **not row-unique**.

**Why the size explanation was rejected the first time.** The walk pads the WIDTH by SIMD
alignment but not the height, so the rule is asymmetric and does not look like "too
small": `54x96` is FULL (54 → 64 by `simd_padded_width`) while `96x54` is SHORT, and
`62x96` is FULL while `48x64` is SHORT. Reading "min side < 64" off the data therefore
fails on real counterexamples. The true rule is **`simd_padded_width(W) ≥ 64 AND H ≥ 64`**,
i.e. `W ≥ 49 && H ≥ 64` for `W < 497`.

§3.26's *size* figures do not correspond to the short rows at all, under any parse:
across all three slices the short rows span **13 distinct `(W, H)` classes**, min side
`{36, 41, 42, 43, 44, 45, 47, 48, 54, 55}` and max side `{64, 96}` — not "168 distinct
sizes spanning 36…1024", and `512x384` is **not** among them. Whatever population that
figure was computed over, it was not the short rows.

## 2. Mechanism (file:line)

The pyramid walk stops on either dimension:

```
zensim/src/streaming.rs:862   pub(crate) fn compute_multiscale_stats_streaming(...)
zensim/src/streaming.rs:889       for scale in 0..num_scales {
zensim/src/streaming.rs:890           if w < 8 || h < 8 { break; }   // w starts at simd_padded_width(width), h at height
```
(the same `< 8` break guards the `*_with_ref` walk at `streaming.rs:3381` and the
`PrecomputedReference` dim table at `streaming.rs:2807`).
`combine_scores` then sizes the output from what survived — `metric.rs:4916`
`let n_scales = scale_stats.len();` — so 3 scales emit `3·3·31 = 279`, 2 emit 186, 1 emits 93.
The returned **score** was wrong too, by the same mechanism: `try_score_from_features`
re-derives `n_scales = features.len() / features_per_scale` (`metric.rs:393`) and divides
the raw distance by it, so a truncated vector was scored as the mean over the 3 surviving
scales instead of 4. Stated from the code, not measured — the only entry that could
produce such a vector is the `training` free function, whose `score()` no shipped consumer
reads (both extractors take `.features()`; the benches take timings).

`compute_with_config_inner` (`metric.rs:3145`, behind every `Zensim::compute*`) prevents
that by reflect-padding any sub-`MIN_PYRAMID_DIM` side first. **Three entries did not:**

| entry | kind | pre-fix behaviour at `W < 49` or `H < 64` |
|---|---|---|
| `compute_zensim_with_config` (`metric.rs:4800`) | `training` free fn | **silent short vector** — 93 / 186 / 279 wide, no error |
| `compute_zensim_with_ref_and_config` (`metric.rs:706`) | `training` free fn | **panic** `scale 0 width mismatch` (unpadded distorted vs padded reference) |
| `Zensim::compute_with_ref_into` (`metric.rs:2271`) | **PRODUCT API**, not gated | **panic**, same assertion — not previously reported |

The reported defect is entirely the first row: **both** v1-372 extractors call it —
`zensim-bench/examples/extract_features_372col.rs:195` and
`zensim/examples/v2_ab_extract.rs:319` — which is exactly why §8.5(d) saw "the identical
short-row set from BOTH extractors and from both the grouped and per-pair flows". It was
never a tool quirk or a flow quirk; it was the one function underneath both.

## 3. The fix

One owner for the decision, in `zensim/src/metric.rs`:

- `min_pyramid_dim_for_scales(num_scales)` → `8 << (num_scales-1)`; `MIN_PYRAMID_DIM` is
  now this at `NUM_SCALES = 4`, i.e. **64, unchanged**.
- `needs_pyramid_pad(width, height, num_scales)` — the single predicate.
- `reflect_pad_for_scales(src, num_scales)`; `reflect_pad_to_min` is a wrapper.

All seven pyramid entries now ask it: the four that already padded
(`compute_with_config_inner`, `Zensim::compute_with_ref`, the two strip routers,
`PrecomputedReference::new`) share the predicate instead of open-coding it, and the three
above gained it. Making the threshold `num_scales`-aware also closes the same hole for
`--num-scales 5/6` (a live `zensim-validate` CLI knob and
`ProfileParamsBuilder::num_scales`), which previously truncated silently at 64–127 px.

## 4. Gates

`zensim/tests/v1_feature_width_pure_function.rs` — 5 of its 8 tests fail at `6d0a393a`,
all 8 pass at `f9fac41e`:

- 372 at every one of the 21 R1B size classes, on the free fn and the with-ref free fn;
- alone vs 5-batch vs full batch vs reversed batch vs rayon-parallel batch — **bit-identical**;
- every side `8..=130` (square + both oblong orientations against 64 and 96) is 372;
- the whole public v1 surface (buffered / precomputed / strip / diffmap / training) handles a sub-64 side;
- `compute_with_ref_into` == `compute_with_ref` bit-for-bit at sub-64.

Unchanged and green: `v1_golden_bytes` (incl. `v1_same_class_determinism_bitexact`),
`size_invariance`, `feature_v2::tests::folded720_v1_pools_match_v1_path`,
`cargo test --release --workspace` (0 failures), `cargo clippy` clean.

**Bit-identity on real data.** All 20,812 R1b slice pairs re-extracted with the fix and
diffed line-by-line against the stored pre-fix CSVs (`format!("{v}")` on `f64` is
shortest-round-trip, so byte-equal ⇒ bit-equal):

| slice | rows | previously-372 | byte-identical | differ | previously-short | now 372 |
|---|---:|---:|---:|---:|---:|---:|
| imazen26 | 6,953 | 6,500 | **6,500** | 0 | 453 | **453** |
| nonphoto | 6,142 | 5,720 | **5,720** | 0 | 422 | **422** |
| hfnlproxy | 7,717 | 7,224 | **7,224** | 0 | 493 | **493** |
| **total** | 20,812 | 19,444 | **19,444** | **0** | 1,368 | **1,368** |

**Fold parity (the fixed value is the right value).** For all 453 previously-short
imazen26 pairs, the fixed v1 `f0..f155` is **bit-identical** to the stored 944 fold's
`f0..f155` (`r1b-pools944-2026-08-30/ext_imazen26.parquet`, row-aligned, 0 basename
mismatches). Control on the same run: every row whose `simd_padded_width(W) == W` is
bit-exact (2,209/2,209 full + 453/453 short); rows with `pad ≠ W` fall in the
pre-existing, documented "padded-width class" divergence
(`feature_v2.rs` `folded720_*` parity tests) and are unrelated to this fix.

## 5. Blast radius — measured

| surface | exposed? | evidence |
|---|---|---|
| **Canonical 372 parquets** (25 tables under `2026-05-15-full-features/`) | **NO** | every one has exactly `f0..f371`. A short row cannot become a parquet — the builder raises on a ragged column (`ArrowInvalid`), so a short row is always a build failure or an explicit drop, never a padded row. |
| The 11 canonical **corpora** (149,195 pairs: sdr25, aic4, konjnd, aic3, live, csiq, tid, cid22val, kadid, cid22_train201, safesyn_full) | **NO** | header-level dimension scan of every reference: **0 pairs** with `simd_padded_width(W) < 64` or `H < 64`. The defect could not fire on any of them. |
| cid22val / kon504 / safesyn, re-extracted (250 / 504 / 250 rows) | **NO** | all rows 372 wide; **alone == batch byte-identical** (250/250, 504/504, 250/250); **pre-fix binary == fixed binary BYTE-IDENTICAL** on all 1,004 rows — the fix is a no-op there. |
| **944 folded tables** | **NO** | the fold emits a fixed width by construction and pads through `feature_v2`'s own `reflect_pad_to_min`. Measured: `foldapp2pools` on the 453 sub-64 pairs gives 946 columns on **every** row, pre-fix and post-fix, and the two CSVs are **BYTE-IDENTICAL**. |
| `bake_verdict` runtime path | **NO** | it loads pre-extracted feature parquets and never extracts (module doc + no `compute_*` call site). It is exposed only through the tables it reads, and those are all full width. |
| **Product `Zensim::compute*` API** | **NO truncation; ONE panic** | no product entry ever returned a short vector (`product_compute_path_was_always_372`, and the full-surface sweep). The one product defect was `compute_with_ref_into` **panicking** at sub-64 — loud, not silent — with no in-tree or cross-repo caller found. |
| Cross-repo callers of the three defective entries | **NONE** | grep over `~/work/zen`, `~/work/imageflow`, `~/work/squintly`: only zensim's own benches/examples and the archived `_zensim-pu-panel` copy. |

**Where the defect DID land, and the one thing still open.** The three R1b eval slices are
the only place sub-64 renditions met the defective extractor. `build_r1b_samepair_roots.py`
dropped the 1,368 short rows (counted in its manifest, never silently), so
`r1b-samepair372-2026-08-30` is a 6,500 / 5,720 / 7,224-row restriction of a
6,953 / 6,142 / 7,717-row population — a **6.5 % row loss whose selection is
size-correlated** (every dropped row has a side under 64), which is the honest caveat on
any family-axis number read on those roots. Consequently
`/mnt/v/zen/zensim-training/r1b-372root-2026-08-30/` carries **three dangling symlinks**
(`ext_hfnlproxy`, `imazen26_test_120k_2026-07-16`, `nonphoto_features_372col_2026-07-15`)
into `r1b-372slices-2026-08-30/`, which only ever received `ext_konjnd_jpeg_val.parquet`.
The full-width replacements now exist as CSVs at
`/mnt/v/output/zensim/v1width-fix-recheck-2026-08-30/` (sha-manifested); promoting them to
parquet and re-cutting the same-pair roots at 6,953 / 6,142 / 7,717 rows belongs to R1b's
keyed-rebuild lane and is **registered, not executed** here.

Six shipped 372 tables built from the same picker corpus **do** contain sub-64
renditions and are nonetheless full width — `imazen26_test_120k` 687, `nonphoto_features_372col`
598, `ext_hfnlproxy` 255, `imazen26_valbucket` 18,585, `nonphoto_valbucket` 8,229,
`hfnlproxy_valbucket` 1,357 rows (29,711 total). Full width means they were **not** produced
through the defective free function; they came from a padding path.

**Not caused by this defect, and not fixed by this change:** a fresh v1-372 extraction of
cid22val and kon504 differs from the *stored* tables on essentially every slot
(masked/IW blocks differ on 100 % of rows, max_rel 0.34 / 1.0). That is the drift §8.5(b)
already priced ("+0.0060 SROCC for B on cid22, three and a half months of extractor
evolution") and §8.5(a) measured between the two extractors (|d| up to 0.0927 on f129).
It is pre-existing: the pre-fix and post-fix binaries agree **byte-for-byte** on those
same rows. Quantified here so nobody re-attributes it to the width fix.

## 6. Reproduction

```sh
cargo test --release -p zensim \
  --features custom-profiles,feature-regime-v2,threads,training \
  --test v1_feature_width_pure_function            # 8/8; 5 fail at 6d0a393a

# corpus-level, with a pre-fix binary built from 6d0a393a:
ZENSIM_AB_MODE=v1 ./v2_ab_extract_PREFIX short5.tsv   out.csv   # 5 rows, all 281 fields
ZENSIM_AB_MODE=v1 ./v2_ab_extract_PREFIX short453.tsv out.csv   # 453 rows, all 281 fields
ZENSIM_AB_MODE=v1 ./v2_ab_extract_FIXED  short453.tsv out.csv   # 453 rows, all 374 fields
```
Pair lists: `/mnt/v/output/zensim/v1width-fix-recheck-2026-08-30/short_pairs_<slice>.txt`
(row index + ref + dist + human_score for every affected row of all three slices) and
`short5.tsv` / `short453.tsv`.
