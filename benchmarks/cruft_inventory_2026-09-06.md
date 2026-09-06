# The cruft inventory — measured, with a class per finding

Plan: [`../docs/PLAN_CRUFT_PURGE_2026-09-06.md`](../docs/PLAN_CRUFT_PURGE_2026-09-06.md),
increment A. Read at `main@origin b546985e` in the `purge` workspace.
Every count below is a grep or a tool run, not an estimate. Production vs
`#[cfg(test)]` was split by brace-balancing the 42 `cfg(test)` attributes, not by
"line number > the first `cfg(test)`" — which is wrong for five files in this
crate (`attribution.rs` has 1,162 production lines AFTER its test module ends).

Classes: **DELETE** (dead or fully superseded) · **REPLACE-BY-PLAN** (its job is
now `feature_plan::Plan`'s) · **DEPRECATE-SHIM** (public, needs one release) ·
**KEEP** (load-bearing).

---

## 0. The headline, and it inverts the obvious prior

**The production positional layer is already thin. The debt is in the tests, the
consumers and the scripts.**

| | production | test |
|---|--:|--:|
| width literals in `zensim/src` | **11 — every one of them already a `const`** | **343** |
| `ComputeSet::from_block_profile` callers | **1** | 9 |
| `fold_engine::wide_bake_v2_read` callers | **1** | **0** |
| `FeatureRegime` readers of `.regime()` | **0** | 13 |
| `FeatureRegime::V1` constructions | **0** | 0 |

So the smallest cut available is also the one phase 5 already named: `Plan::for_bake`'s
`if layout.is_identity()` branch is the whole of `from_block_profile`'s production
reach, and its `else` branch — the id-space derivation — is already written, already
tested, and already the path every dense bake takes.

## 1. The shipped bakes — the ruling's own example, measured

`bake_block_profile` over `zensim/weights/*.bin`
(`/mnt/v/output/zensim/purge-2026-09-06/shipped_bake_block_profiles.txt`):

| bake | profile | declares | layer-0 rows | caller lines READ | carried and unread |
|---|---|--:|--:|--:|--:|
| `d_sdr_add156_id100_negrich_dial_2026-09-05` | **D (default)** | 372 | 372 | **28** | **344 (92.5 %)** |
| `d_sdr_add156_dense_dial_2026-08-31` | D (era-1) | 372 | 372 | 28 | 344 |
| `b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07` | **B** | 372 | 372 | 95 | 277 |
| `b_sdr_linear_cid80_dense_dial_2026-07-05` | retired | 372 | 372 | 95 | 277 |
| `b_sdr_linear_cid80_anchored_2026-07-04` | retired | 372 | 372 | 95 | 277 |
| `bhdr_linear_shaped_cvvdpmix_2026-07-12` | **BHdr** | 372 | 372 | 133 | 239 |
| `bhdr_linear_shaped_anchored2_2026-07-04` | retired | 372 | 372 | 50 | 322 |
| `v47_strict_qat_native_2026-05-27` | **A** | 372 | 372 | 285 | 87 |
| `c_sdr_purity944_2026-08-29` | **C** | 944 | 667 | 667 | **277 `FeatureTransform::Drop`** |
| `c_sdr_mlp944_corrmix_2026-08-05` | retired | 944 | 667 | 667 | 277 `Drop` |
| `c_hdr_l1t1944_2026-08-29` | **CHdr** | 944 | 697 | 697 | **247 `Drop`** |

**0 of 11 carry a `zentrain.feature_set_id`**, though `feature_layout::declared_layout`
has read one since phase 4.

## 2. `zensim/src` — the library

### 2.1 `FeatureRegime` — DEPRECATE-SHIM

`feature_v2.rs:1064`, `#[non_exhaustive] pub`, in the **supported** surface
(`docs/public-api/zensim.txt:150-156`) because `feature-regime-v2` is default-on
since 2026-09-01. Six variants (`:1067 :1069 :1079 :1086 :1091 :1098`).

* Constructed at exactly 2 sites, `feature_v2.rs:9305-9310` and `:9503`.
* **`FeatureRegime::V1` is constructed ZERO times** anywhere in the crate.
* Matched in 4 expressions / 13 arms, ALL inside `ZensimV2Result`'s own accessors
  (`:1142 :1170 :1195 :1212`), and each arm derives a block offset by arithmetic on
  `self.features.len()` minus a computed tail (e.g. `:1150-1152`) — the job
  `Layout::pos_of` / `Layout::slot_at` now owns.
* **In-crate production readers of `.regime()`: 0.** All 13 are test assertions.

### 2.2 Width literals — 11 production, all `const`

`attribution.rs:139 :142 :144 :146 :150` (`BLOCK_END_*`) → REPLACE-BY-PLAN;
`fold_engine.rs:253 :254 :255` (`V1_PEAKS/MASKED/IW`) → REPLACE-BY-PLAN;
`profile.rs:1381 :1618` + `metric.rs:5345` (`[f64; 228]` weight arrays) → **KEEP**,
these are shipped bytes. **Zero inline production width literals.** `504`, `265`,
`289`, `956` appear only in tests; `376` appears nowhere.

343 test literals, by file: `feature_v2.rs` 93 · `attribution.rs` 63 ·
`feature_plan.rs` 52 · `research.rs` 42 · `metric.rs` 36 · `feature_defs.rs` 21 ·
`profile.rs` 13 · `feature_layout.rs` 13 · `fold_engine.rs` 10. REPLACE-BY-PLAN.

### 2.3 Structural-zero producers — 12 sites

THE producer is `feature_v2.rs:9073`, `vec![0.0f64; v12_total + append_total +
append2_total + csfw_total]`: `f156..372` stays 0.0 unless `v1_pools != Off`, `f372+`
unless the block toggle is on. KEEP the allocation — what was missing is a record of
*which* positions are fills, and `Plan::emit` is now it. Also `:9214-9231` (Peaks /
Carriers partial writes), `:1801-1806` (`transducers_luma_only` masks PJND X/B to
0.0), `:7243-7244` + `:7343` + `:9156-9159` (`APPEND_SKIP_B_SCALE0`) — all KEEP,
all deliberate. `:9303-9304` and `metric.rs:3545-3560` / `:5296` → REPLACE-BY-PLAN.
Truncation: `fold_engine.rs:166` and `:209` (the latter takes no plan at all).

**Load-bearing caveat: the structural fill is NOT always 0.0.**
`research.rs:845-866` derives 12 `PJND_FRAGILITY` slots (defect F15) whose finaliser
returns **1.0** on the degenerate no-samples case whether or not the kernel ran. Any
purge that assumes "unpopulated ⇒ zero" is wrong on those twelve.

### 2.4 Pool modes and skip policy

| symbol | prod / test sites | visibility | class |
|---|--:|---|---|
| `V1PoolsMode` (+ `CARRIER_SLOTS`) | 56 / 59 | **pub, supported** | DEPRECATE-SHIM |
| `V1FreeExtras` | 26 / 26 | **pub** type; the `free_extras` FIELD is `#[doc(hidden)]` and its own doc says *"TEST/BENCH INSTRUMENTATION — NOT A PRODUCT MODE"* | DEPRECATE-SHIM |
| `skip_unread_pools` (`metric.rs:1240`) | 5 / 0 | `pub(crate)` | KEEP |
| `with_unread_feature_skipping` (`metric.rs:1331`) | 3 / 3 | `pub` + `#[doc(hidden)]` | KEEP |
| `score_pool_mode` (`fold_engine.rs:512`) | 3 / 7 | `pub(crate)` fn, doc-hidden method | REPLACE-BY-PLAN — its only production consumer is the doc-hidden read-back at `metric.rs:1344`; the live path is `score_plan` |
| `fast_by_default` (`metric.rs:1284`) | 4 / 0 | local | KEEP — the whole of Profile D's speed wiring |

### 2.5 The two superseded derivations

`ComputeSet::from_block_profile` (`feature_v2.rs:2143`, `pub(crate)`) — **1
production caller** (`feature_plan.rs:317`), 9 test callers.
`fold_engine::wide_bake_v2_read` (`fold_engine.rs:395`, `pub(crate)`) — **1
production caller** (`feature_v2.rs:2189`), **0 test callers**.

Deleting both collapses `Plan::for_bake` to its own `else` branch. One perf risk,
named honestly: `wide_bake_v2_read` is what keeps a 944-wide free-set bake off the
full 944 walk (`fold_engine.rs:386-394`); the id-space derivation must reproduce
that or the fallback is `everything`. Neither symbol is in either public-API file.

### 2.6 `caller_input_width` vs `n_inputs`

7 production `caller_input_width()` calls, **all POSITION counts**
(`feature_layout.rs:221`, `feature_plan.rs:287`, `feature_v2.rs:2172`,
`research.rs:249`, `metric.rs:591 :4694 :4734`). 2 production `n_inputs()` calls,
**both COLUMN counts** (`fold_engine.rs:279`, `metric.rs:4729` — the one site that
uses both readings in one condition). The local binding *named* `n_inputs` in
`metric.rs` is bound from `caller_input_width()`; that name is the debt, not the
distinction. KEEP.

`prep_bake_input_f32` (`metric.rs:4811`, private) has four branches — exact / `+4`
size-axes / **silent positional prefix** / loud mismatch. The prefix branch is the
`position == id` assumption in its purest form; it is load-bearing for legacy
300/228-input bakes and is bypassed entirely by the dense gather at `metric.rs:4793`.

## 3. Consumers — `zensim-validate`, `-bench`, `-regress`, scripts

### 3.1 `--regime`: 4 parsers, 105 invocations

Parsers: `bake_verdict.rs:1140` (the only real one — **three names collapse to two
booleans**), `scripts/register_ladder_instrument.py:70`,
`scripts/canonical_corpus/build_tbig_200k.py:314` (a metadata tag, not a scorer
regime), `scripts/harvest_bakes.sh:68`.

**7 more files derive a regime with no flag**: `scripts/run_full_eval.sh:69`
(and remaps a **fourth** value, `924`, that `bake_verdict` cannot parse),
`scripts/dialgate_arms.sh:117`, `benchmarks/wave_r4_2026-09-01/score_arm.sh:8`,
`scripts/add156_pools.sh:115,148` and `scripts/linbandvis/x_pools.sh:172,205,229`
(ROOT LETTER → regime), `scripts/v_next/measure_dial_zones.py:57-59`, and
`scripts/gaddr_board_regrade.py:145-150` — which infers the regime from a corpus
**filename** because "`--regime` is not printed". It is printed now (phase 5); that
inference is **DELETE**.

10 Rust sites map the value to a width/root/grid/**corpus list**
(`bake_verdict.rs:1445-1487 :3516 :4145 :4228-4252 :4269-4285 :4521 :5603`,
`zensim_mlp_train.rs:4443-4453`). 105 invocation lines across 61 files; 4 of them
are forwarding OWNERS every wave script reaches `bake_verdict` through.

**Class: the FLAG is DEPRECATE-SHIM** (it already resolves through
`feature_set::registry().regime()` and prints its meaning); the `regime_720` /
`regime_944` booleans and the 10 mapping sites are the actual REPLACE-BY-PLAN debt.

### 3.2 `eval_roots.rs` — KEEP, with one duplicate owner

6 root constants (2 with zero Rust callers, both registered era labels backed by a
real root on disk — KEEP), `KONJND_JPEG504_372_SLOTS`, `ResolvedSlot`, `resolve_slot`.
`era_of()` (`:85-110`) is a hand-written `match` on six path literals answering the
same question as `feature_set::Registry::era_for_root` (`feature_set.rs:472`) —
**two era labellers, one not registry-backed → REPLACE-BY-PLAN.** Six root-class
grid paths live OUTSIDE the one-owner module, as private consts in `bake_verdict.rs`
(`:399 :401 :408 :443 :445 :453`) → REPLACE-BY-PLAN.

### 3.3 `block_profile.rs` — KEEP the mechanism, REPLACE the tables

`used_caller_lines` (`:439`) is the single definition of "what a bake reads" and
`feature_set::bake_feature_set_ref` calls it deliberately so the id derivation and
the block table cannot fork. KEEP. The two hardcoded `FAMILIES` tables (`:75-80`,
`:91-96`) and `beyond_f943_cols` (`:496`) restate numbering the registry already
carries as `Registry::token_slots` → REPLACE-BY-PLAN. Same shape in
`bake_contrib.rs:276-299` (13 ranges), `inspect_l0_input_norms.rs:183-188`, and
`zensim_mlp_train.rs:3592-3594` (the `--coarse-l2-mult` indices are valid only for
the 924/944 layout and are silently clipped by `if i < nf`).

### 3.4 The silent widener — REPLACE-BY-PLAN, and it is the consumer-side twin of the
`--regime 944` bug

`bake_runtime.rs:369-375`: `let take = n_inputs.min(row.len()); … for f in
&mut f32_features[take..] { *f = 0.0; }`. A 944-input bake handed 372-wide rows
gets 572 zeros, scores, and returns a plausible number. It is the DEDUP-M canonical
dispatch, so every scorer inherits it (`bake_verdict.rs:1634`, `qsweep_eval.rs:485`,
`bake_contrib.rs:591 :1221`, `ensemble_score_rows.rs:114`, `preview_stats_demo.rs:249`,
and four thin bins).

**The asymmetry is the finding.** Dial grids, corruption grids, per-pair metric
sources and the corruption head ARE width-gated and skipped loudly
(`bake_verdict.rs:5015 :5173 :5255 :5284 :5862`). **Corpora are not** —
`render_corpus` (`:3510-3535`) never compares `g.n_features` to `n_inputs`. The two
guards added since are narrower than the hole: `folded_root_conflict` fires only
under `--regime 944` and only for `f156..371`, and `feature_set::check` REPORTS
unless `--require-feature-set-match` is passed — **which no committed script passes
(0 hits).** Closing it at `render_corpus` costs one comparison.

### 3.5 Table builders — 48 write a manifest, ONE writes a `feature_set_id`

`scripts/canonical_corpus/pack_eval372_root.py:57 → :237-238` is the only one, and
there it is **optional**. Every other root falls back to the registry path table or
to a free-prose `"regime"` string, both of which set `inferred: true` — evidence
about a NAME, never about BYTES. That is why `bake_verdict` prints `table id NOT
ESTABLISHED` on most roots and why `--require-feature-set-match` cannot yet default
to on.

The producer side has a **second, unregistered vocabulary**: `ZENSIM_AB_MODE` ∈
`foldapp | foldapp2 | foldapp2pools | foldapp2carriers | foldcsfw`. Four of those five
emit **944 columns**, `--regime 944` cannot tell them apart, and the only thing that
does is `bake_verdict.rs:4217-4220`'s `root_regime.ends_with("pools")` — a
**string-suffix heuristic standing in for an id**. REPLACE-BY-PLAN.

### 3.6 Smaller items

DELETE: `zensim-validate/src/main.rs:200-220` `FeatureTier{Basic=156, Peaks=228,
Extended=300}` + its two `row.truncate(tier_features)` sites (`:1060`, `:1203`) —
all three widths are ones the trainer BANS at `zensim_mlp_train.rs:2656`, and the two
live callers (`build_eval372_root.sh:20`, `r6_extract_arms.sh:23`) return before
reaching them. DELETE: `examples/mlp_cross_check.rs:26-37` (228-wide vectors).

DEPRECATE-SHIM: five probe binaries' `const NF = 372`
(`train_minmax.rs:26`, `embedding_distance_probe.rs:38`,
`unconstrained_mlp_probe.rs:21`, `monotone_subspace_probe.rs:34`,
`residual_identity_probe.rs:31`); `bake_quant_stats.rs:243` and
`concat_three_way.rs:191` labels; `extract_features_372col_omni.rs`.

KEEP: all of `feature_set.rs` (this is the replacement, not the debt);
`parquet_loader`'s width-dynamic contiguous-prefix scan (it never pads and never
truncates — but it IS why a table's identity is "how many contiguous `f<N>` columns
it happens to have", so a table missing `f7` silently becomes a 7-wide table);
the trainer's `< 372` ban and `--expect-n-feat` guard; `scripts/sota944/slice_*.txt`
(11 positional index files, 3,000+ lines — these ARE the registered compute sets and
should be REGENERATED from named slot sets, never hand-edited); `zensim-regress` has
**zero** feature-width literals.

## 4. Totals

| class | sites | principal items |
|---|--:|---|
| DELETE | 6 | `wide_bake_v2_read`; `FeatureTier` + 2 truncations; `mlp_cross_check` 228 vectors; `gaddr_board_regrade` regime-from-filename |
| REPLACE-BY-PLAN | ~90 prod + 343 test | `from_block_profile`; the `regime_720/944` booleans + 10 map sites; `BLOCK_END_*` + `V1_*` + both `FAMILIES` tables + `bake_contrib::family_of`; `era_of`; `bake_runtime::score_row`'s zero-pad; 9 of 10 table builders; `ZENSIM_AB_MODE` → ids |
| DEPRECATE-SHIM | ~125 | `FeatureRegime` (+ `.regime()`), `V1PoolsMode`, `V1FreeExtras`, `FeatureView`; the `--regime` flag and its 105 call sites; 5 probe `const NF` |
| KEEP | — | `feature_set.rs`; `used_caller_lines`; every `eval_roots` constant; `parquet_loader`; `prep_bake_input_f32`'s branch table; `skip_unread_pools` / `fast_by_default`; the `[f64; 228]` weight arrays |

**One more supported-API instance of the same bug class, not previously named:**
`FeatureView::new` (`metric.rs:5049-5071`) **guesses the tier from the vector's
length** and returns `None` otherwise — 34 supported-API lines. Its own v2 successor
documents the contrast at `feature_v2.rs:1219-1221`: *"Unlike v1's `FeatureView`,
`FeatureViewV2::new` VALIDATES the exact expected length rather than guessing a tier
from an ambiguous length."* DEPRECATE-SHIM.
