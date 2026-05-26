# DATA INTEGRITY ROOT CAUSE: kadid/tid iwssim + ssim2_gpu corruption (task #215)

Forensic follow-up to
[`DATA_INTEGRITY_kadid_tid_metric_columns_2026-05-25.md`](DATA_INTEGRITY_kadid_tid_metric_columns_2026-05-25.md),
which found the symptom and traced propagation but pinned the v24 build as
"an ad-hoc step, NOT a committed script". **That is wrong — the buggy code
is committed.** Both corruption modes have a named, committed origin, and
the two modes are *independent bugs in two different scripts*.

Census reproducible via
[`scripts/canonical_corpus/audit_metric_columns.py`](../scripts/canonical_corpus/audit_metric_columns.py);
full TSV at [`metric_column_census_2026-05-25.tsv`](metric_column_census_2026-05-25.tsv).

---

## TL;DR

| Question | Verdict |
|---|---|
| Found the buggy code? | **YES — both modes, committed.** |
| iwssim leak origin | `scripts/v_next/v0_22_iw_make_mock_val_csvs.sh:56` (`iwssim := human_score` mock, val-only by design; mock qualifier later lost) |
| ssim2_gpu corruption | **Pure JOIN bug**, NOT a metric bug. `scripts/v_next/build_ex3_mix_corpus.py:add_ssim2_to_372feat_corpus()` L233-242 |
| Is the ssim2-gpu METRIC sound? | **YES.** `zen-metrics batch` scores correct (ref,dist) pairs; the per-pair sidecar `*_ssim2_local.parquet` is correct (9882/10125 unique values). The Python join collapsed it. |
| Scope | **2 root corpora × ~14 derived parquets affected; safesyn + LARGE + all score sidecars CLEAN.** |
| Shipped bakes safe? | **YES.** V39 (PreviewV0_3) + the whole shipped lineage train on `human_score`, never the corrupt columns. |
| scores/ sidecars correct? | **YES — they are the correct source the v24 build should have joined against (full 4-key), but didn't.** |

---

## 1. The buggy code (both modes, committed)

### Mode A — `iwssim` = literal `human_score` copy (a *mock*, mispropagated)

**Origin:** [`scripts/v_next/v0_22_iw_make_mock_val_csvs.sh:50-58`](../scripts/v_next/v0_22_iw_make_mock_val_csvs.sh)

```awk
NR==1 { print "ref_basename,human_score,iwssim," substr(...); next }
{ print $1 "," $2 "," $2 "," substr(...) }   # field $2 (human_score) duplicated into field 3 (iwssim)
```

The script's own docstring (L1-22) is explicit: KADID/TID have **no real
IW-SSIM** (only safesyn got the Wang & Li 2011 compute —
`iwssim_targets_safesyn_2026-05-16.parquet`, 196,086 rows). So a *mock*
`iwssim := human_score` copy was created **deliberately, for
validation-SROCC monitoring only** (RankNet loss is rank-invariant within a
group, so the mock just preserves the val group's native rank order). The
output files are even named `*_iwssim_mock.csv`.

**The bug is not the mock — it's that the mock leaked into TRAINING parquets
with the "mock" qualifier stripped.** Propagation chain (each step verified
by census):

1. `v0_22_iw_make_mock_val_csvs.sh` → `2026-05-16/{kadid,tid}_features_372col_2026-05-15_iwssim_mock.csv` (iwssim==human, 100% identical)
2. `scripts/v_next/v0_22_iw_v2_add_log_target.py` → `2026-05-16/v2/{kadid,tid}_features_iwssim_log_372col.parquet` (adds `iwssim_log_norm` derived from the mock — see L83-87, reads the `_iwssim_mock.csv` path explicitly)
3. (intermediate join) → `2026-05-17-cvvdp/{kadid,tid}_features_iwssim_cvvdp_372col.parquet` and `_mix_targets_372col.parquet` — **"mock" qualifier dropped from filename; `iwssim` now indistinguishable from a real column** (census: iwssim corr 1.000, 100% identical)
4. `scripts/v_next/build_ex3_mix_corpus.py` (next section) → `2026-05-18-ssim2/{kadid,tid}_4target_372col.parquet`
5. `scripts/v_next/build_v24_mix_target_corpus.py` → `2026-05-18-v24/` (pure passthrough rename `mix_cv33_iw33_sm33 → mix_target`)
6. `scripts/canonical_corpus/build_canonical_parquets.py` + `build_canonical_2026_05_21.py` → `canonical-2026-05-{18,21}/train/{kadid,tid}.parquet`

The original doc's claim that the base DataFrame "was `…, iwssim, …` and never
overwritten with real IW-SSIM" is correct in spirit but had the wrong origin
file. The real origin is step 1, three corpus-generations upstream of v24.

### Mode B — `ssim2_gpu` ref-misjoin (pure JOIN bug)

**Origin:** [`scripts/v_next/build_ex3_mix_corpus.py:add_ssim2_to_372feat_corpus()`](../scripts/v_next/build_ex3_mix_corpus.py) L191-260, specifically L233-242:

```python
join_keys = ["ref_basename", "codec", "quality"]
available_keys = [k for k in join_keys if k in targets.columns and k in local_lite.columns]
...
local_agg = local_lite.groupby(available_keys, as_index=False)["ssim2_gpu"].mean()   # L241
merged = targets.merge(local_agg, on=available_keys, how="left")                       # L242
```

**The 372-feat targets parquet has only `ref_basename` — no `codec`, no
`quality`, no `q`** (verified: its non-feature cols are
`ref_basename, human_score, iwssim, iwssim_log_norm, cvvdp_score, …mix_cv*`).
So `available_keys` collapses to `["ref_basename"]`, and L241 averages **all
125 distortions of each reference into one mean ssim2**, broadcast onto every
row of that ref by L242. Result: `ssim2_gpu` constant within every ref group
(census: `ssim2_constant_per_ref = 1.000`, corr 0.013/-0.002 with MOS).

This is structurally unavoidable in this script: KADID/TID distinguish
distortions by `knob_tuple_json` (codec="kadid" and q=0 are constant), and
the targets parquet carries no per-pair key at all. The only correct join is
**positional** (row order == dmos.csv / mos_with_names.txt order), which is
exactly what the 2026-05-25 fix used.

---

## 2. ssim2_gpu: metric sound, join broken — proof

The corruption is **NOT** in the SSIMULACRA2 metric or the batch CLI. Three
independent confirmations:

1. **batch CLI scores (ref,dist):**
   [`zenmetrics/crates/zen-metrics-cli/src/main.rs:619-677`](../../zenmetrics/crates/zen-metrics-cli/src/main.rs)
   reads `ref_path` + `dist_path` columns from the pairs TSV and scores each
   pair; output keyed on `(image_path, codec, q, knob_tuple_json)`. No
   ref-vs-ref pairing anywhere.

2. **the per-pair sidecar is correct:**
   `/mnt/v/zen/zensim-training/2026-05-18-ssim2/kadid_ssim2_local.parquet`
   has **9,882 / 10,125 unique** `ssim2_gpu` values (range −367.20 … 99.99),
   with **~122 distinct values per `image_path`** (one per distortion). The
   metric produced correct, varied, per-pair SSIMULACRA2. (TID local sidecar:
   3,000/3,000 unique.)

3. **post-fix recompute recovers signal:** the 2026-05-25 fix recomputed
   ssim2 on the correct positional pairs → KADID SROCC 0.803, TID 0.849 vs
   MOS (census `*_fixed_2026-05-25.parquet` rows). A broken metric could not
   produce that.

**Verdict: (c) downstream join-by-ref-only bug** in `build_ex3_mix_corpus.py`.
NOT (a) metric bug, NOT (b) batch-CLI pairing bug.

A separate, independent leak exists on `cid22_train.parquet`: there
`ssim2_gpu` is a **100% copy of `human_score`** (MCOS), corr 1.000, 0%
varied — a different defect (target-as-score copy), unused by any shipped
recipe (see §4). Flagged for completeness; not the kadid/tid bug.

---

## 3. Complete scope census

Filtered to training/canonical parquets (full TSV excludes the ~550 per-chunk
`ssim2_imazen/*.parquet` raw sidecars, which are clean by construction).
`s2_const` = fraction of ref-groups with one unique ssim2 (1.00 = misjoin
signature; "N/A" when each ref appears once → per-pair sidecar, not a misjoin).

| Parquet (under /mnt/v/zen/zensim-training/) | iwssim | ssim2_gpu | Verdict |
|---|---|---|---|
| `2026-05-16/{kadid,tid}_..._iwssim_mock.csv` | HUMAN-COPY | absent | **iwssim leak ORIGIN** |
| `2026-05-16/v2/{kadid,tid}_features_iwssim_log_372col.parquet` | HUMAN-COPY | absent | iwssim leak |
| `2026-05-16/v2/safesyn_..._iwssim_log_372col.parquet` | real (0.896) | absent | clean |
| `2026-05-17-cvvdp/{kadid,tid}_features_{iwssim_cvvdp,mix_targets}_372col.parquet` | HUMAN-COPY | absent | iwssim leak |
| `2026-05-17-cvvdp/safesyn_..._{iwssim_cvvdp,mix_targets}_372col.parquet` | real (0.896) | absent | clean |
| `2026-05-18-ssim2/{kadid,tid}_4target_372col.parquet` | HUMAN-COPY | REF-MISJOIN | **ssim2 misjoin ORIGIN** |
| `2026-05-18-ssim2/{kadid,tid}_ssim2_local.parquet` | absent | **real, per-pair** | clean (correct source) |
| `2026-05-18-ssim2/large_3target_300feat_minus_ssim2.parquet` | real | real | clean |
| `2026-05-18-ssim2/ssim2_imazen_consolidated.parquet` | absent | real, per-pair (N/A) | clean |
| `2026-05-18-v24/{kadid,tid}_4target_372col.parquet` | HUMAN-COPY | REF-MISJOIN | YES (passthrough of ssim2-dir) |
| `2026-05-18-v24/{safesyn,large}_*.parquet` | real | real | clean |
| `canonical-2026-05-{18,21}/train/{kadid,tid}.parquet` | HUMAN-COPY | REF-MISJOIN | YES |
| `canonical-2026-05-21/train/{kadid,tid}_dial100.parquet` | HUMAN-COPY (corr 1.0) | REF-MISJOIN | **YES (doc missed these)** |
| `canonical-2026-05-21/train/{kadid,tid}_cvvdptgt.parquet` | real (0.55/0.79) | REF-MISJOIN | partial (ssim2 only) |
| `canonical-2026-05-{18,21}/train/{kadid,tid}_fixed_2026-05-25.parquet` | real (0.72/0.76) | real (0.80/0.85) | **FIXED** |
| `canonical-2026-05-{18,21}/train/safesyn.parquet` (+`_dial100`,`_cvvdptgt`) | real (0.896/0.81/0.48) | real | clean |
| `canonical-2026-05-21/train/cvvdp_iwssim_LARGE.parquet`, `cid22_train*` | real | real (cid22: =human-copy, unused) | clean for shipped targets |
| `canonical-2026-05-21/train/{konjnd-dense*,pipal}.parquet` | null | null | no real metric cols (harmless) |
| `canonical-2026-05-21/train/multiband_anchor_dial100.parquet` | real (0.79) | real (0.97), s2_const 0.72 | clean (small anchor set) |
| `canonical-2026-05-{18,21}/val/*.parquet` | null | null | no (carry human_score only) |
| `canonical-*/scores/{iwssim,ssim2}_imazen.parquet` | real, per-pair | real, per-pair (N/A) | **clean — the correct join source** |

**One-line scope verdict:** 2 root corpora (kadid, tid) with the bug baked at
2 distinct upstream stages, propagated into **~14 derived parquets** across
`2026-05-16` → `2026-05-18-v24` → `canonical-2026-05-{18,21}` (including
`_dial100`/`_cvvdptgt` variants the prior doc did not catalog). safesyn,
LARGE, cvvdp_iwssim_LARGE, all `scores/` sidecars, and all `val/*` are clean.

---

## 4. Shipped-bake safety — CONFIRMED SAFE

Every shipped bake's recipe uses `human_score` (or `cvvdp_*`, which is real
on kadid/tid) — never the corrupt `iwssim`/`ssim2_gpu`/`mix_*` columns.

| Shipped bake (`zensim/src/profile.rs`) | Recipe target | kadid/tid as train? | Safe? |
|---|---|---|---|
| **V39 / PreviewV0_3** (`v39_v32plus_spline_seed17`) | `human_score` (per-group normalized) — see `v5_vs_v03_comparison_2026-05-25.md:93-108` | yes, but on `human_score` | **SAFE** |
| `zensim_b_desktop` / `zensim_b_phone` | `cvvdp_*` (real on kadid/tid) | — | SAFE |
| PreviewV0_5* family (`v22_mix…`, `v_compression…`, `v05_ensemble`, `v_tuner`, `v_cross_codec`) | `mix_cv40_iw60`, but trained on **safesyn(real iwssim) + cid22_train(real iwssim) + LARGE** — kadid/tid val-only or absent | per recipe | SAFE for the shipped variants |

**Prior shipped V0_3 (`v_tuner_v11`)** recipe
(`benchmarks/v_tuner_v11_methodology_2026-05-24.md:90-96`):
`--target-column mix_cv40_iw60` on `safesyn:1.0:0.0` + `cid22_train:0.5:0.0`
**only** — kadid/tid not training groups → safe (safesyn/cid22_train iwssim
are both real).

### INVALID experiments (trained ≥1 epoch on the leaked eval label)

Recipes that set **kadid/tid train_weight > 0 AND** a contaminated target
(`mix_cv40_iw60` / `mix_cv35_iw65` / `mix_cv30_iw40_sm30` / `iwssim`) trained
30–60 % of the kadid/tid signal on the leaked `human_score` copy (via the
`iwssim_log_norm` term) and the misjoined ssim2. These results are invalid
and must be re-run on the `_fixed_2026-05-25` parquets:

- `run_persample_mix3_seed.sh` (target `mix_cv30_iw40_sm30`, kadid `0.3:1.0`) — **worst: iwssim leak + ssim2 misjoin**
- `run_metric_inputs_seed.sh`, `run_chunkc_perpair_seed.sh`, `run_exp_larger_large_seed.sh`, `run_seed.sh` (`mix_cv40_iw60`, kadid `0.3:1.0`)
- `run_tuner_v11_attempt7_seed.sh`, `run_tuner_v11_yj_autotransforms_2026-05-25.sh`, `arch_eval_matrix.sh` (`mix_cv40_iw60`, kadid `0.5:0.0`)
- `run_v11a*`, `run_v11av2*`, `run_v12a_cvvdp_seed.sh`, `launch_sweep.sh` (`mix_cv35_iw65`, kadid `0.6:0.4` or `0.3:1.0`)
- `run_chunkc_anchor_only_seed.sh` (`mix_cv40_iw60`, kadid `1.0:1.0`)

NOT invalid: `run_v13_cvvdp_distill_seed.sh` / `run_v14_cvvdp_lognorm_seed.sh`
(`cvvdp_*` targets — real on kadid/tid), `run_distill_seed.sh`
(`ensemble_teacher`). `run_metric_inputs_fixed_*` were the prior partial
repair (commit `f370b97`) and used a separately-fixed corpus.

Note: commit `f370b97` (2026-05-18) already independently diagnosed the
"ssim2 constant per ref (first-row-wins join key collision)" for the
`metric-inputs` corpus and built fixed parquets at
`2026-05-18-metric-inputs-fixed/` — but that fix did **not** propagate back
into the canonical train set, which kept shipping the corrupt columns until
the 2026-05-25 `_fixed_` siblings.

---

## 5. scores/ sidecar verdict — they are the CORRECT source

`canonical-2026-05-{18,21}/scores/{iwssim_imazen,ssim2_imazen}.parquet` are
**clean and per-pair correct**, keyed on the full
`(image_path, codec, q, knob_tuple_json)`:

- `iwssim_imazen`: 74,801 / 75,300 unique values (range 0.727–1.0)
- `ssim2_imazen`: 15,189 / 55,000 unique values (range −50.5–99.99), 55,000
  distinct `image_path` = 55,000 rows (each a distinct distorted-image cell)

The census flags `ssim2_imazen` as `s2_const=1.00`, but that is a **false
positive of the heuristic** — its `image_path` is the full distorted path
(one row per cell), so "one ssim2 per image_path" is trivially true, not a
misjoin. The audit script now gates this with `mean_group_size > 1.5`.

**These sidecars cover the codec-sweep corpus (safesyn/LARGE), NOT
KADID/TID** (KADID/TID images aren't in the codec q-sweep). For KADID/TID the
correct per-pair source is `2026-05-18-ssim2/{kadid,tid}_ssim2_local.parquet`
(also clean). The v24 build's mistake was joining the per-corpus features
against a per-pair source **on `ref_basename` alone** (because the features
parquet had no other key), instead of doing the only correct join available —
**positional row-alignment** — which is what the 2026-05-25 fix did.

---

## 6. Recommended permanent fix

The 2026-05-25 `_validate_metric_columns()` guard in
`build_canonical_2026_05_21.py` (raises if iwssim==human-copy or ssim2
constant-per-ref) is a **good last-line backstop but insufficient on its own**
— it catches the symptom at the *final* build stage, three generations
downstream of where the bug is introduced, and it can't catch the iwssim
mock-propagation (the mock is a legitimate val-only artifact; the guard would
have to know the corpus's *role*).

The real fixes, in priority order:

1. **Never join a per-pair metric onto a per-corpus features table on
   `ref_basename` alone.** `add_ssim2_to_372feat_corpus` should have *failed
   loudly* when `available_keys == ["ref_basename"]` (i.e. when the target
   parquet lacks a per-pair key). Add an assertion to any join helper:
   `assert set(join_keys) <= set(available_keys) or use_positional`, and make
   KADID/TID metric attachment **positional-only** (row order == dmos.csv /
   mos_with_names.txt), as the fix script
   `fix_kadid_tid_apply_scores.py` already does. Delete the
   `groupby(ref_basename).mean()` codepath entirely — it can only produce a
   ref-broadcast.

2. **Carry a per-pair key (or row-provenance) end to end.** The deepest cause
   is that the 372-feat targets parquets dropped `codec`/`q`/`knob_tuple_json`
   /`dist_path`, leaving `ref_basename` as the only key. Any future feature
   table that will later be joined to a metric MUST retain a per-pair
   identifier (the full distorted path or the dmos row index). Without it, a
   correct join is impossible and the next agent will re-broadcast.

3. **Forbid silent mock columns in training corpora.** Rename the mock
   helper's output column `iwssim_MOCK_VAL_ONLY` (or carry a
   `iwssim_is_mock: bool` sidecar flag), and make `zensim_mlp_train` refuse a
   `--target-column iwssim*` when train_weight > 0 on any group whose iwssim
   is flagged mock. The leak survived three corpus generations *purely
   because the "mock" qualifier lived only in a filename that got renamed
   away.*

4. **Run `audit_metric_columns.py` in CI / pre-train.** The census takes
   seconds on the train parquets. Wire it as a pre-flight in every
   `run_*_seed.sh` that uses a `mix_*`/`iwssim`/`ssim2` target, aborting if
   any train-weighted group's column is flagged HUMAN-COPY or REF-MISJOIN.
   This is the only check that fires *before* compute is spent, regardless of
   how many build generations the corpus has been through.

The `_validate_metric_columns()` guard (#0) should stay — but #1 and #2 are
what make the bug *structurally impossible to reintroduce*, rather than merely
*detectable after the fact*.

---

## Provenance

Forensic root-cause + complete census: data-integrity forensic agent,
2026-05-25 (task #215). Build code recovered from committed worktrees
(`zensim--v24-alpha`, main `scripts/v_next/`); no build script lives in R2
(only the clean per-chunk score sidecars under `s3://zentrain/ssim2-backfill-2026-05-18/`).
Census reproducible via `scripts/canonical_corpus/audit_metric_columns.py`.
