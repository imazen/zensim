# join_safety adoption + CI grep-gate — migration evidence (2026-05-26)

Chunk L of cross-repo dedup initiative. Routes the historically-corruption-prone
metric-join builders through `scripts/canonical_corpus/join_safety.py`
and lands a CI grep-gate so new bare-`pd.merge` calls cannot creep back in
without explicit per-line opt-out.

## Background

2026-05-25 — kadid/tid metric corruption shipped because a builder did a
ref-only `pd.merge` that broadcast-joined a per-pair metric onto a per-source
features table. `join_safety.py` (added post-incident, 18 self-tests) encodes
both that failure mode and the mock/human-copy leak as hard `assert` failures.

VERIFIED synthesis (2026-05-26, `dedup_VERIFIED_synthesis_2026-05-26.md`):
this is the single highest-leverage correctness action. 35 of 36 builders
were bypassing the wrapper.

## Migrations landed (5 of 6 listed, all with measurable surface)

### 1. `zensim/scripts/v_next/build_unified_parquet.py`

Two unguarded `.merge` sites:

| Line (before) | Shape | Migration |
|---|---|---|
| `145: tsv.merge(feat, on=key, …)` | Per-pair full-key merge | Added explicit `feat.duplicated(subset=key)` uniqueness assert + `# joinsafety-ok` allow-list. |
| `195: df.merge(corpus_features, on="image_basename", how="left")` | Per-source 1-to-many attach (legitimate broadcast) | Routed through new `attach_per_source_features(...)` helper + post-attach `guard_metric_table(...)`. |

The line-195 site was the exact Mode-B shape; `attach_per_source_features`
refuses if the source side has duplicate rows on the ref key (silent
broadcast corruption mode).

Import verification:

```
$ python3 -c "from join_safety import attach_per_source_features, guard_metric_table, safe_metric_join; print('OK')"
OK
```

### 2. `zensim/scripts/v_next/v11_ssim2_v2/build_v11_substrate_v2.py`

One unguarded `.merge` site at `98: omni.merge(feat, on=OMNI_KEY_COLS)`.

| Migration step | Where |
|---|---|
| `feat.duplicated(subset=OMNI_KEY_COLS)` assert | `join_chunk` before the merge |
| `omni.duplicated(subset=OMNI_KEY_COLS)` assert | `join_chunk` before the merge |
| `# joinsafety-ok` allow-list comment | on the merge line |
| `guard_metric_table(...)` post-write | `build_unified_omni` before `pq.write_table(...)` |

### 3. `zenanalyze/zentrain/tools/zensim_metric_train.py`

One unguarded `.merge` site at `222: df.merge(sub, on="ref_basename")`.

Replaced with `attach_per_source_features(df, sub, "ref_basename", how="left")`
+ post-attach `guard_metric_table(...)`. Both raise on the documented
corruption modes (mock leak / human-copy leak / duplicate sidecar broadcast).

Cross-repo import via `sys.path.insert(0, "/home/lilith/work/zen/zensim/scripts/canonical_corpus")`.

### 4-5. `zenmetrics/scripts/sweep/build_per_codec_training{,_extended}.py`

DuckDB SQL joins with proper per-pair-key + explicit dedup (already
correct on the join shape). Added pre-write `guard_metric_table(...)`
call before each `pq.write_table(...)` so a future schema/source change
that re-introduces a mock or constant-per-ref metric column fails loud.

Soft cross-repo import (try/except) so the script still runs if the
zensim repo isn't present on a fleet worker; warns on stderr but does
not crash.

### 6. Anchor-builder family (`build_v8/v9/v10_anchor_parquet.py`)

**Audit miscategorisation, not actually migratable.** `grep` confirmed
no `.merge`/`.join`/`pd.concat`/`DuckDB JOIN` in these files; they
write anchor selections out of in-memory dicts, not joins. No
migration target.

## New helpers added to `join_safety.py`

| Helper | When to use |
|---|---|
| `attach_per_source_features(target, source, source_key, *, how="left", suffixes=("", "_src"))` | Legitimate 1-to-many per-source attach (image_basename/ref_basename → many distortions). Refuses on source-side duplicates. |
| `guard_metric_table(label, table, *, source_key=None)` | Post-join wrapper. Runs Mode-A check (mock + human-copy leak) on any pyarrow.Table OR pandas.DataFrame. With `source_key`, also runs Mode-B constant-per-ref check on every ssim2/cvvdp/butter/dssim/iwssim column. |

Test coverage: 6 new tests in `test_join_safety.py` (`test_attach_per_source_features_*` + `test_guard_metric_table_*`), total now 18 passing.

## CI grep-gate

`.github/workflows/joinsafety.yml` runs on every push to `main` + every
PR. Steps:

1. `scripts/canonical_corpus/joinsafety_gate.py` — scans `scripts/`,
   strips per-line `# comment` before pattern match, flags any
   `pd.merge(` / `.merge(` outside `join_safety.py` / `test_join_safety.py`
   / `joinsafety_gate.py` without a `# joinsafety-ok: <reason>` allow.
2. `python3 scripts/canonical_corpus/test_join_safety.py` — 18 invariants.

Local PASS:

```
$ python3 scripts/canonical_corpus/joinsafety_gate.py
PASS: scanned scripts/ — no unguarded pd.merge / .merge( found.

$ python3 scripts/canonical_corpus/test_join_safety.py
... ALL 18 join_safety tests passed.
```

## Packaging decision: option (a), minimal `setup.py` shim

Added `setup.py` + `zen_corpus_join.py` thin re-export module in
`scripts/canonical_corpus/`. Cross-repo consumers can either:

- `pip install -e /home/lilith/work/zen/zensim/scripts/canonical_corpus`
  then `from zen_corpus_join import safe_metric_join, ...`
- OR add `sys.path.insert(0, "/home/lilith/work/zen/zensim/scripts/canonical_corpus")`
  then `from join_safety import ...` (the pattern used by in-tree
  builders and by the zenanalyze/zenmetrics migrations above).

The shim is a thin re-export of `join_safety` — DO NOT add helpers to
it. Helpers go in `join_safety.py` so both import patterns see them.

## What is NOT covered

- ~30 eval-side join scripts (`eval_v*_pjnd_check.py`, `cross_codec_*`,
  `picker_agreement.py`, etc.) — these read parquet + merge for
  read-only verdict/PLCC analysis. Lower blast radius (their output
  doesn't feed a downstream training corpus). The CI grep-gate WILL
  flag them on next change, so they get migrated opportunistically.
  See follow-on chunk size at the end of this evidence.

- `audit_metric_columns.py` / `audit_mix_columns.py` — audit-only
  tools that DON'T write parquet outputs. Could be migrated to
  reduce noise from the CI gate, but the audit-tool category was
  documented as LOW risk in the cross-repo audit.

## Follow-on chunk size

A follow-on chunk migrating the remaining ~30 eval-side joins (mostly
single-line `df.merge(sub, on="ref_basename")` patterns in
`scripts/v_next/eval_v*_pjnd_check.py` family + `cross_codec_*` +
`zenanalyze/zentrain/tools/{train_hybrid.py,refresh_features.py}` +
`zenanalyze/tools/v14_metapicker_train.py` family) is ~3-5 hours of
mechanical rewrite, blocked only by deciding per-script whether the
attach is per-pair (safe_metric_join) or per-source (attach_per_source_features).

The CI gate will surface each one on its next CI run; this is the
intended forcing function. Migration can be opportunistic from there.
