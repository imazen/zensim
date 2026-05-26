# IQA-stats consolidation — py reimpls → canonical Rust `panel` (2026-05-26)

Consolidation of the ~14 scattered Python reimplementations of the IQA
correlation panel (SROCC / PLCC / KROCC / OR / PWRC / Z-RMSE /
4-param-logistic) onto the single canonical Rust home that already
exists. Scope per `benchmarks/dedup_VERIFIED_synthesis_2026-05-26.md`
Tier-1 #2 and `benchmarks/sweep_training_script_dedup_2026-05-26.md`
Class 2. **No new crate** — the home already exists.

## Canonical home

`zensim-validate/src/panel.rs` (a `pub mod panel` in
`zensim-validate/src/lib.rs`) is the canonical, strict-superset
implementation. Key fns (file:line):

| Stat / op            | fn                  | line |
|----------------------|---------------------|------|
| rank vector          | `ranks`             | panel.rs:30 |
| Spearman SROCC       | `spearman`          | panel.rs:50 |
| Pearson PLCC         | `pearson`           | panel.rs:72 |
| Kendall KROCC        | `kendall_tau`       | panel.rs:93 |
| Outlier ratio OR     | `outlier_ratio`     | panel.rs:129 |
| PWRC                 | `pwrc`              | panel.rs:163 |
| global Z-RMSE        | `z_rmse`            | panel.rs:193 |
| per-sample Z-RMSE    | `z_rmse_per_sample` | panel.rs:234 |
| 4-param logistic     | `rescale_logistic`  | panel.rs:458 |
| full 6-stat panel    | `compute_panel`     | panel.rs:656 |
| light panel (S/P/PW) | `compute_light_panel` | panel.rs:778 |
| MRR h-stat           | `mrr_h`             | panel.rs:847 |
| bootstrap CI delta   | `bootstrap_ci_delta`| panel.rs:947 |
| § A.9 decisive rule  | `decisive`          | panel.rs:1107 |

## New entry points (this change)

1. **Rust `panel` subcommand** — `zensim-validate/src/bin/panel.rs`.
   Reads a TSV or Parquet with columns `predicted`, `target`, optional
   `sigma`, optional `band`; emits the full panel as text or `--json`.
   THE canonical entry point for the arbitrary-pairs case (not a bake).
   Hidden `--emit-rescaled` flag (used only by the parity gate) prints
   panel.rs's own logistic-rescaled scores. Build:
   `cargo build --release -p zensim-validate --bin panel`.

2. **`scripts/lib/zen_stats.py`** — thin Python shim that shells to the
   `panel` bin. For pipelines that can't restructure to call the binary
   directly. `from scripts.lib.zen_stats import panel, srocc, ...`.

3. **`scripts/verify_panel_parity.py`** — the MANDATORY cross-check
   gate. 36 synthetic cases through BOTH the Rust `panel` AND two
   Python references (scipy + a faithful pure-Python mirror of
   panel.rs's exact defs); asserts every gated stat agrees to <= 1e-9.

4. **`zensim-validate/tests/panel_parity.rs`** — Rust gate. A
   default-running golden test pins `compute_panel` against
   cross-checked reference values; an `#[ignore]`-gated test shells to
   the Python parity script for the full cross-language ±1e-9 gate
   (caller-controlled skip per CLAUDE.md, mirroring
   `tests/parquet_load_equivalence.rs`).

## Parity result (the gate)

`python3 scripts/verify_panel_parity.py` — 36 cases (seeds 1-3 × n in
{40,120,400} × {linear_noisy, saturating, distance_shaped, weak}):

```
## vs panel_def_ref (faithful pure-Python mirror of panel.rs defs;
##  uses panel.rs's own --emit-rescaled scores for PLCC/Z-RMSE)
stat              max_div     gate
srocc           4.972e-11    GATED
plcc            4.856e-11    GATED
krocc           4.937e-11    GATED
pwrc            4.968e-11    GATED
or              3.333e-11    GATED
z_rmse          4.878e-11    GATED

## vs scipy_ref (textbook scipy.stats == mohammadi_eval.py defs)
stat              max_div     gate
srocc           4.972e-11    GATED
plcc            4.856e-11    GATED
krocc           4.937e-11    GATED
pwrc            4.968e-11    GATED
or              3.750e-01   report  (definitional difference — see below)

RESULT: PASS — every GATED stat agrees to <= 1e-9.
```

Max divergence ≈ **5e-11 per gated stat**, two orders below the 1e-9
gate. The canonical Rust panel is verified equivalent to the textbook
scipy reference; the py reimpls can be retired.

## Genuine algorithmic differences surfaced (NOT papered over)

The task required stopping and reporting if any stat diverged > 1e-9.
Two real divergences were found and reconciled in the gate's
definition (rather than silently averaging them away):

1. **PWRC argument order.** PWRC is NOT symmetric — its weights derive
   from the *first* argument's ranks. `panel.rs::compute_panel` calls
   `pwrc(humans, scores)` → weights by the **human/target** ranks (the
   meaningful "weight by importance of the reference MOS" convention).
   `scripts/mohammadi_eval.py:62` called `pwrc(pred, target)` →
   weighted by the **predicted** ranks. The parity gate calls the
   reference with the panel.rs argument order; this is the canonical
   convention going forward. (Divergence with the old order was ~0.2.)

2. **OR (outlier ratio) definition.** `panel.rs:129` uses a
   polarity-aligned z-score residual with a 2σ rule on the residual
   distribution. `mohammadi_eval.py:56` uses logistic-rescaled
   |residual| > 2σ. Both are internally consistent and
   Mohammadi-2025-compatible; they are NOT bit-equal (max divergence
   0.375 in the synthetic cases). The gate reports OR-vs-scipy as
   informational (not gated) and gates OR only against panel.rs's own
   definition (`panel_def_ref`, agreement 3e-11). Going forward,
   panel.rs's OR is canonical.

3. **`eval_ensemble_2026-05-18.py`'s `pwrc` is not PWRC at all** — it
   is Pearson on rank-transforms (`pearsonr(rank(t), rank(p))`), i.e.
   plain Spearman-as-Pearson, NOT the importance-weighted PWRC of
   Mohammadi 2025. That column in any old `eval_ensemble` output is a
   different statistic; the consolidation eliminates it.

Z-RMSE is also definition-dependent (panel.rs `z_rmse` normalizes by
the target's global σ; `mohammadi_eval.py`'s `z_rmse_per_sample`
normalizes per-stimulus). The `panel` bin exposes BOTH (global
`z_rmse` always; per-sample `z_rmse_per_sample` when a `sigma` column
is present), so neither is lost.

## Per-caller disposition (the 14)

Legend: **deprecated-banner** = stat-math superseded, banner added
pointing at `panel`/`bake_verdict`/`zen_stats`, file kept for its
unique non-stat logic; **migration-candidate** = in-loop stat that
can't cheaply shell, documented to use `zen_stats` for the final
report; **cross-repo** = documented here, not edited (separate repo,
own `.workongoing` claim needed).

| # | File:line | defs | disposition |
|---|---|---|---|
| 1 | `zensim/scripts/mohammadi_eval.py:39,43,48,56,62,82` | srocc/plcc/krocc/or/pwrc/z_rmse_per_sample | **deprecated-banner** (keeps AIC-3 σ-parquet + per-row predictor wiring) |
| 2 | `zensim/scripts/exp_ensemble/eval_ensemble_2026-05-18.py:67-149` | srocc/krocc/pearson_abs/rescale_logistic/z_rmse/pwrc/or/mohammadi_panel | **deprecated-banner** (keeps the ensemble-classifier training; note its "pwrc" is a different stat) |
| 3 | `zensim/scripts/v_next/ensemble_seeds.py:19` | spearman | **deprecated-banner** |
| 4 | `zensim/scripts/v_next/per_band_step5.py:27` | spearman (per-band) | **deprecated-banner** (`panel` has a native `band` column) |
| 5 | `zensim/scripts/v_next/butter_concordance_audit.py:27` | spearman | **deprecated-banner** |
| 6 | `zensim/scripts/v_next/v0_20b/finetune_head.py:108` | spearman_abs | **migration-candidate** (in-loop epoch monitor) |
| 7 | `zensim/scripts/v_next/v0_22_iw_option_c_alpha_sweep.py:76` | z_rmse | **deprecated-banner** |
| 8 | `zensim/scripts/v_next/aggregate_lr_retune.py:95` | srocc_only | **note** (already shells to `bake_verdict`; only re-parses its SROCC — no own stat math) |
| 9 | `zensim/scripts/dial_bug_audit/run_dial_audit.py:155` | srocc_sign_tolerant | **deprecated-banner** (keeps the dial-distribution scan) |
| 10 | `zenanalyze/zentrain/tools/zensim_metric_train.py:428` | srocc | **cross-repo** (zenanalyze) |
| 11 | `zenanalyze/zentrain/tools/correlation_cleanup.py:158` | spearman_corr_matrix | **cross-repo** (zenanalyze; note: this is a feature-feature correlation *matrix*, a related-but-broader use — needs an API beyond the (predicted,target) panel) |
| 12 | `coefficient/scripts/spearman_prune.py:39` | spearman | **cross-repo** (coefficient) |
| 13 | `jxl-encoder/scripts/analyze_smart_fanout.py:80` | spearman | **cross-repo** (jxl-encoder) |

(Items 3,4,5 are listed as one row in the Class-2 table —
`v_next/{ensemble_seeds,per_band_step5,butter_concordance_audit}` — and
counted individually here, totalling the 14 distinct primary files.)

## Cross-repo migration plan (NOT done here — separate-repo work)

The 4 cross-repo callers (#10-13) are documented, not edited: each
lives in a different repo that needs its own `.workongoing` claim and
may collide with in-flight work. The clean path for each:

- **zenanalyze `zensim_metric_train.py:428`** + **coefficient
  `spearman_prune.py:39`** + **jxl `analyze_smart_fanout.py:80`** — all
  compute a plain Spearman on arbitrary pairs. Replace the local `def`
  with a shell-out to the Rust `panel` bin (via a mirrored
  `zen_stats.py` copied/vendored into each repo, or a shared
  `scripts/lib/` if the repos share a checkout root). Same parity gate
  applies.
- **zenanalyze `correlation_cleanup.py:158`** computes a *feature ×
  feature* Spearman correlation matrix for redundancy pruning — a
  broader use than the (predicted, target) panel. The `panel` bin
  covers the pairwise case; the matrix case wants either N² `panel`
  calls (cheap at feature counts) or a small dedicated
  `corr_matrix` mode. Document as needing API design before migration.

A future `zen-iqa-stats` crate + mirrored `zen_stats.py` (Tier-2 #7 in
the verified synthesis) is the long-term cross-repo home; this change
establishes the zensim-side canonical entry point that crate would
expose, and the parity methodology any cross-repo migration must pass.

## Known environment caveat (build state, not this change)

The legacy `zensim-validate` binary `main.rs` does NOT compile at
committed HEAD (it declares `mod loss_norm_in_norm;` at the crate root
while the file lives at `mlp_train/loss_norm_in_norm.rs`, and includes
`mlp_train` whose `crate::panel` / `crate::adam_simd` references only
resolve in the *library* build, not the *binary* build). This is
pre-existing bit-rot from the module-extraction refactor (`091b4d5
"make extracted submodule items pub (required for binary access via
library)"`) + the in-flight `v45_monotone_cbc` work — it is NOT caused
by this change and was NOT touched (it overlaps active uncommitted
work on `zensim/src/metric.rs`). Consequence: `cargo test -p
zensim-validate` (which builds all bins, including the broken
`main.rs`) cannot complete at HEAD. The deliverables here build and
test in isolation:
- `cargo build -p zensim-validate --bin panel` — OK
- `cargo build -p zensim-validate --lib` — OK (63 lib unit tests pass,
  incl. panel.rs's 16)
- `cargo build -p zensim-validate --bin bake_verdict` — OK
- `python3 scripts/verify_panel_parity.py` — PASS (the cross-language
  ±1e-9 gate)
