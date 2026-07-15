# Duplication audit — 2026-07-15

**User directive:** *"rust not python for loading, collapse to/fix rust-native
code for all tasks"*, *"I want duplication killed off and prohibited in both
projects claude.md, ensure key bakes are reproducible with the rust paths and
update rust when needed to do so"*, *"zenstats crate should be the home for
statistical math"*.

Rule of record: **NO DUPLICATE IMPLEMENTATIONS** in `CLAUDE.md`.

---

## 1. The count

`scripts/v_next/` held **134 Python scripts**. Classified against the canonical
Rust owners:

| Bucket | n | Meaning |
|---|--:|---|
| **A — duplicates a Rust owner** | 34 | Re-implements loading / stats / training / bake-bytes |
| **B — already deprecated** | 7 | Header says superseded; several proven byte-identical |
| **C — legitimately unique** | 89 | Dashboards, corpus builders, log parsers, orchestrators |
| **D — unclear / real gap** | 4 | No Rust owner exists for the specific thing |

**89 of 134 are fine.** Python is not the problem; duplication is. The 89 are
plotting, corpus joins, log aggregation, and orchestrators that shell out to
the Rust binaries — all tasks where Python IS the owner.

Cross-cutting counts (a file can hit several):

- **30** hand-roll IQA stats — *the thing an existing CLAUDE.md rule already banned*
- **69** load parquet in Python
- **11** run parallel torch trainers
- **33** edit bake bytes after that work was migrated to `bake_dial_refit`

## 2. Why the existing rule failed

The rule said:

> Do NOT hand-roll srocc/plcc/krocc/pwrc/z_rmse in Python — that re-creates the
> 14-fork divergence this consolidates.

It named a **symptom** (a specific stat, in a specific language). The
**principle** — one owner per task — was left implicit, so every new script
re-derived the forbidden thing under a slightly different name, and nothing in
the rule's wording covered a second *Rust* site. zensim currently carries ~10
private Rust copies of `spearman` across probe binaries plus a separate impl in
`zensim-train-core/src/stats.rs`.

**Extraction is not migration.** The 2026-05-26 consolidation *succeeded*
architecturally: `zenstats` shipped, both siblings consume it, the parity gate
passes at ~5e-11 (two orders under its 1e-9 bar). It still failed behaviorally,
because the old call sites were never migrated and new ones kept appearing. In
zenanalyze, `load_features_raw` adoption went from 7-of-25 to ~15-of-35 — the
lib exists, the forks kept coming. Landing the owner is half the job; deleting
the callers is the other half, and it is the half that gets skipped.

## 3. What duplication actually cost — measured

**It reports quietly wrong numbers for months.** `bake_verdict` had its own
inline copy of every stat. When `panel.rs`'s OR + PWRC were rewritten to the
paper-correct ITU-T P.1401 / Mohammadi SA-ST forms (`83e7ff70`), the copy was
not. Every `bake_verdict` output before that fix reported the wrong OR + PWRC
while the `panel` binary reported correct ones **on the same fixture**. Nothing
crashed. The same shape showed up three more times in the 2026-05-26 audit:
PWRC argument order off by ~0.2, an OR definition off by 0.375, and one script
whose "pwrc" was Spearman-as-Pearson and not PWRC at all.

**It hides capability gaps — the least obvious cost, and the largest.**
`blend_lib.py` grew a within-ref RankNet term in Python because nobody checked
whether the Rust trainer could do it. It could not: `zensim_mlp_train` drew
every pair uniformly across a group (cross-image). That gap sat invisible
behind a working Python script. Extending the owner surfaced it in an hour and
fixed it for every future recipe. **The duplicate did not just cost
maintenance — it cost knowledge.**

**It re-pays solved problems.** `blend_lib._load` OOM'd on a 5.3 GB parquet
(one `read_table`, ~2× peak; errno 12 at a 40G cap with only 9.5 GiB RSS,
because the box had 45 GiB in page cache and 4.7 GiB free).
`parquet_loader.rs` had never had that bug.

**Stale "missing tool" claims are worse than no claim.** CLAUDE.md said
*"Missing v3 equivalent for the affine op — build a v3 affine tool when
needed."* False: `zensim-validate/src/bin/affine_calibrate.rs` has handled
v2/v3 since 2026-06-18. The claim told the next session to rebuild a shipping
tool, and it excused `affine_calibrate_bake.py` as filling a gap that was never
open.

## 4. Gaps closed this session (Rust extended, not worked around)

| Gap | Fix | Proof |
|---|---|---|
| Trainer could only draw cross-image RankNet pairs | `:withinref` group flag + `RefBuckets` | Existing recipes byte-identical (md5 `346c5a6d…` from binaries built at the parent commit and at HEAD, same recipe) |
| Loader rejected `feat_`-prefixed parquets | Both prefixes accepted | 3 loader tests on the real corpora |
| Loader dropped reference identity | `ref_ids` from `ref_basename`/`image_path` | 200 refs read from the HF corpus |
| **Per-ref SROCC had no Rust owner** — `--per-pair-output` existed so an external script could reduce it | `zenstats::panel::per_group_srocc`, surfaced as a first-class `bake_verdict` column | Independently reproduces the documented AIC-3 confound: **0.7774 pooled / +0.9236 per-ref** vs the recorded "0.79 / 0.93" |
| HF corpus was a raw parquet + sidecar TSV needing a hand join | `canonical-2026-07-15/train/hf_nearlossless*.parquet` via `join_safety` | Rust trainer loads it directly |

`per_group_srocc` landed in **zenstats** (`zenmetrics@0ce69492`), not in the
caller that wanted it — statistical math has one home. zensim's pin was bumped
to it; it builds from the pinned rev with no local patch.

That work also caught a real bug **in its own first implementation**: it
dropped degenerate groups by filtering `is_finite()`, but `spearman` returns
**0.0**, not NaN, on a vanishing denominator. Constant-target/constant-score
groups would have been averaged in as 0.0 — reporting a ranking failure where
there was nothing to rank. The test now guards the premise itself.

## 5. Deleted — and a correction worth recording

**Deleted (6):** `bhdr_bottom_extend.py`, `dense_dial_refit_b.py`,
`winsorize_bake.py` (all three proven byte-identical to their
`bake_dial_refit` replacements first), `w11_webp_ood_refit_2026-07-05.py` (a
falsified campaign), `verify_bake_srocc.py` and `yj_at_l0_per_block.py` (their
target binaries have no source anywhere).

CLAUDE.md's deprecated list also claimed `affine_calibrate_znpr_v2.py`,
`score_unified_with_bake.py`, and `soft_iso_smooth.py` were "deprecated but
present" long after they had been deleted.

**A wrong deletion, caught and reverted.** I first deleted **25** scripts on
the grounds that they shelled to binaries in deleted sibling worktrees
(`zensim--cross-codec-metric`, `--v10`, `--eval-accel`, ...). That reasoning
was wrong: **a worktree is a copy of this repo**, so every one of those
binaries — `ensemble_score_rows`, `predict_features_with_bake`, `bake_verdict`
— exists right here in `zensim-validate/src/bin/`. The scripts were not dead,
they were *mis-pointed*. The fix was `zensim--whatever` → `zensim`: one sed, 25
scripts recovered, zero deletions.

They were restored from the working copy (uncommitted, so `jj restore`
recovered all 25) and repointed. Of the original 25, exactly **2** were
genuinely dead — the two above, whose targets have no source anywhere.

The lesson generalizes past this incident: **"the binary it calls is missing"
does not mean "the script is dead."** It usually means the path is stale or the
artifact is unbuilt. `scripts/lint_scripts.py` now encodes the distinction —
a missing artifact whose source still exists is a `cargo build`, not a fossil,
and failing that case would train people to ignore the linter.

## 5b. The fossil mechanism (root cause)

The dead references were not carelessness; they were structural:

1. an agent opens sibling worktree `zensim--foo`
2. it writes a script hardcoding `/home/lilith/work/zen/zensim--foo/target/release/bar`
3. the worktree is cleaned up — **correctly**, the cleanup rule is mandatory
4. the script remains, permanently dead, and nothing notices

CLAUDE.md's worktree-cleanup rule covers the worktree. It never covered the
scripts pointing *into* it. That gap is now closed by a rule ("never hardcode a
sibling-worktree path in a committed script") plus `just lint-scripts`.

The same audit found `metric_compare_report.py` had not **parsed** since commit
`731cf0eb` ("utf-8 charset on all 293 report pages") bulk-inserted an
unescaped `<meta charset="utf-8">` into a Python string literal — and then a
correctly-escaped one right after it. Broken for weeks, unnoticed, because
nothing ever asked whether these scripts still run. Now fixed, and
`just lint-scripts` asks.

**Net: 201 scripts, all runnable, verified by a check that runs in CI.**

## 6. Still open

- **`pack_and_calibrate.py` is a doc conflict, not a clean delete.** It is a
  duplicate on the facts (own `fit_spline_knots` + `PchipInterpolator`, own
  zerobias/dtype repack, scipy `spearmanr`) but CLAUDE.md **mandates** it as
  the standard non-QAT packing fallback. Either `bake_dial_refit` gains a
  `pack-and-calibrate` subcommand or CLAUDE.md stops blessing it. Highest-value
  single item remaining.
- **The torch trainer cluster** (~10 files) is the largest. `blend_lib.py` is
  the keystone — the only file reimplementing the *entire* panel including
  per-band — and `bandwise_dashboard.py` imports it, so it can't be deleted
  until the dashboard is ported to `bake_verdict` output.
- **`linear_projections_2026-07-03.py`** is a load-bearing duplicate:
  `bake_mlp_negatives.py` imports its `fit_spline_knots`.
- **Blocked deletions:** `bake_outlier_gate.py` (← `xmetric_consensus.py`),
  `shared_anchor_refit.py` (← `hdr_anchor_dense_refit.py`).
- **~10 private Rust `spearman` copies** in probe binaries +
  `zensim-train-core/src/stats.rs` → route to zenstats.
- **Real gaps with no owner** (D bucket): `affine_per_sample_alpha.py` edits
  the `zentrain.per_sample_alpha_head` metadata payload, which no Rust
  subcommand handles; `strip_spline_metadata.py` needs a `bake_dial_refit
  strip`; `convert_features_bin.py` reimplements the ZSFC v3 `features.bin`
  reader. `bake_to_znpr.py` is likely dead (its trainer no longer exists).

## 7. Scope note — why zenanalyze and not zenmetrics

The companion rule goes in **zenanalyze**: it is the other trainer-owning repo
(4,347-line `zentrain/tools/train_hybrid.py`, `zenpicker_train.rs`, ~15 Python
torch trainers including a frozen v06→v10→v12→v14→v15 copy-the-last-one chain),
it has no anti-duplication rule of any kind, and both cross-repo items deferred
by the 2026-05-26 audit are still un-migrated seven weeks on
(`zensim_metric_train.py:467`, `correlation_cleanup.py:161`).

**zenmetrics is the supplier, not the patient.** It *owns* `zenstats`, which
both siblings consume (dep direction: zensim → zenmetrics, zenanalyze →
zenmetrics). It has 2 stat strays vs zensim's ~30, no MLP trainers at all, and
already enforces single-source rules — including one it imposed *on zenanalyze*
(`origin_split.py` hard-errors rather than allow a leaky fallback). That
hard-error is the enforcement pattern worth copying, not replacing.

## 8. The one exception: gated mirrors

A second implementation is legitimate when it exists for a measured engineering
reason **and** a test holds it bit-exact against the owner:

- `zenpicker-train/src/picker_eval.rs` — `pwrc_sa_st_auc_lowmem` (O(n²)→O(1)
  memory), gated by `pwrc_lowmem_matches_canonical_exactly`.
- `zensim-validate/src/panel.rs` — `compute_light_panel_subsampled`, which
  fixed a 307 GB OOM.

Without that test it is not a mirror; it is a fork with a good story.

---

Prior art: `benchmarks/iqa_stats_consolidation_2026-05-26.md`,
`benchmarks/cross_repo_duplication_audit_2026-05-26.md`.
