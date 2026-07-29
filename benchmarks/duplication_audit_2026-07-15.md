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

**Net: 309 scripts, all runnable, gated by a `lint-scripts` CI job.**

(When first written this line said "verified by a check that runs in CI" while
`lint_scripts.py` existed only as a justfile target. It was not in CI. The claim
was false for as long as it took to notice — corrected by adding the job, not by
softening the sentence.)

## 5c. The fork chain, measured

A normalized-similarity scan (strip comments/docstrings, fold version/date/seed
tokens, then compare) over all scripts found 61 near-duplicate pairs. Three
families were collapsed: **2,905 lines -> 1,011, 17 files -> 4.**

| family | before | after |
|---|--:|--:|
| `eval_v{4,4b,6,7,8}_pjnd_check.py` | 845 | 221 |
| `eval_v{5,6,7,8}_multi_band_check.py` | 943 | 307 |
| `summarize_v4.py` + `summarize_v4b.py` | 612 | 317 |
| `eval_cross_codec_v{4,4b,5,6,7,8}.sh` | 505 | 166 |

Every number in every report was verified identical against every generation
first; v6/v7/v8's PJND reports reproduce byte-for-byte.

**The cost was never the line count.** Each `cp` + sed updated the glob and left
the prose, and the prose is what humans read:

| copy | claimed | did |
|---|---|---|
| `eval_v4_pjnd_check` | "Gate (**relaxed start**, per task brief)" | `<= 5.0` — the standard gate, never relaxed |
| `eval_v4b_pjnd_check` | report titled **V4** | globs `cc4v4b_*` |
| `eval_v8_pjnd_check` | docstring "Verifies **V6**", usage -> v6 dir | globs `cc4v8_*` |
| `summarize_v4b` | report titled **V4**, "Gates per **V4** ship criteria" | summarizes V4B |
| `eval_cross_codec_v4b.sh` | titles itself **V4**; prints "**V4** native", "each **V4** bake" | evaluates V4B |
| `eval_cross_codec_v7.sh` | prints "**V6**" in three phase banners | evaluates V7 |
| `eval_lr_retune.sh` | header lists a PJND phase 4 and a multi-band phase 5 | has **neither**; its phase 4 aggregates the lr grid |

Seven instances. Every `v4b` copy in the family mislabeled itself as `v4`.

**And a fork chain can move a gate.** `eval_v8_multi_band_check` passes a band
only on `cc_std_median <= 5 AND |achieved_mean - target| <= 5`; v5/v6/v7 pass on
the first alone. Both print "PASS". Comparing a v6 report to a v8 report meant
comparing different criteria under one label. v6-vs-v8 read as 0.98 "similar" —
the ship gate underneath them differed. That is why the normalized scan is the
right instrument and eyeballing a diff is not.

Two more shapes worth naming:

- **The copy-and-sed factory, automated.** `eval_cross_codec_v4.sh` and `v4b.sh`
  copied a prior driver to `/tmp` and sed'd it into the next generation at
  runtime. The file each sed synthesized already existed on disk — presumably
  created by running the factory once and committing the output, leaving
  generator and generated both shipping and free to drift. Both were wrapped in
  `if [ -x ]` over a source that had not existed for months, so **Phase 3
  silently skipped on every run**.
- **False completion in shell.** Every phase ended `|| echo "... failed"`, which
  swallows the error despite `set -e`, then printed "All eval phases complete"
  unconditionally.

## 5d. The index rots because nothing reads it

`scripts/v_next/README.md` — the thing that makes 89 scripts findable — offered
as live options: `vastai_iwssim/` (committed `5ccea813`, later deleted; its
cited deployment plan gone too), `ensemble_seeds.py`, `per_band_step5.py`
(deleted in `4d6715f9`), `score_unified_with_bake.py`, `soft_iso_smooth.py`,
`train_v_next_mlp.py` (deleted in `34f796f4`), and `affine_calibrate_znpr_v2.py`.

**This is the mirror image of §2.** Those two commits deleted the scripts and
left the index advertising them; the 2026-05-26 audit named the duplicates and
deleted nothing. Extraction without migration; deletion without de-indexing;
documentation without deletion. Each half of the job keeps getting done alone,
and each half alone is worth roughly zero — a stale index sends the next session
to rebuild a tool that exists, or hunt for one that never will.

`CYCLE_7_DSSIM_COTRAIN_PLAN.md` stated an "Expected outcome" and never recorded
the actual one, so for two months it read as *authorized, pending* work — while
CLAUDE.md recorded that same hypothesis as FALSIFIED (`4ed499e`, all 5 variants
regressed CID22 by 0.04–0.07, "don't retry"). Exactly how a session retries a
dead idea, and exactly what CLAUDE.md Step 10 exists to prevent.

`lint_scripts.py` now covers `.md` under `scripts/`. Its historical-record
exemption is opt-in and visible — a doc is skipped only if its header *declares*
itself falsified/superseded/historical. `HISTORY_v06_rebalance_falsified.md`
already did; the cycle-7 plan did not, so it got a status header rather than an
exemption, and `README.md` carries no marker, so it stays fully linted.

## 5e. Every linter gap was found by the linter being wrong

Four in a row, each caught by disbelieving a green result:

1. It read only `.py` and reported "all runnable" while three `.sh` pointed into
   deleted worktrees. **68 of 316** were dead — 4× the Python count.
2. `find_source` knew only `src/bin/<name>.rs`, so it cried wolf on
   `zenmetrics`, declared via `[[bin]] name`. A linter that cries wolf on a
   shipping binary gets muted, and then it protects nothing.
3. No DEAD-SCRIPT check, so consolidating the `.py` silently broke six `.sh`
   callers. Adding it immediately surfaced three **pre-existing** breaks.
4. No `.md` check — §5d.

**Self-inflicted, recorded:** the mass repoint sed (`zensim--*` -> `zensim`,
69 files) rewrote `lint_scripts.py`'s own docstring and deleted the cautionary
example it was illustrating — the same mechanism (a bulk sed that does not read
context) that left `metric_compare_report.py` unparseable for weeks, committed
by me an hour after documenting it.

The `zen-metrics` -> `zenmetrics` rename (13 scripts) is the same shape: the
binary is `[[bin]] name = "zenmetrics"`, no dash, per the no-dash-after-zen
rule. Nothing had called it successfully since the rename, and nothing said so.

## 5f. A guard nobody reads is not a guard

The Join-Safety Gate — the forcing function against recurrence of the 2026-05-25
kadid/tid metric corruption — had failed **100 of its last 100 runs**. Not
flaky, not infrastructure. One real violation:

```
scripts/v_next/knob_consistency_atscale.py:40
    df = rf.merge(f4, on="encoded_filename", how="inner")
```

`ac158352` shipped the gate on 2026-05-26. `001b3c54` **added** that script on
2026-07-11, six weeks later. So the gate fired on day one, correctly, and the
commit merged anyway — and every push since has been red.

This is the failure mode the rest of this document keeps circling, in its purest
form. §2: the lib shipped, the call sites never migrated. §5d: the scripts were
deleted, the index never updated. Here: the gate was built, and then ignored so
consistently that red became the normal colour — at which point the next
violation is free, because nobody can tell it apart from the standing one. A
guard's value is entirely in someone acting on it.

The join itself was sound: `encoded_filename` is the full per-pair key, and the
sidecar is 4,214,382 rows / 4,214,382 unique keys, so an inner merge cannot
ref-broadcast. It could have been allow-listed with `# joinsafety-ok`. It was
routed through `safe_metric_join` instead, which re-checks metric-side
uniqueness on every run — the difference between a guard and a promise, and the
reason not to re-implement that check inline.

Numerically identical, which mattered: this script produced `001b3c54`'s "B wins
decisively" conclusion, so a silent change would have silently moved a
benchmark. cvvdp 0.675→0.6749, butteraugli 0.582→0.5824, dssim 0.525→0.5246,
iwssim 0.485→0.4846, n=678,435 unchanged.

Gate is green for the first time in its recorded history.

**The generalization.** Four instruments in this repo were built correctly and
then left un-acted-on: `zenstats` (extracted, callers not migrated), the
2026-05-26 audit (duplicates named, none deleted), the two cleanup commits
(files deleted, index not updated), the join-safety gate (violation caught, red
ignored). Building the instrument is the satisfying half and it is worth nothing
alone. **The half that gets skipped is the half that closes the loop**, and it is
the whole job.

## 5g. A red gate hides the next failure — measured, four times over

§5f found the Join-Safety Gate red for 100 straight runs. The main CI workflow
was in the same state, and this section is what that cost.

At `69b53715` — before this session touched CI — **Clippy, Format, Lint scripts,
Coverage, WASM SIMD128 and every Test platform were failing simultaneously.**
When every light is red, a new breakage changes nothing observable, so four
independent defects landed and sat:

| # | defect | how long | why invisible |
|---|---|---|---|
| 1 | A→B deprecation (`d2953d92`) left **14 files** naming `ZensimProfile::A` without `#![allow(deprecated)]` | since 2026-07-12 | Clippy already red |
| 2 | Same commit put `A` behind a default-on feature, but the workspace pins `default-features = false`, so `A` **ceased to exist** in zensim-wasm-tests — which still names it | since 2026-07-12 | WASM already red |
| 3 | Same commit left two examples mapping the user-facing name `latest` to the **deprecated** profile, while `latest_preview()` returns `B` | since 2026-07-12 | not a build error at all — just wrong |
| 4 | A file literally named `"$LOG"` (an unexpanded shell redirect, `9a4e2272`) broke **both Windows runners at `git checkout`** | weeks | Tests already red; the log says only `invalid path`, which reads like a runner fault |

Defect 4 is the sharpest: CLAUDE.md requires `windows-11-arm` CI **with no
exceptions**, and a stray 2,786-byte file silently removed that platform —
plus `windows-latest` — from coverage entirely. Not "the tests failed there";
the tests never ran. Three of the four trace to one commit, and none of the
three were noticed by it.

**Two guards were also found to be prose, not guards** — the same shape as §5f
but worse, because both *claimed* to be enforced:

- `--mse-weight`'s reachability check sat **inside**
  `train_mlp_per_sample_alpha_head` testing `!per_sample_alpha_head` —
  unreachable there by construction. Its doc said "trainer panics if set on
  other heads"; in fact `--mse-weight` on any other head was silently
  discarded and the run trained pure rank. Dead code masquerading as a
  guarantee. Caught by a `should_panic` test that did not panic. (Audited: 0 of
  142 weight manifests affected — every one that sets `mse_weight` also sets
  `per_sample_alpha_head`, so the uselessness was latent.)
- `zensim-train-core/src/stats.rs` declared "both impls must be kept in
  lock-step" with zenstats and cited a verifier — `test_zen_stats_rust_python_parity.py`
  — that **never shipped**. The duplication was documented, declared safe, and
  had never once been checked. Now enforced by a real bit-identity test.

**The generalization, sharpened.** §5f said building the instrument is the half
that gets skipped. This adds the sequel: *an instrument left un-read decays into
an instrument that cannot be read.* A gate that is always red carries zero bits
— it cannot distinguish "broken" from "still broken," so the cost of the next
breakage drops to nothing and they accumulate. Four did. Restoring signal was
not a matter of fixing four bugs; it was a matter of getting back to a state
where a red light **means** something. Green is not a vanity metric — it is the
precondition for the gate having any value at all.

## 6. Still open

> **2026-07-29 status sweep** (commits `36fd508c..26ee566c`; details in
> `benchmarks/pack_rust_migration_2026-07-29.md` +
> `benchmarks/key_bake_repro_verification_2026-07-29.md`): **22 scripts
> retired in one day.** CLOSED since this audit: `pack_and_calibrate.py`
> (→ `bake_dial_refit pack`, byte-identical 3 ways incl. the shipped
> packed30k artifact); both "blocked deletions" (`bake_outlier_gate.py` →
> `gate` with `xmetric_consensus.py` migrated to the canonical Rust
> forward; `shared_anchor_refit.py` — its claimed importer was STALE,
> docstring-only); the ENTIRE D bucket (`strip_spline_metadata.py` →
> `strip` byte-identical ×2 fixtures; `affine_per_sample_alpha.py` — its
> only consumer was the concluded EXP-CROSS-CODEC-V3 harness, both
> deleted, no speculative Rust port for a zero-consumer capability;
> `convert_features_bin.py` + `bake_to_znpr.py` dead); plus 13 concluded
> one-shot stat scripts (g5 trio, cvvdp probes [falsified campaign],
> A_Phone pair, pre-QAT v47 pair, V0_2/3 pair, lr-retune pair).
> STILL OPEN: the torch-trainer cluster below (`blend_lib.py` keystone —
> unchanged), `linear_projections` (BHdr fit chain → task #68 with a
> concrete lasso-CD + npz-reader port plan), and the two Rust `spearman`
> stragglers in zensim-regress/zensim-experimental (the published-crate
> dep trade — still a trade to raise, not sneak in).

- **`pack_and_calibrate.py` — PARTLY closed.** Its private
  `fit_spline_knots` is gone: it was AST-identical to
  `linear_projections_2026-07-03.py`'s (proved by comparing parsed bodies, not
  by eye), which five other scripts already import as `lp.fit_spline_knots`. It
  now imports that one — 3 copies of the function down to 2 (1 Python + the
  port-verified Rust in `bake_dial_refit`). Notably `linear_projections`
  *documented* the duplication in its own docstring ("Same knot logic as
  pack_and_calibrate.py") and copied anyway.

  The FILE stays: it is not a duplicate of `bake_dial_refit` — it also does
  per-layer zerobias with `--protect-last` and enforces the pack-THEN-calibrate
  order, neither of which Rust has (`zenpredict repack`'s global `--zerobias`
  uses the wrong order and drops identity). Collapsing it needs a
  `bake_dial_refit pack` subcommand. Still the highest-value single item.
- **The torch trainer cluster** (~10 files) is the largest. `blend_lib.py` is
  the keystone — the only file reimplementing the *entire* panel including
  per-band — and `bandwise_dashboard.py` imports it, so it can't be deleted
  until the dashboard is ported to `bake_verdict` output.
- **`linear_projections_2026-07-03.py`** is a load-bearing duplicate:
  `bake_mlp_negatives.py` imports its `fit_spline_knots`.
- **Blocked deletions:** `bake_outlier_gate.py` (← `xmetric_consensus.py`),
  `shared_anchor_refit.py` (← `hdr_anchor_dense_refit.py`).
- **Private Rust `spearman` copies — CLOSED, and the count was wrong.** A
  `fn spearman` grep returns 14 hits, but the Dedup-K pass (2026-05-26) had
  already converted 5 to thin `zenstats` wrappers — the grep was matching
  *wrappers*, not duplicates. Classifying each first left 6 real copies. The 4
  in zensim-validate (embedding_distance / unconstrained_mlp /
  monotone_subspace / residual_identity probes) now call zenstats, verified
  value-identical over 400 tie-heavy cases at n = 3..10,000 (400/400
  bit-identical, max delta exactly 0) rather than argued from
  shift-invariance.

  `zensim-train-core/src/stats.rs` is deliberately NOT routed: it is a
  standalone WASM-targeted bit-exact core and zenstats is not WASM-vetted. That
  boundary is now *enforced* instead of asserted — see §5g; the lock-step rule
  it already claimed was backed by a verifier that never shipped.

  Remaining: `zensim-regress/examples/slice_real_codec_localization.rs` and
  `zensim-experimental/tests/feature_distortion_direction.rs`. Neither crate
  deps zenstats and zensim-regress is **published**, so adding a dep to its
  tree to dedup an example is a trade to raise, not to sneak in.
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
