# Cross-repo dedup — VERIFIED synthesis (Phase 2)

**Date:** 2026-05-26
**Status:** SUPERSEDES the two first-pass grep-and-count docs
(`cross_repo_duplication_audit_2026-05-26.md`,
`dedup_inventory_master_2026-05-26.md`,
`sweep_training_script_dedup_2026-05-26.md`) — those are downgraded to
first-pass candidates. This doc is built from **6 deep-read per-file
ledgers** (`benchmarks/dedup_ledgers/*.md`, each file read in full or
read-by-structure with line-range evidence, role-characterized) PLUS
parent spot-verification of the headline claims.

## Why this supersedes the earlier docs

The grep-and-count pass mischaracterized severity and remedy because it
never read what files ARE. Confirmed errors it made:

| Shallow claim | Verified reality (ledger + read) |
|---|---|
| "zenmetrics orchestration = 2,454 LOC of duplicate shell" | `crates/zenfleet-vastai` is a **4957-LOC tested Rust binary** (13-test JSON parser, async tokio worker). The 11 bash "forks" are the **DEPRECATED** legacy chain it replaced (`onstart_v2/v3` marked deprecated); migration is in progress, not two live systems. |
| "2 GPU metric backends, no parity test" | **Every** zenmetrics GPU crate has a parity test (ssim2/butter/dssim/zensim/iwssim/cvvdp + a `cvvdp-conformance` gate); coefficient has `gpu_zensim_verify`. The only gap is no **cross-repo** check between CubeCL and cudarse — narrow, not "untested." |
| "3 broadcast joins in coefficient" | All 3 (`optimal_tree.py:49`, `fit_selector_model.py:53`, `feature_utility.py:327`) are **keyed inner `pd.merge`**, not broadcasts. Lower risk: they just bypass `join_safety`'s cardinality guard. |
| "55 R2 blocks across codec sweeps" | Codec-side R2 blocks = **24, ALL in jxl-encoder**; zero in zenwebp/zenjpeg/zenavif. The 55 conflated zensim+zenmetrics+jxl. |
| codec repos "5256 / 605 rs" duplication | Inflated by `.claude/worktrees/` noise + vendored `third_party` (jpegli-cpp). Real hand-written codec sweep glue is small. |
| trainers lumped as "scripts" | `mlp_train/mod.rs` (9897 LOC, ~30 tests), `train_hybrid.py` (3310 LOC), zenfleet-vastai — all correctly `lib-api`. |

**Lesson:** grep-and-count cannot distinguish a tested crate from a
shell pile, deprecated-mid-migration from live-fork, keyed-merge from
broadcast, or worktree-noise from real code. Every consolidation
decision needs the role-characterized read.

## The verified inventory, ranked by (actionability × value × safety)

### Tier 1 — intra-repo, safe, high-value (do these first)

These are within a single repo, low-risk, and a shared home already
exists or is obvious. Verified by reading both sides.

| # | Cluster | Evidence | Single home | Effort |
|---|---|---|---|---|
| 1 | **zensim bins re-roll `score_row`/head-extraction** (4-7 bins: bake_verdict, ensemble_score_rows, preview_stats_demo, qsweep_eval, …, ~90-95% shared) | spot-verified | call the **public** `metric.rs:402 score_features_with_profile` | LOW — cleanest win |
| 2 | **spearman/pearson/rank reimplemented 7×** in zensim (ensemble_mix, mlp_train/utils, eval_bake_per_band, bake_verdict, profile_compat_report, main.rs, panel.rs) | spot-verified (7 > ledger's 4) | `panel.rs` is canonical; make the 6 others `use panel::*` | LOW |
| 3 | **35/38 parquet builders bypass `join_safety.py`** | task #220/#223 verified; the corruption surface | route builders through `join_safety.safe_merge` + CI grep-gate on bare `pd.merge(` | LOW — **highest correctness value** |
| 4 | **zensim recipe-driver fork families** — 9 `run_cross_codec_vN_seed.sh` (~55-70%), 7 `eval_cross_codec_vN.sh` (~50-60%, "Mirrors v5"), 5 `eval_vN_pjnd_check.py` (~80%) | zensim ledger, diff-measured | one parameterized recipe + per-experiment config (proven by `_picker_lib.py`) | MED |
| 5 | **zenanalyze metapicker copy-forward forks** — v10→v12→v14→v15 standalone copies, no shared base, 60-80% skeleton each; `classify_stem` byte-identical v14↔v15 | root-tools ledger, diff-verified | extract shared base; v14 canonical (4-codec), v15 only for PNG | MED |
| 6 | **zenanalyze ablation/probe forks** — 4 tools carry byte-identical `load_pareto`/`load_features` predating `_picker_lib.py`; 3 reimplement numpy forward-pass | zenanalyze ledger | finish the `_picker_lib.py` migration | LOW |

### Tier 2 — cross-repo, real, needs careful API design

| # | Cluster | Verified scope | Single home | Caveat |
|---|---|---|---|---|
| 7 | **IQA stats across repos** — zensim 7, zenanalyze ~9, coefficient + zenmetrics more. **But the stat SETS differ**: full Mohammadi panel (SROCC/PLCC/KROCC/PWRC/ZRMSE/DS-AUC) lives only in zensim `panel.rs`; the rest compute argmin/R²/spearman subsets. | a `zen-iqa-stats` crate + mirrored `zen_stats.py`, CI-cross-checked | highest *reach* but needs the API to cover both the full-panel and the picker-overhead subset use cases |
| 8 | **target-quality loop** — zenwebp `zensim_target.rs` **explicitly mirrors** zenjpeg `zq.rs` (verified, `:3`). zenavif `auto_tune.rs` is NOT a mirror (single-shot MLP, no loop). | `zentarget` crate over a pluggable `Scorer` trait | 2 of 3, not 3; adjustment mechanism genuinely differs (per-block vs global-q) so only the control skeleton consolidates |
| 9 | **CodecFamily enum order** — zenpicker canonical; coefficient is **internally inconsistent**: `constraints.rs:30` diverges (Avif/Webp swapped), `oracle_picker.rs:99` matches. | single `CodecFamily` in zenpicker; coefficient depends on it | silent bake-mislabel risk; small fix |
| 10 | **jxl `zenjxl-tuning-sweep` ≈ zenmetrics sweep infra** — both `onstart.sh` + tuning-runner `lib.rs` name zenmetrics as mirror source | the same bash→zenfleet-vastai migration zenmetrics is doing | jxl should adopt zenfleet-vastai, not fork its bash |
| 11 | **coefficient vastai ↔ zenmetrics zenfleet-vastai** — two Rust vast.ai orchestrations (coefficient also has GCP+DO that nothing else has) | shared `zen-vastai` crate; coefficient keeps GCP/DO on top | LIB-level, not script-layer |

### Tier 3 — already handled or NOT duplication (no action / finish-in-flight)

- **zenfleet-vastai bash→Rust migration** — already underway; action is "delete the 11 deprecated bash forks once migration completes," not "consolidate."
- **GPU metric parity** — already parity-tested in both repos. Only add a **cross-repo** CubeCL-vs-cudarse conformance check (1 test), not a rewrite.
- **coefficient feature extraction** — correctly *calls* `zenanalyze`; no kernel dup. No action.
- **R2 boilerplate** — real (~24 in jxl + the zensim/zenmetrics copies) but low-risk; a sourced `zen-r2.sh` / finishing zenfleet-vastai's R2 module absorbs it.
- **onstart_*.sh** — MUST be shell (boot-time); shared hydrate/verify boilerplate → `zen-fleet.sh`, low priority.

## The single highest-priority action (unchanged, now reconfirmed)

**`join_safety` adoption + CI grep-gate** (Tier-1 #3). It's the literal
recurrence surface of the kadid/tid corruption, the fix exists, and
35/38 builders still bypass it. Independent of every other item here.
The structural fixes (#20) already guarded the *canonical* builders;
the grep-gate catches the other 35 the moment anyone runs CI.

## Recommended sequence

1. **CI grep-gate on bare `pd.merge(`** + mandate `join_safety` in
   builders (Tier-1 #3) — correctness, cheap, independent.
2. **zensim `score_row` → `score_features_with_profile`** (Tier-1 #1)
   + **stats → `panel.rs`** (Tier-1 #2) — cleanest intra-repo wins.
3. **CodecFamily single source** (Tier-2 #9) — small, kills a
   silent-mislabel risk.
4. **Cross-repo CubeCL-vs-cudarse parity test** (Tier-3) — converts
   the one remaining GPU unknown into a known.
5. Recipe/metapicker fork consolidation (Tier-1 #4/#5), `zentarget`,
   `zen-iqa-stats`, `zen-vastai` — as capacity allows; these need
   API design, not just extraction.

## Ledger provenance

| Repo | Ledger | Files | Read-in-full |
|---|---|---|---|
| zensim | `dedup_ledgers/zensim_ledger_2026-05-26.md` (`eb987c09`) | 224 | ~155 (12 large by-structure) |
| zenanalyze/zentrain | `zenanalyze_ledger_2026-05-26.md` (`af5bf047`) | 81 | most; configs by header |
| zenanalyze/tools (root) | `zenanalyze_root_tools_ledger_2026-05-26.md` (`41b87ae4`) | 27 | all 27 |
| zenmetrics | `zenmetrics_ledger_2026-05-26.md` (`9749065`) | zenfleet-vastai 15 mods + cli 19 + sweep ~35 | core in full; GPU kernels skimmed |
| coefficient | `coefficient_ledger_2026-05-26.md` (`b345d38`) | cloud 9 + bins 6 + flagged py | core in full; ~130 RD examples clustered |
| codec slices | `codec_sweep_ledger_2026-05-26.md` (`cdbb5962`) | jxl/webp/avif/jpeg sweep glue | fleet core + target loops in full; ~50 jxl benches by header |
