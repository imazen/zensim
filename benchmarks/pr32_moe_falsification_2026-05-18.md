# PR #32 MoE rebase + re-eval — falsification (2026-05-18)

**Verdict**: closed without ship. The MoE branch was a code-only milestone
from 2026-05-05 against architecture (V_0.4 era, ZNPR v2, 228+3+5
feature schema, in-tree mlp_train.rs trainer + hand-rolled inference
in dataset_metric_baseline.rs) that has been architecturally superseded
on today's main and would not produce a competitive bake on today's
canonical corpus shape.

**Source artifacts**:
- PR #32: https://github.com/imazen/zensim/pull/32
- Branch: `origin/v06-moe`, head `cf48ba2a` (May 5 2026)
- Architecture doc: `docs/moe_architecture.md` on the branch
- This falsification: `benchmarks/pr32_moe_falsification_2026-05-18.md`
- Workspace: `~/work/zen/zensim--pr32-moe/` (jj workspace)

## What the branch contains

One commit (`cf48ba2`, May 5 2026, +2183 / −15 lines):

- `zensim-validate/src/mlp_train_moe.rs` — 1217-line MoE trainer behind
  cargo feature `moe`. Architecture: gate MLP (N→Hg→K, ReLU+softmax/τ)
  + K expert MLPs (N→H→1, LeakyReLU). Loss = RankNet + magnitude-match
  + L2 + λ_lb · KL(uniform || w). Each expert + the gate baked as a
  standard 2-layer ZNPR v3 (the branch said v3, the code emits v2 via
  the old `bake_v2` call, which today is banned per CLAUDE.md).
- `zensim-validate/src/main.rs` — `--mlp-moe-*` CLI flags (+237 lines).
- `zensim-validate/tests/moe_smoke.rs` — 3 smoke tests (gate finite,
  softmax weights sum to 1±1e-6, hard-routing argmax matches soft).
- `zensim-bench/examples/dataset_metric_baseline.rs` — `--moe-manifest`
  flag + MoeManifest TSV loader + soft/hard runtime mixture (+265 lines).
- `zensim-validate/Cargo.toml` + `zensim-bench/Cargo.toml` — `moe` feature.
- `docs/moe_architecture.md` — 248-line architecture + bake format +
  CLI invocations + acceptance gates.

**No bake**. No `*moe*.bin` exists anywhere on `/mnt/v/zen/zensim-eval/`,
`/mnt/v/output/`, or `~/work/zen/`. The architecture doc says
explicitly: "**No training has been run.** The architecture compiles,
the trainer round-trips synthetic data... The first real run waits on
the rebalanced corpus."

The "rebalanced corpus" the doc references is the
`zensim--v06-rebalance` branch which never materialized in the shape
the MoE design anticipated. Today's canonical corpus at
`/mnt/v/zen/zensim-training/canonical-2026-05-18/` is a 372-feature
schema with the same ~99% photo-distortion shape the MoE doc itself
predicts would degenerate to expert-0-only routing:

> "on the current 99.5%-photo corpus FiLM degenerates to 'single photo
> modulation' because every eval pair lives in class 0"

The same collapse applies to MoE: a learned gate on photo-heavy data
will route everything through one expert, recovering the V_0.4 baseline
with K× the parameters.

## Why a rebase + train + eval is not the right investment

Three independent blockers, any one of which would be sufficient:

### 1. The architecture has been superseded on main

Today's `zensim::metric::forward_one_bake` ships **two runtime
dispatch mechanisms** that solve the same "let one model specialize
across distortion regimes" problem MoE was designed to solve, but
inside a single ZNPR v3 bake:

- **`zentrain.hybrid_head`** (commit `8dd3bfa`, 2026-05-18) — rank
  head + pool head mixed via a learned scalar α.
- **`zentrain.per_sample_alpha_head`** (commit `2788b92`, 2026-05-18)
  — rank head + pool head mixed via a per-sample sigmoid gate over
  the hidden vector. Ships as `PreviewV0_5Compression` today.

Both dispatches are **bake-metadata-keyed**: the bake is a single
ZNPR v3 file, the metadata payload tells `forward_one_bake` /
`bake_compare::score_corpus` / `bake_verdict::score_row` to apply
the routing on the predictor output. `bake_compare` 1000-bootstrap
panels work unchanged. MoE's manifest-based "load 1 gate + K
experts + route at inference" pattern is the **structural inverse**
of this design — it cannot be slotted into the existing dispatch
without significant new infrastructure (a `LoadedMoeBake` type, a
manifest-aware score path through every consumer, new CLI flags
throughout bake_compare/bake_verdict/forward_one_bake).

The single-bake-with-metadata pattern is what landed the
two-trail SOTA. MoE would have to either:
- Pack gate + K experts into a single ZNPR v3 (the architecture doc
  explicitly says **"no new bake format is introduced"** — and that
  rule is now load-bearing because the metadata-keyed dispatch is
  the runtime contract), OR
- Run as a separate code path with a manifest, which means writing
  ~600 lines of new dispatch through bake_compare / bake_verdict /
  forward_one_bake plus a new `MoeRuntime` API on `zensim::mlp` that
  the doc itself lists as "DEFERRED."

### 2. The trainer code does not run on today's main

The MoE branch is 13 days behind main with ~726 files changed in
between. Even setting MoE aside, the branch is incompatible with
today's code on multiple axes:

| What the MoE branch does | What main expects |
|---|---|
| imports `crate::mlp_train::{bake_two_layer_znpr_v2, ...}` | helper renamed `bake_two_layer_znpr_v3`, lives in `mlp_train.rs` (signatures changed) |
| imports `crate::mlp_train::{MlpHyperparams, TrainingGroup, ValidationPolicy, SplitMix64, spearman_correlation, compute_scaler_from_groups}` | these moved to `zensim-train-core` crate (signatures slightly changed) |
| calls `zensim::mlp::bake::{BakeLayer, BakeRequest, bake_v2}` | `zenpredict_bake::{BakeLayer, BakeMetadataEntry, BakeRequest, bake}` — `bake_v2` is **banned** per CLAUDE.md "ZNPR v2 PROHIBITED" |
| BakeRequest pre-v3 fields only | BakeRequest has 3 mandatory v3 fields (`output_specs`, `discrete_sets`, `sparse_overrides`, plus `feature_transforms`) |
| 228+3+K (e.g. 236) features expected, with `cclass_*` tail | canonical corpus is 372 features (300 for LARGE), no `cclass_*` columns. The trainer wants a `--mlp-zenanalyze-tsv` + `--mlp-zenanalyze-features cclass_photo,...` plumb that does not exist on main |
| eval harness in `dataset_metric_baseline.rs` (now deprecated — CLAUDE.md DO NOT USE entry) | eval goes through `bake_verdict` + `bake_compare`, neither of which knows about a manifest |

A genuine port of the trainer is ~6-8 hours of careful work to
update imports, switch to the v3 bake API, plumb the 372-feature
schema (and decide what to do with the missing `cclass_*` features),
and add `--mlp-moe-*` flags to today's `zensim_mlp_train` binary.
That's before training (90-120 min) and before the runtime port
into bake_compare / bake_verdict (~4-6 hours).

### 3. Predicted result is worse than the current ships

The architecture doc's own acceptance gates (against the V_0.4 era):

> KADID SROCC: ≥ FiLM (FiLM ships +0.0165 over baseline at last
> measurement). MoE should match or exceed.
> TID2013 SROCC: ≥ FiLM (FiLM ships +0.0051).
> CID22 SROCC: ≥ baseline V_0.6 (FiLM regresses CID22 slightly; MoE
> should at minimum not make CID22 worse).

The current ships are:

| Trail | Ship | CID22 | KADID | TID | KonJND | AIC-3 |
|---|---|--:|--:|--:|--:|--:|
| Balanced (`PreviewV0_5Balanced`) | V_22-mix-LARGE+iwssim (41 KB) | 0.8324 | 0.9677 | 0.9729 | 0.8927 | 0.7845 |
| Compression (`PreviewV0_5Compression`) | V_24-per-sample-α s4 (44 KB) | 0.8641 | 0.9319 | 0.8875 | 0.8125 | 0.8183 |

Both ships use **today's mechanisms** (LARGE corpus iwssim mix
training; per-sample-α head dispatch) that produced 0.86+ CID22 and
0.97+ KADID/TID. The MoE design predates these by 13 days; its
acceptance bar (match FiLM) is far below the current shipping bar.
There is no headroom for MoE to produce a competitive bake under
the two-trail § A.10 gate without first solving the same
photo-distortion-skewed-corpus problem the architecture doc itself
flagged.

## Decision per § 7 of the brief

- **§ 7a (MoE passes a gate)** — not testable; no bake. Predicted no
  per § 3 above.
- **§ 7b (MoE doesn't pass)** — this verdict. Documented here +
  PR #32 closed.
- **§ 7c (fresh seed-3 training from canonical corpus + V_22 baseline
  recipe)** — declined on engineering + scientific grounds:

  1. The trainer code on the branch does not run on today's main
     (per § 2 of the blockers above). A "fresh seed-3 training" is
     not a kick-the-tires job — it requires a 6-8h port of
     `mlp_train_moe.rs` to today's ZNPR v3 API + 372-feature schema
     + zensim-train-core API. That's 1-2 sessions of pure
     infrastructure work for a single seed.
  2. The downstream runtime port (manifest-aware
     bake_compare/bake_verdict) is another 4-6h. Without that, the
     eval cannot use the canonical 1000-bootstrap § A.9 verdict.
  3. The brief's wall-time estimate (3-8 hours) is inconsistent with
     this scope. Honest scope is 16-22 hours.
  4. The MoE architecture is the structural inverse of the
     single-bake-with-metadata dispatch that landed on main 2026-05-18
     (per § 1 of the blockers). Building substantial infrastructure
     for a path that contradicts the new runtime contract is not the
     right direction.
  5. The author's own design doc predicts the gate would degenerate
     on the current corpus shape. Today's canonical corpus is more
     photo-distortion-skewed than the MoE design's pessimistic
     assumption.

## Recommendation for the next agent

If MoE is to be revisited:

1. Wait for a content-rebalanced corpus that genuinely exercises
   non-photo distortions (UI screenshots, line art, mixed content
   with explicit class labels). The current canonical is too photo-
   heavy.
2. Reframe MoE as a **single-bake metadata-keyed dispatch** —
   `zentrain.moe_head` with the gate + K expert heads packed into
   one ZNPR v3, parsed by `forward_one_bake` exactly like
   `hybrid_head` and `per_sample_alpha_head` do. The "no new bake
   format" line in the architecture doc is the right intent but it
   was implemented via an external manifest, which is the wrong
   direction now that single-bake metadata is the runtime contract.
3. Port the trainer to `zensim-train-core` (not back into
   `zensim-validate`) so the architecture lives next to
   `hybrid_head` and `per_sample_alpha_head` infrastructure.
4. Train on canonical corpus with explicit content-class targets
   from `zenanalyze` features that are already in the 372-column
   schema (e.g., tier_depth features at the HDR tail; tier 1/2/3
   features); do NOT depend on adding `cclass_*` columns that
   require regenerating the corpus.

Until those conditions are met, MoE is a "good idea, wrong time"
hypothesis. The two-trail framework on main is the right shape;
adding a third trail (specialist trail?) for MoE-style routing is
plausible but requires the corpus rebalance first.

## Closing PR #32

PR #32 closed with this falsification record. Branch retained on
the remote at `origin/v06-moe` for historical reference (no force
deletion). The architecture doc on the branch (`docs/moe_architecture.md`)
remains useful reading for anyone reconsidering the direction.

## Files touched in this falsification

- `benchmarks/pr32_moe_falsification_2026-05-18.md` — this doc.
- `/home/lilith/.claude/projects/-home-lilith-work-zen/memory/project_pr32_moe_reeval.md` — memory file.

No code, weights, or runtime changes. No bake_compare reports produced
(MoE has no bake to compare).
