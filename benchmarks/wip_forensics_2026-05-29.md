# Forensic analysis: the "uncommitted f64→f32 WIP" — 2026-05-29

User flagged a large uncommitted change in the working tree as "probably
quite valuable" and asked to test/eval it, trace the commit log + worktrees,
and piece together purpose + timeline. **Conclusion: it was a false alarm —
the valuable work is already committed + tested; the uncommitted part is a
stray `cargo fmt` pass.**

## What the uncommitted diff actually is

A pure **`cargo fmt --all`** pass across 35 files (all mtime
`2026-05-28 18:08:29`, within 0.5 s — the signature of one fmt invocation).
**Zero logic changes.** Verified four independent ways:

1. **rustfmt round-trip:** `rustfmt(git show origin/main:streaming.rs)` is
   byte-identical to the working-copy `streaming.rs`. Same for `metric.rs`
   and `per_sample_alpha_head.rs`. The working copy IS `rustfmt(committed)`.
2. **Token-stream equality:** whitespace-stripped `metric_invariants.rs` and
   `iw_perf_baseline.rs` are md5-identical committed-vs-working.
3. **Diff inspection of the token-differing files:** every difference is
   line-reflow / import alphabetization / trailing-comma insertion. Example
   (`monotone_cbc_projection.rs`): `.fold(0.0f32, f32::max)` collapsed from
   2 lines to 1 — `0.0f32` and `f32::max` unchanged.
4. **Subagent token-level audit** of all 35 files: every change is fmt;
   zero changed identifiers/literals/types/operators.

**Test integrity intact:** no assertion, threshold, tolerance, or `#[ignore]`
changed in any test file. Every numeric gate (`t_strip <= t_full * 6.0`,
`med_both < 4.0`, `rel < 1e-6 || diff < 1e-9`, `max_rank <= 1e-6`, etc.) is
byte-identical, only reflowed.

## The valuable work — already committed (2026-05-26)

The f64→f32 port the file set belongs to was completed and **pushed to
main two days before the fmt pass**:

| commit | date | what |
|---|---|---|
| `3947afe` | 2026-05-26 08:29 | `feat(simd_encoder_f32): add 2-layer fns + parity test` |
| `9c1edab` | 2026-05-26 09:08 | `refactor(simd_encoder): consolidate to single f32 module, drop f64 SIMD dup` |
| `d6ae255` | 2026-05-26 08:40 | `feat(simd_encoder_f32): skip_forward_f32 + skip_backward_f32` |
| `ca6d47a` | 2026-05-26 19:15 | `perf(trainer): f32-native forward in PSAH hot path (1.44× epoch speedup)` |
| `6eccb93` | 2026-05-26 19:30 | `refactor(arch_f32): strip unused backward machinery, keep lean forward-only` |

All are ancestors of `origin/main` (verified `git merge-base --is-ancestor`).

**Purpose:** trainer throughput. `ca6d47a` measured a **1.44× per-epoch
speedup** in the per-sample-alpha-head (PSAH) trainer by routing the forward
pass through f32 SIMD primitives and dropping the duplicate f64 SIMD module
— eliminating ~1 M short-lived `Vec<f64>→Vec<f32>` casts and ~76 GB of
memory-bandwidth waste per epoch (per the `arch_f32.rs` module header). Adam
gradient accumulators stay f64. A companion streaming win (also committed)
is the zero-copy borrowed-strip slicer (`slice_rows_view` /
`PrecomputedReferenceView`) that drops the ~65 MB per-strip memcpy.

## Eval — the committed port passes all tests (run 2026-05-29)

- `zensim-train-core`: **42 passed**, incl. f32↔f64 parity:
  `encoder_2layer_f32_matches_f64`, `encoder_forward_f32_production_shape`,
  `dot_bias_f32_matches_scalar`. (`encoder_f32_speedup_vs_f64` is `#[ignore]`
  perf microbench — pre-existing.)
- `zensim` lib: **81 passed**, incl. streaming byte-exactness:
  `strip_aggregator_byte_exact_single_pair`,
  `strip_aggregator_byte_exact_safesyn_99`, `streaming_matches_full_image`,
  `precomputed_ref_matches_streaming`.

The f32 port is correct and shipped.

## Timeline

```
2026-05-26 08:29–19:30  f64→f32 PSAH-trainer port — 5 commits, pushed to main
2026-05-27 …            v47 ship, jxl loop retune, corruption work (this session's lineage)
2026-05-28 05:14        last corruption-corpus commit (853b2abe), pushed
2026-05-28 18:08        `cargo fmt --all` run (35 files reformatted) — NOT committed
2026-05-28 18:25        jj auto-snapshot captured the fmt diff into the default workspace @
2026-05-29 01:47        session resumes ("best bakes?" → Cell 5 ship → this forensic pass)
```

The 18:08 fmt pass fell in a ~13 h gap with no commit activity — a fmt run
(likely a /loop or pre-push-hook style invocation) that never got committed.

## Worktree inventory (the user also asked)

The repo carries a large multi-agent R&D archive — **23 jj workspaces +
~25 git worktrees** (16 are locked `.claude/worktrees/agent-*`). They are
the experiment wave, not lost work:

- **jj workspaces**: mostly *described* checkpoint commits — `V_24` sweeps
  (α-sweep, stdpool, konjnd-densify, pjnd-pairweighting — all marked
  FALSIFIED/Pareto-FAIL in their own messages), per-sample-α + hybrid runtime
  dispatch, two-trail ship, bake_compare tool, zenblend phase-4. Each is a
  named experiment commit; none is undescribed valuable WIP at risk.
- **`zensim--principled-activity`**: a git worktree sharing the main `.git`,
  HEAD at current main (`d2a7697`) — its 3 `M` files are the *same* fmt pass.
- **`zensim--372feat`, `zensim-cpu-gpu-bench`**: plain directories, NOT git
  repos (data/scratch dirs).
- **Undescribed non-empty workspace `@`s** worth a glance if reused:
  `cross-codec-v8` (ruunnwnv), `diffmap-public-ctors` (vmzxyzoo). Not
  touched here; left intact.

No worktree holds uncommitted valuable logic that isn't either (a) already on
main or (b) a described experiment checkpoint recoverable via jj op log.

## Action taken

The fmt pass is committed as a standalone `style:` commit (mechanical, safe,
byte-verified = rustfmt of committed code) to clean the working tree and
unblock `cargo fmt --check` in CI. No logic touched.
