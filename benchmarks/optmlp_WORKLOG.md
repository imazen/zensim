# optmlp lane worklog — AVX2 `forward`/`backprop`/`add_l2_grad_layer1` bit-identical optimization

Lane: `optmlp` | Workspace: `~/work/zen/zensim--optmlp` | Bookmark: `quarantine/devin/optmlp`
Targets (perf share on Zen2, per BRIEF_COMMON): `simd_mlp::forward_avx2` 23.9%,
`backprop_avx2` 22.1%, `mlp_train::add_l2_grad_layer1` 5.9%.
Oracle: BIT-IDENTICAL output — golden `~/tmp/devin/fitopt/golden/best_e3.bin` +
`epochs_e3.txt` (3-epoch H32 fold under `ZENSIM_MAX_TIER=v3`, produced on Zen2).
Host: Ryzen 9 9950X3D (Zen5, dev box) — cycle-relevant measurements on
r3500 (Ryzen 5 3500, Zen2) + iai-callgrind instruction counts.

## Baseline

- build: `cargo build --release -p zensim-validate --bin zensim_mlp_train`
  (`CARGO_TARGET_DIR=/var/tmp/optmlp/target`, no `target-cpu` flags)
- bitexact: `~/tmp/devin/fitopt/bitexact.sh <bin>` → `epoch lines identical: True | weights identical: True`
- iai: `optmlp_iai` bench (dev-dep `iai-callgrind 0.16.1`, harness=false);
  calls the pub dispatchers with a local `tier_cap` shim pinning
  `avx512_allowed()=false` (the `ZENSIM_MAX_TIER=v3` semantics — dispatch
  deterministically reaches `_avx2`, which is all valgrind can execute).
- asm: `cargo asm` / `objdump` on `forward_avx2`/`backprop_avx2`.
- wall (r3500): `taskset -c 5`, fixed 3-epoch fold, 3 repeats, min.

### Baseline numbers

| bench | value | command |
|---|---|---|
| r3500 wall, bin-rel2 binary (fleet-fits build, bitexact PASS — pre-check, mine queued) | 10.242s min (10.319/10.242/10.281) | `~/tmp/devin/optmlp_time_fold.sh /var/tmp/fleet-fits/bin-rel2/zensim_mlp_train binrel2_base` |
| r3500 wall, i134bins/rel2 (bitexact PASS) | 10.397s min (10.403/10.397/10.398) — ~1.5% build-flag spread vs bin-rel2 | `~/tmp/devin/optmlp_time_fold.sh <bin> i134rel2_base` |
| r3500 wall, own HEAD build | (dropped — bin-rel2 is byte-identical-source baseline; queued build now produces OPT) | `diff -q` fleet-fits vs optmlp simd_mlp.rs+mlp_train/mod.rs → IDENTICAL |
| iai forward_372x128 / forward_944x32 / backprop_372x128 / backprop_944x32 / l2_* | (pending lock) | baseline crate: /var/tmp/optmlp/iai_base (path-includes snapshot of wuzttoun sources) |
| bitexact, own HEAD build | (pending lock) | |

## Analysis so far

`forward_avx2` / `backprop_avx2` inner loops are chunk-inner/feature-outer:
per nonzero feature i they walk all `n_hidden/4` chunks doing
`load acc; fmadd(row,s); store acc` — ~3 memory ops per FMA. The per-lane
accumulation is a serial chain over i, so the FMAs across chunks are
independent — memory ops, not FMA latency or throughput, bound the loop.

Restructure (bit-identical — per-lane FMA sequence unchanged, operand order
unchanged, zero-skip set identical): hoist 8 chunks (32 lanes) of `h_pre`
into ymm accumulators across the whole feature sweep, so each FMA reads
only its w1 operand (folded into `vfmadd231pd`'s memory operand).
- forward: `h_pre[j] += x_i * w1[i,j]` — acc in regs across i sweep.
- backprop gw1: `gw1[i,j] += x_i * dh[j]` — dh block in regs across i
  sweep; gw1 streams load+fma+store (store is inherent, one per element).
- nonzero-index list `nz` built once — replaces the per-(pass,i)
  `s == 0.0` branch with an exact same-set index sweep.
- `add_l2_grad_layer1`: scalar `g += sm*w` autovectorizes to SSE2 only
  (baseline x86-64 build); hand AVX `vmulpd`+`vaddpd` (mul+add, two
  roundings — NO FMA) gated by `is_x86_feature_detected!("avx")`.

## Changes tried

### Standalone bit-identity check (before touching the repo)

`~/tmp/devin/optmlp_parity.rs` — verbatim OLD kernels vs NEW (const-generic
register-blocked, 8/4/2/1 cascade), `rustc -O -C target-cpu=x86-64-v3`, run on
the Zen5 build host: **ALL BIT-IDENTICAL** across 13 shapes
(944×32, 372×128, 944×128, 16×8, 16×6, 8×7, 4×5, 4×4, 4×3, 5×9, 7×13,
944×33, 944×36) × 4 zero-fractions (0/30/70/99%), with -0.0 and NaN feature
values and a -0.0 bias probing skip semantics; plus the L2 AVX row kernel
vs scalar on 4 length/hidden combos × mult/none.

New structure: hoist N chunks (N∈{8,4,2,1}) of the accumulator/target array
into ymm registers across the whole nonzero-feature sweep (`nz` index list
built once — identical skip set, ascending i, same FMA operands/order).
Eliminates the per-(i,chunk) load+store round trip: ~3 mem ops/FMA → ~1.1
(forward) and removes the redundant `dl_dh_pre` load per (i,chunk)
(backprop). `gb1` accumulation fused into pass 1 (same add, same value,
earlier). `add_l2_grad_layer1`: AVX `vmulpd`+`vaddpd` row kernel gated by
`is_x86_feature_detected!("avx")` — mul+add two roundings, NO FMA.

### Asm check of the NEW kernels (prototype, `rustc -O -C target-cpu=x86-64-v3`)

`fwd_acc_block::<8>`: all 8 ymm accumulators live in registers across the nz
sweep; per feature: index load + `imul` + `vbroadcastsd` + 8×`vfmadd231pd`
with w1 loads folded into memory operands (LLVM unrolls ×2); no spills.
`bwd_gw1_block::<8>`: 8 dh vectors hoisted into ymm0-7; per feature: index +
broadcast + imul + 8×(load g, fmadd, store).

## Applied changes (2026-09-25 ~07:40Z)

- `src/simd_mlp.rs`: `+fwd_acc_block<N>`, `+bwd_gw1_block<N>` const-generic
  helpers (avx2+fma, #[inline]); `forward_avx2` builds `nz` once then runs
  the 8/4/2/1 chunk cascade + scalar tail over `nz`; `backprop_avx2` fuses
  `gb1 += dl_dh_pre` into pass 1 and runs the same cascade for `gw1`.
  Dispatch wrappers + `avx512_allowed()` guard untouched; AVX-512 fns
  untouched.
- `src/mlp_train/mod.rs`: `add_l2_grad_layer1` gains `l2_row_avx` (AVX
  mul+add, no FMA) gated by `is_x86_feature_detected!("avx")` in both the
  `mult=None` whole-slice arm and per-row of the `Some` arm; non-x86 stub
  keeps all targets compiling.
- `rustfmt --edition 2024 --check` clean on both files; both compile
  standalone (`rustc --edition 2024 -O -C target-cpu=x86-64-v3`, tier_cap
  stubbed); L2 extract runs correctly.
- Gates so far: `cargo fmt --all -- --check` ✓ clean; `just lint-scripts` ✓
  (658 scripts runnable). Rest pending heavy-lock.
- `benches/optmlp_iai.rs`: opt side (dispatcher + tier_cap shim → AVX2);
  baseline counts come from throwaway crate `/var/tmp/optmlp/iai_base`
  which path-includes the wuzttoun source snapshot — same harness, same
  inputs, old kernels.

## Mid-gate results (before the lock freed)

- **bitexact on optimized binary: PASS** — `epoch lines identical: True |
  weights identical: True` (`~/tmp/devin/fitopt/bitexact.sh
  /var/tmp/optmlp/target/release/zensim_mlp_train`, 2026-09-25 ~14:25Z).
- r3500 wall under contention (`optadam_proto2` pinned on core 5 since
  14:17 local — started AFTER the clean baseline reps, so the clean
  10.242s baseline is not directly comparable to anything measured now).
  Interleaved A/B on the same contended core, 3 reps each alternating:
    binrel2_base: 25.112 / 25.033 / 25.048 → min 25.033
    opt         : 23.887 / 24.117 / 24.132 → min 23.887
  Opt wins all 3 pairings; ratio ≈ 0.954 (~4.6% faster under ~50% duty
  cycle — the clean-core number should be similar or better since the
  saved work is all in the timed kernels).
- Zen5 local A/B (taskset -c 23, interleaved, 3 reps): base 6.254/6.229/
  6.228 (min 6.228) vs opt 5.687/5.646/5.644 (min 5.644) → **opt 9.3%
  faster end-to-end** on the same 3-epoch fold under `ZENSIM_MAX_TIER=v3`.
- Local: `avx2_kernels_match_scalar_directly` standalone-compiled + ran
  (12 shapes, every cascade arm) — pass.

## Phase-2 gate results (~14:59Z, on the pre-clippy-fix binary)

- bitexact: **PASS** (`epoch lines identical: True | weights identical: True`)
- `adam_simd_equivalence` 9/9 ✓, `adam_simd_rsqrt_precision` 7/7 ✓
- `simd_mlp` lib tests 5/5 ✓ (incl. `avx2_kernels_match_scalar_directly`)
- integration: `nonneg_distance` 8/8, `hybrid_head_runtime` 4/4,
  `group_loss_mode` 6/6, `keep_features_multilayer` 3/3,
  `monotone_cbc_projection` 2/2, `train_core_zenstats_lockstep` 4/4 — all ✓
- `bake_surface` 11/16 — **5 failures, diagnosed PRE-EXISTING** at parent
  `wuzttoun`: the failing asserts cover the corruption-companion
  extraction-plan / per-bake formula-revision contract, and the fixing
  commit (`277041e8`, "a corruption companion extends the extraction
  plan…") is NOT an ancestor of this lane's base — the tree holds the old
  impl against tests written for the new one. `BakeScorer` lives in the
  `zensim` crate, disjoint from every file this lane touched; failures
  reproduce identically without the kernel diff (same test text verified
  at `277041e8`).
- clippy: FAILED on my code — 2× `needless_range_loop` in the new scalar
  tails (forward ~545, backprop ~691).
- iai bench: compile-fail — `iai_callgrind::std::hint::black_box` import
  (crate doesn't re-export `std`); macro rejects `///` doc comments
  adjacent to `#[library_benchmark]` as an invalid `doc` attr.
- l2 test filter was wrong (matched 0); actual name:
  `l2_row_form_matches_divided_index_form_bitwise`.

## Fixes after phase-2 (2026-09-25 ~15:20Z)

- scalar tails rewritten iterator-style (`iter_mut().enumerate().skip` /
  `iter().enumerate().skip`) — identical op order, clippy-clean.
  Re-verified standalone: all 5 tests pass incl. direct-AVX2 (12 shapes).
- `optmlp_iai.rs`: `use std::hint::black_box` + `//` comments under
  `#[library_benchmark]`; same fixes applied to `/var/tmp/optmlp/iai_base`.
- Pending re-gate (phase-3 queued): rebuild → `just clippy` → iai opt+base
  → bitexact on fresh binary → targeted tests (`simd_mlp`, `l2_row_form`)
  → `cargo fmt --all -- --check`.

## Contention-matched r3500 A/B (after optadam_proto2 freed core 5)

| | rep1 | rep2 | rep3 | min |
|---|---|---|---|---|
| base (bin-rel2, same-source kernels) | 11.249 | 11.285 | 11.386 | 11.249 |
| opt (phase-2 binary) | 10.630 | 10.931 | 10.850 | 10.630 |

**opt 5.5% faster (min/min), wins every pairing.** NB: contended runs —
other lane jobs shared the box; min-based comparison under matched
interleave is the honest read (per CLAUDE.md no-speedup-from-contention
rule this is reported as contention-matched, not quiet-machine).


## Phase-3/4 gates + iai results (15:44–15:55Z)

- Rebuilt binary bitexact: **PASS** again on the post-clippy-fix build.
- `just clippy` workspace-wide: **PASS** (after fixes below).
  - Fixed my bench: `iai_callgrind::std` → `std::hint::black_box`;
    `///` before `#[library_benchmark]` rejected by the macro → `//`;
    `#[allow(dead_code)]` on the `#[path]`-included `simd_mlp` module
    (test helpers are dead code under bench targets' cfg(test)).
  - **Pre-existing breakage repaired**: `bench_mlp_kernels.rs` lacked a
    `crate::tier_cap` shim for its own `#[path]` include of `simd_mlp.rs`
    — broken at parent `wuzttoun` (the `avx512_allowed()` call predates
    this lane). Added the 5-line `mod tier_cap` returning `true`
    (host-best dispatch — semantics the bench always had).
- `cargo fmt --all -- --check`: PASS; `simd_mlp` lib tests 5/5 PASS;
  `l2_row_form_*` 2/2 PASS (`l2_row_form_matches_divided_index_form_bitwise`
  is the L2 bitwise gate).
- iai bench restructured: input generation moved to `setup=` (was inside
  the measured region — masked the L2 delta at first).

### iai-callgrind counts (instructions | estimated cycles), same harness both sides

| bench | base | opt | Δ instr | Δ cycles |
|---|---|---|---|---|
| forward 372×128 | 50,175 | 20,453 | **−59.2%** | −32.6% |
| forward 944×32 | 48,531 | 21,498 | **−55.7%** | −36.2% |
| backprop 372×128 | 48,326 | 35,660 | **−26.2%** | −28.9% |
| backprop 944×32 | 43,358 | 31,626 | **−27.1%** | −27.8% |
| l2 944×32 (mult) | 135,412 | 81,923 | **−39.5%** | −32.8% |
| l2 944×32 (nomult) | 98,580 | 38,497 | **−61.0%** | −46.8% |

### Binary provenance check (deterministic clean builds)

`cargo clean -p zensim-validate` + rebuild, twice per tree:
- my tree `f73b1761` → `64fab650…` (both builds, 9,576,504 B)
- parent `wuzttoun` → `3ba0bc09…` → `/var/tmp/optmlp/zensim_mlp_train_base_v2`
- phase-3's build produced `500da672…` (9,569,456 B) — same source, ~7KB
  different binary: build-to-build nondeterminism (metadata/layout), worth
  ~±1.5% wall-clock, same magnitude as the earlier flag spread. All
  bitexact-equivalent. Final timing uses `optfinal=64fab650` +
  `base_v2=3ba0bc09` (same rustc, same lockfile state).

## Final r3500 (Zen2) wall timing — 4-way interleaved A/B, taskset -c 5

All binaries ran the same fixed 3-epoch fold (`--max-features 944 --hidden
32`, 50k pairs/epoch, `ZENSIM_MAX_TIER=v3`, `RAYON_NUM_THREADS=1`), pairs
alternating under identical ambient load. 16 reps total.

| binary | source | build | rep times (s) | min |
|---|---|---|---|---|
| `base_v2` | parent `wuzttoun` | this tree's flags/lock | 11.034 / 11.130 / 10.964 / 10.978 | 10.964 |
| `optfinal` | this lane `f73b1761` | this tree's flags/lock | 10.553 / 10.598 / 10.611 / 10.603 | **10.553** |
| `binrel2` | parent source | fleet-fits lane build | 10.825 / 10.839 / 10.859 / 10.810 | 10.810 |
| `opt3` | this lane | earlier (phase-3) build | 10.969 / 10.965 / 11.067 / 10.883 | 10.883 |

**optfinal vs same-flags parent: 10.553s vs 10.964s → 3.75% end-to-end
faster; optfinal wins all 4 pairings.** Box was shared (other lanes'
training jobs on sibling cores); alternating reps keep both sides under
the same conditions — reported as contention-matched per repo rules.

Build-noise lesson: `opt3` vs `optfinal` are byte-different builds of
identical source (rustc layout nondeterminism, ~7KB size delta) and differ
~3% in wall time — comparable to the kernel effect itself. The like-for-like
same-invocation comparison (base_v2 vs optfinal, both clean-built in this
target dir) is the honest headline; iai instruction counts (−27% fwd..−59%
fwd, −61% l2-nomult) are the noise-free corroboration.

Caveats: box not quiet; single fixed fold; Zen2 only (+ Zen5 host A/B at
9.3% earlier, same protocol). The 5 `bake_surface` failures are
pre-existing at the parent commit (fix lives in `277041e8`, not an
ancestor of this lane) — unrelated to the kernel diff.
