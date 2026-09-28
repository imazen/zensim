# fusedstep lane worklog — fused K=1 pair update (backprop gw1 + L2 + Adam)

Brief: `/home/lilith/tmp/devin/fitopt/BRIEF_fusedstep.md`
Workspace: `zensim--fusedstep` on `quarantine/devin/fusedstep` (jj), from `main@origin`.
Scratch/build: `/var/tmp/fusedstep/` (CARGO_TARGET_DIR=/var/tmp/fusedstep/target).

## Objective

Replace the per-pair layer-1 chain

    backprop_step(A): gw1[i] += xa[r]·dha[j]     (row-skip on xa[r]==0)
    backprop_step(B): gw1[i] += xb[r]·dhb[j]
    add_l2_grad_layer1: gw1[i] += sm[r]·w[i]     (mul+add, never FMA)
    adam_update(w1):  consume gw1, write w/m/v, store g=+0.0

with a single pass that keeps `g` in a register. Bit-identical output is the
hard requirement: every element must see the identical IEEE op sequence in
the identical order.

## Arithmetic contract (mapped before implementing)

- Backprop gw1 sweep (all tiers): per-row `s = x[r]`; if `s == 0.0` the row
  is skipped. Fused `s.mul_add(dh[j], g)` on lanes `j < n4h` where
  `n4h = nh - nh%4`; mul+add on the per-row tail `[n4h, nh)`.
- `add_l2_grad_layer1`: `*g += sm * w` — SEPARATE mul then add both in the
  AVX helper (`l2_row_avx`: `_mm256_add_pd(g, _mm256_mul_pd(sm,w))`) and the
  scalar loop. Per-row `sm = scale * mult[feat]` hoist when `mult=Some`.
- Adam (`adam_update_inner_v3` canonical): fused domain `[0, n-n%4)` uses
  `mul_add` for m/v updates, `g*g` product, `inv_bc` multiplies, sqrt+div;
  flat tail `[n4, n)` uses `adam_update_scalar_ref` (mul+add, `/bc` divides).
- `n = n_features * n_hidden`; when `n_hidden % 4 == 0` then `n % 4 == 0` and
  the Adam tail is empty; every row's `n4h == nh` so the whole backprop
  sweep is fused-domain. → call-site gate: `k == 1 && nh > 0 && nh%4 == 0
  && w1.len() % nh == 0`.
- gw1 entry value: `+0.0` in the K=1 cadence (zeroed by the prior step), but
  the kernel loads it — bit-identical even under TV-regularizer residue.

## Changes

### `src/simd_mlp.rs`

Split each tier's `backprop_*` into a **head** (gw2 += dl_dy·h; dh_out =
dl_dy·w2 with leaky gate; gb1 += dh; gb2 += dl_dy) and the retained **gw1
sweep**:

- `backprop_scalar_head` — head of `backprop_scalar`, verbatim ops
  (FMA domain `o < n4`, mul+add tail).
- `backprop_avx512_head` — head of `backprop_avx512`; preserves the
  AVX2-parity boundary handling (8-lane body, 4-lane masked group when the
  canonical fused domain has a 4-lane remainder, scalar tail after n4).
- `backprop_avx2_head` — head of `backprop_avx2`.
- Existing `backprop_scalar`/`avx512`/`avx2` now call their head then run
  the unchanged gw1 sweep — no arithmetic change to `backprop_step`.
- New dispatcher `backprop_grad_head(...)` mirrors `backprop_step`'s tier
  selection (`tier_cap::avx512_allowed()` + `X64V4Token` → `X64V3Token` →
  scalar).

### `src/adam_simd.rs`

- `AdamW1FusedArgs` — w/g/m/v + xa/dha/xb/dhb + l2_scale/l2_mult/n_hidden +
  Adam params.
- `adam_update_w1_fused` — `incant!` dispatch `[+v4]`; off-grid shapes
  (`nh==0`, `nh%4!=0`, `w.len()%nh!=0`) route to the composition fallback.
- `adam_pair_fused_inner_v3` (`#[arcane]`): per row, `sa`/`sb`/`sm` splats;
  per 4-lane chunk: `g = load(gc)`; `if do_a: g = fma(sa_v, dh_a_c, g)`;
  `if do_b: g = fma(sb_v, dh_b_c, g)`; `if l2: g += sm_v*w` (mul+add); then
  the verbatim v3 Adam arithmetic; stores m/v/w and +0.0 to g.
- `adam_pair_fused_inner_v4`/`_neon`/`_wasm128`/`_scalar` →
  `adam_pair_fused_fallback`: scalar replication of the unfused elementwise
  sequence into `g` (row-skip + n4h FMA domain + mul+add tail + L2) then the
  regular `adam_update` dispatch — bit-identical by composition (all tiers'
  adam are AVX2-parity under the tierparity contract).
  v4 takes the fallback deliberately: `incant!` would give v4 an 8-lane
  domain boundary that AVX2-parity forbids; scalar accumulate + dispatched
  adam is exactly the unfused op sequence.

### `src/mlp_train/mod.rs`

- `AdamState::step_w1_fused` — same `t`/bias-correction as `step`; calls
  `adam_update_w1_fused` for w1, ordinary `adam_update` for b1/w2/b2.
- Call site: hoisted `fuse_w1` gate + `dh_a`/`dh_b` scratch (allocated once
  instead of the per-call `dl_dh_pre` Vecs). K=1 gated branch runs
  `backprop_grad_head`×2 → w2-L2 loop → `step_w1_fused`; everything else
  (k>1 accumulation, off-grid shapes) keeps the unfused sequence verbatim.
- `apply_post_adam_penalties` + `nonneg_project` run in both arms at the
  same point.

## Standalone verification (before heavy gates)

`/var/tmp/fusedstep/proto1.rs` — scalarized replica of the fused elementwise
sequence vs unfused oracle, 12 shapes × {mult none/some} × {l2 0/1e-5/3} ×
{dense/zero-rows}, non-zero gw1 residue: **ALL BIT-IDENTICAL**.

## Gates (filled as they run)

- [x] cargo check lib+tests+benches (01:33 log — clean)
- [x] cargo fmt --check — clean
- [x] lib test `fused_w1_per_tier_bit_identical` — PASS after black_box(t) fix
      (cargo-built binary, dbgtest.log 04:5x)
- [x] standalone `rustc --test` of fusedstep_equiv.rs (19/19 PASS — incl.
      fused_pair_update_bit_identical end-to-end, both negative controls,
      all simd_mlp/adam tier-parity tests)
- [ ] full cargo test suite (queued in gates_all.sh)
- [ ] clippy -p zensim-validate --all-targets -D warnings (queued)
- [ ] cargo asm / objdump on adam_pair_fused_inner_v3 (queued)
- [ ] bitexact.sh + bitexact_uncapped.sh (release binary; queued)
- [ ] cell identity H32 + H128 vs verified blobs
- [ ] iai fusedstep_iai (before/after; queued)
- [ ] wall-clock: dev pinned ≥5 interleaved alone + loaded; r3500 ×3

## Numbers

(iai/wall tables land here.)

## 2026-09-26 — v3 "divergence" root cause: test-harness `powi` const-fold, kernel clean

`fused_w1_per_tier_bit_identical` failed in cargo-built test binaries (lib +
`#[path]`-included copies) at `(944,32,mult=false)` with 741 × ±1-2ulp `w`
diffs — but the identical source passed under `rustc --test` harnesses.

Isolation (standalone harnesses linking the real rlibs):
- `adam_update_w1_fused` ≡ unfused oracle under both v4 and v3-forced
  dispatch — 0 diffs on the exact failing shape/seed.
- `adam_update_inner_v4` ≡ `adam_update_inner_v3` on the accumulated
  gradients — 0 diffs.
- Both `__arcane_adam_pair_fused_inner_v3` CGU copies in the failing binary
  disassemble to the correct op sequence (vfmadd A, vfmadd B, vmulpd+vaddpd
  L2, adam chain, zero-store).
- Element-80 analytic replay: every g-ordering/adam-ordering of the
  documented ops gives w=0xbfb9f54303cb2a82 — the *fused* value. The
  *oracle* produced …a81.

Root cause: **`f64::powi` const-fold vs runtime disagreement.** The fused
side built `bc1 = 1.0 - 0.9f64.powi(5)` with a literal exponent — CTFE/
const-prop evaluates it to `0x3fda35696e58a32e`; the oracle's `make_args`
evaluates `beta1.powi(t as i32)` with a runtime `t` → `llvm.powi` →
`0x3fda35696e58a32c` (2ulp lower). The two sides ran Adam with different
bias corrections; the resulting 1-ulp `w` diffs were a TEST BUG, not a
kernel bug. (rustc 1.98.1; literal `powi` folds at cargo's dev mir-opt
level but not under `rustc -g`, which is why the standalone harness
passed.)

Fix: `let t = std::hint::black_box(5u64);` ahead of both calls; fused args
now compute `1.0 - 0.9f64.powi(t as i32)` — runtime-evaluated on both
sides at every opt level. Same treatment in the dispatch test (t=3).
Production `AdamState::{step, step_w1_fused}` already share the same
runtime expression `beta1.powi(self.t as i32)` — unaffected.

The failing test now also prints first-diff index + `g0/g_bp/g_pre_adam`
bits instead of dumping 30k-element vectors.

## Canonical gates — 2026-09-26 (dev box, heavy lock, `/var/tmp/fusedstep/target`)

`gates_all.sh` (first consolidated pass):

- `cargo fmt --check` — PASS (FMT_EXIT=0).
- `cargo test -p zensim-validate --lib` — **268 passed / 0 failed**
  (includes `fused_w1_per_tier_bit_identical`, post-powi-fix).
- Integration tests (`--tests`, fusedstep_equiv + path-included modules) —
  **19 passed / 0 failed**.
- `cargo build --release --bin zensim_mlp_train` — BUILD_EXIT=0.
- `bitexact.sh` on the release binary, `ZENSIM_MAX_TIER=v3`
  (`--max-features 944 --hidden 32 --epochs 3 --pairs-per-epoch 50000
  --init-seed 1101 --sample-seed 101`, `RAYON_NUM_THREADS=1`) — **PASS**
  (BITEXACT_EXIT=0, epochs+weights vs golden).
- Same run **uncapped** (AVX-512 host path) — **PASS**
  (BITEXACT_UNCAPPED_EXIT=0).
- asm gate: `__arcane_adam_pair_fused_inner_v3` found and dumped from the
  release binary (prologue shown; inner-loop vfmadd/vmulpd/vaddpd sequence
  verified earlier on the standalone-built copy — same source, same
  codegen flags).
- `cargo bench --bench fusedstep_iai` — bench binary built but the runner
  wasn't on PATH in that job; rerun queued in `final.sh` with
  `IAI_CALLGRIND_RUNNER` pointed at the built dep.
- clippy steps ran but their exit codes were swallowed by a `| tail`
  pipeline bug — they actually FAILED with 7 `too_many_arguments` errors
  on the two test-only oracle helpers (`unfused_w1_oracle`,
  `fused_wrong_order`, both 12 params mirroring the kernel signature).
  Fixed with `#[allow(clippy::too_many_arguments)]` (codebase convention);
  verification rerun queued (`clippy2.sh` with PIPESTATUS capture).

Post-gates_all source edits (require rebuild before cell runs — folded
into `final.sh`):
- `fmult` fetch in the fused call-site arm is now gated on
  `l2_lambda > 0.0` — matches the unfused arm's call site exactly
  (skips a mutex+Arc clone when L2 is off; no arithmetic change).
- per-tier test's verbose first-diff panic compressed to a compact
  `assert_eq!` carrying tier/shape/first-index context.
- `#[allow]`s noted above.

## Cell identity — staged

- `/var/tmp/fusedstep/run_cell.sh` — same as cellprofile's `run_cell.sh`
  (`--network=none`, `ZENSIM_MAX_TIER=v3`, instrumented exec/scripts
  bind-mounted, single-CPU `--cpuset-cpus`) plus one extra mount:
  `/var/tmp/fusedstep/target/release/zensim_mlp_train` over
  `/opt/fleet-fits/program/bin/zensim_mlp_train`.
- H32 cell (kadid_train/r0/h32, manifest copied from
  `runs/h32l`, data `5db2419e` hardlinked) → compare vs
  `gate7-dev/verified/196561ec…`.
- H128 cell (aic3/minus_basic/h128, manifest from `runs/h128a`) →
  compare vs `gate7h128-dev/verified/c54b7e52…`.
- My binary verified to launch inside `fit-p0-v7` (container glibc 2.43,
  binary needs ≤2.39).

## iai-callgrind — canonical (final.sh, release build, IAI_CALLGRIND_RUNNER dep)

Whole-pair sequence per bench body: `forward`×2 + gw1-path (sweep×2 +
`add_l2_grad_layer1` + `adam_update`) vs `forward`×2 +
`backprop_grad_head`×2 + `adam_update_w1_fused`.

| bench | Ir | est. cycles |
|---|---|---|
| unfused 944×32  | 299,804 | 733,201 |
| fused   944×32  | 338,947 | 725,694 |
| unfused 372×128 | 420,917 | 1,088,441 |
| fused   372×128 | 471,572 | 1,058,504 |
| `fused_matches_unfused` verify | 3,306,931 | ran under callgrind — no assert → **bit-identical** |

Deltas: 944×32: **+13.1% Ir, −1.0% est.cyc**; 372×128: **+12.0% Ir,
−2.7% est.cyc**.

Read: the fused v3 kernel spends ~3.3 extra instr per 4-chunk on
row-boundary bookkeeping (per-row `sa`/`sb` broadcast + zero-checks live
inside the chunk loop) — visible as Ir growth — while eliminating the
`gw1` store→load round-trips that dominate real-cache cost. Callgrind's
simple cache model credits only part of the traffic win; the
authoritative numbers are the r3500 / dev wall-clock A/B.

## Wall-clock A/B — final.sh (interleaved, taskset, v3)

- BASE binary missing in final.sh's first pass — `git archive wrnwwpls`
  fails on a jj change-id (tar error → BASE_BUILD_EXIT echoed 0 wrongly,
  all `WALL base*` rows are ~0.005 s instant-fails). Proper commit-hash
  build rerun queued in `wallclock.sh`.
- FUSED (dev box, cpu 9): 10.54, 10.23, 10.70, 9.97, 10.68 s —
  median 10.54 s (CONTENDED — fleet load ~25-30).
- FUSED (r3500, cpu 5): 8.69, 8.67, 8.70 s — median 8.69 s. **Verified
  output**: `/tmp/fs-check.bin` pulled back, masked-compare vs
  `golden/best_e3.bin` → identical (fused binary is bit-correct on the
  real Zen2 AVX2 host).
- Baseline numbers land when wallclock.sh runs.

## Cell identity — H32 result (P0 kadid_train/r0/h32/o0_r0)

`final.sh` ran `run_cell.sh h32` (image `fit-p0-v7`, `--network=none`,
`--cpuset-cpus=3`, `ZENSIM_MAX_TIER=v3`, instrumented exec/scripts,
my release binary bind-mounted over `program/bin/zensim_mlp_train`).
`CELL_H32_EXIT=0`; blob `be6454…`.

`diff -r` vs verified `gate7-dev/verified/196561ec…` after the cellprofile
mask (two metadata-length fields) plus tail-metadata inspection:

- **All 17 `.bin` payloads identical** (`inner0..3/best.bin`,
  `refit/best.bin`, `refit/ckpt_epoch{000..055}.bin`) — the masked
  compare covers the whole weight prefix; the tail `argv` JSON differs
  only in `hostname` / `timestamp_epoch` / `trainer_source_dir`.
- `train.log` ×5: identical loss/srocc/plcc/pwrc at every epoch; only
  `t=` differs (my run was slower — contended dev box under `--cpuset 3`).
- `best.bin.spec.json` ×5: only `timestamp_epoch`.
- `importance.json` / `result.json` / `fleet_receipt.json`: value-equal
  except `*_sha256` fields, which are sha-of-timing (logs/bin headers
  embed timestamps → hash chain differs by construction).
- `fleet_stdout.log`: timing lines only.

**H32 cell identity: PASS** — output bit-identical modulo the allowed
hostname/timing/sha-of-timing set, matching the cellprofile identity rule.

H128 cell (aic3/minus_basic/h128) started ~11:0x, still running.

## clippy `-D warnings` — PASS (direct run, 2026-09-26)

After fixing all lints on owned code (`too_many_arguments` allows on the
two test oracles; `is_multiple_of` in the dispatcher + `fuse_w1` gate;
`assign_op` in the v3 kernel; iterator form in the oracle row loops):

- `cargo clippy -p zensim-validate --lib --tests -- -D warnings` →
  Finished, 0 warnings/errors (11.77 s incremental).
- `cargo clippy -p zensim-validate --all-targets -- -D warnings` →
  Finished, clean (1.56 s — incl. `bench fusedstep_iai` + all
  `#[path]`-including targets).

## Cell identity — H128 result (P0 aic3/minus_basic/h128/o0_r0)

Blob `7e58d1f5…`, exec_total 3235 s under `--cpuset-cpus=3` on the loaded
dev box. `cellcmp.py` vs `gate7h128-dev/verified/c54b7e52…`:
**47 files exact-identical, 17 `.bin` masked-equal, 14 real-diffs** —
all inside the allowed set: `fleet_receipt/importance/result.json`
(sha-of-timing chains), `*.spec.json` (`timestamp_epoch`), `train.log`
(`t=` + one byte-count line), `fleet_stdout.log` (timing). Every
`inner*`/`refit` weight payload bit-identical.

**H128 cell identity: PASS.** Both P0 cells verified.

## Wall-clock A/B vs parent baseline `921d072b` (wallclock.sh, DONE)

Baseline built via `git archive 921d072b` from the main repo into
`/var/tmp/fusedstep/base-tree` → `target-base/zensim_mlp_train`.
Interleaved runs, `ZENSIM_MAX_TIER=v3`, `RAYON_NUM_THREADS=1`.

r3500 (Zen2 AVX2, taskset -c 5):
- base:  11.020 / 11.071 / 11.067 → median 11.067
- fused:  9.772 /  9.736 /  9.650 → median  9.736
- **Δ −12.0% median; ranges disjoint; fused output bit-identical vs
  Zen2 golden (verified earlier).**

dev box (taskset -c 9, CONTENDED — load 25-51):
- base:  10.36 / 9.84 / 10.25 / 9.90 / 9.82 → median 9.90
- fused: 10.09 / 10.34 / 10.06 / 10.18 / 10.06 → median 10.09
- Δ ≈ +2% — Zen5's caches hide the gw1 traffic win; extra per-row
  instruction cost shows through instead.

Verdict: the fusion wins where it was designed to — memory-bound AVX2
hosts (fleet = Zen2). −12% wall-clock on a full e3 fold at identical
bits.

## Command evidence log (C5 review correction — added 2026-09-26)

Heavy-queue commands (`~/tmp/devin/heavy --`, shared lock
`/home/lilith/tmp/devin/heavy.lock`). Start times reconstructed as
(log-mtime − reported duration); end = log mtime; both converted to UTC
from file timestamps (host TZ −0600). sha256 of each full log follows.

| command (script) | cwd | start UTC | end UTC | rc | dur | log sha256[:16] |
|---|---|---|---|---|---|---|
| `cargo check -p zensim-validate` (check.sh) | /home/lilith/work/zen/zensim--fusedstep | 2026-09-26T07:20:55Z | 07:21:26Z | 0 | 31 s | 38c6049864c53e41 (check.log) |
| `cargo test --lib` + `--test fusedstep_equiv` (tests.sh) | same | 07:33:28Z | 07:33:57Z | 0* | 29 s | 54a9e6c66fc40a05 (tests.log) |
| diagnostic rebuild of lib test (dbgtest.sh) | same | 11:01:03Z | 11:01:17Z | 0 | 14 s | 884f4f58c76b6d18 (dbgtest.log) |
| gates_all.sh: fmt→lib tests→equiv tests→release build→bitexact capped+uncapped→asm→iai bench | same | 11:14:39Z | 11:20:42Z | 0 | 363 s | b1617a314908cd95 (gates_all.log); runner tail in gates_all.out |
| clippy2.sh: `clippy --all-targets` (missing `-D warnings` — superseded) | same | 11:26:33Z | 11:26:44Z | 0 | 11 s | 67105dd6f8fc2f3a (clippy2.log) |
| clippy3.sh | — | — | — | cancelled (removed from queue; superseded by direct runs below) | — | e3b0c44298fc1c14 (empty) |
| final.sh: release rebuild→iai bench→run_cell.sh h32→run_cell.sh h128 | same + docker | 11:26:25Z | 12:40:55Z | 0 | 4450 s | 1cf3aa05db5d288b (final.log) |
| wallclock.sh: baseline build (git archive 921d072b→/var/tmp/fusedstep/base-tree)→dev A/B ×5→r3500 A/B ×3 | same + r3500 ssh | 12:52:01Z | 12:56:10Z | 0 | 249 s | e2d509a269d5d5cc (wallclock.log) |

*tests.sh ran before the `black_box(t)` fix; the failing per-tier test
was subsequently verified by dbgtest + the gates_all rerun.

Direct commands (no heavy lock — incremental/read-only):
- `cargo clippy -p zensim-validate --lib --tests -- -D warnings` → 0,
  cwd zensim--fusedstep, ~12:2xZ (during final.sh docker wait).
- `cargo clippy -p zensim-validate --all-targets -- -D warnings` → 0,
  same cwd, ~12:2xZ.
- `rustc --test` standalone harnesses (`/var/tmp/fusedstep/tharness.rs`,
  `dbg*.rs`, `proto1.rs`, `equiv_test`) — cwd `/var/tmp/fusedstep`,
  linked against `target/debug/deps` rlibs; all pass after the
  `black_box(t)` fix; the pre-fix failure repro is in dbgtest.log.
- `cellcmp.py` compares + `run_cell.sh` — invoked inside final.sh
  (timings above); docker cell runs pinned `--cpuset-cpus=3`,
  `ZENSIM_MAX_TIER=v3`, `--network=none`.

## Opus review corrections (REVIEW_FUSEDSTEP.md, PROMOTE WITH CORRECTIONS)

- **C1 — v4 arm defect fixed.** `adam_pair_fused_inner_v4` was routing to
  the scalar-emulating fallback every pair (reviewer measured +315%
  uncapped on Zen5: 41.4 s vs 10.0 s, bit-identical). The kernel is
  elementwise and `nh%4==0` keeps all rows in the 4-lane fused domain,
  so the fix is `adam_pair_fused_inner_v3_body(token.v3(), args)`
  (`X64V4Token::v3()` — archmage 0.9.28 `x86.rs:1800`). Verified
  standalone: `fused_w1_per_tier_bit_identical` (now with a v4 arm)
  passes on this AVX-512 host.
- **C2 — real-code equivalence now covers every tier.**
  `fused_pair_update_bit_identical` wrapped in
  `archmage::testing::for_each_token_permutation` +
  `lock_token_testing()`; 9 permutations on this host, all pass
  (standalone `rustc --test` run).
- **C3 — "negative controls" relabelled as input-sensitivity checks**
  in both code comments and the report; they compare replicas and never
  call the kernel. Kernel-level evidence = the reviewer's mutant table,
  stated as measured: M1–M5 all caught by the replica-based per-tier
  test; on the fixed tree the permuted `fused_pair` test additionally
  catches M1; on `11eef724` the capped bitexact gate caught M2;
  M2–M5 not re-run against the permuted test; M6 caught by the
  dispatch + equivalence tests.
- **C4 — footprint cleaned.** dev `/tmp/fusedstep-ab-*`(20), 
  `/tmp/fs-check-r3500.bin`; r3500 `/tmp/fs-{ab,check}.bin[.spec.json]`;
  `r3500:~/tmp/optmlp-exp/zensim_mlp_train.{fusedstep,fsbase,fsfused}`;
  empty `~/tmp/devin/fusedstep_note` — all removed.
  Manifest: `~/tmp/devin/rev4_fusedstep_manifest.tsv`.
- **C5 — `jj metaedit --update-author` → author lilith@imazen.io;
  commit message gets What/Commands/Outputs/Numbers.
- **C6/C7 — DONE text + bench header corrected** (per-host verdict,
  MISSING additions, tier-cap comment no longer claims it pins
  `incant!` dispatch).

Re-verification pending: `bitexact_uncapped.sh` + uncapped dev A/B on
the fixed binary (c1verify.sh, heavy queue).

## C1 re-verification results (c1verify.sh, run 2026-09-26 ~15:0xZ)

- `cargo fmt --check -p zensim-validate` → 0
- `cargo build --release --bin zensim_mlp_train` → 40.55 s, rc 0,
  binary sha `e3ccb135…` (pre-test-refactor build; the later clippy
  `type_complexity` refactor touched only `#[cfg(test)]` code — final
  binary rebuilt to `fa088cc4…` and re-gated True/True on both
  bitexact scripts)
- `cargo test --lib fused_w1` → 3/3 pass (per-tier incl. new v4 arm)
- `cargo test --test fusedstep_equiv --release` → 19/19 pass (the
  equivalence test now runs under 9 token permutations on this host)
- `clippy --all-targets -D warnings` → initially FAILED
  (`type_complexity` on the Vec-based arms table, surfaced in every
  `#[path]`-including target); refactored to a plain `check()` closure
  → now clean (direct run rc 0)
- `bitexact.sh` True/True; `bitexact_uncapped.sh` True/True
- Uncapped interleaved A/B (dev Zen5, taskset -c 9, CONTENDED):
  base 9.734/7.510/9.799 → median 9.734;
  fused 10.892/7.527/9.454 → median 9.454.
  **−2.9% median ≈ parity** (pre-C1: fused ≈ 41 s, +315%).
  All six outputs mask-identical to the Zen2 golden.
- c1verify.log sha256: cf1d8adbbd2dab27

## Post-review text corrections (R1–R4, 2026-09-26)

- R1: commit message H32 blob hash corrected to `be6454032701ff0e…`
  (the `…f5145…` fragment belonged to the H128 blob).
- R2: cell-identity provenance noted — the H32/H128 cells ran on the
  pre-C1 build (11:26–12:41Z); the result transfers because under
  `ZENSIM_MAX_TIER=v3` the v4 arm is never dispatched and the v3 source
  is unchanged between the two builds.
- R3: C3 mutation claims restated as measured (above).
- R4: uncapped post-C1 result restated as "parity within contention —
  neither sign established" (my −2.9% median vs reviewer's +3.4%
  median / −4.2% min, overlapping ranges).
