# TRAINEROPT3 worklog — fuse the pair forward into the fused Adam row walk

Lane: SWE-2 (Devin), 2026-10-02. Bookmark `quarantine/swe2/traineropt3` (local only, never push).
Base: `main@origin` a62edff3. Target dir: `/var/tmp/traineropt3/target`. Scratch: `/var/tmp/traineropt3/`.
Every build/test/bench: `taskset -c 8-15 nice -n 19 ~/work/zen/scripts/run-heavy --mem 16G --jobs 4 -- ...`
Gate reference: `/var/tmp/fitv2/bin-v6/zensim_mlp_train` sha256 f9d076c6d68374404a28370d1447f9697958b3589244a673a55ae9d19ecc8451.

## Recon findings (source-verified)

- Pair loop: `mlp_train/mod.rs` `for epoch` / `for pair_i` at ~2710. Draw via `sampling::draw_pair` (RNG
  contract documented at sampling.rs:282-289); the ONLY other RNG consumer in the loop is the TV draw at
  ~3072. Non-Pair draws (`GroupTooSmall`, `SameRow`) consume RNG and `continue`.
- Forward pair site ~2821-2842: `forward(xa)` + `forward(xb)` back-to-back, then loss/drop checks, then
  the `fuse_w1` branch (heads via `backprop_grad_head` → `step_w1_fused` ~2980 → optional
  `apply_post_adam_penalties` (coarse_on only) → `nonneg_project`).
- `step_w1_fused` (~6987): fused w1 walk FIRST, then adam(b1), adam(w2), adam(b2) — all disjoint arrays.
- `nonneg_project` (~1283): `b1 := 0.0` (+0.0), `w2 := min(w,0)`, `b2 := pin`. b1's projected value is
  known statically (+0.0) → lookahead seed needs no projection ordering, only the flag.
- Keep-mask: bin zeroes dropped raw columns BEFORE the scaler (bin ~3452-3481) ⇒ standardized dropped
  columns are exactly `+0.0` (mean 0, `(0−0)/1e-8` = +0.0; LazyStd template likewise). `forward`'s
  `s == 0.0` skip always bypasses them; `-0.0` also skipped.
- `active_rows` skips rows where w=m=v=g=+0.0 AND xa==xb==0.0 — same invariant covers lookahead x_next.
- `forward` arithmetic: per-lane ascending-row chain `h_pre[j] = fma(x[i], w1[i,j], h_pre[j])` on the
  fused domain `[0, nh−nh%4)`, mul+add tail; hpre seeded `b1.to_vec()`; finish = leaky + 4-lane fused
  accumulator + pairwise `(a0+a1)+(a2+a3)` + sequential tail + `b2[0] + lane_sum + tail_sum`.
- TV draws consume rng AFTER the step — lookahead must be gated off whenever TV can fire
  (`weight>0 && apply_every>0 && pairs nonempty && tv_std.is_some()`), plus `coarse_decay_rate() > 0`
  (penalty pass writes w1 after the step). Everything else between the step and the next forward
  (nonneg, eval, checkpoints) is read-only on w1 or handled via the b1-zeroed seed flag.

## Design

- Task 1: `forward_pair(xa, xb, ...)` — one ascending row walk, each w1 row loaded once, applied to
  side A then side B per the `x != 0.0` skips. scalar/avx2/avx512 twins; shared `forward_finish`
  (leaky+reduce) extracted so the lookahead path can finish an already-accumulated h_pre.
- Task 2 (Option B — pre-draw, no deferred step): inside `if fuse_w1` when `lookahead_ok` and
  `pair_i+1 < ppe`, draw pair p+1 early (same RNG sequence — nothing consumes between step_p and the
  next draw; TV gated off), copy its standardized rows into owned bufs, and pass `FwdAccumW1` into
  `step_w1_fused`. The step runs adam(b1) FIRST (disjoint arrays — w1 walk never reads b1), seeds
  hpre with final b1 (+0.0 when nonneg pins it), then the w1 walk applies each post-Adam/post-prox
  row to hpre_a/hpre_b. Consume at next iter top; `forward_finish` on current w2/b2 reproduces the
  baseline's forward at that program point. Never crosses an epoch boundary.

## Implementation status (2026-10-02 ~18:20Z)

- Task 1 DONE: `simd_mlp.rs` `forward_pair_scalar`/`forward_pair_avx2`/`forward_pair_avx512`/dispatch +
  `finish_{scalar,avx2,avx512}` extracted + `forward_finish`. Wired at all four pair sites:
  main sequential loop, TV block, parallel-chunk loop, `PairForward` accumulation. Tests:
  `forward_pair_all_tiers_bit_identical`, `forward_finish_matches_forward_tail` — pass.
- Task 2 DONE (conditional): `adam_simd::FwdAccumW1` + `fwd_accum_row_{v3,scalar}`; v3 body applies
  each row post-Adam (no prox) or post-prox per 8-row block; composition fallback applies post-prox
  post-pass over all rows (x==0 guard reproduces `forward`'s skip regardless). `step_w1_fused`
  gained `nonneg_pin`+`next_fwd`; armed mode runs adam(b1)+`nonneg_project_b1` BEFORE the w1 walk
  (disjoint arrays — bit-identical), seeds hpa/hpb from final b1, caller runs
  `nonneg_project_w2b2` only. Loop: `lookahead_ok = fuse_w1 && !parallel && !nin_on && !tv_can_fire`
  (+ `!coarse_on` per step + not-last-pair); stash carries `drawn` + optional `LookaheadFwd`;
  consume replays drawn and either `forward_finish`(armed) or `forward_pair`(unarmed).
- `nonneg_project` split into `nonneg_project_b1` + `nonneg_project_w2b2` (elementwise, disjoint).
- Oracle tests pass: `fused_w1_fwd_accum_bit_identical` (v3+fallback × prox × active_rows × L2,
  armed state bit-identical + hpa/hpb == forward replay on final w), `step_w1_fused_lookahead_
  bit_identical` (16 buffers incl. all Adam state, nonneg × prox × active, 3 chained steps),
  `fwd_skipped_rows_zero_contract`.
- `cargo fmt -p zensim-validate --check`: clean. `just lint-scripts`: 793 checked, clean.
  `cargo clippy -p zensim-validate --all-targets`: clean. `just clippy` (workspace, -D warnings):
  running.
- Fixed-test lesson: fallback walks all rows (ignores `active_rows`), so the loader's "unkept ⇒
  x == +0" invariant must hold for BOTH pairs in tests — nonzero x on unkept rows legitimately
  updates them in the fallback.

## Pending

- just clippy (workspace) result; cargo test --release; release binary build; quick gate (e4, a few
  cells) then full gate (≥10 cells, 120ep on ≥4 incl. ≥2 N + ≥2 group-l1, one AVX-512 uncapped run);
  benchmarks 3 specs × ±gl; DONE report; local commit on quarantine/swe2/traineropt3.

## Quick gate (e4, ZENSIM_MAX_TIER=v3 unless noted) — 2026-10-02 ~18:50Z

new binary `/var/tmp/traineropt3/bin/zensim_mlp_train` sha256 605d20e0ecebb1586779e2cf586818e24d12db88feca0f225ecc09d3636d1033
vs reference `/var/tmp/fitv2/bin-v6/zensim_mlp_train`. All 12 cells weights+pred+dev IDENTICAL:

| spec | head | heldout | seed | H | gl | kept | eps | match |
|---|---|---|---|---|---|---|---|---|
| r0 | N | konfig | 1 | 128 | off | 944 | 4 | IDENTICAL |
| r0 | F | aic3 | 2 | 64 | off | 944 | 4 | IDENTICAL |
| r0 | N | kadid | 0 | 128 | 1 | 944 | 4 | IDENTICAL |
| r0 | F | konfig | 3 | 128 | 2.8 | 944 | 4 | IDENTICAL (log-every 8) |
| r0 | N | kadid | 0 | 128 | off | 944 | 4 | IDENTICAL (AVX-512, cap lifted both sides) |
| screen_main | N | kadid | 0 | 128 | off | 1853 | 4 | IDENTICAL |
| screen_main | F | tid2013 | 1 | 64 | off | 1853 | 4 | IDENTICAL |
| screen_main | F | aic3 | 1 | 128 | 2.8 | 1853 | 4 | IDENTICAL |
| csfw | N | cid22_a25 | 2 | 128 | off | 956 | 4 | IDENTICAL |
| csfw | N | tid2013 | 2 | 64 | 1 | 956 | 4 | IDENTICAL |
| minus_basic | F | konfig | 3 | 128 | off | 716 | 4 | IDENTICAL |
| minus_basic | N | kadid | 1 | 64 | 2.8 | 716 | 4 | IDENTICAL |

Gate script gained a `GATE_MAX_TIER` env passthrough (`""` = uncapped) for the AVX-512 run;
default remains v3.

## Final gate (120 epochs, running ~19:00Z)

| spec | head | heldout | seed | H | gl |
|---|---|---|---|---|---|
| screen_main | N | kadid | 0 | 128 | 2.8 |
| r0 | N | tid2013 | 1 | 128 | 1 |
| screen_main | F | aic3 | 2 | 64 | off |
| csfw | N | konfig | 3 | 64 | 2.8 |

## Paired interleaved benchmarks (e4, log-every 100, box loaded w/ gates — ratios)

3 reps per config, alternating variant order (B,N / N,B / B,N). wall_s = trainer
subprocess only (predict excluded).

| spec (H128) | gl | base walls s | new walls s | ratio |
|---|---|---|---|---|
| minus_basic (716) | off | 39.28 / 40.06 / 38.38 | 38.17 / 39.83 / 37.52 | ~1.02 |
| minus_basic (716) | 1.4 | 59.68 / 54.55 / 53.60 | 51.83 / 50.72 / 52.76 | ~1.09 |
| r0 (944) | off | 69.00 / 67.65 / 67.60 | 60.00 / 61.01 / 63.44 | ~1.11 |
| r0 (944) | 1.4 | 97.11 / 90.75 / 91.84 | 74.50 / 84.60 / 73.60 | ~1.19 |
| screen_main (1853) | off | 298.27 / 304.52 / 283.07 | 277.94 / 280.27 / 246.60 | ~1.10 |
| screen_main (1853) | 1.4 | 300.86 / 290.96 / 287.56 | 305.52 / 304.26 / 281.56 | ~0.99† |

† screen_main+gl1.4 reps ran on the same 2-core band as the running 120-epoch
screen_main gate (3 procs / 2 cores) — heavy timeslicing; the quick-gate e4
single-shot walls for the same kind of cell (gl 2.8, different seed/heldout)
read 1.19× (289.5 → 242.5), and the 120-epoch final gate walls give the
least-loaded signal. Ratios under sustained load, not quiet-box numbers.

## 120-epoch final gate — walls so far

| spec | head | gl | base s | new s | ratio |
|---|---|---|---|---|---|
| csfw H64 | N | 2.8 | 1165.4 | 1088.3 | 1.07 |
| screen_main H64 | F | off | 1937.0 | 1534.4 | 1.26 |
| r0 H128 | N | 1 | 2429.8 | 2163.3 | 1.12 |
| screen_main H128 gl2.8 | N | 2.8 | pending | | |

All 120-epoch cells so far byte-identical (weights sha, pred sha, all 120 dev
values). screen_main H128 gl2.8 still running its base variant ~19:35Z.

## Final gate complete — 2026-10-02 ~22:09Z

All four 120-epoch cells byte-identical (weights sha + pred sha + all 120 dev values):

| spec | head | heldout | seed | H | gl | base s | new s | ratio |
|---|---|---|---|---|---|---|---|---|
| screen_main | N | kadid | 0 | 128 | 2.8 | 7981.4 | 6250.2 | **1.28×** |
| r0 | N | tid2013 | 1 | 128 | 1 | 2429.8 | 2163.3 | **1.12×** |
| screen_main | F | aic3 | 2 | 64 | off | 1937.0 | 1534.4 | **1.26×** |
| csfw | N | konfig | 3 | 64 | 2.8 | 1165.4 | 1088.3 | **1.07×** |

Grand total: 16 gate cells, 16/16 IDENTICAL. 120-epoch runs confirm the fusion is
exact at full depth including group-lasso epochs, epoch boundaries, LR schedule, dev
evals and checkpointing in between steps.

## Follow-up: set:/sel: scattered-keep coverage — 2026-10-02 ~22:30Z

Coordinator follow-up: production E9″ cells are `set:` specs and E9′ `sel:` specs —
scattered kept rows with large `active_rows` gaps (where `fwd_skipped_rows_zero` and the
in-walk apply carry the load). Gate change: `resolve_keep(core_spec, args.columns, lists)`
(zensim bfd22c24, in base) replaces direct `lists["specs"]` indexing; new `--columns`
passthrough validates sel: subsets via `v2_common.selection_id`.

Quick gate (e4, ZENSIM_MAX_TIER=v3), all IDENTICAL:

| spec | head | heldout | seed | gl | kept | result |
|---|---|---|---|---|---|---|
| set:csfw | N | konfig | 0 | off | 12 | IDENTICAL |
| set:v2+basic | F | aic3 | 1 | off | 504 | IDENTICAL |
| set:v2+basic | N | konfig | 2 | 1.4 | 504 | IDENTICAL |
| set:v2+basic+c3 | N | tid2013 | 2 | off | 648 | IDENTICAL |
| set:v2+basic+p3 | F | kadid | 3 | 1.4 | 686 | IDENTICAL |
| sel:57ac1bb6d6bd | N | cid22_a25 | 1 | off | 289 | IDENTICAL |
| sel:57ac1bb6d6bd | F | konfig | 0 | 1.4 | 289 | IDENTICAL |

sel:57ac1bb6d6bd = every 7th column of 0..1852 plus 1825..1852 (289 scattered columns,
maximal gap pattern). Harness hiccup: first sel: N run lacked exported COLS (shell var
not env) — reran with prefix form, IDENTICAL.

120-epoch final cells running: set:v2+basic+c3 N (gl off) on 8-11, sel:57ac1bb6d6bd F
gl1.4 on 12-15; ~650-kept paired bench (set:v2+basic+c3) on 12-15 after a head start.

## Follow-up complete — 2026-10-02 ~22:40Z

120-epoch scattered-keep cells, both IDENTICAL (weights + pred + all 120 dev values):

| spec | head | heldout | seed | gl | kept | base s | new s | ratio |
|---|---|---|---|---|---|---|---|---|
| set:v2+basic+c3 | N | kadid | 0 | off | 648 | 802.1 | 584.9 | **1.37×** |
| sel:57ac1bb6d6bd | F | konfig | 0 | 1.4 | 289 | 456.9 | 401.8 | **1.14×** |

Paired interleaved bench, set:v2+basic+c3 (~650 kept, H128, no gl, e4, le=100):
base 27.06 / 19.65 / 20.33 vs new 21.65 / 20.33 / 21.39 — r1 cold-start outlier;
r2-r3 ~1.0 under gate contention. The same cell at 120 epochs reads 1.37× —
the scattered-keep shape is where the fused pass pays most.

Grand total: 25 gate cells, 25/25 IDENTICAL (12 contig + 7 set:/sel: at e4 incl.
one uncapped AVX-512; 6 cells at full 120 epochs).
