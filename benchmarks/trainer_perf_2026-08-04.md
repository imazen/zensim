# Trainer performance: fresh profile + the honest ceiling (2026-08-04)

**Host** `lilith` — WSL2, AMD Ryzen 9 7950X, 28 cores visible, 58 GiB.
**Commit** `8099ea562c64`. **Binary** `zensim_mlp_train`, release, no `-C target-cpu=native`.
**Raw numbers** `benchmarks/trainer_perf_2026-08-04.tsv`.

This supersedes any pre-`0ce3e2f2` profile. The `idx / n_hidden` integer-divide fix
(43 % of cycles) is already in; everything below is measured on top of it.

## How the profile was taken

`perf record -F 499 -g -p <pid>` attached to a **live wave-10 trainer** for 35 s
(21,632 samples), plus `perf stat` for 20 s on the same process. Attaching to
production work rather than a synthetic run costs no extra RAM, takes no lane, and
profiles the recipe people actually run. Wave 10's recipe: 11 groups / 779,290 rows /
944 features, `--n-hidden-layers 0`, `--pairs-per-epoch 50000`, `--epochs 120`.

Note `--n-hidden-layers 0` does **not** mean "no hidden layer" — only `>= 2` selects a
different architecture, so the trained net is `944 → 128 (LeakyReLU) → 1`, confirmed in
the trainer's own banner. Adam therefore updates 120,832 f64 parameters per step.

## The profile

| Symbol | % cycles |
|---|---|
| `adam_simd::adam_update_inner_v4` | **44.39** |
| `simd_mlp::forward_avx512` | 23.57 |
| `mlp_train::train_mlp_strategy` | 10.66 |
| `simd_mlp::backprop_avx512` | 8.05 |
| `zenstats::panel::sa_st_curve` (PWRC) | 7.19 |
| `mlp_train::apply_coarse_decay` | 2.78 |
| `zenstats::panel::run_lm` | 1.17 |
| `zenstats::panel::rescale_logistic` | 0.58 |

Adam alone costs more than forward + backward combined.

## Why the "1–5 % of peak" framing misleads

The arithmetic that motivated this pass (≈3.6 GFLOP/s against a 50–100 GFLOP/s
single-thread peak) assumes the trainer is doing FLOPs. It is not. Two counters settle it:

- **IPC 1.46**, L1-dcache load-miss rate **17.48 %** (18.6 G misses / 106.3 G loads),
  LLC misses only 3.03 % of references (~3.7 GB/s to DRAM). The trainer streams
  ~59 GB/s out of L2/L3 and barely touches DRAM.
- `perf annotate` on the Adam kernel puts **53 % of the kernel in `vsqrtpd` + `vdivpd`**
  (15.6 % + 37.8 %, skid-attributed to the following instruction). That is ~23.7 % of
  total trainer cycles sitting on the FP divide unit.

So the trainer is **divider-port-bound and L2/L3-streaming-bound**, not FLOP-starved.
A GFLOP/s figure was never the right yardstick for this workload.

### The FMA question is already answered — that 25.5× does not transfer

`zenpredict::inference::forward` was 25.5×-able because it was scalar and emitting
software `fmaf`. This trainer is not in that state:

- `adam_simd.rs` is AVX-512 `f64x8` via archmage `#[arcane]`, with `vfmadd231pd` for
  both moment updates and hardware `vsqrtpd`.
- `simd_mlp.rs` is AVX-512 `f64x8` for forward and backprop.
- `simd_encoder.rs` carries a full f32 AVX-512/AVX2/NEON/WASM tier set.

There is no software-`fmaf` scalar loop left to convert. Applying the archmage
treatment again buys nothing here.

### Why the divides cannot simply be removed

The update is `w -= lr * m_hat / (sqrt(v_hat) + eps)`. Both `vsqrtpd` and `vdivpd` are
IEEE correctly-rounded and go to the same, poorly-pipelined divide unit (~9 cycles
throughput each per 512-bit op on Zen 4). Any `rsqrt`/`rcp` + Newton-Raphson substitute
lands within ~1 ULP but is **not bit-identical** — which is exactly why the existing
`adam_update_inner_v4_rsqrt` is opt-in and unused by default. Measured cost model:
15,104 chunks × ~18 cycles ≈ 272 k cycles per Adam call, which reproduces the observed
44 % share. **The divider cost is irreducible at fixed math.**

## What is actually available — measured, with prices

### 1. Minibatch K — 3.63×, but it changes the model

`minibatch_size` defaults to **1**, and `parallel = parallel_batch && k > 1`, so the
default path is fully sequential: one Adam step over 120,832 parameters *per pair*,
50,000 times per epoch. Raising K amortises Adam and turns on the rayon batch path.

Grid: kadid + tid + konjnd_val (13,125 rows), same 944-feature arch. Per-epoch time
derived as `(wall_9ep − wall_3ep) / 6`, which cancels dataset load and bake.

| K | s/epoch | speedup |
|---|---|---|
| 1 (default) | 8.199 | 1.00× |
| 8 | 3.660 | 2.24× |
| 32 | 2.258 | **3.63×** |

**Price, measured, not argued:** `verify_bake_identity.sh` on K=1 vs K=32 →
352,829 model bytes differ, `best_val` 0.95893 vs 0.96602. K>1 is a **recipe change**,
not a free optimization. It is the largest single-run lever available and it needs a
science decision (train matched seeds at K=1 and K=32, compare `bake_verdict`), not a
perf decision. On this evidence K=32 is worth that experiment.

### 2. Lane count is capped by memory, and half of that memory is dead

Exact accounting for wave 10's 779,290 rows × 944 features:

| | GB |
|---|---|
| raw `Vec<Vec<f64>>` (`TrainGroup.features`) | 5.91 |
| `std_features` flat f64 | 5.89 |
| predicted total | 11.80 |
| **measured RSS** | **11.88** |

The prediction matches measurement to 0.7 %, so the accounting is complete — there is no
third consumer.

The raw copy is **dead after standardization**. Verified by reading every `.features`
use in the strategy path after the `std_features` build at `mlp_train/mod.rs:1820`:
only `.len()` remains (the `t.features` hits belong to the TV regulariser, which wave 10
does not use). Half of every trainer's 11.88 GB is feature data that is never read again.

This is the binding constraint on throughput: the box has 28 cores and runs 3 lanes,
because 4 lanes × 11.88 GB ≈ 47.5 GB against 58 GiB shared with several agents. **The
machine is CPU-idle and RAM-bound.** Removing the dead copy would take a lane from
11.88 GB to ~5.9 GB and roughly double lanes per box at zero math change.

Two routes, neither landed here (see "not done"):
- Free the raw rows after standardization — needs `&mut [TrainGroup]` and a cached
  row-count vector, because `g.features.len()` is read in the hot loop of four trainer
  variants.
- Store raw features as `f32` and widen at standardization. Bit-identical **only** when
  the parquet column is `Float32` (widening f32→f64 is exact); needs a dtype guard, and
  `std_features` must stay f64 since the standardized value is an f64 division result.

> **LANDED 2026-08-04 — and route 1 above was tried first and MEASURED-DEFEATED.**
> Freeing each row `Vec` as it is standardized empties the rows at the Rust level but
> moved full-recipe peak RSS only 11.94 → 10.97 GB: the ~7.5 KB row chunks are allocated
> interleaved across the loaders' glibc arenas, and once freed they become interior
> free-list holes that never return to the OS (the standardized buffers are fresh mmaps
> that cannot reuse them). What shipped instead: the trainer bin flattens each group to
> ONE row-major `Vec<f64>` at the loader boundary, and standardization TAKES that buffer
> and transforms it in place — the run never materializes a second copy at all, and no
> row-count vector was needed (`FeatureRows` caches `n_rows`, which is all
> `g.features.len()` reads). Full method, both attempts' measurements, and the full-data
> bit-identity gate:
> [`trainer_mem_release_2026-08-04.md`](trainer_mem_release_2026-08-04.md).

### 3. PWRC is O(n²) and runs every epoch

`sa_st_curve` is 7.19 % of cycles. It is computed per epoch per validation group and
feeds `GeomeanSPP` model selection, so it is load-bearing — it cannot simply be dropped.
It is already rayon-parallel above `PAR_MIN_N = 512` at the pinned zenstats rev
(`29f1d61e` *is* the parallelisation commit), so there is no free flag to flip. The cost
is structural: wave 10 validates on 208 k-row groups, i.e. ~2.2 × 10^10 pair-operations
per group per epoch. `group_eval_cap` (currently 0 = uncapped) would cut it, but capping
changes which epoch wins and therefore the bake — another recipe decision.

## Bit-identity gates

Run with the committed `scripts/verify_bake_identity.sh`.

| Gate | Result |
|---|---|
| Same recipe + seed, two runs, same `--out` path | **PASS** — 492,125 model bytes identical, only `timestamp_epoch` differs, `best_val` identical |
| K=1 vs K=32 | **FAIL as expected** — 352,829 model bytes differ |

**Gate usage caveat, learned the hard way:** comparing two runs written to *different*
`--out` paths reports a 1-byte diff at offset 68. That is the ZNPR section-length table
shifting because the embedded `argv` string lengths differ — the model is identical and
`best_val` matches. The script prints a hint saying so. Always run both trainings to the
**same** `--out` path and move the file between runs.

## Fleet assessment

| Node | Cores | RAM | ISA tier | Lanes (11.88 GB each) |
|---|---|---|---|---|
| wsl (7950X) | 28 | 58 GiB | AVX-512 | 3 live, ~4 max |
| lianli (7900X) | 24 | 29 GiB (7 GiB free when probed) | **AVX-512 — same tier as wsl** | 2 when idle, 0 as found |
| tower (TR 2950X) | 32 | 62 GiB / 34 avail | **AVX2 only** (`avx512f` = 0) | 2–3, Docker-only, media priority |

Two findings that matter more than the core counts:

- **lianli is a drop-in identical-bits node.** It is Zen 4 with AVX-512, so archmage
  summons the same `X64V4Token` tier as this box. Bakes produced there should be
  byte-comparable with local ones under the same gate.
- **tower is not, without a test.** It is AVX2-only, so it dispatches `..._v3` (f64x4)
  instead of `..._v4` (f64x8). For *this* architecture the kernels look tier-invariant —
  Adam is elementwise, forward accumulates per-`j` with the same order over `i`, and
  `forward_simd` deliberately keeps the final reduction scalar "so the public output is
  bit-identical". That is an inference from the source, **not a measurement**. Before
  enrolling tower for training, run the same recipe on both tiers and diff with
  `verify_bake_identity.sh`. (The per-sample-α head is the known risk: `dot_bias_f32`
  does a cross-lane reduction whose order *does* change with lane count.)

### Should training go through zenfleet? Not yet — and here is the N

Per-run cost is ~20 min. Adding a `JobKind::Train` plus an executor image is real work:
a new job kind and ledger schema, an image carrying the trainer, staging ~6 GB of
parquets to each node, and a cross-ISA identity test before tower can be trusted.
Call it 1.5–2 days.

The ad-hoc two-lane ssh pattern stays cheaper until a wave exceeds roughly
**60–80 runs**, i.e. the point where saved wall-time (~3 extra lanes × 20 min per run)
repays two days of build. Today's 23-run LOO sweep at ~3.7 h on two lanes is well under
that. **Recommendation: do not build the zenfleet training path yet.**

The cheaper ordering, in payback order:

1. **Cut the dead 5.91 GB** (§2). Doubles lanes per box, zero math change, no new
   infrastructure. A 23-run wave goes from ~3.7 h to ~1.9 h on this box alone.
2. **Add lianli as a second ssh lane host** — same ISA tier, no image, no ledger.
   With (1) it contributes ~4 lanes instead of 2.
3. **Then** reconsider zenfleet, once a wave is large enough to pay for it.

Marginal value of a node is currently `min(RAM/11.88, cores)` lanes — which is why
lianli's 24 cores are worth only 2 lanes and why (1) is worth more than any node.

## What was deliberately not done

- **No `rsqrt`/`rcp` Adam substitution.** It is the obvious way to attack the 23.7 % of
  cycles on the divide unit, and it breaks bit-identity. The opt-in kernel already
  exists if that trade is ever wanted.
- **No f32 optimizer state or f32 `std_features`.** Same reason — halves the streaming
  traffic, changes every published number.
- **No K default change.** Measured at 3.63× and proven to change the model; that is a
  science call.
- **The dead-copy removal was not landed *in this pass*.** It touches `g.features.len()`
  in the hot loop of four trainer variants across a 12,738-line module, and it needed to
  land with a full-data identity gate. Wave 10 was mid-sweep on both lanes, so a
  full-data A/B could not be run cleanly, and shipping an invasive refactor to a 12 k-line
  trainer without its own gate is exactly the failure mode this repo's rules exist to
  prevent. **Landed later the same day with that gate** —
  [`trainer_mem_release_2026-08-04.md`](trainer_mem_release_2026-08-04.md); see the
  correction note in §2 (freeing *after* standardization would not have moved the peak).
- **`apply_coarse_decay` fusion into the Adam loop** (2.78 %, and it would be
  bit-identical since per-element order is preserved) was scoped and dropped: it adds a
  range test to the hottest loop in the trainer to chase 2.78 %, and the fused branch
  risks slowing the common path.
