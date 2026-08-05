# Trainer memory: removing the dead raw-feature copy (2026-08-04)

Lands the highest-value item from [`trainer_perf_2026-08-04.md`](trainer_perf_2026-08-04.md)
§2 — "half of every lane's RAM is dead" — with its full-data bit-identity gate,
and documents the first attempt that **did not work**, because the reason it
failed is the durable lesson here.

**Host** `lilith` — WSL2, Ryzen 9 7950X, 28 cores, 58 GiB, shared with a live
wave-10 sweep. **Baseline** `ae2a3838`. **Change** see the commit carrying this
file. **Binary** `zensim_mlp_train`, release, no `-C target-cpu=native`.

## What was dead

The trainer held two full copies of the feature matrix for the whole run: the
raw rows the caller handed it, and the flat standardized buffer it derives from
them. Both are `n_rows × n_features × 8 B`. For wave 10's 11-group recipe
(779,290 rows × 944 features) that is 5.91 GB + 5.89 GB, which is the entirety
of a lane's ~11.9 GB RSS.

The raw copy is dead the instant it has been standardized. Audited directly,
not taken on the report's word: across all four heads (plain / pool / hybrid /
per-sample-α), every read of `TrainingGroup::features` after the
standardization call is `.len()` — the row count the pair sampler draws
against. There is no post-standardization read of a row's values.

## Attempt 1 — free each row `Vec` as it is standardized: MEASURED-DEFEATED

The report's route 1: keep the loader's per-row `Vec<Vec<f64>>`, and have the
standardization loop drop each row's allocation right after copying it into
the flat buffer. Rust-side it works — the rows verifiably empty (a unit test
asserted it). Allocator-probe stand-ins (clean single-thread allocation of
400k × 944 rows, then the flat buffer, `VmRSS`/`VmHWM` sampled from
`/proc/self/status`) said the peak would halve: 5.64 → 3.05 GB.

**On the full wave-10 recipe it moved peak RSS by ~0.4 GB** (11.94 → 10.97 GB
by `/usr/bin/time -v` max RSS; at epoch 0 the run already sat at 10.88 GB):
the trainer freed 5.9 GB of rows and the OS got almost none of it back. The
probes had allocated their rows cleanly on one thread in one glibc arena,
where freed chunks coalesce and trim. The real loaders don't: row `Vec`s
(~7.5 KB each — heap chunks, far under the mmap threshold) are allocated
interleaved with parquet/arrow scratch and across rayon threads' arenas, so
the 779k freed chunks become **interior free-list holes** that glibc cannot
return, while the standardized buffers are fresh mmaps that cannot reuse
them. RSS keeps counting the holes.

**Lesson (why this file exists):** for scattered small allocations, "freed at
the Rust level" and "returned to the OS" are different facts, and only the
second one raises lane count. An allocator probe that builds its allocations
cleanly will overstate the win; gate memory claims on the real process.

## What shipped — one flat buffer per group, standardized IN PLACE

Two structural changes, which together mean the two copies never both exist:

1. **The trainer binary's `LoadedGroup` stores each group's features as ONE
   flat row-major `Vec<f64>`**, flattened at the loader boundary (dropping
   each row `Vec` as it is consumed — the freed same-size chunks are
   immediately reused by the *next* group's loader rows, so the per-row stage
   stays bounded by one group instead of the dataset).
2. **`TrainingGroup::features` becomes `FeatureRows`**: `Borrowed`
   (read-only row table — tests and in-process callers, nothing consumed) or
   `Releasable` (the flat buffer + cached `n_rows`/`n_features`). The four
   heads' four byte-identical standardization loops collapse into
   `standardize_groups_releasing_raw`, which **takes** a `Releasable` buffer
   (`std::mem::take`) and standardizes it in place: each element is read,
   transformed, and written back to the same slot. No second copy is ever
   allocated, so there is nothing for the allocator to fail to return.
   `len()` reports the cached `n_rows`, which is all the hot loop reads after
   standardization.

Bit-identity is structural: same expression `(x − mean[d]) / scale[d].max(1e-12)`,
same element order, each raw value read before its slot is overwritten; row
values are never modified anywhere else. Flattening preserves value order
exactly (`flat[i·nf+d] == rows[i][d]`), and `--max-features` narrowing keeps
the same kept-values semantics as the old per-row `truncate`.

Also removed on the same path: the `--feature-transform` NaN/Inf sweep
deep-cloned each group (`g.feature_rows.to_vec()`) to hand `sweep_nan_inf` a
copy it never mutates — up to ~1.6 GB transient on the largest wave-10 leg.
It now reads row views of the flat buffer (`sweep_nan_inf` takes any iterator
of row slices and does one pass instead of one pass per checked feature).

## Bit-identity gate — FULL data, full recipe

`scripts/verify_bake_identity.sh` on the complete wave-10 L0 argv
(`WAVE10_ECHO=1 scripts/wave10_seed.sh L0 4001`): 11 groups, 779,290 rows,
944 features, 120 epochs, seed 4001. Not a reduced fixture — the dead copy
only exists at scale. All runs used the **same binary path and the same
`--out` path** (binary copied over a fixed name, bake moved aside between
runs); this is load-bearing — the ZNPR section-length table embeds `argv`, so
two runs whose paths differ report a spurious 1-byte diff at offset 68.

Three arms against one baseline bake: `base` (commit `ae2a3838`),
`attempt-1` (per-row release), `flat` (shipped). Gate outputs verbatim:

```
# base (ae2a3838) vs FLAT (shipped)
A = /home/lilith/tmp/memfix/gate/bake_base.bin
B = /home/lilith/tmp/memfix/gate/bake_flat.bin
model bytes (outside zentrain.repro): IDENTICAL (502471 bytes)
repro non-volatile fields: IDENTICAL (ignored provenance keys that differ: ['timestamp_epoch'])
best_val: 0.7363318750190634 vs 0.7363318750190634 -> SAME
RESULT: PASS — same model

# base (ae2a3838) vs attempt-1 (per-row free)
A = /home/lilith/tmp/memfix/gate/bake_base.bin
B = /home/lilith/tmp/memfix/gate/bake_new.bin
model bytes (outside zentrain.repro): IDENTICAL (502471 bytes)
repro non-volatile fields: IDENTICAL (ignored provenance keys that differ: ['timestamp_epoch'])
best_val: 0.7363318750190634 vs 0.7363318750190634 -> SAME
RESULT: PASS — same model

# attempt-1 vs FLAT
A = /home/lilith/tmp/memfix/gate/bake_new.bin
B = /home/lilith/tmp/memfix/gate/bake_flat.bin
model bytes (outside zentrain.repro): IDENTICAL (502471 bytes)
repro non-volatile fields: IDENTICAL (ignored provenance keys that differ: ['timestamp_epoch'])
best_val: 0.7363318750190634 vs 0.7363318750190634 -> SAME
RESULT: PASS — same model
```

The in-CI version of the same property is
`releasable_rows_train_identically_to_borrowed_and_are_released` (mlp_train
unit tests): Borrowed-vs-Releasable arms must produce byte-identical bakes and
the Releasable buffer must actually be taken.

## Measured

Full wave-10 L0 recipe (11 groups / 779,290 rows × 944 features / 120 epochs),
one run at a time on the shared box, `/usr/bin/time -v` for peak RSS, a 2 s
`/proc/<pid>/status` sampler for phase attribution, per-epoch time from the
trainer's own `t=` log line (Δt over 119 epochs — concurrent wave-10 lanes on
the box add noise; treat small deltas as neutral):

| arm | peak RSS (`time -v`) | RSS at epoch 0 | s/epoch | wall |
|---|---|---|---|---|
| base `ae2a3838` | 11.38 GiB (11,935,940 kB) | 11.33 GB | 8.57 | 18:18.90 |
| attempt-1 (per-row free) | 10.97 GiB (11,504,800 kB) | 10.88 GB | 7.97 | 17:20.84 |
| **flat, in-place (shipped)** | **8.12 GiB (8,514,140 kB)** | **7.46 GB** | 8.04 | 17:21.93 |

Phase attribution for the shipped arm: the load phase runs at ~5.2-5.7 GB
(the raw flat buffers, with the per-row loader stage bounded by one group via
chunk reuse), and the peak (8.12 GB, briefly, during load) is the transient where the
largest group (`ttbig`, 208k rows ≈ 1.57 GB) exists as loader rows and flat
buffer at once during flattening, on top of the groups already flattened.
Training then settles at ~7.4-7.5 GB — the 5.89 GB standardized matrix plus
eval scratch and the load-phase arena retention that has no successor
allocation to reuse it.

**Lanes.** The box is 58 GiB shared with agents and a live sweep; wave 10 ran
**3 lanes at 11.9 GiB each** (perf report: "3 live, ~4 max" — 4 × 11.88 ≈
47.5 GiB is the practical envelope). At 8.12 GiB peak per lane that envelope
packs **5 lanes with worst-case-simultaneous peaks, 6 with staggered starts**
(the 8.12 peak is a ~90 s load transient; the 17-minute steady state is
7.46 GB, and 6 × 7.46 = 44.8 GiB fits). So the box goes from 3-4 lanes to
5-6 — a 23-run wave at ~17.4 min/run drops from ~2.3 h on 3 lanes to ~1.2 h
on 6.

**Next lever — TAKEN 2026-08-05 (`d9e336ec`): the loader emits flat.**
`load_parquet_impl` = one shared scan behind two shapes: `load_parquet`
(per-row `Vec<Vec<f64>>`, unchanged signature and values — the compat shape
every non-trainer consumer keeps) and the new `load_parquet_flat` →
`OwnedLoadedGroupFlat` (ONE row-major buffer, pre-reserved exactly from the
parquet footer's `num_rows` so it is allocated once and never
realloc-copied). The trainer's parquet branch adopts the flat shape;
`From<OwnedLoadedGroupFlat>` is a field move, so the per-group rows+flat
flatten transient (this file's remaining peak) no longer exists.

Gates, measured on the **wave-11 recipe** (seed-4101 argv extracted from the
production bake's embedded `zentrain.repro`: 10 groups, 729,290 rows × 944,
120 epochs — wave-10 L9 with the corrected KADID table and `tkadis`
dropped), same methodology as the table above (same binary path, same
`--out` path, both runs from one workspace state; quiet box):

```
# scripts/verify_bake_identity.sh w11_old.bin w11_new.bin
model bytes (outside zentrain.repro): IDENTICAL (502471 bytes)
repro non-volatile fields: IDENTICAL (ignored provenance keys that differ: ['timestamp_epoch'])
best_val: 0.9197662869686666 vs 0.9197662869686666 -> SAME
RESULT: PASS — same model
```

The new arm's bake also matches the LIVE wave-11 production bake
`W11_s4101.bin` byte-for-byte outside provenance (1 byte at offset 68 = the
documented section-length shift from different `--out` argv lengths, plus
`trainer_head_at_train`; `best_val` bit-identical) — the loader change
reproduces what the wave actually shipped. Loader-level bit-identity is
separately gated by `tests/parquet_flat_equivalence.rs` (`to_bits` equality
on every element; generated fixture + TID 3,000×372 + ext944 KADID
10,125×944 all bit-identical).

| arm | peak RSS (`time -v`) | wall |
|---|---|---|
| rows loader (`22e37ce3`, pre-change) | 8.12 GiB (8,510,208 kB) | 11:58.79 |
| **flat loader (`d9e336ec`, shipped)** | **7.10 GiB (7,442,616 kB)** | 12:17.05 |

**−1.02 GiB (−12.5%).** The earlier ~6.2 GB estimate was optimistic: with
the load transient gone the peak moves to the training steady state itself
(standardized matrix + eval scratch — the ~7.4-7.5 GB regime the phase
attribution above measured for L0; wave-11's smaller mix lands at 7.10),
which the loader cannot reduce further. Wall-clock delta (+18 s) is within
single-run noise. Consumers other than the trainer keep the per-row shape
and are byte-unchanged.

## Not changed

- No change to the Adam kernel, the SIMD paths, the RNG, the sampler, or any
  hyperparameter. The `rsqrt` Adam substitution stays opt-in and unused.
  (`rsqrt_path_precision_vs_scalar` fails identically at `ae2a3838` and on
  this change — pre-existing, untouched by this work. *Resolved 2026-08-05,
  `22e37ce3`: the failure began at `aaf9b808`'s archmage/magetypes lock
  movement 0.9.26→0.9.28, whose 0.9.27 reciprocal-contract change turned the
  kernel's `_approx`+1-NR shape from ~full precision into ~28-bit; the kernel
  was repaired to the new `rsqrt()`/`recip()` contract and the bound was NOT
  moved — measured 1.117e-12 vs the 1e-9 gate, with the 0.9.26/0.9.28 A/B
  matrix in the `chore: archmage/magetypes minimum 0.9.28` commit.*)
- `parquet_loader::load_parquet` still returns per-row `Vec`s — it has many
  non-trainer consumers (bake_verdict and friends); the trainer bin flattens
  at its own `LoadedGroup` boundary. Anchor / pjnd / konjnd-aggregation /
  equiv / triplet pools keep their per-row storage — they are small next to
  the training groups.
- The GPU path (`--gpu-runtime`) materializes a borrowed row table for
  `zensim_train_core::TrainingGroup` (16 B/row, only on that path); its rows
  are not consumed.
