# CPU 372-feature extraction is NOT a memory hog — measured 2026-06-25

## Hypothesis under test

zenmetrics commit `4bb5febf` (2026-06-25) asserts:

> "The CPU zensim **372-feature extraction** is the per-image memory hog that
> OOMs large frames even at SWEEP_JOBS=2."

and drops `--feature-output` from the Hetzner CPU sweep on that basis.

The code contradicts the claim: the sweep's feature path
(`zenmetrics .../metrics/zensim.rs:55 score_with_features` →
`Zensim::compute_extended_features`) funnels through the **same** streaming
inner function as the plain score (`zensim metric.rs:2200`
`compute_with_config_core` → `streaming::compute_zensim_streaming`).
`extended_features` / `compute_iw_features` only add **fixed-size
accumulators** (`streaming.rs:325-334`, `[f64;3]` per channel), not full-image
maps. The prior heaptrack TSV (`refresh_2026-05-28`) only ever measured the
228 `compute` score (`cpu_profile/src/main.rs:341`), never the 372 path — so
the claim was never measured.

## Method

`zensim-bench/examples/peak_entry.rs` isolates ONE call per process so
`heaptrack` reports a clean process peak. Replicates the sweep's exact call
(`StridedBytes` + `Srgb8Rgb` + `ZensimProfile::latest_preview()` +
`compute_extended_features`). Three public entries × 4 sizes. Parse + bytes
conversion lifted from the 2026-05-28 harness for comparability. Runner:
`run.sh`. Built + run under `run-heavy` (peak-RSS 1.85 GiB, 6 s total).

## Result (peak heap, heaptrack 1.3.0)

| entry  | 1 MP      | 4 MP       | 12 MP      | 30 MP        |
|--------|-----------|------------|------------|--------------|
| score  | 61.5 B/px | 46.3 B/px  | 41.5 B/px  | 38.3 B/px    |
| ext372 | 61.5 B/px | 46.3 B/px  | 41.5 B/px  | 38.3 B/px    |
| strip  | 63.6 B/px | 66.7 B/px  | 68.4 B/px  | 70.9 B/px    |

`ext372` peak_heap_bytes is **byte-identical** to `score` at every size
(1,148,903,751 B at 30 MP for both). Score values match too
(strip score == full score, bit-exact).

## Verdict

1. **`4bb5febf` is FALSE / misattributed.** The 372-feature CPU extraction is
   NOT a per-pixel memory hog — it streams at the SAME ~38 B/px asymptotic as
   the score. At 30 MP it peaks 1.07 GB; a "~1 KB/px" reading would be ~30 GB.
   Streaming is honored, including for features.

2. **The real per-cell sweep memory driver is elsewhere** — the ENCODE
   (separately measured: jxl-modular ~1.5 GB @ 3.15 MP ≈ ~500 B/px, lossy
   ~0.2 GB; `a447e87d`) × per-cell rayon parallelism (SWEEP_JOBS). 2 jobs ×
   ~500 B/px ≈ the ~1 KB/px observation. Dropping `--feature-output` did not
   address that.

3. **Wiring the feature path through the strip aggregator is unnecessary AND
   counterproductive** at ≤30 MP: `strip` measured HIGHER peak than the
   already-streaming base path (70.9 vs 38.3 B/px at 30 MP) because the default
   parallel-strip geometry holds several concurrent per-strip working sets. The
   strip path only helps at the very-large tail where the base path's absolute
   GB × workers would OOM.

## Recommended follow-ups (not yet done)

- Correct the `4bb5febf` claim in `zenmetrics scripts/sweep/hetzner_cpu_sweep.sh`
  (cross-repo — surface to user, don't edit unprompted).
- Measure a full sweep CELL (decode + encode all codec variants + score) under
  heaptrack to confirm the encode × SWEEP_JOBS is the ~1 KB/px driver.
