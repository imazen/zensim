# Streaming foldapp — acceptance-gate measurements (2026-07-26)

Gates from `STREAMING_FOLDAPP_PLAN_2026-07-26.md` (plan commit `6676aaf2`),
measured on the C3 state (commit `3d4d19c9`; chunks C0/P1/P2/C1/C2/C3 —
design note `docs/STREAMING_FOLDAPP_C0_DESIGN_2026-07-26.md`).

Host: 7950X (Zen4, AVX-512), WSL2, single-thread (`RAYON_NUM_THREADS=1`,
`with_parallel(false)`), `nice -n19 ionice -c3`. Pairs: first 100 of
`/mnt/v/output/zensim/v2-ab-2026-07-19/aic3_pairs_ab.tsv` (`~/tmp/
aic3_100.tsv`). Driver: `v2_ab_extract` (`ZENSIM_AB_MODE=none|fold|foldapp|
foldstream|foldappstream`; cached modes grouped+moments, stream modes have
nothing to prepare). Raw timing log: 4 interleaved rounds
(`~/tmp/streamfold-c4-timing.log`, committed inline below).

## G-PARITY — PASS (byte-exact, fixtures AND real pairs)

- `streamed_foldapp_bitwise_vs_materialized`: all 924 (foldapp) / 720
  (fold) slots `to_bits`-equal on 7 fixture dims incl. odd (127×93), exact
  strip multiple (96×256), one-past (96×257), 1-row second strip (72×129),
  tall (96×517). Plus sub-64 reflect-pad, parallel==serial, scratch-reuse
  gates; producer wide-window byte-equality vs `gather_strip_halo`
  (`feature_v2_stream` C1 gate); full suite 212 passed / 0 failed.
- Real pairs: the aic3-100 output CSVs of `foldapp` vs `foldappstream`
  (924 cols × 100 pairs, full f64 round-trip formatting) are
  **byte-identical** (`cmp` clean; likewise fold vs foldstream at 720).
- Contract note: bitwise equality is against the P1+P2 canonicalized
  materialized path (blockiness row-ordered; bs2 strip-tiled — both landed
  as pre-chunks on ALL entry paths with no trained consumer affected; see
  the design note §2). The v1-basic fold parity class vs the frozen v1
  path is untouched (`folded720_v1_basic_matches_v1_path` green).

## G-CPU — MISS at 1.33× (gate ≤ 1.15× cached / ≤ ~62 ms; floor ≈ 1.27×)

Wall seconds for 100 pairs (4 interleaved rounds, medians; decode baseline
`none` = 2.06 s subtracted for ms/pair):

| mode | r1 | r2 | r3 | r4 | median | compute ms/pair |
|---|--:|--:|--:|--:|--:|--:|
| none (decode) | 2.08 | 2.06 | 2.05 | 2.06 | 2.06 | — |
| fold (cached) | 6.62 | 6.53 | 6.53 | 6.54 | 6.535 | 44.75 |
| **foldapp (cached)** | 7.38 | 7.21 | 7.18 | 7.08 | **7.195** | **51.35** |
| foldstream | 7.79 | 7.72 | 7.78 | 7.75 | 7.765 | 57.05 |
| **foldappstream** | 8.97 | 8.88 | 8.87 | 8.86 | **8.875** | **68.15** |

- Ratio streamed/cached foldapp (same session): **68.15 / 51.35 = 1.327×**
  — the gate is ≤ 1.15× (≤ ~62 ms/pair on the plan's 54.1 anchor;
  absolute miss +9.9%, ratio miss +15.4%).
- Fold-only: 57.05 / 44.75 = 1.275×. Append marginal: streamed 11.1
  ms/pair vs cached 6.6 (per-strip bs2 replaces the cached plane).
- Today's cached baseline (51.35) is slightly FASTER than the doc's 54.1
  — the P1 blockiness row-ordering is a locality win and the box is
  lightly loaded; the ratio is the load-bearing number.

### Where the 16.8 ms/pair goes (perf, dense-kernel-anchored units)

The optimization history (169 → 68 ms/pair, all parity-preserving):
two-phase channel split undone (fold bands were re-reading evicted H
planes: 21.6 → ~13 ms), ONE shared scratch set serially (3 per-channel
sets cycled 27 MB/strip through L3), zero-copy interior windows, producer
buffer pool in `V2Scratch` (~1.7k page faults/pair). After those, the
residual split:

- **Honest cache-deleted work ≈ +14 ms/pair**: ref-side mu1 V-blur +
  activity chain per strip (~8.6, ablation-measured), per-strip
  `bs2 = blur(src²)` square+H+V (~4.5 net of the cached-plane reads it
  replaces), ref-side XYB conversion + pyramid (~2.6, amortized per-group
  before). These are work the moments cache existed to amortize across
  8-10 variants/reference; a cache-free design must do them per pair.
- **Residual locality/overhead ≈ +3 ms**: fold-band + fused-H reads of
  rolling planes (vs L2-hot gathered copies; copying instead measured
  net-worse), producer bookkeeping, stash copies.

**Floor analysis: the honest component alone puts cache-free streaming at
≈ 65-66 ms/pair = ~1.27× cached — ABOVE the 1.15× gate.** No locality
tuning removes required work; the gate presumed v1's +15.9% cache-free
cost class, but v1's `PrecomputedReference` only amortizes XYB+pyramid
(v1 re-blurs the reference every pair), while the v2 moments cache also
amortizes the mu1/activity chains and bs2 — deleting it necessarily costs
more than deleting v1's. This is the plan's anticipated honest-stop
condition, not an implementation gap.

### Counterpoint: single-variant / large-image shapes FAVOR streaming

12 MP single pair, wall incl. decode (heaptrack runs below): streamed
**1.75 s** vs cached **2.57 s** — the cache costs more to fill than it
saves with one variant (the bench-doc's MOMENTS=0 observation, amplified).
The 1.33× penalty is specific to the many-variants-per-reference batch
shape at ~1 MP.

## G-RAM — PASS (221 MB @ 12 MP vs gate ≤ 250; cached path 1.03 GB)

heaptrack, single 4000×3000 pair (`~/tmp/pair12mp.tsv`), 1-thread:

| config | peak heap | peak RSS |
|---|--:|--:|
| **foldappstream** | **221.04 MB** | 170.03 MB |
| foldapp (cached, pre-C5 path) | 1.03 GB | 1.02 GB |

**4.7× reduction**; composition ≈ decoded inputs 72 + `V2Scratch` strip
sets 100 (3 × 14 wide buffers at scale-0 width) + producer rolling planes
~35 + decode/misc ~15. The walk is O(width), not O(pixels).

80 MP synthetic (8000×10000, `foldapp_stream_bigpair` example — the
`make_pair` pattern, run under heaptrack): peak heap **777.97 MB**, of
which 480 MB is the two synthetic input `Vec<[u8;3]>`s themselves — walk
≈ 298 MB at 8000 px width. Completed in ~7 s single-thread. (The
materialized path at 80 MP was not measured — not extrapolated either;
its 12 MP cache footprint already exceeds the entire streamed peak.)

## G-SIMPLER — deletion inventory ready, NOT executed (see verdict)

The C5 deletion list (design note §6: moments/bs2 cache + fill/replay
helpers, `ensure_append` replay planes, `prepare_v2_reference_with_
moments[_append]`, cached-ref blur variants, materialized foldapp walk)
is implemented-around and ready, but C5 is **decision-gated** by the
G-CPU miss below.

## Verdict + recommendation

3 of 4 gates pass (PARITY bitwise even on real pairs; RAM 4.7× under
gate; SIMPLER staged). **G-CPU misses at 1.327× vs the 1.15× gate with a
measured structural floor of ~1.27×** — per the plan's honest-stop rule
this is a wall, so the C5 switchover (cache deletion) is NOT executed
unilaterally. The streaming path is landed, parity-locked, and publicly
callable (`Zensim::compute_folded720[_append]_features_streaming`);
nothing regresses by its existence.

Decision for the user (either is a small change from here):

1. **Accept ~1.33× batch CPU and complete C5** — delete the cache
   machinery per plan. Costs ≈ +17 ms/pair on many-variant ~1 MP batch
   extraction (+9.4 CPU-h per million pairs, 1-thread); buys the 4.7×
   memory reduction everywhere, O(width) large-image scaling, faster
   single-variant scoring, and the full G-SIMPLER deletion.
2. **Keep the cached materialized path for the batch shape** alongside
   the streaming entries (status quo after this work) — retains the
   maintenance surface the plan wanted gone; batch drivers keep 51 ms/pair.

Recommendation: (1) if the corpus-extraction budget tolerates +33% on the
batch shape — the deleted surface (moments/bs2 cache, replay planes, two
prepare variants, two blur variants, path-parity test matrix) is exactly
the class of machinery that keeps costing sessions; the CPU delta is
bounded and measured, and 12 MP+ extraction gets FASTER. But this is the
plan-anchored gate failing by design-relevant margin, so it is the user's
call, not this session's.

## Reproduce

```
cargo build --release -p zensim --features feature-regime-v2,training --example v2_ab_extract
head -101 /mnt/v/output/zensim/v2-ab-2026-07-19/aic3_pairs_ab.tsv > ~/tmp/aic3_100.tsv
for r in 1 2 3 4; do for m in none foldapp foldappstream fold foldstream; do
  /usr/bin/time -f "r$r $m %e" env RAYON_NUM_THREADS=1 ZENSIM_AB_MODE=$m \
    nice -n19 ionice -c3 target/release/examples/v2_ab_extract ~/tmp/aic3_100.tsv /tmp/$m.csv; done; done
cmp /tmp/foldapp.csv /tmp/foldappstream.csv   # G-PARITY on real pairs
RAYON_NUM_THREADS=1 ZENSIM_AB_MODE=foldappstream heaptrack target/release/examples/v2_ab_extract ~/tmp/pair12mp.tsv /tmp/x.csv
cargo build --release -p zensim --features feature-regime-v2 --example foldapp_stream_bigpair
heaptrack target/release/examples/foldapp_stream_bigpair 8000 10000
```
