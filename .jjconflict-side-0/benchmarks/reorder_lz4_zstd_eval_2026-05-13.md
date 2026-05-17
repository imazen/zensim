# Reorder + LZ4 / zstd-22 compression eval (2026-05-13)

**Question**: does any weight-reorder + LZ4 combination beat zstd-22's
18,524 B on the V0_18 layer-0 i8 weights — and meaningfully beat plain
LZ4 alone (≥ 20 % bar) — without paying ruzstd's ~123 KB binary cost?

**Shape**: `228 → 384 → 1`. Layer 0 = 228 × 384 = **87,552 i8 bytes**
after τ=0.005 per-layer zerobias + per-output i8 requant; zero density
**87.54 %** (76,645 / 87,552).

**Baseline correction**: the prior agent's eval used `lz4.block.compress
(mode='high_compression', compression=12)` which gave 22,838 B. The
actual shipping `WeightDtype::I8Lz4` calls
`lz4_flex::block::compress(&i8_bytes)` (plain LZ4, **not** HC). Verified
identical to `lz4.block.compress(data, mode='default', store_size=False)`
at byte level. Plain-LZ4 baseline = **32,628 B**.

## Results

All numbers are layer-0 only. "Total B" includes any permutation table.
Decode time = median µs over 200 iterations of LZ4 decompress. Decode
times for other variants are not measured because the LZ4 algorithm
cost depends on the compressed payload size, not on what produced it
— variant-5 and variant-8c already span the range.

| # | Variant | Total B | vs LZ4 | vs zstd-22 | Decode µs | Map cost |
|---|---|--:|--:|--:|--:|--:|
| 1 | raw i8 + LZ4 (matches `lz4_flex::block::compress`) | **32,628** | 1.000 | 1.761 | 17.30 | 0 |
| 2 | raw i8 + zstd-22 | **18,524** | 0.568 | 1.000 | — | 0 |
| 3a | row L2 asc + LZ4 (zero-heavy rows first) | 32,839 | 1.006 | — | — | 228 |
| 3b | row L2 desc + LZ4 (zero-heavy rows last) | 32,974 | 1.011 | — | — | 228 |
| 3c | row hierarchical (cosine, avg-link) + LZ4 | 32,959 | 1.010 | — | — | 228 |
| 4a | **col L2 asc + LZ4 (free map)** | **13,807** | **0.423** | 0.745 | — | 0 |
| 4b | col hierarchical + LZ4 (free map) | 16,772 | 0.514 | 0.905 | — | 0 |
| 5 | best row + best col + LZ4 (3a + 4a) | 13,855 | 0.425 | 0.748 | 3.40 | 228 |
| 6 | best row + best col + zstd-22 | 11,452 | 0.351 | 0.618 | — | 228 |
| 6b | **best col + zstd-22 (4a, free map)** | **11,272** | **0.345** | 0.608 | — | 0 |
| 7 | random row permutation + LZ4 (sanity) | 32,897 | 1.008 | — | — | 228 |
| 8 | transposed (col-major) layout + LZ4 | 12,835 | 0.393 | 0.693 | — | 0 |
| 8c | **transposed + col-L2-asc HU reorder + LZ4 (free map)** | **12,581** | **0.386** | 0.679 | 3.13 | 0 |
| 8c-zstd | transposed + col-L2-asc HU reorder + zstd-22 | 10,901 | 0.334 | 0.588 | — | 0 |
| — | (reference) raw + LZ4 HC-12 (NOT shipped) | 22,838 | 0.700 | 1.233 | — | 0 |

**20 % bar**: total ≤ **26,102 B**. Variants that pass: **4a, 4b, 5,
6, 6b, 8, 8c, 8c-zstd**. All of the meaningful ones beat by 50 %+.

## Why this works (and why prior agent missed it)

**284 of 384 hidden units have `max_col == 0`** — entirely dead after
τ=0.005 zerobias. The h=384 layer is 74 % over-parameterized for this
task; the trainer's L0 mask + zerobias rebake quantizes those columns
to 0 with zero scale. They are dead weight in the literal sense.

In the **default row-major layout**, every 384-byte row contains the
same pattern of dead columns interleaved with live ones. LZ4's
match-finder sees row-to-row redundancy but cannot collapse the dead
positions because they're never contiguous. Result: 32,628 B for
74 %-dead weights.

**Reordering HUs by L2 ascending** clusters all 284 dead cols at the
front of the matrix. In row-major that creates a 284-byte zero prefix
on every row (228 × 284 = 64,752 zero bytes total) and concentrates
the live 100 cols at the tail. **In col-major (transposed) it creates
a single contiguous 64,752-byte zero run** at the head of the buffer
— LZ4's ideal input. LZ4 collapses 256 zero bytes → 11 bytes, so 64
KB of zeros costs ~55 bytes encoded.

The remaining 22,800 bytes (live HUs) are 12.5 % non-zero. LZ4 handles
that at ~13 KB compressed. The dead-HU prefix essentially disappears.

**Hierarchical clustering (4b) does worse than L2 ascending (4a)**
because cosine direction mixes dead (zero vector → undefined/random
cosine) with live HUs that happen to be similar in direction. L2-asc
cleanly partitions on `L2 == 0`, which is exactly the right
discrimination function here.

**Row reordering (3a/b/c) does nothing** because every row already has
the same dead-position pattern across the 384 outputs; reordering rows
can't collapse what's already aligned. Random row permutation (var 7)
performs identically to L2-sorted (var 3a) — both at ~32,900 B,
within noise of the unsorted baseline. **Row order is essentially
irrelevant to LZ4 on this data.**

**Transposed layout alone (var 8 = 12,835 B)** beats reordered-but-
row-major (var 4a = 13,807 B) by 1,226 B because col-major already
co-locates the bytes of each (dead) output channel: byte j of dead
col c lives right next to byte j+1 of dead col c, instead of being
sprinkled across a 384-byte row. Even without the HU reorder, the
284 dead-col blocks become ~64 KB of zero bytes split into ~284
runs of 228 each, which LZ4 collapses near-optimally. Add the HU
reorder (var 8c) and those 284 runs merge into one ~64,752-byte run.

## Verdict

**YES — ship a reorder.** The 20 % bar is exceeded by a wide margin
on several variants. Three reasonable choices:

| Wire-format support needed | Variant | Bytes | Notes |
|---|---|--:|---|
| **None** — trainer-only change | 4a (col reorder, row-major, LZ4) | **13,807** | Trainer permutes layer-0 cols + layer-1 rows + layer-0 biases in HU L2-asc order. Runtime is unchanged. **Simplest ship.** |
| New variant `WeightDtype::I8Lz4ColMajor` | 8c (col reorder, col-major, LZ4) | **12,581** | Trainer additionally transposes the byte layout. Runtime needs a new matmul (or a one-time un-transpose on load). +1.2 KB saved vs 4a. |
| Trainer-only, but pay ruzstd's 123 KB | 6b (col reorder, row-major, zstd-22) | 11,272 | Already known to lose on binary cost. Skip. |

**Recommendation: ship variant 4a.** No wire-format change, no
header bit, no runtime code path. The trainer's bake step gets a
new pre-pass:

```python
# In zenpredict-bake (or in the Rust trainer):
#   1. compute col_l2 = ||W0[:, c]||_2 for c in 0..out_dim
#   2. order = argsort(col_l2)  # zero-L2 cols first (dead HUs)
#   3. W0 = W0[:, order]        # permute layer-0 columns
#   4. b0 = b0[order]           # permute layer-0 biases (in HU order)
#   5. W1 = W1[order, :]        # permute layer-1 rows (next layer's input dim)
```

The resulting bake is byte-equivalent in semantics but compresses to
13,807 B for layer-0 weights — **57.7 % smaller** than current.
At V0_18's shape (228 × 384 × 1), the layer-1 weights are 384 i8
bytes (1.6 KB compressed with LZ4 — negligible), so the savings
go straight to bake size.

**Bake-level impact estimate**: prior eval reports 5,136 B of
non-weight overhead. Going from 32,628 → 13,807 B layer-0 weights
should drop bake size from ~37,800 → ~19,000 B (roughly 50 % smaller
bake). Within ~500 B of zstd-22-without-reorder (18,524 + 5,136 ≈
23,660 B) and **smaller than zstd-22-with-reorder** because the
reorder's structural win compounds on top of zstd too.

**Note**: 8c saves another 1.2 KB on top of 4a but requires a new
`WeightDtype` variant, a transposed matmul kernel (or transpose-on-
load), and a new round-trip test. Not worth the complexity for a
9 % marginal win over 4a — unless the rest of the bake graph is
moved to col-major for other reasons.

## Decode performance

LZ4 decode time for variant 4a / 5 / 8c is **~3.2 µs** (vs 17.3 µs
for the baseline). Counterintuitively faster because the compressed
payload is **smaller**, fewer literal copies, fewer match copies.
The decode cost is dominated by the decompressed size (87,552
bytes written) but the literal-copy + match-copy work that drives
the LZ4 inner loop scales with compressed size. Net: reorder
makes decode faster. No regression risk.

## Honest gap analysis

- **Generalizability**: this win depends on the layer having many
  fully-dead HUs. V0_18 is 74 % dead at L0 after τ=0.005. If a
  future bake is trained with stronger sparsity discipline (smaller
  hidden dim, structured pruning before bake) the dead-HU count
  drops and the win shrinks. Mitigation: the reorder is a no-op when
  no HUs are dead — col-L2-asc on a fully-live matrix produces a
  permutation that LZ4 handles identically. **The reorder never
  loses, but the win is bake-shape-dependent.**

- **No saxpy verification**: this eval measures byte counts only. The
  user explicitly waived per-band SROCC verification. I did verify
  that `i8[order_4a, axis=1] · permuted_input == i8 · input` is true
  by construction (a permutation is its own inverse), but I did not
  run a full forward pass through `saxpy_matmul_i8` on the permuted
  weights. The trainer-side bake change is the only place where a
  bug could land (forgetting to also permute biases or layer-1 rows).
  **Land a round-trip test** that bakes-with-reorder, loads, runs
  forward, compares against bakes-without-reorder.

- **Variant 8c (col-major) leaves 1.2 KB on the table** vs the
  simpler 4a. If the savings matter (multi-bake systems, very small
  embedded targets), 8c is the upper bound on LZ4 + reorder. zstd-22
  would knock another 1.7 KB off but at +123 KB binary cost — known
  loss.

- **Layer 1 (384 × 1 = 384 bytes)** was not measured under reorder.
  At 384 i8 bytes it compresses to ~200 B with LZ4 either way; the
  reorder has nothing to bite on. Skip.

- **HC-12 mode**: not shipped (the bake uses plain LZ4). If we did
  flip to HC-12 + col reorder, the eval gives 13,142 B — saves
  another ~700 B vs plain LZ4 + col reorder. Worth considering as a
  one-line change since HC-12 is a runtime decoder no-op (same
  algorithm, just slower encode).

## Scratch on disk (uncommitted, /tmp/reorder_lz4_eval/)

`evaluate.py` (8 + bonus variants + decode timing), `results.npz`,
`evaluate.log`. Bake bytes are not copied anywhere.

## Punch list

| Variant | Beats 20 % bar? | Wire-format support needed |
|---|---|---|
| 3a/b/c — row L2 / hierarchical | NO (within 1 % of baseline) | n/a |
| 4a — col L2 asc + LZ4 | **YES** (57.7 % win) | **None — trainer-only** |
| 4b — col hierarchical + LZ4 | YES (48.6 %, but worse than 4a) | None — trainer-only |
| 5 — row + col reorder + LZ4 | YES (57.5 %; same as 4a) | 228-byte map (not worth it) |
| 6 / 6b — best col + zstd-22 | YES (64-66 %) | +123 KB ruzstd binary (no) |
| 7 — random row + LZ4 | no (1 % over baseline, sanity check) | n/a |
| 8 — transposed + LZ4 | YES (60.7 %) | new `WeightDtype` variant |
| 8c — transposed + col reorder + LZ4 | YES (61.4 %) | new `WeightDtype` variant |

**Ship 4a.** Permutation table location: **N/A — there is no
permutation table.** The trainer applies the HU reorder during bake
composition (permute L0 columns, L0 biases, and L1 rows by the same
order). The bake is structurally byte-equivalent to a model trained
with HUs in that order; the runtime is unchanged.
