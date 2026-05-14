# Reorder + zero-RLE feasibility eval (2026-05-13)

**Question**: would per-output reordering (sort columns so zeros cluster
into runs) plus the user's zero-only RLE format beat the existing LZ4
weights path (~32,840 B compressed weights, 37,976 B full bake, +40 µs
decode at V0_17 → V0_18 shape)?

**Shape**: V0_17 → V0_18 is `228 → 384 → 1` (layer 0 = 228×384 = 87,552
i8 weights; layer 1 = 384×1 = 384 i8 weights; total **87,936 i8 bytes**).
After τ=0.005 per-layer zerobias and per-output-column i8 requant, zero
density is **87.51 %** (76,953 zero bytes; layer 0 87.56 %, layer 1
76.56 %).

**Format** (user-specified): non-zero byte emits literally; zero byte
emits `[0x00, count_1..=255]`. Run of N zeros → ⌈N/255⌉ records of 2 B
each. Best case: 2 B per 255 zeros (1.0078 B/zero). Worst case: 2 B per
isolated zero (200 % expansion on that byte).

**Mapping-table widths**: layer 0 in_dim = 228, fits in `u8`. Layer 1
in_dim = 384, needs `u16`. (The user's spec quoted "u16 → 175 KB"; with
the right width the worst-case naive table for layer 0 is 87,552 B —
one byte per (row, output) pair — which already dominates the RLE
savings and is the verdict driver below.)

## Variants vs baselines (weights only; full bake adds ~5,136 B overhead)

All numbers compress the i8 weight stream **after** τ=0.005 zerobias.
"Map" is the permutation/index side-channel needed to un-reorder at
runtime. zstd-22 row matches prior eval (18,597 B) within rounding.

| Variant | RLE bytes | Map bytes | Total bytes | Ratio (raw=87,936) |
|---|--:|--:|--:|--:|
| Raw i8 (no compression) | 87,936 | 0 | **87,936** | 1.000 |
| **A** — no reorder, zero-RLE | 29,487 | 0 | **29,487** | 0.335 |
| **B** — per-output \|w\|-sort, zero-RLE | 11,755 | 88,320 | **100,075** | 1.138 |
| **C** — per-output sign-aware, zero-RLE | 11,755 | 88,320 | **100,075** | 1.138 |
| **D** — per-layer row-mean sort, zero-RLE | 29,375 | 996 | **30,371** | 0.345 |
| lz4 HC-12 (Python `lz4.block`) | 23,036 | 0 | **23,036** | 0.262 |
| zstd-22 (Python `zstandard`) | 18,608 | 0 | **18,608** | 0.212 |

**B and C produce identical RLE output** — both place zeros in a single
contiguous run at the tail of each column, and the zero-RLE primitive
ignores non-zero byte identity, so reorder within the non-zero prefix
doesn't change run structure. Sign-aware sorting cannot help a
zero-only RLE.

## Map-table cost dominates per-output reorder

For layer 0 (228 × 384, u8 indices): one byte per (row, output) =
**87,552 B** of side-channel data. Layer 1 (384 × 1, u16 indices) adds
768 B. Naive map total = **88,320 B** — bigger than the raw i8 stream
it permutes.

### Variant B with cheaper map alternatives

Two natural compressions of the mapping table:

- **Smart map** (store K non-zero row indices per column; decoder
  fills the zero suffix from the complement, ascending): 11,459 B for
  the table. Combined with variant B's 11,755 B RLE = **23,214 B
  total**. Still loses to zstd-22's 18,608 B by 24 %.
- **zstd-22 over the naive permutation table**: 13,037 B. Combined
  total = **24,792 B**. Same conclusion.

Per-column non-zero counts: mean K = 28.5, P95 = 220, max = 226 (some
columns have almost no zeros — for those the smart-map approach
collapses back to the naive cost).

### Variant D is the only mapping-cheap reorder, and it doesn't help RLE

Per-layer row-mean-magnitude sort costs only 996 B of map but doesn't
materially improve clustering — the per-column zero distribution
across rows is mixed enough that a single global row order can't
co-locate zeros for all 384 outputs simultaneously. D's 30,371 B is
**larger than variant A's 29,487 B** (the additional 0.4 % comes from
the run structure shifting unfavorably under the global sort) and is
**26 % worse than lz4** with a brand-new format extension.

## Bitmask + non-zero values (sanity reference, not a proposed variant)

For context, a sparse encoding that stores 1 bit per weight (zero/non-zero
mask) plus the non-zero i8 values densely:

- Bitmask: 10,992 B; non-zero values: 10,983 B; **total 21,975 B**.

Still 18 % worse than zstd-22, confirming the prior eval's finding
that custom structural sparse codings need entropy coding to compete
with zstd — and at that point you have rebuilt zstd.

## Verdict

**No.** Reorder + zero-RLE is not competitive with the existing LZ4
path on this shape, and not within 20 % of LZ4 on any variant
including map-compressed B.

- Best total **without** mapping table (variant A, no reorder):
  29,487 B — **28 % larger than lz4 HC-12 (23,036 B)**.
- Best total **with** any per-output reorder: 23,214 B with the
  smart-map prefix encoding — barely beats lz4 (23,036 B) by 174 B
  (0.8 %), while requiring (a) a new wire format, (b) a permutation
  decoder, (c) a per-output un-permute pass on the loaded i8 column
  before saxpy, and (d) custom round-trip + fuzz coverage.
- Bar for adoption was "must beat lz4-on-zerobias's 37,976 byte total
  bake by > 20 % to justify the format extension". A 0.8 % win or a
  28 % loss both fail that bar.

**Variant D** (per-layer row-mean reorder, 996 B map) loses to plain
variant A by 0.4 % on RLE, so the reorder isn't even strictly helpful
without per-output specificity.

**Variants B and C are not distinguishable** under zero-only RLE — the
RLE primitive is blind to non-zero byte identity, so any reorder that
clusters zeros at one end produces the same compressed stream
regardless of how the non-zero prefix is sorted. Sign-aware sorting
needs a richer RLE (run-of-equal-value, not run-of-zero) to pay off,
which is a different proposal.

**Recommendation: do not implement `WeightDtype::I8Reorder`.** The
existing zerobias + lz4 path (37,976 B full bake) already lands a 59 %
shrink and is within 0.8 % of the best mapping-aware variant here.
Effort is better spent on the levers identified in the prior
2026-05-13 eval: (a) bake zstd-flag support (75-80 % shrink at
runtime cost of one ruzstd decompress call), or (b) shrink the
architecture from 228×384 to 228×128 (88 % of layer 0 zeros under
τ=0.005 is direct evidence the layer is over-parameterized).

## Honest gap analysis (called out per task)

- The user's "175 KB" estimate for the per-bin u16 table was an upper
  bound; layer 0 in_dim = 228 fits in u8, giving **87,552 B** for that
  layer alone — still well above the raw weight stream it permutes.
  This is the architectural problem, not a sizing detail.
- Map-table compressibility cuts the table to ~13 KB (zstd) or
  ~11.5 KB (smart prefix encoding), but the *combined* compressed
  weights + map total still trails zstd-22 on the raw stream.
- The user's intuition was right that per-column zero-clustering is
  achievable (12,000 B of RLE'd column data is a real number). The
  problem is the metadata cost of the permutation, not the encoding
  primitive.

## Scratch on disk (uncommitted, /tmp/reorder_rle_eval/)

- `evaluate.py` — variants A/B/C/D + lz4/zstd baselines
- `explore_map_compress.py` — smart-map prefix encoding + zstd-on-map
- `explore_bitmask.py` — bitmask + non-zero list sanity reference
- `results.npz`, `evaluate.log`
