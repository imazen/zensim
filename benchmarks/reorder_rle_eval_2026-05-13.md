# Reorder + zero-RLE feasibility eval (2026-05-13)

**Question**: would per-output reordering plus the user's zero-only RLE
beat the existing LZ4 weights path (~32,840 B compressed weights,
37,976 B full bake) at V0_17 → V0_18 shape?

**Shape**: `228 → 384 → 1`. Layer 0 = 228×384 = 87,552 i8 weights;
layer 1 = 384×1 = 384; total **87,936 i8 bytes**. After τ=0.005
per-layer zerobias + per-output i8 requant, zero density **87.51 %**
(layer 0 87.56 %, layer 1 76.56 %).

**Format**: non-zero byte → literal; zero → `[0x00, count_1..=255]`;
runs split into ⌈N/255⌉ records.

**Map widths**: layer 0 in_dim = 228 fits in `u8` (not `u16`). Layer 1
needs `u16`. The user's "u16 → 175 KB" is a loose upper bound; with
right widths the naive layer-0 map is **87,552 B** alone — already as
large as the raw weight stream.

## Variants vs baselines (weights only; full bake adds ~5,136 B)

| Variant | RLE B | Map B | Total B | Ratio |
|---|--:|--:|--:|--:|
| Raw i8 | 87,936 | 0 | **87,936** | 1.000 |
| **A** — no reorder, zero-RLE | 29,487 | 0 | **29,487** | 0.335 |
| **B** — per-output \|w\|-sort | 11,755 | 88,320 | **100,075** | 1.138 |
| **C** — per-output sign-aware | 11,755 | 88,320 | **100,075** | 1.138 |
| **D** — per-layer row-mean sort | 29,375 | 996 | **30,371** | 0.345 |
| lz4 HC-12 baseline | 23,036 | 0 | **23,036** | 0.262 |
| zstd-22 baseline | 18,608 | 0 | **18,608** | 0.212 |

**B and C are identical**: zero-only RLE is blind to non-zero byte
identity, so any reorder that clusters zeros at the column tail
produces the same stream regardless of how the non-zero prefix is
sorted. Sign-aware sorting needs a richer RLE (run-of-equal-value) to
pay off — different proposal.

## Map-table cost dominates

Two natural compressions of the per-output map:

- **Smart map** (store K non-zero row indices per column; decoder
  fills the zero suffix as ascending complement): 11,459 B. Combined
  with B's 11,755 B RLE = **23,214 B**. Beats lz4 by 0.8 % (174 B).
- **zstd-22 over the naive permutation table**: 13,037 B. Combined =
  **24,792 B**.

Non-zero counts per column: mean K = 28.5, P95 = 220, max = 226. Some
columns are nearly dense, so smart-map collapses to naive cost there.

Variant D (1 KB map) doesn't help RLE — a single global row order can't
co-locate zeros for all 384 outputs simultaneously. D's 30,371 B is
*worse* than A's 29,487 B.

**Sparse reference**: bitmask (1 bit/weight) + dense non-zero values =
10,992 + 10,983 = **21,975 B**. 18 % worse than zstd-22, confirming the
prior eval's finding that custom sparse codings need entropy coding to
compete with zstd — at which point you've rebuilt zstd.

## Verdict

**No on `WeightDtype::I8Reorder`.**

- **A** (29,487 B) is 28 % larger than lz4 (23,036 B). Worse than
  status quo with zero added complexity.
- **B/C** with naive map (100,075 B) are catastrophic.
- **B + smart map** (23,214 B) beats lz4 by 0.8 % — not the 20 %
  required to justify a new format + permutation decoder + per-output
  un-permute pass before saxpy + custom round-trip coverage.
- **D** (30,371 B) is dominated by A.

The user's intuition that per-column zero-clustering produces a small
RLE stream is correct: 11,755 B of compressed columns is real. The
problem is metadata cost. The permutation table is fundamentally as
large as the matrix shape it permutes; compressing it to ~11-13 KB
still leaves the combined total at ~23 KB, where lz4 on the raw
stream lands with no format change. zstd-22 on the raw stream lands at
18,608 B, which no variant matches.

**Recommendation**: keep the existing zerobias + lz4 path. Levers that
actually move the needle (from `zenpredict_rle_zerobias_eval_2026-05-13.md`):

1. A `flags` bit "weight section is zstd-compressed" + ruzstd
   decompress at load — ~24 KB total bake, 75 % shrink.
2. Shrink the architecture 228×384 → 228×128. 88 % of layer 0 under
   τ=0.005 is direct evidence of over-parameterization. Stacked with
   zstd: ~10 KB.

## Honest gap analysis

- The "u16 → 175 KB" map estimate is a loose upper bound; right-width
  is **87,552 B** for layer 0 alone — still larger than the weights it
  permutes. The architectural cost of per-bin permutation is the
  verdict driver.
- Map compressibility brings the table to 11.5-13 KB; the combined
  total still trails zstd-22 on the raw stream by 24 %.
- B and C are indistinguishable under zero-only RLE. Sign sorting
  needs a different RLE primitive (run-of-equal-value) to test.

## Scratch on disk (uncommitted, /tmp/reorder_rle_eval/)

`evaluate.py` (variants + baselines), `explore_map_compress.py`
(smart-map + zstd-on-map), `explore_bitmask.py` (sparse reference),
`results.npz`, `evaluate.log`.
