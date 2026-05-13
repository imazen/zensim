# Pure-Rust compression candidates for zenpredict runtime weights (2026-05-13)

**Question**: which pure-Rust compression library is the right trade-off
across (compressed size, decode allocations, decode wall time,
decode-only binary cost) for un-rebaked production weights?

**Method**: extract weight bytes from each bake via the ZNPR layer
table, compress with each candidate, decode 1,000 times and take the
median, verify byte-for-byte round-trip, count global-allocator calls
around a single warm decode (warmup call first so once-cell init isn't
charged), measure a stripped `codegen-units=1 lto="thin" strip=true
panic="abort"` decode-only binary per candidate. Scratch project:
`/tmp/compression_eval/`. Encode-side `zstd-22` uses C-backed
`zstd 0.13.3` (measurement only); runtime decode uses pure-Rust
`ruzstd 0.8.3`. Bake files read-only.

**Bake dtypes** (correcting the task spec): only V0_18 is I8. Both
pickers are **F16** across all layers. Measured byte entropy: V0_18
7.60 bits/B, zenwebp 7.39, zenavif 7.27 — all ≥ 90 % of max. F16
mantissas look uniform-random to a general compressor.

## V0_18 zensim (87,936 B raw, I8)

| candidate | comp B | ratio | dec µs | allocs | lib KB |
| --- | --: | --: | --: | --: | --: |
| ruzstd (zstd-22 input) | 83,899 | 0.954 | 351 | 30 | 123 |
| lz4_flex block | 88,271 | 1.004 | 1.8 | 1 | 4 |
| miniz_oxide deflate-9 | 83,893 | 0.954 | 166 | 2 | 22 |
| snap Snappy raw | 87,945 | 1.000 | 1.7 | 1 | 8 |
| bitpack-±1 custom | 106,904 | 1.216 | 74 | 1 | 0.4 |

## zenwebp picker v0.1 (92,672 B raw, F16)

| candidate | comp B | ratio | dec µs | allocs | lib KB |
| --- | --: | --: | --: | --: | --: |
| ruzstd | 85,904 | 0.927 | 334 | 28 | 123 |
| lz4_flex | 92,971 | 1.003 | 1.9 | 1 | 4 |
| miniz_oxide | 85,848 | 0.926 | 179 | 2 | 22 |
| snap | 92,681 | 1.000 | 1.8 | 1 | 8 |
| bitpack-±1 | 114,658 | 1.237 | 77 | 1 | 0.4 |

## zenavif rav1e picker v0.1.1 (215,040 B raw, F16)

| candidate | comp B | ratio | dec µs | allocs | lib KB |
| --- | --: | --: | --: | --: | --: |
| ruzstd | 196,035 | 0.912 | 780 | 39 | 123 |
| lz4_flex | 215,885 | 1.004 | 4.2 | 1 | 4 |
| miniz_oxide | 196,235 | 0.913 | 441 | 2 | 22 |
| snap | 215,055 | 1.000 | 4.1 | 1 | 8 |
| bitpack-±1 | 267,524 | 1.244 | 178 | 1 | 0.4 |

## Summary (mean across three bakes)

| candidate | mean ratio | mean µs | max allocs | lib KB | no_std |
| --- | --: | --: | --: | --: | :-: |
| ruzstd (zstd-22) | 0.931 | 488 | 39 | 123 | alloc |
| miniz_oxide deflate-9 | 0.931 | 262 | 2 | 22 | alloc |
| lz4_flex block | 1.004 | 2.6 | 1 | 4 | alloc |
| snap | 1.000 | 2.5 | 1 | 8 | **std-only** |
| bitpack-±1 | 1.232 | 110 | 1 | 0.4 | alloc |

`lib KB` is decode-only stripped size minus an empty-Rust floor
(decode-baseline minus its blob: 329 KB) minus the candidate's blob.
ruzstd built without `hash` (no twox-hash, saves ~1 KB); lz4_flex
without `frame` for same reason.

Cross-reference: prior agent's report
(`zenpredict_rle_zerobias_eval_2026-05-13.md`) measured `zstd-22` at
83,902 B on V0_18 raw (τ=0), matching this run within rounding. The
75 % shrink that report shipped required a **τ=0.005 rebake** zero-
biasing 87.5 % of weights — decoder choice is the same, the rebake is
what unlocks savings.

## Round-trip verification

All 15 cases (5 candidates × 3 bakes) round-tripped byte-for-byte.

## Verdict

**No candidate meets the four-way bar** (ratio ≤ 0.40, ≤ 100 µs,
≤ 2 allocs, ≤ 30 KB binary) on raw un-rebaked weights.

- zstd / deflate plateau at 0.93 — unmodified weight bytes are
  near-incompressible (V0_18 only 3.4 % `{-1,0,+1}` and 1.35 % zeros;
  F16 mantissas uniform-random). No `flags="zstd"` bit alone unlocks
  the 75 % headline shrink — that needs a τ-rebake on top.
- lz4_flex and snap are tiny and fast (~2 µs, 1 alloc, 4-8 KB) but
  **expand** every bake by 0.3-0.4 %.
- bitpack-±1 expands by 23 % because 96.6-99.4 % of weights miss the
  fast path and pay 10 bits each (2 code + 8 escape) vs. 8 raw.
- ruzstd's 28-39 allocations per decode is the worst single number —
  structural to the zstd state machine (Huffman cache, FSE, decode
  buffer, ring buffer, sequence scratch), not the optional checksum.
  123 KB stripped is 4× over the 30 KB budget.

**Recommendation for `feature = "compressed-weights"` in zenpredict
0.2.0**: do not ship one. 4.5-9 % shrink doesn't justify ruzstd's
123 KB or miniz_oxide's 22 KB — every codec crate wrapping zenpredict
pays those bytes. Compressing a 141 KB picker to ~130 KB saves ~11 KB
and adds 22 KB of decoder. Net loss.

If a τ-rebake lands first (where the 75 % shrink lives), **prefer
miniz_oxide over ruzstd** for picker-shaped (≤ 500 KB) bakes: 22 KB +
2 allocs vs. 123 KB + 30+ allocs at a gap (prior agent's τ=0.005:
zstd-22 18,597 B vs. gzip-9 20,190 B, 7-9 % edge) too small to erase
the library cost on every bake under ~2 MB. lz4_flex, snap, bitpack-±1
are non-starters — each expands every tested bake.
