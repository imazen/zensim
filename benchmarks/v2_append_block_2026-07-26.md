# f720+ append block — perf + RAM gates (2026-07-26)

**Ask (user):** add the 2026-07-26 gap-audit features
(`zenpapers/docs/zensim-720-feature-gaps-2026-07-26.md` §5) to the folded v2
path; benchmark and optimize to keep perf within 10%; RAM must not grow
significantly — heaptrack before and after.

**Result: both gates pass on the production (append-moments) extraction path.**
Compute **+9.2%** (49.55 → 54.1 ms/pair, gate ≤10%); heaptrack peak heap
**+11.9 MB = +12.7%** (93.99 → 105.92 MB), all of it the new `blur(src²)`
moments-cache plane — charged only to append-mode preparation, linear in
reference megapixels, zero change to every existing path.

Host: 7950X (Zen4, AVX-512), WSL2, single-thread (`RAYON_NUM_THREADS=1`),
`nice -n19 ionice -c3`. Pairs: first 100 of
`/mnt/v/output/zensim/v2-ab-2026-07-19/aic3_pairs_ab.tsv` (~1 MP photos, the
`fold_extraction_2026-07-24.md` grid). Driver:
`v2_ab_extract` (`ZENSIM_AB_MODE=none|fold|foldapp`, grouped + moments —
foldapp prepares via `prepare_v2_reference_with_moments_append`). Commit:
this change (parent `e4b7edf7` first landing, `5027cf49` base).

## Timing (4 rounds each, medians; decode baseline = `none` mode)

| mode | wall (s) | compute ms/pair | vs fold |
|---|--:|--:|--:|
| none (decode only) | 2.04 | — | — |
| fold (720, baseline) | 6.995 | 49.55 | 1.000 |
| **foldapp (924)** | **7.45** | **54.1** | **1.092** |

Round spread: fold 6.93–7.16 s, foldapp 7.38–7.50 s; worst-case cross-pairing
1.117, mean-based 1.085 — the median ratio 1.092 is the reported number.
The fold baseline itself matches (slightly beats) the documented 59.9 ms/pair
from `fold_extraction_2026-07-24.md`, so the baseline is not inflated.

## heaptrack (same runs, single-thread)

| mode | peak heap | peak RSS (incl. heaptrack) | allocation calls |
|---|--:|--:|--:|
| fold | 93.99 MB | 108.91 MB | 128,342 |
| foldapp | 105.92 MB | 120.68 MB | 165,446 |

Δ peak heap = **+11.93 MB (+12.7%)** ≈ 12 B/pixel-of-reference: the `bs2`
cache is one plane-kind × 3 channels × 1.328 pyramid − the skipped (B, scale 0)
cell ≈ 2.98 plane-equivalents × 4 B. `/usr/bin/time` maxrss agrees
(102.8 → 114.2 MB, +11.1%). Non-heaptrack RSS deltas in the timing table's
runs: same.

## How the gates were reached (the first attempt failed CPU by 4.7×the budget)

The naive implementation — per-pair `blur(dst²)` via `square +
box_blur_1pass_into` per strip + an unconditional 19-lane append kernel —
measured **+46.6%** (74.9 ms/pair) with RAM +4%. perf attributed ~5.5% of
wall to the standalone d² H-blur and 8.5% to the append kernels. Three
structural changes got under the gate:

1. **σ-split from the reference side.** `blur(dst²)` is per-pair and
   unavoidable as a blur; `blur(src²)` is reference-side and cacheable. With
   `bs2` in the (append-variant) moments cache, the kernel derives
   `var₁ = bs2 − mu1²` and `var₂ = (ssq − bs2) − mu2²` — the per-pair blur
   chain disappears entirely. Pair-path (no cache) replays `bs2` per scale
   through the same helper, keeping all entry paths bit-identical
   (`append_ref_paths_bit_identical`, 4 legs).
2. **Evidence-backed kernel trims.** Luminance bins pool `mse_i` instead of
   a recomputed SSIM map (the conditioning is the feature; −1 division,
   −2 plane loads); luminance + cross-channel transducers are Y-only (the
   2026-07-19 luma-gate ablation measured chroma transducers as a CID22
   cost); the mid bin is derived at finalize from the Bernstein partition
   of unity (−1 lane pair); the activity load sits inside the Y-only branch.
3. **(B, scale 0) skipped** (`APPEND_SKIP_B_SCALE0`): yellow-violet foveal
   resolution is ~53 ppd vs 94 achromatic (Ashraf/Chapiro/Mantiuk 2025) and
   butteraugli carries no B in its two highest-frequency bands — the cell
   would model signal the eye cannot resolve, at ~25% of the append block's
   pixel cost. Slots emit 0.0, index-stable; its `bs2` cache plane is not
   filled either (that's the −4 MB between the +15.1% first cache
   measurement and the shipped +11.1%).

## Tradeoff record

| variant | compute vs fold | peak heap Δ | verdict |
|---|--:|--:|---|
| per-pair blur(dst²), full kernel | +46.6% | +4% | CPU gate fail |
| + kernel trims (measured mid-state) | +10.4% | +15.1% RSS | CPU still over, RAM over |
| **bs2 cache + trims + B-s0 skip (shipped)** | **+9.2%** | **+12.7% heap / +11.1% RSS** | **both gates** |

The +11.9 MB is the price of the CPU gate; it is confined to append-mode
extraction (plain fold/v2/v1 preparation allocates exactly what it did
before) and scales as ~12 B per reference pixel (≈ 48 MB at 4 MP). If a
future consumer needs it smaller, the next lever is quantizing the cached
plane (bf16 halves it at ~0.4% var precision — rejected here to keep
bitwise path parity simple).

## Feature-value sanity

`foldapp` CSV emits 926 columns (basename, human_score, f0..f923); on real
distorted pairs 162/204 append slots are nonzero — the structural zeros are
the X/B transducer slots, the (B, scale 0) cell, and one-sided
gain/loss polarity pairs.

## Reproduce

```
cargo build --release -p zensim --features feature-regime-v2,training --example v2_ab_extract
head -101 /mnt/v/output/zensim/v2-ab-2026-07-19/aic3_pairs_ab.tsv > ~/tmp/aic3_100.tsv
RAYON_NUM_THREADS=1 ZENSIM_AB_MODE=fold    target/release/examples/v2_ab_extract ~/tmp/aic3_100.tsv /tmp/f.csv
RAYON_NUM_THREADS=1 ZENSIM_AB_MODE=foldapp target/release/examples/v2_ab_extract ~/tmp/aic3_100.tsv /tmp/fa.csv
RAYON_NUM_THREADS=1 ZENSIM_AB_MODE=foldapp heaptrack target/release/examples/v2_ab_extract ~/tmp/aic3_100.tsv /tmp/fa.csv
```

## 12 MP peak memory (measured 2026-07-26, same host/commit + `72156df5`)

Single 4000×3000 pair (aic3 ref resized + JPEG-82 roundtrip; memory is
dimension-, not content-, driven), heaptrack peak heap. One f32 plane at
12 MP = 48.0 MB.

| config | peak heap | composition check |
|---|--:|---|
| fold, grouped+moments | 882.2 MB | pyramids+scratch+decode ≈ 499 + mu1/act cache 383 |
| **foldapp, grouped+moments** | **1.03 GB** | + bs2 cache 143 (2.98 planes) + misc |
| fold, `ZENSIM_AB_MOMENTS=0` | 499.7 MB | cache term gone (882 − 383 exactly) |
| foldapp, `ZENSIM_AB_MOMENTS=0` | 835.7 MB | 499.7 + 336 replay planes (7 × 48) exactly |
| foldapp, pair path (`GROUPED=0`) | 835.7 MB | same replay shape |

Timing at 12 MP, ONE variant per reference (3 rounds, wall incl. ~decode):
fold+moments ~1.74 s; foldapp+moments ~2.1 s; **foldapp MOMENTS=0 ~1.90 s —
faster AND 194 MB smaller than foldapp+moments**, because with a single
variant the cache costs more to fill than it saves. The cache pays off only
when many variants share one reference (the aic3-100 batch shape).

### Bringing 12 MP peak down — options by measured MB

1. **Zero code, available now:** score single photos (or few-variant refs)
   with `ZENSIM_AB_MOMENTS=0` / the pair entry — 1.03 GB → **835.7 MB**
   (−19%), and faster in that shape. Driver policy worth adopting: enable
   the moments cache only when variants-per-reference is high enough to
   amortize it (and/or below a megapixel cap).
2. **Strip-tile the bs2 fill/replay** (moderate, do it BEFORE any corpus
   freezes append values — it shifts bs2 by f32 ULPs): replaces the two
   48 MB whole-plane temps with strip-local scratch → replay-path peak
   835.7 → ~740 MB. Same CPU.
3. **bf16 the moments cache** (invasive): mu1+act+bs2 526 MB → 263 MB on
   the cached path (1.03 GB → ~790 MB). ~0.4% stored-value precision;
   breaks the simple bitwise path-parity story — needs both paths to
   quantize identically. Only worth it if 12 MP batch extraction with
   cache becomes a real workload.
4. **Driver: drop the decoded reference RGB8 after prepare** in fold modes
   (−36 MB; r_px is only consumed by the v1/None-prepared fallbacks).
5. **`STRIP_ROWS` 128 → 64**: scratch 92 → ~52 MB (−40 MB) for ~+13% blur
   halo overhead (~+3-5% compute) — a knob, probably not worth the trade.

Floor context: any 12 MP scoring here carries ≈ 500 MB regardless of the
append block (ref+dst XYB pyramids 335, strip scratch 92, decode buffers +
misc ~70). The append block itself adds +152 MB cached / +143 MB replayed
at 12 MP (linear in reference megapixels).

## Follow-ups (not blockers)

- The append block has no trained consumer yet: next training round should
  extract `foldapp` (924) and let the head prune; the diffmap fold for the
  foldable append families (luminance bins, MSCN, contrast/texture — all
  `Σw·v/Σw` or plain means with reference-only weights) wires up when a
  model actually weights them.
- The pair path (no prepared reference) pays a per-scale `bs2`+activity
  replay — fine for tests/dev, not for batch use; batch drivers should
  prepare with `prepare_v2_reference_with_moments_append`.
- `v2_family_map.json`-style tooling and zenmetrics extraction plumbing
  don't know the 924 layout yet; wire when the first foldapp corpus lands.
