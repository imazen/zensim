# Rev5 dense-pool precision: what the f64 lanes buy and cost (2026-10-24)

Question: how much do Rev5's sixteen virtual f64 lanes (`featcanon::Lanes16F64`) in the v2 dense
kernel matter, for accuracy and for time, and what can replace them.

Host i265 (Core Ultra 7 265K, AVX2/v3 tier, no AVX-512), one thread pinned to core 2, load < 1.2,
Rust 1.99.0, no `target-cpu=native`. Runtime main `ac627e86`; bake default B `89cb4861…`;
`ZENSIM_FORMULA_REV=5`. Driver: the out-of-repo probe described in
[the inventory appendix](feature_paths_inventory_2026-10-22.md) (`BakeScorer::compute`, median of N
calls per process, five interleaved processes per variant). This is diagnostic timing, not a SPEEDQ
qualification. Not measured on Zen 5 or on any AVX-512 tier.

## Provenance of the f64 choice

The Rev5 spec (A2, [rev5_spec](rev5_spec_2026-10-04.md)) says accumulator precision, f32 or f64
lanes, "is chosen per family by measurement against `exact` and recorded". The implementation
log ([rev5_WORKLOG](rev5_WORKLOG.md), Step 3) records f64 as the "initial choice [that] preserves
the Rev4 per-element error floor; FEATACC measurement … remain[s] pending". No later entry measures
f32 lanes at Rev5. The Rev5 accuracy gains (v2 max relative error 3.06e-1 → 1.09e-3 against exact)
come from the local-window blur (A1) and stable moments (F1). At Rev4, [FEATACC](featacc_WORKLOG.md)
found f32 against f64 pools visible only on csfw (not a Rev5 family) and below the element-error
floor elsewhere.

## Where the time goes (perf, 1 MP, score loop)

`dense_block_kernel_canon64_vec_v3` is 27.7–31.8% of samples. Per-instruction annotation puts about
62% of the kernel's samples on the 29-pool update: for every 8 columns, 29 × 2 `vcvtps2pd` from the
spilled term buffer, `vaddpd` from the stack pool and `vmovupd` back. The 29 pools × 16 f64 lanes
(3.7 KB) cannot live in 16 ymm registers.

## Prototypes ([patch](dense_pool_proto_2026-10-24.patch), compile-time `ZENSIM_PROTO_POOL`)

* **f32**: 16 f32 virtual lanes per row, folded lane-wise into the f64 pools before the existing
  per-row tree. Value-changing (a Rev6 candidate).
* **ilvb**: the term vectors of 16 chunks are buffered in L1; each pool then accumulates them in
  register-resident `f64x8` pairs (even chunks → lanes 0..8, odd → 8..16) and stores once. The
  per-lane addition sequence is unchanged.
* A first whole-row interchange (`ilv` rows in the timing file) streamed a 29 × width buffer and
  was slower: 40.6 ms at 1 MP, 411 ms at 4096×2048.

## Bits and accuracy (score + all 720 emitted feature slots)

Ten pairs: synthetic `test_pair` 64², 97×63, 255×129, 1024², 4096×2048, and five gb82 photographs
(576², baby/bulb/city/dog/flowers) with a deterministic quantize-plus-±8-hash-dither distortion.

* **ilvb: bit-identical to main on 10/10 pairs** (score and every slot).
* **f32**: 78–208 of 720 slots differ per pair. Maximum relative change 6.05e-8 (baby), median
  of differing slots 0.8–6.8e-9, maximum absolute 1.58e-8. Score: unchanged bits on 8/10; city
  +3.57e-6, synthetic 4096×2048 −7.11e-6. Because the f64 pools are effectively exact
  accumulation, these deltas are the f32 lanes' own accumulation error. They are four orders of
  magnitude below Rev5's measured element/blur error against exact (v2 max relative 1.09e-3).
  This is not a FEATACC exact-ruler run or a registered-pair replay.

## Time (median of five per-process medians, ms)

| Geometry | main (f64) | ilvb (bit-identical) | f32 lanes |
|---|---:|---:|---:|
| 256² | 2.262 | 2.073 (−8.4%) | 2.007 (−11.3%) |
| 1024² | 33.70 | 30.85 (−8.5%) | 29.08 (−13.7%) |
| 4096×2048 | 284.9 / 283.0 | 260.3 (−8.6%) | 245.3 (−13.9%) |

4096×2048 has two blocks of rounds (5 and 3); both are shown for main. Raw rows:
[dense_pool_timing_2026-10-24.txt](dense_pool_timing_2026-10-24.txt). After either prototype, the dense kernel's share
falls to 18.8% (ilvb) or 14.0% (f32), and `OnlineMoments::block` (F1's two-pass 16-element blocks)
rises to about 10% (≈3 ms at 1 MP). The other Rev5 f64-pool kernels, `fused_vblur_ssim_canon64`
(8.5–10%) and `gradient_block_kernel_canon64` (5–6.6%), were not changed.

## Limits

One host and tier. The magetypes body also generates v4/v4x; their bits and speed were not
measured. Scalar/NEON/wasm keep separate bodies, untouched. The ilvb prototype is not
landing-quality code (compile-time switch, no tier-parity or CI run).

## Build/run

`ZENSIM_PROTO_POOL={f32,ilv}` (unset for main) at build time with the patch applied; then
`ZENSIM_FORMULA_REV=5 PROBE_BAKE=<B bake> PROBE_W=… PROBE_H=… PROBE_N=… [PROBE_IMG=<png>]
[PROBE_DUMP=<file>] taskset -c 2 <probe>`. Builds: run-heavy rc=0, peak RSS 1.01 GiB each.
Probe sha256: base `5244dafc…`, f32 `4ff852bb…`, ilvb `90f0c413…`.
