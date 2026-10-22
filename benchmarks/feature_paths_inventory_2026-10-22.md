# Feature extraction paths: inventory, measured costs, clean-break options (2026-10-22)

Owner request (verbatim): "Let's inventory zensim - all of its scalar and map feature extraction code paths and
parallel implementations, and what a clean break from the past could look like and what performance we could achieve
with which compromises".

Source: main `e33a638b`. Measurements: host `i265` (Core Ultra 7 265K, x86-64-v3 / AVX2, no AVX-512), Rust stable,
release profile with thin LTO and `codegen-units = 1`, no `target-cpu=native`, single thread pinned to P-core 2 unless
stated, quiet box (load1 0.2–1.5). Input: the SPEEDQ synthetic pair (`ssim2_speed_bar.rs::test_pair`), 1024×1024 unless
stated. Model: default B, frozen diff+frag-820 Rev5 (`89cb4861…`, 420 raw reads → 820 derived inputs). Medians of
30 calls (4,000 at 64²) after one warm-up call. These are diagnostic, unpaired observations from the probe in the
appendix; they are not SPEEDQ-qualified latency and were not measured on Zen 5 or with AVX-512. PERF3
(`perf3_2026-10-18.md`, r7900x) and COSTCMP (`costcmp_2026-10-09.md`, dev) own the qualified numbers.

## 1. Inventory

### 1.1 Who runs what today

| Caller | Walk | Arithmetic |
|---|---|---|
| `Zensim::new(B)` (default, `codec_target()`) and `BakeScorer` with a Rev5 bake | fold (`feature_v2` + `feature_v2_stream` producer) | Rev5 local-window, f64 16-lane pools |
| `Zensim` C / CHdr (plans wider than 372) and D | fold | the bake's declared revision (Rev1 when undeclared) |
| `Zensim` A, PreviewV0_1/0_2, LegacyLinearV0_2, BHdr, and stoppable requests below Rev5 | buffered (`streaming.rs` + `fused.rs`) | Rev1–Rev4 per-tier or canonical |
| `research::extract{,_hdr}` | fold, full plan | any revision |
| Legacy diffmaps (`compute_with_diffmap`, `compute_with_ref_and_diffmap*`) | buffered, basic-plane proxy map | B's scalar in the result is the full Rev5 score |
| `compute_v2_diffmap` | materialized per channel/scale (`compute_v2_diffmap_channel_scale`) | v2 kernels |
| `compute_attribution_density*` (caller gradient, basic block) | buffered attribution planes | per request |
| `Fused944Session` (`compute_with_ref_score_and_attribution*`, `compute_folded944_*`) | fold with retention, pass B | fold arithmetic |
| `BakeScorer::compute_with_ref_and_attribution`, `prepare_steering{,_hdr}`, `SteeringSession`, `local_refine` | fold with retention, MLP finite-difference sensitivities, pass B, SAT bins, neighbour-exact replay | Rev5 for B |
| `compute_v2_features*`, `compute_folded720_*` (append, append2, csfw, HDR, streaming) | fold | per toggles |
| PU-linear / HDR (`compute_pu_linear*`) | Rev5: PU fold; Rev1–4: buffered PU | per revision |

`ZensimProfile::jnd_map` is a score-to-JND mapping, not a spatial path.

### 1.2 Leaf kernels and their parallel implementations

Every leaf below exists in several arithmetic forms. Each form is generally dispatched to v4x, v4, v3, NEON, wasm128
and scalar (hand-written `#[arcane]` x86 bodies for Rev1–3, `#[magetypes]`/generic bodies elsewhere).

| Leaf | Production per-tier (Rev1–3) | Canon f32 (Rev4 era, `c32`) | Canon64 / rec64 (Rev4) | Local window (Rev5) | Exact f64 (oracle) |
|---|---|---|---|---|---|
| sRGB / linear / PU → XYB (`color.rs`) | v4, v4x, v3 + generic with padded tail; separate positive, signed, unclamped, PU variants | `srgb_xyb_canon`, `linear_xyb_canon`, `pu_xyb_canon` (f32x16) | — | uses the canon f32 body | `opsin_px_exact`, `pu_xyb_px_exact` |
| 2× downscale (`blur.rs`) | in-place v4, v4x, v3, generic; `_into` f32x16 and f32x8 | — | — | — | — |
| Horizontal box blur | `box_blur_h`, `_abs_diff`, `fused_blur_h_mu`, `fused_blur_h_ssim{,3}` (v4, v4x incl. strided body, v3, generic), tiled and untiled | `fused_blur_h_ssim_canon` | `fused_blur_h_rec64_*` (hand-written AVX-512 and AVX2 transposes plus scalar/NEON/wasm) | `*_local11`, `*_local11_wide` (v4x), `*_local_general`, `blur_local_rows.rs` | — |
| Vertical box blur | `box_blur_v_copy` (v4, v4x, v3, generic) | — | — | `box_blur_v_local11{,_wide}` | — |
| v1 basic 13-slot V-blur + SSIM/edge/HF (`fused.rs`) | `fused_vblur_ssim_inner` (v4, v4x, v3, generic), edge-only variant | `fused_vblur_ssim_canon` | `fused_vblur_ssim_canon64_vec` + three fallbacks | `fused_vblur_ssim_local` | `fused_vblur_ssim_exact` |
| v1 masked/IW/free extras | band pools in `fold_v1_one_band`, `ext_accumulate{16,8,scalar}`, raw moments, bounded error, luma bins (16/8/scalar each) | Pool leaves | Pool64 | — | Neumaier/Shewchuk oracle |
| v2 dense (29 slots per cell) | era1 generic, era2 (+entries), `pools_scalar` | `dense_block_kernel_canon` | `dense_block_kernel_canon64_vec` + three fallbacks | uses canon64 with `Lanes16F64` | `dense_block_kernel_exact` |
| v2 gradient (GMS, band-visibility, gmsbank) | generic × six const instantiations | `gradient_block_kernel_canon` | `_canon64_vec` + three fallbacks | uses canon64 | `_exact` |
| append, append2, CSFW, gridblk, blockiness, dst edge mask, Rev4 banks, palette | generic + canon (+ exact for append/gridblk) | yes | — | — | yes (some) |
| SSIM reference forms (`ssim_form.rs`) | `ssim_dissim{16,8,scalar}`, `ssim_direct*` | — | — | — | `stable_ssim_plane{,_tiled}`, `precision_reference`, `ssim_dissim_exact` |
| Map pass B (`attribution.rs`, `feature_v2*`) | `attr_pass_b_main/grad/blockiness`, f64 and f32 forms, `combine_basic_and_l8`, `BinAccum`, `box_spread_merge_f32` | — | — | — | — |

Training-only extractors outside this matrix: `cvvdp_features`, `xyb_lms_features`, `palette`.

### 1.3 Size

Non-test, non-oracle lines in `zensim/src` (comments included): 86,206; test and oracle lines: 43,428. Function-name
classification gives lower bounds per arithmetic form (generic bodies shared by several forms are counted under
"other"): hand-written per-tier x86 Rev1–3 bodies 8,765 lines in 64 functions; canon64/rec64 2,794 in 34; canon f32
leaves 1,944 in 48; Rev5 local-window 1,241 in 21. Largest files: `feature_v2.rs` 11,964, `blur.rs` 8,176, `fused.rs`
6,837, `feature_v2_kernels.rs` 6,804, `metric.rs` 5,798, `streaming.rs` 5,425, `attribution.rs` 4,605. There are 93
`incant!` dispatch sites and 160 `#[arcane]`/`#[magetypes]` items.

## 2. Measured cost of the production path (B, Rev5)

### 2.1 Entry points, 1 MP, one thread

| Measurement | Median |
|---|---:|
| `BakeScorer::compute` | 33.59 ms |
| `Zensim::new(B).compute` | 37.13 ms |
| `BakeScorer::compute_with_ref_and_attribution`, cached reference, bin 8 | 47.63 ms |
| `BakeScorer::precompute_reference` | 4.38 ms |
| Same four with `MALLOC_TRIM_THRESHOLD_=MALLOC_MMAP_THRESHOLD_=4e9, MALLOC_TOP_PAD_=1e9` | 32.45 / 33.88 / 45.49 / 2.20 ms |
| `BakeScorer::compute`, 64×64 | 0.236 ms |
| `Zensim::new(B).compute`, 64×64 | 0.347 ms |

The allocator setting changes no feature bit (identical scores); it stops glibc from returning freed pages to the OS
between calls. The named-profile path therefore spends about 3.2 ms per megapixel on fresh scratch allocation and first
touch, and `precompute_reference` spends half its time there.

At 64², `perf` attributes about 28% of the named call to per-call plan derivation: `feature_defs::def_at` 16.0%,
`derived_inputs::from_model` 3.9%, `feature_layout::declared_ids` 2.1%, `bake_read_slots` 1.2%, metadata line parsing,
sorting and `Plan::for_bake`. `BakeScorer` caches its plan; the named profile path does not.

### 2.2 Where a score's time goes (perf, 80 calls, one thread)

| Symbol | Share |
|---|---:|
| `dense_block_kernel_canon64_vec` (v2 dense, `Lanes16F64`) | 27.7% |
| `fused_blur_h_local11` | 20.2% |
| `box_blur_v_local11` | 10.9% |
| `fused_vblur_ssim_canon64_vec` (v1 basic) | 9.1% |
| `srgb_xyb_canon_vec` | 8.1% |
| `OnlineMoments::block` | 7.2% |
| `gradient_block_kernel_canon64_vec` | 5.7% |
| `box_blur_h_local11` | 2.7% |
| memmove / blockiness / abs_diff / downscale | 1.7 / 1.5 / 1.5 / 1.4% |

Fold phase timer (`ZENSIM_FOLD_TIMING`, which itself adds about 6 ms per walk at one thread): producer 7.76 ms
(conversion 4.91, downscale 2.81), H blur 6.82, v1 basic fold 3.17, v2 dense 11.42, v2 gradient 2.15, v2 planes 5.53.
The work census confirms no peaks, masked/IW, append or scale-0 X/B planes are computed, and reports 7.9 M scratch
elements zeroed per call at one thread.

### 2.3 Arithmetic eras on the same weights (old checkout `96c3c9bb`, seed-0 `f803b74c` stamped per revision)

| Revision | Walk | v2 dense | H blur | v1 basic fold | v2 planes | Producer |
|---|---:|---:|---:|---:|---:|---:|
| Rev3, per-tier f32 | 37.6 ms | 3.2 | 12.0 | 6.0 | 6.9 | 5.5 |
| Rev4, canon64 | 41.0 ms | 8.4 | 6.4 | 9.4 | 6.7 | 5.2 |
| Rev5, local window | 33.4 ms | 11.5 | 6.8 | 3.1 | 5.3 | 3.6 |

Seed-0 Rev5 scores are bit-identical between `96c3c9bb` and `e33a638b`. The Rev5 dense kernel costs about 8 ms more
than the Rev3 f32 dense kernel; its f64 16-lane pools are the largest single cost in the production walk.

### 2.4 Threads (P-cores 0–7, phase timer on)

Walk time 39.8 / 31.7 / 26.6 / 14.2 ms at 1 / 2 / 4 / 8 threads. At 2–4 threads the walk takes the original
fuse-channels route (queue floor of three slots) with consume occupancy 0.57 and 0.34; at 8 threads the queue route
runs at occupancy 0.52 while the producer (3.07 ms) stays serial. PERF3 on r7900x: 27.8 ms at 1T, 11.5 at 8T, 10.8 at
16T. COSTCMP on dev (v4x, before PERF3): the Rev1 buffered B reached 7.9 ms at 16T where the Rev5 fold reached 12.3.

## 3. Clean-break options

Projections below are arithmetic over the measured component times in section 2. They bound what a change can save;
they are not measurements of an implementation.

**Without changing any feature value:**

1. Keep scratch and the derived plan inside `Zensim` (named profiles), as `BakeScorer` already does. Measured with the
   allocator setting: 37.1 → 33.9 ms at 1 MP. Thumbnails: plan derivation is about 28% of a 64² call.
2. Reuse `precompute_reference` buffers: 4.38 → 2.20 ms measured with the allocator setting; map path 47.6 → 45.5 ms.
3. Pool v1 basic from the v2 dense pass. Both evaluate SSIM/edge/MSE/HF terms over the same Rev5 planes; the v1 fold is
   3.2 ms. It stays bit-identical only if each pool keeps its own accumulation order (not verified).
4. Overlap or parallelize the producer: serial 3.1 ms of 14.2 at 8 threads, and up to 7.8 ms of the one-thread walk on
   the critical path.

**A new formula revision (Rev6), tier-identical, same 420 features:**

5. f32 lane partials per strip, merged in a fixed order into f64, for the dense, gradient and basic pools. Rev3's f32
   dense kernel costs 3.2 ms against Rev5's 11.5, so the bound is about 8 ms (about 25% of the one-thread walk). It
   needs an accuracy check against `exact` (Rev5 F1's stable-moment requirement and the FEATACC cancellation cases) and
   an E25-style replay of the frozen B weights on Rev6 features before any retraining decision.
6. Together with 1–4, the one-thread compute bound on this box is about 33.6 − 8 − 3 ≈ 22 ms per megapixel. That is a
   bound, not a measurement, and it is not transferable to v4x or Zen 5.

**Deleting the past (code, not speed):**

7. Retire Rev1–Rev4 production arithmetic and the buffered walk: the hand-written per-tier Rev1–3 bodies (≥ 8.8k
   lines), canon64/rec64 (≥ 2.8k), the canon f32 leaves (≥ 1.9k), `streaming.rs` (5.4k) and most of `fused.rs` (6.8k),
   plus the legacy diffmap and v2 diffmap owners. Cost: profiles A, PreviewV0_x, LegacyLinearV0_2, BHdr, C, CHdr and D
   lose bit reproducibility and must be rebaked on Rev5/6 features or dropped. Stored Rev1–Rev4 tables can no longer be
   regenerated from main; a pinned historical tag would be the reproduction tool. HDR needs the Rev5 PU fold to become
   the BHdr route first. Stoppable requests need a stop hook in the fold.
8. Keep exactly one oracle: the f64 exact bodies, test-only, as the ruler for the one product arithmetic.

**Compromises that trade accuracy or reproducibility for speed:**

9. Drop cross-tier bit identity (Rev3-style per-tier FMA and reduction order). It goes against the W11 determinism goal
   and the owner's Rev4/Rev5 decisions; the measured gain over option 5 is not established.
10. Remove v2 dense terms the model barely uses (for example the PJND transducer bank). `costpot_2026-10-03` and
    `costset2/3` own that evidence; per project rules this is worth doing only when it removes real kernel work.

## Appendix: probe

Out-of-repo crate `~/tmp/zensim-inventory/probe` (path dependency on `zensim` with `custom-profiles` and
`feature-regime-v2`, zenpredict/zenpredict-bake at the workspace pin `f0903489`). It loads a bake, optionally stamps
`zentrain.formula_revision` with `zenpredict_bake::append_metadata_utf8`, and times `BakeScorer::compute`,
`Zensim::new(B).compute`, `precompute_reference` and `compute_with_ref_and_attribution` on the SPEEDQ synthetic pair.
Environment: `PROBE_BAKE`, `PROBE_MODE=score|named|both|maponly`, `PROBE_W/H`, `PROBE_N`, `PROBE_PAR`, `PROBE_STAMP`.
Profiles: `perf record -F 2999` on `taskset -c 2`, reported with `perf report --no-children --sort symbol`.
