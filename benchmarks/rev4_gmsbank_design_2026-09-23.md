# GMSBANK C8 design (2026-09-23)

**Historical initial definition below; superseded for C8 only by the dated chroma revision appended at the end.**

Initial base `1881409d`, width 1322. The featbank correction review revises C1's magnitude hats and C3's histogram range; this lane rebases onto that corrected tip before final qualification. This note fixes the C8 definition before feature implementation and before the pixel-only constant calibration. No human labels or fit results are inputs.

## Why this form

GMSD's core is a gradient-similarity map followed by deviation pooling. Its paper says `c=0.0026` on 0..1 luma and uses a 2×2 box downsample followed by Prewitt gradients. The local corpus record is `zenpapers` `/mnt/v/input/papers/03/03d268a2140b4dfb8e86d30c6302db6c8eca0f99e524642a5939de291146c9b6.md` (Xue et al., DOI 10.1109/TIP.2013.2293423). The measured KonJND-504 and CSIQ advantage over zensim B in `docs/REV4_EXPERIMENTS_2026-09-23.md` motivates a human-label potential test, while the negative prior screen in `../zensim--gmsd/benchmarks/gmsd_2026-09-22.md` was selected on an SSIMULACRA2-heavy proxy objective.

MS-GMSD (`zenpapers` `/mnt/v/input/papers/25/25effe3f7937f57333471738f104524caab1014d1a271ee70c03ad48798f9131.md`, Zhang et al., DOI 10.1109/ICASSP.2017.7952357) argues for multiple scales, explicit chroma, and a tunable masking term. The four-scale X/Y/B bank covers the first two as separate fitted signals. Its proposed masking formula is materially different from this bank's fixed `c` basis; adopting it would change the local similarity definition and confound the current comparison, so it is not adopted here. MDSI (`zenpapers` `/mnt/v/input/papers/35/35b48f91668bd05676582d72fd846c2033a4dd5beb3de4db22c6a17cc0ad397e.md`, DOI 10.1109/ACCESS.2016.2604042) also combines gradient and chroma similarity with generalized deviation pooling; that is a separate nonlinear family, not this registered C8 candidate. The corpus gap analysis `zenpapers/docs/zensim-720-feature-gaps-2026-07-26.md` §W6/W8 specifically identifies deviation pooling and the central-difference versus Prewitt front-end gap.

## Slots and arithmetic

`gmsbank` is `PerChannel`, scale-major and X/Y/B-minor, with 15 contiguous slots per cell: for k=0..4, `loss_k`, `gain_k`, `dev_k`. Thus `(4 scales)*(3 channels)*(5 constants)*(3 signals)=180`, f1322–f1501, width 1502. `c_k=c_mid*4^(k-2)`; `c_mid` is frozen after label-free calibration in the preregistered manner of `benchmarks/gmsbank_prereg_2026-09-23.md`. All five `c_k` are pinned as source literals after calibration; the hot path performs no transcendental constant construction. Every C8 slot must be nonzero on at least one of the 144 TRAIN pairs, or qualification fails.

The operands `m_r,m_d` are the central-difference magnitudes already computed by `gradient_block_kernel_generic`, in unit-XYB, with its current boundary convention. The local map is `GMS_k=(2m_rm_d+c_k)/(m_r²+m_d²+c_k)`. Loss and gain partition `1-GMS_k` by strict `m_d<m_r`; the tie belongs to gain, where its contribution is zero on identity. Each is divided by the number of real pixels. Deviation is the **population** standard deviation of the GMS map, with per-row Welford in f64 and row-ordered Chan merge. GMSD's published peer score instead uses a sample standard deviation; the control comes from the parity-verified `gmsd` crate with its own convention and gamma-luma input.

For the deviation accumulator, Welford ingests `delta=1−GMS` directly. `std(delta)=std(GMS)` by affine invariance, while the difference form preserves sub-ULP changes near GMS=1 that `1−delta` would erase. The output is still the population standard deviation of the GMS map in the stated real-number definition.

## Pixel calibration implementation fixed before measurement

The RGB8 decode owner is `zensim-bench/examples/shared/score_input.rs` in `LegacyRgb8` mode, the same owner as the bank extractor. Decode digests and `pair_key = SHA256(ref_digest || dist_digest || "legacy-rgb8")` are asserted before a pair enters the sweep. A reference is classified from its decoded RGB8 samples at `(x,y)` every eight pixels where an eight-pixel right neighbor exists. At each site, quantize R/G/B to five bits to count unique colors; a flat site has equal quantized colors at the two positions; an edge site has an integer BT.601 luma difference above 24. `line_art` is at most 64 colors with edge rate over 0.08; otherwise `screen` has flat rate at least 0.45 and over 64 colors; otherwise `mixed` has flat rate at least 0.20; otherwise `photo`. Size uses the original maximum dimension. These thresholds and their order are fixed without reading ratios.

The calibration ratio co-sites interior pixels `(x,y)`, `1 <= x < floor(w/2)-1`, `1 <= y < floor(h/2)-1`. GMSD's side first rounds gamma RGB8 to BT.601 gray as `round(0.299R+0.587G+0.114B)`, averages each aligned 2×2 gray box and divides by 255. It takes the standard 3×3 Prewitt X/Y gradients, normalized by 3. The zensim side converts the same RGB8 pixels to XYB and applies the existing `downscale_2x_inplace` once; its Y central differences are `right-left` and `down-up` on that scale-1 plane, without a factor of two. Both magnitudes must exceed `1e-6`; a pair contributes the median of all eligible reference and distortion site ratios. Borders are excluded from both operators. All output planes, ratios and class records live under `/var/tmp/gmsbank/calibration/`.

Every new slot is exactly zero for an identical pair. Use a difference-form similarity evaluation for the bank so `m_r==m_d` yields a bit-exact zero loss before deviation pooling; this also guards FMA contraction. Keep the existing f0–f1321 arithmetic untouched. Dispatch one `#[rite]` helper from the current `#[arcane]` gradient entry through existing `incant!` tiers. The off toggle takes no extra pixel work. Register the token, kernel, form, direction, cost, revision and 1502 layout. The coordinator authorized exactly two additive public Rust items: `ComputeToken::Gmsbank` (an arm of the existing `#[non_exhaustive]` enum) and `V2NewFeatureToggles::gmsbank` (a field of the existing toggle struct). There is no new public type, function, trait method, feature flag, or crate-root re-export. The extractor's `--full-gmsbank` option is a CLI flag, not a Rust API item.

**SIMD implementation constraint discovered after the design commit.** The gradient pass is generic over `T: F32x8Backend`. The local `zensim/src/fused.rs` and `docs/PLAN_KERNEL_FASTCLASS_2026-09-05.md` establish that `#[rite]` cannot decorate a helper generic over a backend trait: the macro needs one concrete token to attach a target feature. The implementation therefore uses `#[inline(always)]` generic helpers inside the existing `#[magetypes]` tier entries, which expand to `#[arcane]` entries and are chosen by `incant!`. This is the repository's measured pattern for avoiding the 5.3× regression from a surviving cross-feature generic call. The C8 arithmetic stays fused into that gradient walk; there is no second gradient pass. The absence of a literal `#[rite]` on the generic helper is a disclosed deviation from the brief's syntax requirement, to be checked by release disassembly and the tier/cost gates.

## Frozen checks

The preregistered gates, controls and data roles are in `benchmarks/gmsbank_prereg_2026-09-23.md`. The potential arm proposal is separate because the potential owner performs the fit. This design makes no claim that a five-point bank will beat the exact GMSD peer; P1/P2/P3 test that question.


## C8 chroma revision — 2026-09-24 (quarantined)

Preregistration `9016272c`, implementation `f38befaefc524d5aeb3585cef81d81902a47bfb9`;
corrected parent `620a384e06e49422d501bdddc9c0439afc846a98`. This definition
supersedes the earlier unconsumed C8 block only. f0–f1321 keep their definitions.
The same total width 1502 does **not** imply compatibility with prior C8
features: old sidecars must not join or train as this revised era.

Native scale carries the five Y loss/gain/deviation triples (15 slots).
Each of scales1,2,3 carries X/Y/B gradient triples (45 slots), followed by
five joint chromaticity loss/deviation pairs (10 slots). Thus C8 widths for
1/2/3/4 scales are 15/70/125/180. The registry records sparse placement and
round-trips every slot; native X/B and native CS placements are refused.
Chroma begins at half resolution to reflect its lower spatial sensitivity;
Mullen1985 DOI10.1113/jphysiol.1985.sp015591 motivates the coarse placement,
without asserting that image scales equal calibrated retinal frequencies.

X/B gradient stabilizers and X/B value stabilizers were mapped separately
from MDSI's opponent units using TRAIN pixels under the committed prereg.
Y literals remain unchanged. The constants, ratio quartiles, stratum gaps,
empty counts and input hashes are in `gmsd-chroma_calibration_2026-09-24.md`.
The opponent-to-XYB mapping is an empirical unit conversion, not an exact
colour transform identity or a learned performance optimum.

At each retained scale set x=X-f64(0.42f32), b=B-f64(0.55f32), retaining X's
existing factor14. For each literal pair (Cx,Cb), define
L=((xr-xd)^2/Cx+(br-bd)^2/Cb)/(xr²/Cx+xd²/Cx+br²/Cb+bd²/Cb+1).
Emit mean(L) and population std(L), accumulated per row with f64 Welford,
then ordered Chan merges. Identity is exactly zero. For unshifted MDSI H/M
planes and equal constants550 this is 1-CS; equal constants55 are the
negative control. This does not reproduce MDSI's final nonlinear pooling.

The X gradient walk reads co-sited B rows from the existing strip producer.
It computes both chromaticity statistics inside the existing magetypes/arcane
loop, including scalar borders and tails, with no extra image traversal.
The materialized reference path uses the same co-sited rows. Default OFF.
The generic inline helper uses the same established token limitation noted
above. No public API change accompanies this revision.

Differential results: author CS116 cases pass, maximum absolute error
7.771561172376096e-16 and relative5.509513936691231e-12; wrong550/10
constant rejected110 cases. Independent NumPy XYB mirror:8 TRAIN pairs,
1440 C8 cells, maximum relative error1.860909581448716e-15; wrong×16
constants rejected1440 cells. Behavior and registry gates passed; remaining
prefix/cost/full-build status is reported in the lane worklog and DONE file.
