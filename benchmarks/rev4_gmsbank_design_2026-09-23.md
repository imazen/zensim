# GMSBANK C8 design (2026-09-23)

Base `1881409d`, width 1322. This note fixes the definition before feature implementation and before the pixel-only constant calibration. No human labels or fit results are inputs.

## Why this form

GMSD's core is a gradient-similarity map followed by deviation pooling. Its paper says `c=0.0026` on 0..1 luma and uses a 2×2 box downsample followed by Prewitt gradients. The local corpus record is `zenpapers` `/mnt/v/input/papers/03/03d268a2140b4dfb8e86d30c6302db6c8eca0f99e524642a5939de291146c9b6.md` (Xue et al., DOI 10.1109/TIP.2013.2293423). The measured KonJND-504 and CSIQ advantage over zensim B in `docs/REV4_EXPERIMENTS_2026-09-23.md` motivates a human-label potential test, while the negative prior screen in `../zensim--gmsd/benchmarks/gmsd_2026-09-22.md` was selected on an SSIMULACRA2-heavy proxy objective.

MS-GMSD (`zenpapers` `/mnt/v/input/papers/25/25effe3f7937f57333471738f104524caab1014d1a271ee70c03ad48798f9131.md`, Zhang et al., DOI 10.1109/ICASSP.2017.7952357) argues for multiple scales, explicit chroma, and a tunable masking term. The four-scale X/Y/B bank covers the first two as separate fitted signals. Its proposed masking formula is materially different from this bank's fixed `c` basis; adopting it would change the local similarity definition and confound the current comparison, so it is not adopted here. MDSI (`zenpapers` manifest record DOI 10.1109/ACCESS.2016.2604042) also combines gradient and chroma similarity with generalized deviation pooling; that is a separate nonlinear family, not this registered C8 candidate. The corpus gap analysis `zenpapers/docs/zensim-720-feature-gaps-2026-07-26.md` §W6/W8 specifically identifies deviation pooling and the central-difference versus Prewitt front-end gap.

## Slots and arithmetic

`gmsbank` is `PerChannel`, scale-major and X/Y/B-minor, with 15 contiguous slots per cell: for k=0..4, `loss_k`, `gain_k`, `dev_k`. Thus `(4 scales)*(3 channels)*(5 constants)*(3 signals)=180`, f1322–f1501, width 1502. `c_k=c_mid*4^(k-2)`; `c_mid` is frozen after label-free calibration in the preregistered manner of `benchmarks/gmsbank_prereg_2026-09-23.md`.

The operands `m_r,m_d` are the central-difference magnitudes already computed by `gradient_block_kernel_generic`, in unit-XYB, with its current boundary convention. The local map is `GMS_k=(2m_rm_d+c_k)/(m_r²+m_d²+c_k)`. Loss and gain partition `1-GMS_k` by strict `m_d<m_r`; the tie belongs to gain, where its contribution is zero on identity. Each is divided by the number of real pixels. Deviation is the **population** standard deviation of the GMS map, with per-row Welford in f64 and row-ordered Chan merge. GMSD's published peer score instead uses a sample standard deviation; the control comes from the parity-verified `gmsd` crate with its own convention and gamma-luma input.

Every new slot is exactly zero for an identical pair. Use a difference-form similarity evaluation for the bank so `m_r==m_d` yields a bit-exact zero loss before deviation pooling; this also guards FMA contraction. Keep the existing f0–f1321 arithmetic untouched. Dispatch one `#[rite]` helper from the current `#[arcane]` gradient entry through existing `incant!` tiers. The off toggle takes no extra pixel work. Register the token, kernel, form, direction, cost, revision and 1502 layout without a new public API.

## Frozen checks

The preregistered gates, controls and data roles are in `benchmarks/gmsbank_prereg_2026-09-23.md`. The potential arm proposal is separate because the potential owner performs the fit. This design makes no claim that a five-point bank will beat the exact GMSD peer; P1/P2/P3 test that question.
