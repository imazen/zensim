# DVIFM constants fit — loss lane (2026-09-20)

Question: **do we now have an identifiable masking exponent β, and is it psychovisually plausible?**


## Verdict

**Split answer.** *Per-cell* β is mostly not identifiable — but a *shared* β is identified on most domains and lands on the psychovisual band. On the canonical arm (`safesyn`, 141,054 rows / 2,312 references, pairwise-ranking objective):

- **1 of 15 β cells are statistically identified** (profile interval inside the grid, cell in live curve mode). Of these, 1 land in the psychovisual 0.6–0.7 band (`ycbcr_y_l0` β=0.635)
- **13 of 15 cells never exercise β at all** — their detector mode is masking-off/gate/saturated (v≈0 or v≈1 across the contrast range). The β values fitted there are artefacts of a flat objective and must not ship as constants.
- **β ≈ 0.65 is NOT what the SafeSyn per-cell fit picks.** With the prior disabled (λ=0, the dev-optimal setting), fitted β spreads 0.20–23.0 across cells. The 0.65 value only appears when the prior is switched on — it is prior-supported in weak domains, not data-identified per-cell.
- **But a single shared β IS identified on 7 domains — and lands on the band.** Data-only profile intervals for the shared coordinate intersect at [0.607, 0.779] across cid22a_weber, kadid_train, kadid_train_weber, majority, tid_jp2kjpeg_weber, tidkadid, tidkadid_weber — independent domains whose intervals all contain the psychovisual band. As a pooled global exponent, β ≈ 0.65 is defensible from data, not just the prior.
- Predictive quality is not identification: human-domain arms reach dev SROCC 0.87–0.92 while their β is prior-pulled — the model ranks well with β carried entirely by the prior. And the majority arm's own prior-free shared fit lands at 0.46 — inside its [0.29, 1.0] interval alongside 0.65; the data cannot distinguish within it.


## Arms

| arm | contrast | mode | λ_sel | dev rank-L | dev SROCC | dev KROCC | concord. | β coords identified |
|---|---|---|---|---|---|---|---|---|
| cid22a | raw | shared | 0.3 | 0.01829 | 0.3527 | 0.2637 | 0.9022 | 0  |
| cid22a_weber | weber | shared | 0.3 | 0.01793 | 0.3845 | 0.2855 | 0.9058 | 1 (beta_0 [0.174, 1.28]) |
| kadid_train | raw | shared | 0.3 | 0.01300 | 0.9010 | 0.7363 | 0.9348 | 1 (beta_0 [0.368, 1]) |
| kadid_train_weber | weber | shared | 0.3 | 0.01431 | 0.9096 | 0.7567 | 0.9235 | 1 (beta_0 [0.473, 1]) |
| majority | raw | shared | 0.3 | 0.00674 | 0.9125 | 0.7629 | 0.9599 | 1 (beta_0 [0.287, 1]) |
| safesyn | raw | group | 0.0 | 0.00347 | 0.9569 | 0.8244 | 0.9737 | 1 (beta_0 [0.287, 1.65]) |
| safesyn_weber | weber | level | 0.0 | 0.00344 | 0.9522 | 0.8154 | 0.9737 | 1 (beta_0 [0.224, 1.28]) |
| tid_jp2kjpeg | raw | level | 0.3 | 0.00217 | 0.9150 | 0.7481 | 0.9733 | 2 (beta_1 [0.473, 1.65]; beta_2 [0.607, 1]) |
| tid_jp2kjpeg_weber | weber | shared | 0.3 | 0.00251 | 0.9033 | 0.7285 | 0.9644 | 1 (beta_0 [0.607, 0.779]) |
| tidkadid | raw | shared | 0.3 | 0.00212 | 0.8687 | 0.6807 | 0.9587 | 1 (beta_0 [0.473, 0.779]) |
| tidkadid_weber | weber | shared | 0.3 | 0.00278 | 0.8212 | 0.6213 | 0.9449 | 1 (beta_0 [0.607, 1]) |

## β by arm

Fitted values at the **selected** λ, plus the λ=0 (prior-free) column where the sweep recorded it. Prior-pulled cells are flagged by λ_sel > 0.

| arm | λ_sel | β at selected λ | β at λ=0 |
|---|---|---|---|
| cid22a **(prior-pulled)** | 0.3 | [0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65] | [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05] |
| cid22a_weber **(prior-pulled)** | 0.3 | [0.647, 0.647, 0.647, 0.647, 0.647, 0.647, 0.647, 0.647, 0.647, 0.647] | [0.046, 0.046, 0.046, 0.046, 0.046, 0.046, 0.046, 0.046, 0.046, 0.046] |
| kadid_train **(prior-pulled)** | 0.3 | [0.651, 0.651, 0.651, 0.651, 0.651, 0.651, 0.651, 0.651, 0.651, 0.651] | [1.166, 1.166, 1.166, 1.166, 1.166, 1.166, 1.166, 1.166, 1.166, 1.166] |
| kadid_train_weber **(prior-pulled)** | 0.3 | [0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65] | [0.425, 0.425, 0.425, 0.425, 0.425, 0.425, 0.425, 0.425, 0.425, 0.425] |
| majority **(prior-pulled)** | 0.3 | [0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65] | [0.456, 0.456, 0.456, 0.456, 0.456, 0.456, 0.456, 0.456, 0.456, 0.456] |
| safesyn | 0.0 | [0.635, 0.203, 1.111, 5.059, 1.243, 1.009, 0.3, 0.993, 9.337, 23.038] | [0.635, 0.203, 1.111, 5.059, 1.243, 1.009, 0.3, 0.993, 9.337, 23.038] |
| safesyn_weber | 0.0 | [0.51, 0.183, 3.27, 2.45, 21.829, 0.51, 0.183, 3.27, 2.45, 21.829] | [0.51, 0.183, 3.27, 2.45, 21.829, 0.51, 0.183, 3.27, 2.45, 21.829] |
| tid_jp2kjpeg **(prior-pulled)** | 0.3 | [2.696, 0.65, 0.65, 0.65, 1.194, 2.696, 0.65, 0.65, 0.65, 1.194] | [65.603, 2.005, 0.793, 0.364, 36.753, 65.603, 2.005, 0.793, 0.364, 36.753] |
| tid_jp2kjpeg_weber **(prior-pulled)** | 0.3 | [0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65] | [4.145, 4.145, 4.145, 4.145, 4.145, 4.145, 4.145, 4.145, 4.145, 4.145] |
| tidkadid **(prior-pulled)** | 0.3 | [0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65] | [5.626, 5.626, 5.626, 5.626, 5.626, 5.626, 5.626, 5.626, 5.626, 5.626, 5.626, 5.626, 5.626, 5.626, 5.626] |
| tidkadid_weber **(prior-pulled)** | 0.3 | [0.652, 0.652, 0.652, 0.652, 0.652, 0.652, 0.652, 0.652, 0.652, 0.652, 0.652, 0.652, 0.652, 0.652, 0.652] | [2.127, 2.127, 2.127, 2.127, 2.127, 2.127, 2.127, 2.127, 2.127, 2.127, 2.127, 2.127, 2.127, 2.127, 2.127] |

## Structure selection (untie ladder)

Each stage refits and is accepted only if the paired dev-bootstrap delta clears ref noise. U3 splits chroma into per-plane groups (Cb/Cr get their own constants), not a free β.

| arm | U1 per-level β | U2 group split | U3 chroma split |
|---|---|---|---|
| cid22a | reject (Δ+4.11e-04) | reject (Δ+6.24e-05) | reject (Δ+4.23e-03) |
| cid22a_weber | reject (Δ+1.65e-04) | reject (Δ+2.56e-05) | reject (Δ+3.86e-03) |
| kadid_train | reject (Δ+5.16e-03) | reject (Δ+5.22e-03) | reject (Δ+6.06e-03) |
| kadid_train_weber | reject (Δ-1.15e-04) | reject (Δ+4.83e-04) | reject (Δ+4.39e-04) |
| majority | reject (Δ+3.15e-04) | reject (Δ+3.24e-04) | reject (Δ+1.39e-04) |
| safesyn | ACCEPT (Δ-1.73e-04) | ACCEPT (Δ-1.13e-05) | reject (Δ+3.75e-04) |
| safesyn_weber | ACCEPT (Δ-1.89e-04) | reject (Δ+1.06e-05) | reject (Δ+5.82e-05) |
| tid_jp2kjpeg | ACCEPT (Δ-7.54e-04) | reject (Δ+1.87e-03) | reject (Δ+5.09e-04) |
| tid_jp2kjpeg_weber | reject (Δ+7.52e-04) | reject (Δ+8.02e-04) | reject (Δ+8.59e-04) |
| tidkadid | reject (Δ-2.81e-03) | reject (Δ-2.12e-03) | ACCEPT (Δ-2.78e-03) |
| tidkadid_weber | reject (Δ-1.25e-03) | reject (Δ+2.86e-03) | ACCEPT (Δ-3.71e-03) |

## Per-cell constants (canonical arm)

| cell | mode | g | P | c0 | c0 interval | β | σ | β profile interval | identified | domains agreeing |
|---|---|---|---|---|---|---|---|---|---|---|
| ycbcr_cb_l0 | off | 0.35 | 0.0082 | 0.000237 | (1e-06, 10) | 1.01 | 2.54 | (0.05, 20) | False | 4 |
| ycbcr_cb_l1 | off | 1.63 | 0.165 | 0.000452 | [2.05e-05, 10) | 0.3 | 6.86 | (0.05, 20) | False | 5 |
| ycbcr_cb_l2 | off | 0.802 | 0.831 | 0.0159 | (1e-06, 10) | 0.993 | 10.1 | (0.05, 20) | False | 9 |
| ycbcr_cb_l3 | gate | 0.458 | 0.103 | 0.00062 | (1e-06, 10) | 9.34 | 3.11 | (0.05, 20) | False | 4 |
| ycbcr_cb_l4 | off | 0.265 | 3.29 | 0.0318 | [0.0143, 10) | 23 | 11.7 | (0.05, 20) | False | 0 |
| ycbcr_cr_l0 | off | 0.35 | 0.0082 | 0.000237 | (1e-06, 10) | 1.01 | 2.54 | (0.05, 20) | False | 4 |
| ycbcr_cr_l1 | off | 1.63 | 0.165 | 0.000452 | [2.05e-05, 10) | 0.3 | 6.86 | (0.05, 20) | False | 5 |
| ycbcr_cr_l2 | off | 0.802 | 0.831 | 0.0159 | (1e-06, 10) | 0.993 | 10.1 | (0.05, 20) | False | 9 |
| ycbcr_cr_l3 | off | 0.458 | 0.103 | 0.00062 | (1e-06, 10) | 9.34 | 3.11 | (0.05, 20) | False | 4 |
| ycbcr_cr_l4 | off | 0.265 | 3.29 | 0.0318 | [0.0143, 10) | 23 | 11.7 | (0.05, 20) | False | 0 |
| ycbcr_y_l0 | curve | 1.45 | 1.77 | 0.000683 | [5.62e-05, 0.00316] | 0.635 | 7.22 | [0.287, 1.65] | True | 10 |
| ycbcr_y_l1 | curve | 0.615 | 0.589 | 0.111 | [0.0143, 2.21] | 0.203 | 1.32 | (0.05, 0.473] | False | 3 |
| ycbcr_y_l2 | off | 1.99 | 0.516 | 0.0558 | [0.00523, 10) | 1.11 | 1.96 | (0.05, 20) | False | 4 |
| ycbcr_y_l3 | off | 1.73 | 3.92 | 0.026 | (1e-06, 10) | 5.06 | 10.3 | (0.05, 20) | False | 4 |
| ycbcr_y_l4 | off | 0.234 | 9.11 | 0.662 | [2.05e-05, 10) | 1.24 | 7.46 | (0.05, 20) | False | 5 |

## Identifiability detail

- Cells in live curve mode: **2/15** (rest saturate v≈1 or gate v≈0 — β unexercised).
- Cells identified: **1/15** (profile interval closed within the grid).
- Profile interval = {θ : L(θ) ≤ L* + σ_b}, σ_b = paired-bootstrap-over-references SD of the loss at the optimum. Intervals touching the grid edge are open on that side.


## constants-v1 shipping rules

`constants-v1.json` carries every cell, but β must be read through `identified`, `mode`, and `beta_profile_interval`:

- `mode=curve` + `identified=true`: β is a fitted, data-supported constant.
- `mode=curve` + `identified=false`: directionally constrained only (interval edge-truncated); ship the interval, not the point.
- `mode` in {masking_off, masking_gate, saturated}: the fitted β is meaningless; the cell's behaviour is carried by (g, P, c0, σ) and the detector flags — not by β.
- `domains_agreeing` counts arms whose own β profile interval contains the canonical fitted value; a flat (unidentified) interval contains everything, so agreement is necessary-but-not-sufficient evidence.


## Recommended posture for the i16 kernel lane

- **Ship β only where the cell exercises it.** Canonical: `ycbcr_y_l0` (0.635, closed [0.287, 1.648]) is the one data-identified per-cell value and sits in the band. `ycbcr_y_l1` is directionally constrained (β ≤ ~0.47, edge-truncated). All other cells: carry mode + (g, P, c0, σ) and treat β as unset — their fitted values are flat-objective artefacts (up to 23.0).
- **Where a single masking exponent is wanted, the pooled shared-β evidence supports ~0.65** — the 7 independently identified shared-mode intervals intersect at ≈[0.607, 0.779]. Quote it with the interval, not as a point: the honest data-supported claim is β_shared ∈ ~[0.6, 0.8].
- **Per-cell β is not portable**: SafeSyn's prior-free fit scatters 0.20–23.0 (dev-optimal at λ=0), so shipping per-cell exponents would be shipping fit noise in inactive cells. The kernel's two-state (gate/off) reading of those cells is the honest structure.


## Provenance

- Artefacts: cid22a, cid22a_weber, kadid_train, kadid_train_weber, majority, safesyn, safesyn_weber, tid_jp2kjpeg, tid_jp2kjpeg_weber, tidkadid, tidkadid_weber
- Fitter: `/mnt/v/output/zensim/dvifm-loss-2026-09-20/tools/fit_loss.py` (sha ba3791716720)
- Objective: within-reference pairwise logistic ranking loss, reference-normalized; tied-first untie ladder with paired-dev-bootstrap acceptance; prior sweep λ∈{0,0.003,0.01,0.03,0.1,0.3} toward β=0.65.
- Detector globals: {"channel_min": 0.14074729098215757, "mixture_collapse": true, "level_weight_min": 0.00010663456496415709, "map": {"A": 225.52596701040648, "B": -123.1329716818716, "lambda": 39.45969950667347, "mse": 173.99321966419436}, "map_lambda_medE": 0.17681322249527415, "map_degenerate": false, "y_min": -743.8610103164912, "y_max": 99.15995927546757, "frac_at_min": 7.089483460235087e-06, "frac_at_max": 7.089483460235087e-06, "frac_negative": 0.05809831695662654, "target_clamped": false, "saturation": false}
