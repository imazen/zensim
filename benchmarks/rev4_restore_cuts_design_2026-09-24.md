# Restored-cut families: design (2026-09-24, quarantined)

Lane `restore-cuts`. Preregistration: `benchmarks/restore-cuts_prereg_2026-09-24.md`. Implementation commit `2d7b33dd`
(`RESTORE_COMMIT` in `feature_defs.rs`). Slots are append-only after f1501 and default off; f0..f1501 are unchanged.

| token | slots (4 scales) | signals per cell | replication |
|---|---|---|---|
| `mapdev` | f1502..1561 (60) | `mse_dev`, `hfsq_src_dev`, `hfsq_dst_dev`, `hfabs_src_dev`, `hfabs_dst_dev` | `PerChannel`, cell = `scale*3 + channel` |
| `z1max` | f1562..1789 (228) | the 13 `BASIC` then 6 `PEAKS` v1 signals, pooled over block maxima | `PerChannel` |
| `gmsnative` | f1790..1819 (30) | C8's `loss_k, gain_k, dev_k` for k = 0..4 (X cell, then B cell) at scale 0 | `NativeXb` (new replication) |
| `dvifmgate` | f1820..1824 (5) | C7 F1 with the two-state gate, one per DVIFM level | `Flat` |

## Why a band-kernel side pass for mapdev and z1max

The v1 fused kernel is duplicated across four SIMD tiers, three lane widths and an edge-only variant (about 24
accumulation sites). It already stores `mu1`, `mu2` and the SSIM dissimilarity `sd` for a band's inner rows
(`store_mu`, `store_sd`), so the eight per-pixel maps the 228 surface pools are re-derivable per band with the
kernel's own f32 formulas, exactly as the zgeom lane's `v1_channel_maps_and_sums` does. `feature_v2/restore_cuts.rs`
runs that band loop per (scale, channel) cell over the production pyramids (`build_v2_ref_scales`) and feeds each
inner row, in plane order, to two consumers. No kernel is touched, so the existing paths cannot move
(`--full-gmsbank` CSV of 2,000 SafeSyn pairs is byte-identical to main's). Cost: one more V blur and kernel sweep
per cell, reported in the cost record, not gated.

The eight maps: `sd` (kernel side output), `art = max(ed,0)`, `det = max(-ed,0)` with
`ed = (1+|d-mu2|)/(1+|s-mu1|) - 1`, `mse = (s-d)^2`, `hfsq_src = (s-mu1)^2`, `hfsq_dst = (d-mu2)^2`,
`hfabs_src = |s-mu1|`, `hfabs_dst = |d-mu2|`.

## mapdev

Population standard deviation (`sqrt(M2/n)`, the C8 convention; the research patch used the sample std, a factor
`sqrt(n/(n-1))`). Per-row Welford in f64, pixels in ascending `x`, using precomputed reciprocals `1/k`; rows merged
in row order with Chan's formula. Never `sum(x^2) - mean^2`. Five maps, not the record's three: the emitted HF slots
are ratio forms (`var_loss`, `tex_loss`, `contrast_inc`) of the `hfsq` and `hfabs` means, so neither map's second
moment is recoverable from existing columns. The std of the SSIM, art and det maps is (per the record) an exact
function of existing mean and L2 columns and is not carried.

Identity: `mse_dev` is exactly zero on an identical pair (registry `Difference`). The four HF slots are properties
of one image, so on an identity pair `hfsq_src_dev == hfsq_dst_dev` and `hfabs_src_dev == hfabs_dst_dev` bit for bit
and neither is zero (registry `ReferenceOnly` / `Undeclared`). Part B's "identity-exact 0" acceptance applies to the
`Difference` slots only; `bank_sidecar.py` checks each family against its registry form.

## z1max

The record's `Block5Max` at gate weight 1: the (0,0)-anchored 5x5 lattice, partial border blocks dropped, each
block contributing the maximum of each of the eight maps (max initialised at 0.0, as the record), `n` = surviving
block count, accumulated into `V1BasicSums` and finalized by its own `finalize_into` / `finalize_pools_into` (so
`ssim_max`, `edge_*_max` are the global maxima and the L4/L8 slots go through the same `RootForm`). Rows are consumed
in plane order with a 5-row carry of per-block running maxima, so block statistics never depend on band tiling.
The block+global hybrid the record proposes is the two-surface arm `R0 + z1max` (R0 already carries the global 228
surface); no separate slot exists for it.

The 4th/8th-root slots use new `Statistic::BlockL4/BlockL8` and are deliberately not attached to the `v1detroot`/F18
registry gates, whose pinned counts are measurements over the v1/v2 slots
(`registered_defects_cover_exactly_the_audited_slots` failed when they were `L4/L8`). Attaching them is a change to
that gate's pinned counts and belongs to promotion, not to this optional research arm.

## gmsnative

`gmsbank_cell_live` now also runs the C8 gradient bank for X and B at scale 0 when `gmsnative` is on; the same
per-channel stabilisers (`GMSBANK_X_C`, `GMSBANK_B_C`) and accumulators as the coarse X/B cells, emitted into their
own block. No new arithmetic. The walk requires `full_res_xb` for it. Native joint chromaticity (CS) is not carried.

## dvifmgate

Verified before any code: main's `dvifm.rs` computes C7 (f956..985) with the smooth curve
`v(C) = (1+(C/C0)^(beta*sigma))^(-1/sigma)`; the two-state gate exists only on unlanded `943781e8`
(`DvifmVis::{Curve,Gate,Off}`). So the curve form already IS C7 (bank sidecar `features__csfw_dvifm.parquet`) and the
missing arm is the gate. Only F1 depends on `v`; the F2 bins are independent of it, so the whole gate variant is one
extra sum per level: `Sum v_gate * m^P / n`, `v_gate = max over sides of [C <= c0]` with the level's fitted `c0`,
accumulated in `pool_block` next to `f1` in the same block order. It moves no existing value (asserted).

## Planner behaviour worth knowing (measured, not assumed)

`Plan::normalized` derives the compute set from the layout, so a layout that reaches a block computes it
(`a_wide_layout_computes_every_block_it_reaches`): a family cannot be requested "alone" at a wide layout, and the
v1 pools f156..371 are computed only when requested. The nested chain lets a family be requested at the narrowest
layout that reaches it; its values do not depend on later families (asserted by `restore_cuts_parity`). The extractor's
`--restore-cuts prefix,...` also requests f0..f1501 (needed to compare the whole existing surface with the families on).

## Tier policy

Within a SIMD tier every new slot is bit-identical across serial, MT8, strided input and repeated runs (asserted at
Rev1 and at Rev3). Across tiers the new slots are NOT bit-identical: the XYB planes come from tier-specific SIMD cube
roots and the v1 blur/SSIM kernels the side pass calls are also tier-dispatched, so values agree to float precision
only; the responsible stage was not isolated. Following the rev4/C8 policy the drift is measured and reported, with
its revision: the lane's first run was at the shipped default **Rev1** (non-SSIM worst relative 5.7e-3, SSIM-derived
z1max 6.8e-2); the bank is **Rev3**, where the review measured non-SSIM 7.8e-5..5.7e-3 and SSIM-derived
7.6e-5..2.9e-4 (this lane's own Rev3 rerun is in the worklog). The largest relative differences sit on the smallest
values (~1e-4 magnitude at the coarse scales). Rev3 invocation: `just restore-cuts-parity-rev3`.
