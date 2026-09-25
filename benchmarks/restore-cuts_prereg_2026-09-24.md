# restore-cuts preregistration (2026-09-24, before any measurement)

Lane `restore-cuts` (Claude Sonnet), workspace `../zensim--restore-cuts`, bookmark
`quarantine/claude/restore-cuts`, base `main@origin` = `26494c8a` (C8 landed). Brief:
`~/tmp/zensim-paper/rev4/RESTORE_CUTS_brief.md`. Source audit: `COST_CUTS_AUDIT.md`. User ruling
(2026-09-24): "remember not to reject things for the cost budget, and track all things and code and
results of those you have. we can optimize and make things optional". **Cost is measured and
reported, never a gate.** No human label is read. Nothing is fitted or selected here.

## Inputs (sha256)

| file | sha256 |
|---|---|
| `RESTORE_CUTS_brief.md` | `6d40e169199ff274474ad69e9e1bb1ec665c6550c005b050296a39326609dd0d` |
| `COST_CUTS_AUDIT.md` | `63e2a7654c8bf9f04262a7ae896352cc8fec0e426e055397093cd96b994ce132` |
| `research/2026-09-dvifm/gmsd/zgeom_gms_dev_map_dev.patch` | `5784ec8671a8292b3175cd45415a3e67de91f16a98db6289e885ca0e2e7bdd57` |
| `research/2026-09-dvifm/reports/LANE_ZGEOM_DONE.md` | `d322c23ef8c49f80a47977ea839c8d7f2182cb099dfb90c5f0e952f37202f767` |
| `benchmarks/gmsd_2026-09-22.md` | `e1627273753b0903a296fccee10f6f13ff08bcb5d76e8ec5b851614fc238d400` |

Research code read (read-only, other workspaces): `zensim--transplant` @ `cf517c03`
(`keep/transplant-zgeom-block5`): `feature_v2.rs::{v1_channel_maps_and_sums, block_pool_sums}`,
`zgeom.rs`; `keep/dvifm3-int16` `943781e8`: `dvifm.rs::DvifmVis`.

## Families (append-only after f1501; all default OFF; f0..f1501 must stay bit-identical)

| item | token | slots (4 scales) | definition |
|---|---|---|---|
| A1 A_dev | `mapdev` | f1502..1561 = 12 cells x 5 | population std of the maps `mse=(s-d)^2`, `hfsq_src=(s-mu1)^2`, `hfsq_dst=(d-mu2)^2`, `hfabs_src=abs(s-mu1)`, `hfabs_dst=abs(d-mu2)` of the v1 band kernel, per (scale, channel); per-row Welford (f64, x ascending), rows merged in row order with Chan's formula; never `sum(x^2)-mean^2` |
| B2 z1max | `z1max` | f1562..1789 = 12 cells x 19 | the 13 basic + 6 peak v1 signals pooled over the (0,0)-anchored, ungated 5x5 block-MAX lattice of the eight v1 maps (`sd, art, det, mse, hfsq_src/dst, hfabs_src/dst`), partial border blocks dropped, `n` = surviving blocks, finalized by `V1BasicSums::{finalize_into,finalize_pools_into}` |
| Amb. 7 | `gmsnative` | f1790..1819 = X,B x 15 | C8's gradient loss/gain/deviation bank (5 stabilisers x 3 signals) for the X and B channels at NATIVE scale, using the same per-channel stabilisers the coarse X/B cells use |

**Why only the MSE and HF maps for A1.** The record (`gmsd_2026-09-22.md:189-194`) shows the std of
the SSIM, edge-artifact and detail-loss maps is an exact function of two existing columns (mean and
L2), so those groups carry no new information. The emitted HF slots are ratio forms (`var_loss`,
`tex_loss`, `contrast_inc`) of `hfsq` and `hfabs` means, so neither `hfsq` nor `hfabs` map second
moments are recoverable from the existing columns; both HF energy and HF magnitude deviations are
therefore carried (5 maps, 60 columns, not the record's 96). Where each map's std comes from the same
band kernel that pools the 228 surface (`mu1`, `mu2`, `sd` side outputs, kernel f32 formulas).

**B2 hybrid.** The record's proposed block+global hybrid is the two-surface arm: the bank already
carries the global 228 surface inside R0 (`f0..f227`), so `R0 + z1max` IS the hybrid (2 x 228).

**B1 (DVIFM curve form).** Verified before any code: main's `dvifm.rs` computes C7 (f956..985) with
the smooth curve `v(C) = (1+(C/C0)^(beta*sigma))^(-1/sigma)` (`visibility`); the two-state gate
exists only on unlanded `943781e8`. So the curve form is already the C7 form. The missing arm is the
gate form; it is added (item 3) as a default-off family `dvifmgate` carrying the per-level gate F1
(F2 bins do not depend on `v`), appended after `gmsnative`.

## Pooling and numerics contract

Rev3 (`ZENSIM_FORMULA_REV=3`, `ZENSIM_ROOT_FORM=sqrt`) is the era. The maps are read from the v1 band
kernel exactly as the record's `v1_channel_maps_and_sums` does (kernel side outputs `mu1`, `mu2`,
`sd`). Both side passes run a serial band loop per (scale, channel) cell over the production
materializer's pyramids (`build_v2_ref_scales`), consume rows in plane order, and parallelise across
cells only, so results cannot depend on thread count, SIMD tier or stride.

## Gates (each recorded with command and actual output in the DONE file)

1. f0..f1501 `to_bits`-identical with the families on vs off, all tiers, serial and MT8, tight and strided.
2. Each family `to_bits`-identical alone and together; slots outside a request stay structural zero.
3. Identity pair: `mse_dev` exactly 0; `hfsq/hfabs src == dst` bits; every registry `Difference`
   slot exactly 0; `gmsnative` exactly 0.
4. Tier / thread / stride: new slots `to_bits`-identical across serial, MT8, tight, strided and (measured, then asserted) across tiers.
5. Independent NumPy mirror from XYB plane dumps (float64 blur/maps/std/block-max) within a
   stated relative tolerance, with a wrong-definition negative control that must be rejected.
6. Registry: layout arithmetic round-trips, family/populated-slot equality, era revisions pinned.
7. Cost: per family, at 4 image sizes (64, 256, 1024, 4096 px on a side, gradient-heavy synthetic
   plus a real TRAIN pair), single thread and 8 threads, interleaved with the C8-on control; report
   the delta ns/px and % of the C8-on extraction. **Reported only.**
8. A bank sidecar for each family over the 18 promoted bank sets (Part B conventions: pair_key
   binding, f32 RNE, zstd-3 BYTE_STREAM_SPLIT, manifests, identity/finite/coverage checks, 200-pair
   fresh re-extraction).

## Public API delta (repo CLAUDE.md: preregister the concrete caller and exact delta)

Caller: `zensim-bench` extractor `extract_features_372col --restore-cuts <tokens>` and the plan
owner (`feature_plan`), exactly as C8's `--full-gmsbank`. The brief authorises registering the
families and adding an extractor flag; this is the same additive pattern the coordinator approved for
C8 (`ComputeToken::Gmsbank`, `V2NewFeatureToggles::gmsbank`). Additive items only:
`ComputeToken::{Mapdev, Z1max, Gmsnative, Dvifmgate}` (arms of the existing `#[non_exhaustive]`
enum) and `V2NewFeatureToggles::{mapdev, z1max, gmsnative, dvifmgate}` (`#[doc(hidden)]` fields of the
existing toggle struct). No new type, function, trait method or feature flag. Snapshots
(`docs/public-api/*`) and `CHANGELOG.md` are updated in the same change.

## Decision rule

This lane makes no scientific decision. Each family ships as an optional, default-off, registered,
tested extractor arm plus a bank sidecar and an arm spec (R0+family with a size-matched permuted
control) handed to the potential lane as a preregistration-amendment PROPOSAL. Whether any arm helps
is decided there, on human-label bars, not here.
