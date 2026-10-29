# Extraction-plan invariants

The 2026-10-28 audit ran the split parity example with temporary plan logging,
including `compute_all_features`, cached-reference calls, and the public training
configurations in `examples/split_parity/surfaces.rs`. Runtime observations alone
are insufficient to establish constants: the planner and its per-scale projections
also serve the single-slot contracts in `src/feature_v2_tests.rs`.

| ComputeSet field | Observed at walk entry | Why it remains variable |
|---|---|---|
| `formula_revision` | Rev1, Rev7 | Selected by the profile; Rev5 is also used internally |
| `v1_basic` | true | `at_scale` clears it for a scale with no v1 reads |
| `full_res_xb` | false, true | Depends on the declared channel reads |
| `coarse_y_only_scales` | 0 | Single-slot plans can restrict coarse channels |
| `local_only` | false | The planner enables it for basic local-only reads |
| `omit_edges` | false | The planner enables it for SSIM/MSE-only reads |
| `v1_pools` | Off, Peaks, Full | Depends on the requested pools |
| `v1_full_scales` | 15 | Weighted-pool plans can narrow the scale mask |
| `v1_scales` | 15 | Declared slot reads can narrow the scale mask |
| `v2_scales` | 15 | Declared reads and adjacent gradient dependencies select scales |
| `v2_blocks` | false, true | v1-only plans omit v2 work |
| `gradient` | false, true | Depends on v2 work and the per-scale projection |
| `blockiness` | false, true | Depends on v2 work and the per-scale projection |
| `transducer_bank` | false, true | Selected with the v2 family |
| `omit_pjnd_fragility` | false | Single-slot plans may omit the reference-only signal |

The removed `transducers_luma_only` field was false in every constructor, had no
assignment that could change it, and had no public configuration input. Every
active channel now computes its transducer features without that constant gate.

The public training config still controls blur radius, passes and kernel, scale
count, extended and IW features and their strengths, full feature extraction,
score mapping, and threading. Those controls remain intact. Cached references
retain their existing configuration and revision checks.


## Request revision routing

Public custom configurations default to revision 1. `ZensimConfig::formula_revision`
is crate-private, and `ProfileParams::custom` sets its private `mlp_bytes` to `None`.
There is no public conversion from named-profile parameters to either custom input.

Named profiles obtain their revision from the embedded model. Their canonical
configurations use four scales, radius five, one blur pass, and all-feature extraction,
so they satisfy `fold_engine::is_fold_backable`. Both cached diffmap entry points use
these same profile configurations. Full-feature extraction changes only the extraction
flag after configuration construction.

The removed process-mix and raw-config canonical refusals could therefore never return
an error through the public callers. `metric::tests::public_configs_preserve_revision_route_invariants`
pins the named-profile conditions and custom defaults. Cached-reference compatibility
checks remain separate and unchanged.
