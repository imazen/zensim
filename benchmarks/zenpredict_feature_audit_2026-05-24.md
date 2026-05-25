# zenpredict feature audit (zensim consumer view)

**Date:** 2026-05-24
**zenpredict pinned rev:** `8025953e0b4119a0a3fb3e1d330d52138506f45c` (workspace `Cargo.toml:42`)
**zenpredict version on disk:** `0.2.1` (`zenanalyze/zenpredict/Cargo.toml:3`)
**Scope:** zensim, zensim-validate, zensim-bench, zensim-train-core (zensim-regress and
zensim-train-gpu have no zenpredict imports). Read-only audit; no edits.

This is the consumer-side audit. For each item zenpredict exposes from
`zenanalyze/zenpredict/src/lib.rs`, we record whether anything in the zensim
workspace pulls it. Lines are absolute paths.

## Cargo feature flags

`zensim` workspace `Cargo.toml` lines 42-43:

```toml
zenpredict      = { git = "...", rev = "8025953..." }
zenpredict-bake = { git = "...", rev = "8025953..." }
```

**No `features = […]` qualifier on either dep.** zenpredict defaults to
`["std"]`, so the `advanced` feature (which gates `rescue::*`, `safety::*`,
`OutputValue`, `apply_spec`, `OutputBound`, the `_top_k` / `_with_scorer` /
`pick_with_confidence` / `threshold_mask` argmin variants) is **OFF**.

This matches actual usage — none of those advanced items are imported anywhere
in the zensim tree (verified by `grep`). The default-feature posture is correct
for the perceptual-scorer consumer shape that zensim is.

## USED — directly imported and called

| zenpredict item | Where (zensim file) | Why it matters |
|---|---|---|
| `Model::from_bytes` | `zensim/src/mlp/mod.rs:44` (re-export); `zensim-validate/src/bin/{bake_verdict,bake_compare,ensemble_mix,ensemble_score_rows,eval_bake_per_band,inspect_l0_input_norms,predict_features_with_bake,preview_stats_demo,qsweep_eval,score_pair_with_bake,zensim_picker_infer,concat_three_way}.rs`; `zensim-validate/src/mlp_train.rs:8876`; `zensim-bench/examples/profile_compat_report.rs:40`; `zensim-validate/src/output_calibration_spline.rs:13` | Bake-bytes → typed `Model` — the entry point. Everything else hangs off this. |
| `Predictor::new` | `zensim/src/mlp/mod.rs:44`; every binary above | Owns the forward-pass scratch buffer. |
| `Predictor::predict` | `zensim/src/metric.rs:2843`; `zensim-validate/src/bin/score_pair_with_bake.rs:138`; `eval_bake_per_band.rs:133`; `predict_features_with_bake.rs:222`; `preview_stats_demo.rs:176`; `bake_verdict.rs:755`; `mlp_train.rs:8877` (test). | Raw forward when the bake declares no feature_transforms. |
| `Predictor::predict_transformed` | `zensim/src/metric.rs:2838`; `zensim-validate/src/bin/{bake_verdict.rs:753,predict_features_with_bake.rs:220,preview_stats_demo.rs:174,score_pair_with_bake.rs:136}`. | Forward with `feature_transforms` applied first. Required by every bake whose metadata contains a non-identity FT vector. |
| `Model::has_nontrivial_feature_transforms` | `zensim/src/metric.rs:2813`; `zensim-validate/src/bin/{bake_verdict.rs:1246,predict_features_with_bake.rs:508,preview_stats_demo.rs:421,score_pair_with_bake.rs:312}`. | The dispatch flag — picks predict vs predict_transformed. |
| `Model::metadata` (→ `Metadata::get`) | `zensim/src/metric.rs:2749,2757,2762,2765,2769+`; every `*_head_meta` parser in the binaries listed above. | Reads `zentrain.per_sample_alpha_head`, `zentrain.hybrid_head`, `zentrain.tanh_output_head`, `zentrain.output_calibration_spline`, `zentrain.per_codec_calibration`. Hot path in runtime. |
| `Model::n_inputs` | `zensim-validate/src/bin/{inspect_l0_input_norms.rs:79,bake_verdict.rs:1245,eval_bake_per_band.rs:118,predict_features_with_bake.rs:507,preview_stats_demo.rs:426,score_pair_with_bake.rs:311}`. | Sizes feature scratch buffer. |
| `Model::n_outputs` | `zensim/src/metric.rs:2748`; `zensim-validate/src/bin/{score_pair_with_bake.rs:23,72,bake_verdict.rs:663,694,bake_compare.rs:285,315,ensemble_score_rows.rs:54,83,preview_stats_demo.rs:105,132,predict_features_with_bake.rs:100,148,concat_three_way.rs:95}`. | n_hidden for per-sample-α / hybrid metadata parsing (the bake's last layer is an identity passthrough; n_outputs == n_hidden). |
| `Model::n_layers` | `zensim-validate/src/bin/concat_three_way.rs:91`. | Asserts a 2-layer input bake. |
| `Model::scaler_mean` / `Model::scaler_scale` | `zensim-validate/src/bin/concat_three_way.rs:97,98`. | Read scaler from input bakes for the 3-way concat merge. |
| `Model::feature_transforms` / `Model::feature_transform_params` | `zensim-validate/src/bin/concat_three_way.rs:130,132`. | Propagates FT metadata when concating three input bakes. |
| `WeightStorage` (`F32`, `F16`, `I8`) | `zensim-validate/src/bin/{concat_three_way.rs:103-121,inspect_l0_input_norms.rs:97-99,131-133}`. | Type-erased weight access for diagnostics + concat. |
| `f16_bits_to_f32` | `zensim-validate/src/bin/{concat_three_way.rs:104,120,inspect_l0_input_norms.rs:98}`. | F16 → F32 widening for the concat path (F16 inputs widen). |
| `Activation` (`Identity`, `LeakyRelu`) | `zensim-train-core/src/{mlp.rs:47,55,hybrid_head.rs:632,640,per_sample_alpha_head.rs:681,689,775,783,pool_head.rs:589,597}`; `zensim-validate/src/bin/{concat_three_way.rs:280,288,zensim_picker_train.rs:915,923,931}`; `zensim-validate/src/mlp_train.rs:4114,4122`. | BakeLayer activation. Only `LeakyRelu` + `Identity` ever used — `Relu`, `Tanh`, `Sigmoid` variants are NOT used. |
| `WeightDtype::F32` | All bake-producing sites above. | Every zensim-produced bake ships F32 weights. |
| `WeightDtype::F16` / `WeightDtype::I8` | `zensim-validate/src/bin/zensim_mlp_train.rs:1782,1783`. | Trainer CLI accepts `--bake-dtype f16` / `i8`. |
| `FeatureTransform` (enum + `from_token`) | `zensim-validate/src/bin/zensim_mlp_train.rs:977,1635,1668,1677,1693+`; `zensim-validate/src/mlp_train.rs:31`. Variants USED: `Identity`, `ClipThenLog1p`, `WinsorP99`, `QuantileBins` (parameter computation paths). | Trainer parses CLI tokens, computes per-feature transform params, persists to bake metadata. |
| `MetadataType::Utf8` / `MetadataType::Numeric` | `zensim-validate/src/bin/{concat_three_way.rs:332,339,zensim_picker_train.rs:956+}`; `zensim-validate/src/mlp_train.rs:4183,4190`; `zensim-train-core/src/{hybrid_head.rs:648,per_sample_alpha_head.rs:697,792,797,pool_head.rs:605}`. | All zensim metadata writes are Utf8 (for FT JSON-ish payload) or Numeric (for binary per_sample_alpha / hybrid / pool head weights). `Bytes` not used. |
| `keys::FEATURE_TRANSFORMS` / `keys::FEATURE_TRANSFORM_PARAMS` | `zensim-validate/src/bin/concat_three_way.rs:331,338`; `zensim-validate/src/mlp_train.rs:4182,4189`. | Canonical key strings — using zenpredict's constants, not duplicated literals. |
| `BakeRequest`, `BakeLayer`, `BakeMetadataEntry`, `bake` (from `zenpredict-bake`) | `zensim-train-core/src/{mlp.rs,hybrid_head.rs,per_sample_alpha_head.rs,pool_head.rs}`; `zensim-validate/src/bin/{concat_three_way.rs,zensim_picker_train.rs}`; `zensim-validate/src/mlp_train.rs`. | Bake-producing sites. NO Python intermediary — Rust calls `bake()` directly. |

**Note on dispatch correctness:** `zensim::metric::apply_mlp_scoring_with_codec`
(line 2813+) correctly checks `has_nontrivial_feature_transforms()` and dispatches
to `predict_transformed` when true. The other live binaries (`bake_verdict`,
`predict_features_with_bake`, `score_pair_with_bake`, `preview_stats_demo`,
`eval_bake_per_band`) do the same. Per CLAUDE.md "Step 7 — Runtime forward path"
discipline this is the correct gate; no regressions found.

## TRANSITIVELY USED — re-exported but not first-party called

| Item | Notes |
|---|---|
| `zensim::mlp::Model` / `zensim::mlp::Predictor` | `pub(crate)` re-exports in `zensim/src/mlp/mod.rs:44`. Used internally by `zensim::metric`. External consumers depend on `zenpredict` directly per the doc comment at `zensim/src/lib.rs:209-210`. |
| `zensim_train_core::{Activation, WeightDtype}` | `pub use` re-export in `zensim-train-core/src/lib.rs:21`. Trainer-side flexibility. |

## UNUSED — exposed by zenpredict, no zensim consumer

All under the `std` (default) feature unless marked `[advanced]`. Brief
description distilled from zenpredict's docstrings.

| Item | What it is | Why zensim doesn't use it |
|---|---|---|
| `Predictor::predict_with_specs` | Typed multi-output predict that applies `OutputSpec` rounding/categorical mapping to raw outputs. **`[advanced]`-gated.** | zensim bakes are scalar regressors; no `OutputSpec` section. (Picker bakes elsewhere in the zen org use this.) |
| `Predictor::predict_with_specs_transformed` | Same as above with feature_transforms applied. **`[advanced]`-gated.** | Same reason. |
| `Predictor::schema_hash` / `Model::schema_hash` | Schema-fingerprint accessor + `Model::from_bytes_with_schema` constructor for compile-time schema pinning. | zensim bakes all use `schema_hash: 0` (see all 7 call sites of `BakeRequest`). No schema-version gate in the loader. |
| `Predictor::model` | Borrow the underlying `Model` from a `Predictor`. | zensim binaries always keep the `Model` in scope alongside the `Predictor`, so they have a direct `&model` ready. |
| `Predictor::feature_transforms` (accessor on Predictor) | Wrapped over `Model::feature_transforms`. | zensim calls `model.feature_transforms()` directly in concat_three_way. |
| `Predictor::argmin_masked` + variants (`in_range`, `top_k`, `with_scorer`, `pick_with_confidence`, `threshold_mask`) | Codec-config selection: arg-min over masked predicted scores. `top_k` / `with_scorer` / `pick_with_confidence` are `[advanced]`-gated. | zensim is a **perceptual scorer**, not a picker. The picker-shape APIs are out of scope. (zenjpeg / zenwebp / zenavif / zenjxl pickers consume these.) |
| `ScoreTransform` / `ArgminOffsets` / `AllowedMask` (re-exports) | Argmin scoring helpers. | Same — picker-shape API. |
| `Model::from_bytes_with_schema` | Constructor that asserts `schema_hash` matches. | Not used because bakes ship `schema_hash: 0`. |
| `Model::header` / `Model::version` / `Model::flags` | Wire-format header accessors. | Not needed at runtime; zensim only cares about predict + metadata. `affine_calibrate.rs` does raw header byte parsing instead of using these (see Recommendations). |
| `Model::layers` (iterator) / `Model::layer(idx)` | Per-layer view. | The concat_three_way binary reads layers via `read_u32` + `WeightStorage` (it inspects in_dim / out_dim through the higher-level API on a per-bake basis but does not iterate). zensim runtime doesn't introspect layers — `Predictor::predict` handles the loop. |
| `Model::raw_bytes` | Borrow the original `&[u8]` back. | Not needed; all consumers hold the bytes (or `include_bytes!` slice) themselves. |
| `Model::scratch_len` | Required scratch buffer size for an external forward-pass impl. | `Predictor::new` owns scratch internally; no external pass exists. |
| `Model::feature_bounds` + `FeatureBound` + `first_out_of_distribution` | OOD bounds checking from `feature_bounds` section. | zensim bakes ship `feature_bounds: &[]` everywhere. No OOD gate. (Picker bakes use this for runtime safety.) |
| `Model::output_specs` / `Model::has_output_specs` + `OutputSpec` + `OutputTransform` + `OutputValue` (`[advanced]`) + `apply_spec` (`[advanced]`) | Per-output type + rounding for picker codec-config integers. | Picker-shape API. zensim bakes pass `output_specs: &[]`. |
| `Model::discrete_sets` | Pool of allowed discrete output values referenced by OutputSpecs. | Same — `discrete_sets: &[]`. |
| `Model::sparse_overrides` + `SparseOverride` | Lookup-table escape hatch overriding the network for known (feature, expected_output) tuples. | Not used; zensim bakes pass `sparse_overrides: &[]`. Could be used to hard-fix calibration anchors (KonJND PJND midpoints) — see Recommendations. |
| `Model::safety_compact` + `SafetyCompact` + `SafetyProfile` (`[advanced]`) | Compact runtime-safety profile (caller-supplied confidence floors etc.). | `[advanced]`-gated and picker-shape. |
| `Model::cell_rescue_hints` + `CellHint` (`[advanced]`) | Per-cell rescue strategy hints. | Same. |
| `Model::zq_fallback_table` + `FallbackEntry` + `fallback_for` (`[advanced]`) | Zq-keyed fallback table for two-shot rescue. | Same. |
| `Model::output_bounds` + `OutputBound` + `output_first_out_of_distribution` (`[advanced]`) | OOD bounds on prediction outputs. | Same. |
| `rescue::{RescuePolicy, RescueStrategy, RescueDecision, should_rescue}` (`[advanced]`) | Two-shot rescue framework. | Picker-shape. zensim is scalar. |
| `PredictError` | Forward-pass error type. | zensim wraps errors via `Result<…, _>` but never matches variants — it just stringifies. `metric.rs:2839,2843` use `_` and re-emit a fixed `reason: "Predictor::predict[_transformed] failed"`. |
| `LayerEntry`, `LayerView`, `Section`, `Header`, `FORMAT_VERSION`, `LEAKY_RELU_ALPHA` | Wire-format primitives. | zensim never introspects the wire format through these (uses `Predictor::predict` or raw u32 reads). |
| `wire::{HEADER_SIZE, LAYER_ENTRY_SIZE, SECTION_OFF_*, FLAG_COMPRESSED, FLAGS_COMPRESSION_ALGO_MASK, COMPRESSION_ALGO_NONE, COMPRESSION_ALGO_LZ4, OFF_DECOMPRESSED_PAYLOAD_LEN, SECTION_OFF_FEATURE_ORDER, SECTION_OFF_OUTPUT_ORDER}` | Public wire-layout constants. | **Not used.** `zensim-validate/src/bin/affine_calibrate.rs:28-29` re-defines its own `HEADER_SIZE = 128` / `LAYER_ENTRY_SIZE = 48` locally — this is a missed opportunity (see Recommendations). |
| `limits::{MAX_BAKE_BYTES, MAX_DIM, MAX_LAYERS}` | Loader bounds. | Not referenced. zensim doesn't enforce its own bound; relies on `Model::from_bytes` returning Err. |
| `FeatureTransform::{SignedLog1p, SignedSqrt, SignedCbrt, WinsorThenLog, WinsorThenLog1p, WinsorThenSignedCbrt, SignedCbrtThenWinsor, ClipThenLog1pThenWinsor}` | 8 additional FT variants beyond the 5 enabled by trainer. | Trainer at `zensim_mlp_train.rs:1668,1677,1693` enumerates `ClipThenLog1p`, `WinsorP99`, `QuantileBins` for parameter computation; **the bake-side validator in zenpredict-bake also accepts the eight WinsorThen* / SignedCbrt-family variants**, but zensim's trainer doesn't compute params for them. Either the trainer should grow support OR these belong to the bake-side surface and zensim's enumeration over them at training time is intentionally narrow. |
| `apply_feature_transforms` (free fn) | Stand-alone transform applier (operates on a buffer). | zensim always goes through `Predictor::predict_transformed`, which calls the internal applier — the free function is for callers who want to transform without doing inference. Not needed by zensim. |
| `MetadataEntry::get_bytes` / `get_numeric` / `get_utf8` / `get_pod` | Typed metadata accessors. | zensim uses only the un-typed `Metadata::get(...)?.value` byte-slice access (e.g., `bake_verdict.rs:641`). It then byte-parses payloads manually. Could be cleaner with `get_pod` / `get_numeric` (see Recommendations). |
| `MetadataEntry::{key, kind, value}` (public fields) | Direct field access on entries returned from `Metadata::iter`. | zensim never iterates metadata; uses keyed `get` only. |
| `Metadata::{iter, len, is_empty}` | Iterator + length API. | Same — zensim is key-lookup only. |
| `keys::*` (other than `FEATURE_TRANSFORMS` / `FEATURE_TRANSFORM_PARAMS`) | 14 additional canonical key constants (`PROFILE`, `SCHEMA_VERSION_TAG`, `FEATURE_COLUMNS`, `HYBRID_HEADS_LAYOUT`, `PROVENANCE`, `CALIBRATION_METRICS`, `SAFETY_COMPACT`, `CELL_RESCUE_HINTS`, `ZQ_FALLBACK_TABLE`, `OUTPUT_BOUNDS`, `SAFETY_REPORT`, `BAKE_NAME`, `REACH_RATES`, `REACH_ZQ_TARGETS`). | **zensim DOES read `zentrain.tanh_output_head`, `zentrain.per_sample_alpha_head`, `zentrain.hybrid_head`, `zentrain.output_calibration_spline`, `zentrain.per_codec_calibration`, `zentrain.pool_head_reducer` BUT none of these have canonical `keys::*` constants in zenpredict.** Every zensim site duplicates the key literal as a string (see `zensim/src/metric.rs:2154,2162,2194,2358` and the many `md.get("zentrain.…")` sites). Same problem on the bake-write side — `zensim-train-core/src/{hybrid_head,per_sample_alpha_head,pool_head}.rs` and `zensim-validate/src/mlp_train.rs` write the key as a raw string literal. Drift risk: a typo in one site silently disables a runtime path. See Recommendations. |
| `keys::PROFILE`, `keys::SCHEMA_VERSION_TAG`, `keys::FEATURE_COLUMNS`, `keys::PROVENANCE`, `keys::CALIBRATION_METRICS`, `keys::BAKE_NAME` | Generic metadata keys zenpredict offers for picker / training provenance. | zensim could attach these to every bake for traceability (which dataset, which trainer commit, which calibration α/β) but currently doesn't. |

## Recommendations

### Items zensim SHOULD start using

1. **`zenpredict::wire::{HEADER_SIZE, LAYER_ENTRY_SIZE, SECTION_OFF_LAYER_TABLE}` in `affine_calibrate.rs`.** Lines 28-29 of
   `/home/lilith/work/zen/zensim/zensim-validate/src/bin/affine_calibrate.rs` redefine these constants
   locally. They are public in zenpredict. Replace the locals to keep the wire-format dependency
   in one place. (~5-line edit.)

2. **Promote zensim's `zentrain.tanh_output_head` / `per_sample_alpha_head` / `hybrid_head` /
   `output_calibration_spline` / `per_codec_calibration` / `pool_head_reducer` key strings into
   `zenpredict::metadata::keys::*` constants.** Currently each is a raw string literal duplicated
   at 4-6 sites across zensim. Adding them to `zenpredict/src/metadata.rs:312+` (the `keys` mod)
   would centralize the contract — match the existing `FEATURE_TRANSFORMS` / `FEATURE_TRANSFORM_PARAMS`
   pattern. Until that lands (zenpredict-side change, out of scope of this audit), zensim could at
   minimum define a single `mod keys { pub const PER_SAMPLE_ALPHA_HEAD: &str = "..."; ... }` module
   in `zensim/src/metric.rs` so the strings live in exactly one place per repo.

3. **`MetadataEntry::get_pod` and `get_numeric` for typed parsing.** `bake_verdict.rs` and the
   metric runtime byte-parse the per_sample_alpha / hybrid_head / tanh_pin payloads manually
   (e.g., `zensim/src/metric.rs:2467-2480` parses 4-byte chunks). zenpredict offers
   `MetadataEntry::get_pod::<T>` for fixed-size POD payloads and `get_numeric` for length-prefixed
   numeric arrays. Switching to these would shrink the parser code and centralize endianness in
   one library.

4. **`Model::from_bytes_with_schema` once zensim ships a stable schema.** Currently
   `schema_hash: 0` is hard-coded at every bake site. If we ever ship a feature vector whose
   schema can't change silently, swap to `from_bytes_with_schema(bytes, EXPECTED_HASH)` and reject
   mismatched bakes at load time.

5. **Use the eight unused `FeatureTransform` variants in the trainer.** zenpredict-bake's validator
   already accepts `WinsorThenLog`, `WinsorThenLog1p`, `WinsorThenSignedCbrt`,
   `SignedCbrtThenWinsor`, `ClipThenLog1pThenWinsor`, `SignedLog1p`, `SignedSqrt`, `SignedCbrt`.
   `zenpredict/src/feature_transform.rs:99` flags WinsorThenLog as "dominant high-win stack across
   zenjpeg / zenwebp / zenavif in the 2026-05-17 stacks sweep." `zensim_mlp_train.rs` only computes
   parameters for `ClipThenLog1p`, `WinsorP99`, `QuantileBins`. Extending the trainer's
   `--feature-transforms` token parser to compute (p1, p99) winsor bounds in log-domain for the
   8 missing variants unlocks a known-effective search axis.

6. **Provenance / calibration_metrics metadata on every shipped bake.** `keys::PROFILE`,
   `keys::PROVENANCE`, `keys::CALIBRATION_METRICS`, `keys::BAKE_NAME` are exactly the
   trace-fields the V0_X methodology docs already require (CLAUDE.md "Shipping policy" step f-h).
   Currently we keep that lineage in `benchmarks/v0_X_methodology_*.md` files; embedding a copy
   into the bake bytes would survive worktree loss and make `zenpredict inspect` (or a
   `bake_verdict --provenance` flag) self-documenting.

### Items zenpredict could trim

These are read-only inferences — the call is the zenanalyze maintainer's. Several of these are
used by sibling crates (zenjpeg / zenwebp / zenavif / zenjxl pickers) so trimming them is NOT
appropriate based on zensim's view alone. Listed for awareness, not action.

- The `Predictor::model()` accessor — convenient but never called from zensim's tree.
- `Predictor::feature_transforms()` (the wrapper on Predictor) — zensim calls
  `model.feature_transforms()` directly. Removing the Predictor-side wrapper saves
  one accessor.
- `Model::raw_bytes` — every consumer in zensim holds the original `&[u8]` slice (often via
  `include_bytes!`). No first-party call. Picker consumers may use this.
- `Model::scratch_len` — `Predictor::new` owns scratch; zensim never sizes scratch externally.

**None of the `[advanced]`-feature-gated items should be trimmed** — they exist for picker
consumers and the gate already prevents zensim builds from paying for them.

## Bake-producer audit (zenpredict-bake usage)

zensim consumes `zenpredict_bake::{BakeLayer, BakeMetadataEntry, BakeRequest, bake}` directly
from Rust — no JSON intermediary in the bake-producing tools listed below:

| File | Bake shape |
|---|---|
| `zensim-train-core/src/mlp.rs` | Plain 2-layer LeakyReLU → Identity. No metadata. |
| `zensim-train-core/src/hybrid_head.rs` | 2-layer + `zentrain.hybrid_head` numeric blob. |
| `zensim-train-core/src/per_sample_alpha_head.rs` | 2-layer + `zentrain.per_sample_alpha_head` (and a tanh-pin variant adding `zentrain.tanh_output_head`). |
| `zensim-train-core/src/pool_head.rs` | 2-layer + `zentrain.pool_head_reducer` numeric blob. **NB:** zensim runtime never reads this key — only the trainer writes it. The runtime's per-sample-α dispatch treats the bake's hidden vector directly, so `pool_head_reducer` payload is dormant in production. |
| `zensim-validate/src/mlp_train.rs` | Trainer canonical path — 2-layer with optional FEATURE_TRANSFORMS + FEATURE_TRANSFORM_PARAMS metadata. |
| `zensim-validate/src/bin/zensim_picker_train.rs` | 3-layer picker (not consumed by zensim metric — outputs a codec family pick). |
| `zensim-validate/src/bin/concat_three_way.rs` | Concat-merge of three input bakes with metadata propagation. |

**Note:** `BakeRequestJson` from `zenpredict-bake/src/json.rs` (the CLAUDE.md-mandated JSON
pipeline) is **NOT used by zensim** — zensim calls `bake()` directly with a Rust struct. The
CLAUDE.md "JSON pipeline mandate" applies to Python tools that shell out to the
`zenpredict-bake` CLI; Rust tools in-process are fine using the API. References to
`BakeRequestJson` in zensim are only in benchmark logs and methodology docs, not in src.

## What we use exactly (canonical short list)

From the runtime-critical path (the surface that any new zensim bake must rely on):

- `Model::from_bytes(bytes) -> Result<Model, PredictError>`
- `Model::has_nontrivial_feature_transforms() -> bool`
- `Model::metadata() -> Metadata` and `Metadata::get(key) -> Option<&MetadataEntry>`
- `Model::n_inputs() -> usize`, `Model::n_outputs() -> usize`
- `Predictor::new(&Model)`, `Predictor::predict(&[f32])`, `Predictor::predict_transformed(&[f32])`

From the bake-producing path:

- `zenpredict_bake::{bake, BakeRequest, BakeLayer, BakeMetadataEntry}`
- `zenpredict::{Activation::{Identity, LeakyRelu}, WeightDtype::{F32, F16, I8}, MetadataType::{Utf8, Numeric}}`
- `zenpredict::keys::{FEATURE_TRANSFORMS, FEATURE_TRANSFORM_PARAMS}`
- `zenpredict::FeatureTransform::{Identity, ClipThenLog1p, WinsorP99, QuantileBins, from_token}` (trainer)

From the diagnostic / merging path:

- `Model::n_layers`, `Model::scaler_mean`, `Model::scaler_scale`, `Model::feature_transforms`,
  `Model::feature_transform_params`
- `zenpredict::{WeightStorage::{F32, F16, I8}, f16_bits_to_f32}`

Everything else zenpredict exposes is either `[advanced]`-gated (correctly off for zensim) or
serves the codec-picker consumer shape that zensim doesn't share.
