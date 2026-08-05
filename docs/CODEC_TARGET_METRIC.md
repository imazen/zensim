# Codec-target metric — integration guide

> **⚠ STALE — REWRITE PENDING (banner 2026-07-18):** the integration-guide
> prose below predates the A→B default flip and the 'three production
> trails' framing predates A/B. Trust `zensim/src/profile.rs` +
> `docs/MODEL_SELECTION_SCORECARD.md` for anything outside the mapping
> table. **The mapping table itself was refreshed 2026-08-05** (Profile C
> ship) and is verified against `profile.rs` as of that date.

**Stable alias:** `ZensimProfile::codec_target()` → `ZensimProfile::B`
(since 2026-07-12; generation-A is deprecated).

## Variant → backing bake mapping (THE single source of truth)

Naming convention: `docs/NAMING_CONVENTION.md` (external `ZensimProfile`
names are a stable contract; internal bakes rotate; update THIS table
on every rotation — never inline bake identity into variant rustdocs).

| External variant (`name()`) | Backing bake (internal) | Methodology |
|---|---|---|
| `B` (`zensim-b`) — **default**; `codec_target()`/`latest_preview()` | `zensim/weights/b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin` (`ens-Pline-cid80` inclusive-winsor dense-dial, sha `b6fe5233…`) | `benchmarks/profile_b_methodology_2026-07-12.md` |
| `BHdr` (`zensim-b-hdr`) — HDR (absolute-nits) route | `zensim/weights/bhdr_linear_shaped_cvvdpmix_2026-07-12.bin` (`hdrmix-lasso0.0003-shaped`, sha `7d7f2123…`) | `benchmarks/bhdr_improvement_split_lineage_2026-07-12.md` §7 |
| `C` (`zensim-c`) — SOTA-944 wave-11 candidate; 944-regime scoring contract (folded-944 features + `score_features_with_profile`) | `zensim/weights/c_sdr_mlp944_corrmix_2026-08-05.bin` (`W10L9_s4003_packed`, sha `1a2c8d52…`, PRUNED caller 944 / internal 667) | `docs/PROFILE_C_REPRODUCTION_2026-08-05.md` |
| `A` (`zensim-a`) — **deprecated** since 0.3.0; behind default-on `deprecated-profiles` | `zensim/weights/v47_strict_qat_native_2026-05-27.bin` (`v47-strict-QAT`, rotated 2026-05-27 from V39) | `benchmarks/v0_qat_native_methodology_2026-05-27.md` |
| `PreviewV0_1` / `PreviewV0_2` — historical linear profiles | in-source weight arrays (`WEIGHTS_PREVIEW_V0_1` / `_V0_2`), no bake file | README "v0.2" section |

Removed variants (no longer in the enum): `A_Phone` (manifest
`weights/manifests/zensim_b_phone_oled_2026-05-26.toml` +
`benchmarks/zensim_b_phone_oled_methodology_2026-05-26.md` remain; the
`.bin` is no longer in-tree); `PreviewV0_3` (was a deprecated alias of
`A`; V39 bytes preserved in `zensim-experimental`). |

Prior bakes (`v_tuner_v11_2026-05-24.bin`, V_18 lineage, etc.) are kept
on disk under `zensim/weights/archive/` for reproducibility but are no
longer wired to any shipped variant.
**Audience:** authors of zen codec crates (`zenjpeg`, `zenwebp`, `zenjxl`,
`zenavif`, `zenpng`, `zengif`, ...) and the picker pipeline at
`~/work/zen/zenanalyze/zenpicker/`.

This doc is the contract for "the target metric all zen codecs train
and target to." Read this first; the cross-codec baseline doc, the
SOTA trails doc, and the RDO feasibility doc are the supporting
evidence.

## TL;DR

```rust
use zensim::{Zensim, ZensimProfile};

let zensim = Zensim::new(ZensimProfile::codec_target());
let score = zensim.compute(&source_pixels, &distorted_pixels)?.score();
// score ∈ [0, 100]. 100 = lossless. 60 ≈ JND. 30 ≈ JOD.
```

Use `ZensimProfile::codec_target()` everywhere a codec needs "the"
zensim. The const aliases to whichever Tuner variant is the current
production ship; bake rotations (Tuner v5, v6, …) flow through
without any per-codec edit.

## Why one bake, not three

zensim ships three production trails (Balanced, Compression, Tuner —
see `SOTA_TRAILS.md`). Each defends a different Pareto frontier:

- **Balanced** wins KADID + TID + KonJND rank (general perceptual quality).
- **Compression** wins CID22 + AIC-3 rank (within-codec output fidelity).
- **Tuner** wins **monotonic q-dial** + **bit-exact JND/JOD anchors** +
  **smallest cross-codec spread**.

For codec consumers, the load-bearing properties are dial monotonicity
and cross-codec consistency — Tuner's territory. The rank-SROCC
weakness on KADID/TID/KonJND is **acceptable for codec use** because
codecs care about ordering within their own q-sweep (monotonic) and
about cross-codec equivalence at quality targets (Tuner is best),
not about ranking synthetic-distortion families.

**Measured 2026-05-24 — Tuner is 3-6× tighter cross-codec than the
other trails** (`benchmarks/tuner_v10_cross_codec_baseline_2026-05-24.md`):

| Bake | median \|Δ\| | p90 \|Δ\| | p99 \|Δ\| |
|---|--:|--:|--:|
| **Tuner v11** (current ship) | 1.37 | 4.78 | 11.23 |
| Tuner v10 (prior ship) | 1.18 | 3.58 | 8.05 |
| Compression v3 | 2.64 | 15.32 | 39.46 |
| Balanced v3 | 3.06 | 20.71 | 51.49 |

Tuner v11's absolute |Δ| is slightly larger than v10's because v11
uses MORE of the dial (p5=28 vs v10's p5=48). Normalized as % of
dial span: **v11 = 2.36%, v10 = 2.63%** — v11 is proportionally
TIGHTER. Picking Balanced or Compression as the codec-target would
have given users a dial whose precision swings by ±20-50 score
units; v11's ±3-11 + full 0-100 coverage is the production-ready
substrate.

For *general perceptual ranking* (e.g., comparing two competing
codecs' outputs as an A/B quality assessment), use
`ZensimProfile::balanced_v3()` or `compression_v3()` instead.

## What's measured

Per `benchmarks/tuner_v10_cross_codec_baseline_2026-05-24.md`
(68,788 cross-codec equivalence pairs at matched butter anchors):

| Score region | Cross-codec p50 \|Δ\| | Cross-codec p90 \|Δ\| | Ship status |
|---|---:|---:|---|
| 75–90 (high-quality web encodes) | 0.57–0.79 | 1.7–2.4 | ✓ production |
| 60–75 (mid-q, near-JND) | 1.40–1.47 | 3.5–4.0 | ✓ production |
| 30–60 (low-q) | 1.3–1.6 | 3.0–4.0 | △ flat dead zone below ~55 |
| 0–30 (very-low-q) | clamped at floor | — | ✗ **dial dead zone** |

The score-floor pathology below ~55 is the known gap;
task #6 (Tuner v11 retrain) closes it.

## Integration patterns

### Pattern A — Quality-target outer loop (use case A)

User asks: "encode this image at score 70." Codec runs binary search
over its q parameter, encodes-decodes-scores at each candidate, and
returns the encoded output whose decoded score lands closest to 70.

**Reference implementation:** `~/work/zen/zenwebp/src/encoder/api.rs`,
`EncodeConfig::target_zensim`. The plumbing is ~100 LOC per codec.

Skeleton:

```rust
use zensim::{Zensim, ZensimProfile, ZensimResult};

pub fn encode_to_target_zensim(
    source: &[u8],          // source RGB8 / RGBA8 pixels
    width: u32,
    height: u32,
    target_score: f64,      // 0..100
    tolerance: f64,         // typical: 1.0 score units
    max_iters: u32,         // typical: 8
    encoder: &mut MyCodecEncoder,
) -> Result<Vec<u8>, EncodeError> {
    let zensim = Zensim::new(ZensimProfile::codec_target());
    let precomputed = zensim.precompute_reference(&source_view)?;
    // Lookup-table starting point — replace with codec's q ↔ score map.
    let mut q_lo = 1u32;
    let mut q_hi = 100u32;
    let mut best_encoded: Option<Vec<u8>> = None;
    let mut best_score_err = f64::INFINITY;

    for _ in 0..max_iters {
        let q_mid = (q_lo + q_hi) / 2;
        let encoded = encoder.encode_at_q(source, width, height, q_mid)?;
        let decoded = encoder.decode_back(&encoded, width, height)?;
        let score = zensim.compute_with_ref(&precomputed, &decoded)?.score();
        let err = score - target_score;

        if err.abs() < best_score_err {
            best_score_err = err.abs();
            best_encoded = Some(encoded.clone());
        }
        if err.abs() < tolerance { break; }
        if err > 0.0 { q_hi = q_mid - 1; } else { q_lo = q_mid + 1; }
        if q_lo > q_hi { break; }
    }

    best_encoded.ok_or(EncodeError::TargetZensimNoConvergence)
}
```

**Per-iteration cost** at 1920×1080: ~6 ms (precomputed reference)
+ codec encode/decode cost. 8 iterations ≈ 50 ms + codec overhead.

**Tunables**:
- `precompute_reference` once per source — saves ~50 % per iteration.
- `tolerance = 1.0` is the cross-codec p50 floor; tighter than that
  is below the metric's resolution.
- Start `q_lo, q_hi` from a per-codec q↔score lookup table (most
  codecs have one already in their RD curve data) — converges in
  3–4 iterations instead of 6–8.

### Pattern B — Picker training (use case B)

A picker is a model that takes `(image_features, target_score)` and
outputs `(codec, q)` — i.e., "given a 1024² photo and a target score
of 70, encode with WebP at q=82." This is the
`~/work/zen/zenanalyze/zenpicker/` crate.

The training pipeline already runs against the canonical codec-target
bake. When the canonical bake rotates (e.g., task #6 ships Tuner v11),
the picker training command re-runs against the new bytes:

```sh
# Pseudo — actual command in zenanalyze/zentrain/tools/.
zensim_picker_train \
    --bake ~/work/zen/zensim/zensim/weights/v_tuner_v10_2026-05-20.bin \
    --per-codec-parquet-dir /mnt/v/zen/picker-training/2026-05-19/butter/ \
    --out ~/work/zen/zensim/zensim/weights/picker_*_<date>.bin
```

Pickers should pin the codec-target bake's content hash so picker
behavior is reproducible even when the canonical bake rotates.

### Pattern C — In-encoder RDO distortion (deferred)

The codec invokes zensim as the per-block distortion term inside its
trellis / R-D-O loop. **Not feasible with current zensim** at
codec-RDO cadence (5k–20k decisions per image × ~4 ms minimum). See
`docs/RDO_LOSS_FEASIBILITY_2026-05-24.md` for the three deferred paths
(differentiable end-to-end, fast proxy net, or — most realistic —
**skip it** and continue using internal codec proxies like
PSNR/SOS/butter for per-block work; this is what every production
codec does).

**Action:** none. Codec authors should use Pattern A (output-level
target) and not block on this.

## Per-codec target_zensim status

| Codec | Has Pattern A wired? | Notes |
|---|---|---|
| `zenwebp` | ✅ yes | `EncodeConfig::target_zensim`; reference implementation |
| `zenjpeg` | ❌ | tracked: open issue in zenjpeg repo |
| `zenjxl`  | ❌ | tracked |
| `zenavif` | ❌ | tracked |
| `zenpng` (lossless) | n/a | no q dial |
| `zengif`  | ❌ | optional (dither/palette decisions); low priority |

Adding Pattern A to a codec is not a zensim crate task — it's a
1–2 day task per codec, lives in the codec's own repo. The zensim
crate's responsibility is the canonical bake + `codec_target()`
alias, both of which ship today.

## Versioning policy

`ZensimProfile::codec_target()` is **stable** — it points at the
current production Tuner-trail ship. The underlying bake bytes
can rotate (Tuner v4 → v5 → …) without API breakage. Each rotation:

1. Lands a new bake file (e.g., `v_tuner_v11_2026-MM-DD.bin`).
2. Adds a new `ZensimProfile::PreviewV0_5TunerVN` variant.
3. Updates the body of `pub const fn codec_target() -> Self` to
   return the new variant.
4. Old variant remains accessible via its explicit name for
   reproducibility.

What this means for codec authors:

- Use `ZensimProfile::codec_target()` for the live integration —
  you'll get bake improvements for free.
- For *frozen-bytes reproducibility* (e.g., a paper's experiment
  pinning a specific bake), use the explicit variant name
  (`ZensimProfile::PreviewV0_5TunerV4`) — that's locked.

## Open questions

These are not blockers but are worth flagging for any session that
picks up this work:

1. **Should `ZensimProfile::codec_target()` be re-exported at the
   crate root?** Current shape requires `use zensim::ZensimProfile;`
   then `ZensimProfile::codec_target()`. Adding
   `pub use ZensimProfile::codec_target` as a crate-root function
   would make `zensim::codec_target()` valid. Marginal ergonomics;
   defer until a codec author asks.
2. **Does Tuner v11 ship before any codec needs the score 0–55 dial
   range?** Current Pattern A users (zenwebp `target_zensim` for
   "score 70/80/90" workloads) live entirely in the score 60–100
   band where Tuner v10 is solid. Users typing "score 20" would
   hit the floor pathology — but no current codec exposes that
   workflow as a primary use case.
3. **Picker bake versioning** — should pickers pin a specific
   codec-target bake hash, or follow the rolling alias? Pinning is
   reproducible-by-default; rolling auto-updates. Probably pin per
   ship, document the version in the picker bake's metadata.

## See also

- [`SOTA_TRAILS.md`](../zensim/SOTA_TRAILS.md) — the three-trail
  framework and ship matrix.
- [`benchmarks/tuner_v10_cross_codec_baseline_2026-05-24.md`](../benchmarks/tuner_v10_cross_codec_baseline_2026-05-24.md)
  — measured cross-codec consistency numbers; this doc's evidence base.
- [`docs/RDO_LOSS_FEASIBILITY_2026-05-24.md`](./RDO_LOSS_FEASIBILITY_2026-05-24.md)
  — why Pattern C is deferred and what would be needed to revisit.
- [`benchmarks/v_tuner_v9_methodology_2026-05-20.md`](../benchmarks/v_tuner_v9_methodology_2026-05-20.md)
  — training recipe for the current Tuner ship.
- `~/work/zen/zenwebp/src/encoder/api.rs` — Pattern A reference
  implementation.
- `~/work/zen/zenanalyze/zenpicker/` — picker crate, Pattern B.
