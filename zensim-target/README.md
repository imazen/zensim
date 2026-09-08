# zensim-target

CLI and library that choose a codec's encode parameter to approach a requested
zensim score. Given `(image, target_score, codec)`, each probe encodes, decodes,
and scores the reconstruction; the result includes encoded bytes, achieved
score, probe history, and whether it reached the requested tolerance.

**Checked against source on 2026-09-07:** the library default uses
`ZensimProfile::codec_target()`, currently B. The CLI now uses the same alias;
`--profile default` also follows it. Explicit **`--profile tuner-v4`** retains
the historical scorer. Selecting a default profile does not establish
that every image or target is reachable.

The [codec-target integration guide](../docs/CODEC_TARGET_METRIC.md) owns the
current profile mapping, score contract, known limitations, and codec-native
callers. The intended end-user control is one target score; codec quality and
JXL distance are internal search parameters.

## Build and CLI

This is a **standalone Cargo workspace**, excluded from the parent workspace.
Its path dependencies require the sibling codec repositories. From the zensim
repository root:

```bash
cargo run --release --manifest-path zensim-target/Cargo.toml --bin zensim-target -- \
  input.png --target 80 --codec zenjpeg --profile codec-target \
  --tolerance 1.0 --max-iterations 8 --output encoded.jpg
```

The default build enables JPEG, WebP, AVIF, and PNG. Add `--features zenjxl`
**before** `--` to enable the JXL adapter. A smaller JPEG-only build uses
`--no-default-features --features zenjpeg`. The selected codec must be enabled
in the build.

The CLI defaults to target 70, tolerance 1, and at most eight probes. It prints
`codec`, `target`, `achieved`, `knob`, `bytes`, `iters`, and `converged` in its
summary. Without `--quiet`, it also prints the collected per-probe trace.
`--output` writes the returned bytes; omitting it performs the search without
writing an encoded file. The CLI can write a best-effort result and exit
successfully with `converged=false`, so inspect that field when target accuracy
is required.

### Profile selection

| CLI value | Meaning |
|---|---|
| `codec-target` (`codec_target`, `default`) | CLI default; current `ZensimProfile::codec_target()`, matching `TargetSpec::default()` |
| `latest` (`latest-preview`, `latest_preview`) | Current `ZensimProfile::latest_preview()` |
| `a` (`v0_3`, `v03`, `preview-v0.3`) | Deprecated A profile |
| `v0_2` (`v02`, `preview-v0.2`) | Historical linear profile supplied by `zensim-experimental` |
| `tuner`, `tuner-v2`, `tuner-v3`, `tuner-v4` | Historical experimental tuners; explicit selection required |
| `balanced`, `balanced-v2`, `balanced-v3`, `compression`, `compression-v2`, `compression-v3`, `ensemble` | Historical experimental profiles for evaluation |

Names are case-insensitive. The [parser](src/bin/zensim_target.rs) lists all
compatibility aliases. It does not currently accept `b`, `c`, or `d` as CLI
profile names; library callers can set `TargetSpec::profile` directly. Old
JND/JOD anchor labels and May-era profile measurements are historical
calibration conventions, not guarantees for the current default.

### Codec adapters

| `--codec` | Search range | Direction assumed by the search | Fixed settings |
|---|---|---|---|
| `zenjpeg` | q 5–99 | q increases score | `ApproxJpegli`, quarter chroma subsampling |
| `zenwebp` | q 1–100 | q increases score | method 4 |
| `zenavif` | q 1–100 | q increases score | speed 6 |
| `zenjxl` | distance 0.01–15 | distance decreases score | other settings use encoder defaults; requires `zenjxl` feature |
| `zenpng` | no lossy quality knob | one probe | lossless RGB encode |

The CLI loads its reference through the `image` crate and converts it to
packed RGB8. The adapters return decoded RGB8 for scoring. This helper does
not provide HDR, alpha-preserving, or color-managed target search. Decoder
lineage is part of a reproducible measurement; the
[adapters](src/codec.rs) specify the decoder used for each codec.

## Rust API

The input is tightly packed RGB8 with exactly `width * height * 3` bytes and
no row padding. For example:

```rust
use zensim_target::{CodecKind, TargetResult, TargetSpec, target_search};

fn encode_at_target(rgb: &[u8], width: u32, height: u32) -> anyhow::Result<TargetResult> {
    let spec = TargetSpec {
        target: 80.0,
        ..TargetSpec::default()
    };
    target_search(rgb, width, height, CodecKind::Jpeg, spec)
}
```

Inspect `converged`, `achieved_score`, `final_knob`, `iterations`, and `probes`
before using the returned `encoded` bytes. Supply finite target/tolerance
values, a nonnegative tolerance, and at least one iteration; those constraints
are not validated by the current helper.

## Search behavior and limitations

The default algorithm bisects the adapter's quality range. It encodes and
decodes at the midpoint, computes `Zensim::compute(reference, decoded)`, and
returns immediately when `|achieved - target| <= tolerance`. Otherwise it
updates the bracket using the direction in the codec table. When the budget
runs out, it returns the probe with the smallest absolute target error and
sets `converged=false`. PNG always uses one probe and reports convergence
against the same tolerance.

`ZENSIM_TARGET_SECANT=1` (or `true`) enables a bracket-safeguarded secant step
using the two most recent probes. It accepts a finite estimate strictly inside
the current bracket; otherwise it uses the midpoint. This option is off by
default and does not change the metric being targeted.

The search assumes scores follow the codec knob's direction; it does not prove
monotonicity or repair inversions. Quantized knobs, plateaus, codec floors and
ceilings, and a limited probe budget can prevent convergence. It searches one
knob with fixed codec settings, retains the closest observed score, and does
not optimize encoded size among equally acceptable probes. Scores are not
clamped at zero by this helper; negative measurements remain visible.

For product evaluation, record achieved target error, bytes, probe count and
total encode/decode/score time, split by codec and content. Equal achieved
zensim scores alone do not establish equal perceived quality across codecs;
use the independent evidence required by the integration guide.

## Historical measurements

These records describe the profiles and codec versions used on their dates:

- [May 18, 2026 demo matrix](../benchmarks/zensim_target_demo_2026-05-18.md):
  three codecs, three images and four targets.
- [May 19, 2026 V6 cross-codec demo](../benchmarks/zensim_target_v6_cross_codec_2026-05-19.md):
  ten images and four codecs with the historical tuner-v2 calibration.

They do not certify today's library default or current codec adapters.

## Licensing

This internal, unpublished crate is **AGPL-3.0-only or Imazen commercial**
because it links the codec crates. The `zensim` metric library remains
MIT/Apache-2.0.
