# zensim-target

CLI + library that picks codec encode parameters to hit a user-typed
zensim score. Given `(image, target_score, codec)`, runs a binary
search over the codec's quality knob, encodes + decodes at each
probe, scores the round-trip via `zensim::Zensim::compute`, and
returns the encoded bytes that landed closest to `target ± tolerance`.

The runtime side of the "user-facing quality dial" goal documented in
[`zensim/CLAUDE.md`](../CLAUDE.md): the user types "give me zensim 70"
and the codec stack does the binary search.

## Licensing

This crate is **AGPL-3.0-only or Imazen commercial**. It links the
AGPL zen codec crates (`zenjpeg`, `zenwebp`, `zenavif`, `zenjxl`,
`zenpng`); `zensim` itself stays MIT/Apache. `publish = false` — the
crate is internal to the zensim workspace.

## CLI

```bash
cargo run --release -p zensim-target -- <input.png> \
    --target 70 \
    --codec zenjpeg \
    [--profile v0_3] \
    [--tolerance 1.0] \
    [--max-iterations 8] \
    [--output encoded.jpg]
```

Output:

```
codec=Jpeg  target=70.0  achieved=69.456  knob=78.438  bytes=62234  iters=5  converged=true
```

With the default trace (no `--quiet`), each iteration prints a
`iter / knob / achieved / bytes` row before the summary.

### Codecs

| `--codec`  | scale | direction | notes |
|------------|-------|-----------|-------|
| `zenjpeg`  | 5..99 (q) | q↑ → score↑ | `ApproxJpegli` + `ChromaSubsampling::Quarter` |
| `zenwebp`  | 1..100 (q) | q↑ → score↑ | `LossyConfig::with_quality`, method=4 |
| `zenavif`  | 1..100 (q) | q↑ → score↑ | `EncoderConfig::quality`, speed=6 |
| `zenjxl`   | 0.5..15 (distance) | distance↓ → score↑ | encode-only in v0.1; decode plumbing pending |
| `zenpng`   | n/a (lossless) | — | single probe; score reflects PNG↔PNG round-trip |

### Profiles

| `--profile` | `ZensimProfile` | notes |
|---|---|---|
| `v0_2` | `PreviewV0_2` | linear weights, small-image friendly |
| `v0_3` (default) | `PreviewV0_3` | production-grade MLP |
| `balanced` | `PreviewV0_5Balanced` | ⚠ returns near-zero scores for visually-good outputs in this workspace; see "Known limitations" |
| `compression` | `PreviewV0_5Compression` | ⚠ same |
| `ensemble` | `PreviewV0_5Ensemble` | ⚠ same |

## Rust API

```rust
use zensim::ZensimProfile;
use zensim_target::{CodecKind, TargetSpec, target_search};

let rgb: Vec<u8> = /* width*height*3 bytes */;
let spec = TargetSpec {
    target: 70.0,
    tolerance: 1.0,
    max_iterations: 8,
    profile: ZensimProfile::PreviewV0_3,
};
let result = target_search(&rgb, width, height, CodecKind::Jpeg, spec)?;
println!("achieved {:.2} at q={:.1} in {} iterations", 
    result.achieved_score, result.final_knob, result.iterations);
std::fs::write("out.jpg", &result.encoded)?;
```

## Algorithm

Binary midpoint search over the codec's native quality knob:

1. Encode reference at `q_mid = (q_lo + q_hi) / 2`.
2. Decode encoded bytes back to RGB.
3. Score `zensim(reference, decoded)` with the chosen profile.
4. If `|achieved - target| <= tolerance`: done.
5. If achieved > target (too sharp): `q_hi = q_mid`.
6. Else: `q_lo = q_mid`.

Capped at `max_iterations`. Returns the best probe by
`|achieved - target|` if the budget runs out without converging.

For zenjxl the direction is inverted (lower distance → higher
quality → higher score), handled internally via
`lower_quality_means_higher_score`.

## Demo results

See [`benchmarks/zensim_target_demo_2026-05-18.md`](../benchmarks/zensim_target_demo_2026-05-18.md)
for the full 3 codecs × 3 images × 4 targets = 36-cell matrix.
Headline: **33 / 36 (92 %) converged within ±1.5 score units**,
median 5 iterations.

## Known limitations (v0.1)

1. `ZensimProfile::PreviewV0_5*` produces near-zero scores for
   visually-good outputs (their bake maps zero-feature vectors to
   ~0, which the `skip_score_mapping=true` path uses verbatim).
   Default is `PreviewV0_3` until the V0_5 runtime is fixed.
2. zenjxl backend is encode-only; decode plumbing is a follow-up.
3. Single-knob search only — codec-specific knobs (subsampling,
   effort, speed) stay at defaults.
4. Bisection. A secant or Brent's-method update would converge
   faster on smooth RD curves.
