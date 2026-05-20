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
    [--profile tuner-v2] \
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
| `zenjxl`   | 0.01..15 (distance) | distance↓ → score↑ | full encode+decode wired via `zencodec` traits |
| `zenpng`   | n/a (lossless) | — | single probe; score reflects PNG↔PNG round-trip |

### Profiles

| `--profile` | `ZensimProfile` | notes |
|---|---|---|
| `v0_2` | `PreviewV0_2` | linear weights, small-image friendly |
| `v0_3` | `PreviewV0_3` | legacy MLP — useful fallback |
| `balanced` | `PreviewV0_5Balanced` | ⚠ ranking metric — non-monotonic q-step output (V0_5 ranking ships are NOT calibrated for quality-dial use) |
| `compression` | `PreviewV0_5Compression` | ⚠ same |
| `ensemble` | `PreviewV0_5Ensemble` | ⚠ same |
| `tuner` | `PreviewV0_5Tuner` | prior tuner ship (V_tuner-v2-s2 calibrated, 2026-05-18) |
| `tuner-v2` (default) | `PreviewV0_5TunerV2` | EXP-CROSS-CODEC-V6 ship (2026-05-19) — passes every Tuner-trail gate, Pareto-dominates `tuner` on monotonicity / median range / cross-codec PJND parity |

### Default profile: PreviewV0_5TunerV2 (2026-05-19)

The CLI default rotated from `PreviewV0_3` to `PreviewV0_5TunerV2`
because the latter strictly improves the four properties a quality
dial needs:

1. **Strict monotonicity 95.22 %** on the 50-image × 19-q JPEG sweep
   (vs prior `tuner` ship 92.78 %, vs V0_5 ranking ships 71–86 %).
2. **Tied rate 0.00 %** (vs ranking ships 57–76 % clamp-flat dead
   zones).
3. **Median dynamic range 78 score units** across the JPEG q range
   (vs V5 candidate 30.73 — V6 restored range without losing
   monotonicity).
4. **Cross-codec PJND parity** at T=63: mean butter_pnorm3 1.731
   (gate < 2.5), cc_std_median 0.91 across {jpeg, webp, avif, jxl},
   all-band cc_std_max 1.68.

CID22 SROCC sits at 0.8770 (essentially tied with `tuner` at
0.8786). KADID/TID/KonJND drop to 0.72 / 0.75 / 0.20 by design —
the Tuner trail trains on safesyn only and is **NOT** a general
ranking metric. For ranking workloads use `--profile balanced` or
`--profile compression`.

The cross-codec demo at
[`benchmarks/zensim_target_v6_cross_codec_2026-05-19.md`](../benchmarks/zensim_target_v6_cross_codec_2026-05-19.md)
runs 10 images × 4 codecs at T=63 and shows median z_std=0.64,
median p_std=0.10 — both well inside the gates documented in
the V6 methodology.

## Rust API

```rust
use zensim::ZensimProfile;
use zensim_target::{CodecKind, TargetSpec, target_search};

let rgb: Vec<u8> = /* width*height*3 bytes */;
let spec = TargetSpec {
    target: 70.0,
    tolerance: 1.0,
    max_iterations: 8,
    profile: ZensimProfile::PreviewV0_5TunerV2,
};
// Or use the default, which is also PreviewV0_5TunerV2:
// let spec = TargetSpec { target: 70.0, ..TargetSpec::default() };
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

1. `PreviewV0_5{Balanced,Compression,Ensemble}` profiles produce
   non-monotonic scores in the target search loop — they're ranking
   metrics, not quality-dial metrics. Use `tuner-v2` (default) or
   `tuner` for quality-dial workloads.
2. Single-knob search only — codec-specific knobs (subsampling,
   effort, speed) stay at defaults.
3. Bisection. A secant or Brent's-method update would converge
   faster on smooth RD curves.
4. Screen-content images with text-rich regions (UI screenshots,
   code listings) may hit the codec's q-ceiling before reaching low
   targets (e.g. T < 65 on dense-text images via zenjpeg / zenwebp).
   The search returns the best-so-far in that case (`converged=false`).
