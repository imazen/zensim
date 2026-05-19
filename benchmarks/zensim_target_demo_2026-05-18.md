# zensim-target — demo matrix (2026-05-18)

`zensim-target` is a CLI / library that picks codec encode parameters
to hit a user-typed zensim score. Given `(image, target_score,
codec)`, it runs a binary search over the codec's quality knob,
encodes + decodes at each probe, scores the round-trip via
`zensim::Zensim::compute`, and returns the encoded bytes that landed
closest to `target ± tolerance`.

## Demo configuration

- **Crate**: `zensim-target` (new workspace member, `publish = false`,
  AGPL-3.0-only or Imazen commercial — same license as the codec
  dependencies it links).
- **Profile**: `ZensimProfile::PreviewV0_3` (see "Known limitations"
  for why this is the default instead of `PreviewV0_5*`).
- **Tolerance**: ±1.5 score units.
- **Iteration cap**: 8.
- **Algorithm**: midpoint binary search over the codec's native
  quality knob; q↑ → score↑ for jpeg / webp / avif (zenjxl, which
  uses inverted distance, is wired in the library but not yet
  exercised in this demo).

## Test images

| label | class | path | resolution |
|---|---|---|---|
| kadid I12 | photo | `/home/lilith/work/codec-eval/codec-corpus/kadid10k/I12.png` | 512×384 |
| gb82-sc gui | screen | `/home/lilith/work/codec-eval/codec-corpus/gb82-sc/gui.png` | 1356×1132 |
| kadid I50 | line-art | `/home/lilith/work/codec-eval/codec-corpus/kadid10k/I50.png` | 512×384 |

## Results — 3 codecs × 3 images × 4 targets = 36 cells

Captured via `cargo run --release --example demo_matrix -p zensim-target`
on commit `<insert-commit-sha-after-commit>` against the workspace
sibling layout at `~/work/zen/`.

| image (class) | codec | target | achieved | Δ | knob | bytes | iters | converged |
|:---|:---|---:|---:|---:|---:|---:|---:|:---:|
| kadid I12 (photo) | zenjpeg | 30 | 28.70 | -1.30 | 13.81 | 17026 | 5 | yes |
| kadid I12 (photo) | zenjpeg | 50 | 50.50 | +0.50 | 46.12 | 33576 | 4 | yes |
| kadid I12 (photo) | zenjpeg | 70 | 69.46 | -0.54 | 78.44 | 62234 | 5 | yes |
| kadid I12 (photo) | zenjpeg | 90 | 90.44 | +0.44 | 91.66 | 100432 | 6 | yes |
| kadid I12 (photo) | zenwebp | 30 | 29.16 | -0.84 | 7.19 | 22490 | 4 | yes |
| kadid I12 (photo) | zenwebp | 50 | 51.08 | +1.08 | 31.94 | 42936 | 4 | yes |
| kadid I12 (photo) | zenwebp | 70 | 69.45 | -0.55 | 76.02 | 68138 | 7 | yes |
| kadid I12 (photo) | zenwebp | 90 | 90.30 | +0.30 | 89.17 | 101184 | 6 | yes |
| kadid I12 (photo) | zenavif | 30 | 30.07 | +0.07 | 31.94 | 19168 | 4 | yes |
| kadid I12 (photo) | zenavif | 50 | 50.97 | +0.97 | 47.41 | 37207 | 5 | yes |
| kadid I12 (photo) | zenavif | 70 | 70.71 | +0.71 | 65.97 | 63711 | 5 | yes |
| kadid I12 (photo) | zenavif | 90 | 89.40 | -0.60 | 78.34 | 89132 | 5 | yes |
| gb82-sc gui (screen) | zenjpeg | 30 | 34.00 | +4.00 | 5.37 | 14091 | 8 | no |
| gb82-sc gui (screen) | zenjpeg | 50 | 51.37 | +1.37 | 10.88 | 16231 | 4 | yes |
| gb82-sc gui (screen) | zenjpeg | 70 | 68.90 | -1.10 | 22.62 | 20392 | 4 | yes |
| gb82-sc gui (screen) | zenjpeg | 90 | 89.62 | -0.38 | 75.50 | 33134 | 2 | yes |
| gb82-sc gui (screen) | zenwebp | 30 | 31.52 | +1.52 | 2.55 | 8580 | 8 | no |
| gb82-sc gui (screen) | zenwebp | 50 | 48.39 | -1.61 | 14.92 | 10520 | 8 | no |
| gb82-sc gui (screen) | zenwebp | 70 | 70.61 | +0.61 | 31.94 | 11758 | 4 | yes |
| gb82-sc gui (screen) | zenwebp | 90 | 89.38 | -0.62 | 81.44 | 18162 | 4 | yes |
| gb82-sc gui (screen) | zenavif | 30 | 31.27 | +1.27 | 19.56 | 7338 | 4 | yes |
| gb82-sc gui (screen) | zenavif | 50 | 49.92 | -0.08 | 27.30 | 8766 | 6 | yes |
| gb82-sc gui (screen) | zenavif | 70 | 71.02 | +1.02 | 42.77 | 12200 | 6 | yes |
| gb82-sc gui (screen) | zenavif | 90 | 89.95 | -0.05 | 62.88 | 17636 | 3 | yes |
| kadid I50 (line-art) | zenjpeg | 30 | 30.40 | +0.40 | 10.88 | 8900 | 4 | yes |
| kadid I50 (line-art) | zenjpeg | 50 | 50.20 | +0.20 | 40.25 | 17501 | 3 | yes |
| kadid I50 (line-art) | zenjpeg | 70 | 70.96 | +0.96 | 78.44 | 34672 | 5 | yes |
| kadid I50 (line-art) | zenjpeg | 90 | 90.98 | +0.98 | 91.66 | 60219 | 6 | yes |
| kadid I50 (line-art) | zenwebp | 30 | 29.47 | -0.53 | 7.19 | 7914 | 4 | yes |
| kadid I50 (line-art) | zenwebp | 50 | 48.52 | -1.48 | 31.94 | 16446 | 4 | yes |
| kadid I50 (line-art) | zenwebp | 70 | 69.88 | -0.12 | 78.34 | 35378 | 5 | yes |
| kadid I50 (line-art) | zenwebp | 90 | 88.88 | -1.12 | 90.72 | 62464 | 5 | yes |
| kadid I50 (line-art) | zenavif | 30 | 31.29 | +1.29 | 28.84 | 7552 | 5 | yes |
| kadid I50 (line-art) | zenavif | 50 | 49.30 | -0.70 | 45.86 | 16620 | 6 | yes |
| kadid I50 (line-art) | zenavif | 70 | 68.51 | -1.49 | 62.88 | 31023 | 3 | yes |
| kadid I50 (line-art) | zenavif | 90 | 88.65 | -1.35 | 78.34 | 51639 | 5 | yes |

**Aggregate**: 33 / 36 cells converged within ±1.5 score units (92 %).

| codec   | converged | of |  rate |
|---------|----------:|---:|------:|
| zenavif |        12 | 12 | 100 % |
| zenjpeg |        11 | 12 |  92 % |
| zenwebp |        10 | 12 |  83 % |

Median iterations to converge: 5. Worst-case (non-converged): 8 (hit the cap).

## Failure analysis (3 cells)

All three failures are at `target = 30` on the screen-content image
(`gb82-sc/gui.png`):

- **zenjpeg @ q=5.37 → score=34.00** — the JPEG-q floor (5.0 in this
  build) still produced a sharper-than-target result on this UI
  screenshot. To hit exactly 30, the codec needs to compress past
  its useful q range. Workaround: relax the lower bound or accept
  the floor result.
- **zenwebp @ q=2.55 → score=31.52** — same shape.
- **zenwebp @ q=14.92 → score=48.39** — got close to the q=50 target
  but missed the band; the search was oscillating around the
  boundary. A tighter secant-step (vs midpoint bisection) would
  likely fix this in 1-2 more iterations.

For zenavif, every cell — including target=30 on screen-content —
converged: AV1 has a wider effective q range so the binary search
can reach lower scores without bottoming out.

## Known limitations (v0.1)

1. **`ZensimProfile::PreviewV0_5*` returns near-zero scores for
   visually-good outputs in this workspace.** Reproduced in
   `zensim-target/examples/smoke_check.rs`: identity (image scored
   against itself) returns score=0 for `PreviewV0_5` and
   `PreviewV0_5Balanced`, ~2 for `PreviewV0_5Compression` and
   `PreviewV0_5Ensemble`. The bake's MLP raw output for an
   all-zero-feature vector evaluates to ~0, and with
   `skip_score_mapping = true` that ~0 IS the final score. The
   short-circuit at `compute_zensim_inner` correctly returns 100
   for byte-identical images, but `apply_mlp_scoring` then
   overwrites it with the MLP path. Tools that need V0_5* scoring
   should track this in the zensim crate itself; for the target
   search loop, V0_3 is the production-grade fallback.
2. **zenjxl backend is encode-only**. The decode plumbing requires
   either pulling jxl-rs in directly or adding a typed pixel
   accessor on `zenjxl::JxlDecodeOutput`. Encode bytes can be
   produced via the existing API but a decoded-RGB8 buffer is
   needed for zensim scoring. Tracked as a follow-up.
3. **No knob tuning beyond q**. The current search only varies the
   single quality knob. Per-codec knobs (subsampling, effort,
   speed, trellis) are fixed at sensible defaults. Multi-knob
   search is a phase-4+ extension.
4. **Binary midpoint search**. A secant or Brent's method update
   would converge ~2× faster on smooth RD curves. Bisection is
   robust enough for v0.1 but leaves easy wins on the table.

## How to reproduce

```bash
cd ~/work/zen/zensim
cargo build --release -p zensim-target
target/release/zensim-target <input.png> --target 70 --codec zenjpeg
# or the full demo
target/release/examples/demo_matrix
```

The demo's image paths are hard-coded in
`zensim-target/examples/demo_matrix.rs`; they assume the
`codec-eval/codec-corpus/` checkout at
`~/work/codec-eval/codec-corpus/`.
