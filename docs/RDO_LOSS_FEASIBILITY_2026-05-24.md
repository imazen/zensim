# RDO-loss feasibility for zen codecs (2026-05-24)

**Task #3.** The user requirement: zensim is the "target metric all zen
codecs train to" with three use cases — (A) quality-target dial,
(B) picker training, (C) in-encoder RDO loss. This doc scopes whether
zensim can serve each use case as-is, what's already wired, and what
architectural work the third use case would actually require.

## TL;DR

| Use case | Current state | What it needs |
|---|---|---|
| **A. Quality-target dial** (user types score, codec binary-searches q) | ✅ **already shipped** — `zenwebp::EncodeConfig::target_zensim` runs the iterative encode→rescore→adjust loop. Cost: 5–10 forward passes × ~14 ms each at 1024² = ~70–140 ms per image. | Document the pattern; add the same plumbing to zenjpeg/zenjxl/zenavif. **No architecture work.** |
| **B. Picker training** (ML picker takes features + target, outputs `(codec, q)`) | ✅ **already works** — `bake_verdict` + per-codec parquets at `/mnt/v/zen/picker-training/2026-05-19/butter/*.parquet` are exactly this. V11 cross-codec substrate already uses this pattern. | Re-wire when a new bake ships. **No architecture work.** |
| **C. In-encoder RDO** (per-block trellis decisions use zensim as distortion term) | ❌ **not feasible with current zensim** at codec-RDO cadence (~5k–20k RDO decisions per image × 4 ms minimum = 20–80 s per image, before any per-block cost). | Either (1) full reimplementation in an autograd-capable framework, or (2) train a fast **per-block proxy** ("zensim-lite") that predicts the full bake's score from codec-internal coefficients. (3) Most realistic: **skip in-encoder RDO** — codecs continue using PSNR/SOS internally, zensim runs only at the output. This is the current SOTA pattern for non-end-to-end codecs (mozjpeg, jpegli, libjxl all do this). |

**The honest answer for "target metric all zen codecs train to":**
the **dial (A) + picker (B)** use cases are the load-bearing ones,
they work today, and the path forward is documentation +
trail-of-bakes improvement (tasks #4–#6). Use case (C) is a
research direction, not a blocker.

## Measured zensim per-pair latency

From `benchmarks/iw_perf_optimized_{256,512,1024}_v3_2026-05-15.log`
(parallel=true, rayon, 372-feature extended+IW, AMD Ryzen 9 7950X):

| Image | n_features | median ms | per-pixel ns |
|---|--:|--:|--:|
|  256×256   | 372 |  3.85 | 59 ns/px |
|  512×512   | 372 |  6.22 | 24 ns/px |
| 1024×1024  | 372 | 13.81 | 13 ns/px |
| 2048×1024  | 372 | 34.02 | 16 ns/px (CHANGELOG 2026-05-22) |

Sub-linear scaling (per-pixel cost drops with image size) reveals
~3 ms of **fixed-overhead per call**: config setup, color
conversion to positive XYB, 4-scale pyramid allocation, MLP forward
pass. This floor is what makes per-block RDO use infeasible without
a different architecture.

## Use case A — Quality-target dial (already shipped)

### What's wired

`zenwebp::EncodeConfig::target_zensim` (per `src/encoder/api.rs`)
sets a target score `T ∈ [0, 100]`. The encoder runs:

1. Initial encode at the recommended q for T (lookup table or model).
2. Decode result, run `Zensim::compute(source, distorted)`.
3. If `|score − T| > tolerance`: adjust q, goto 1. Bounded to ≤ N
   iterations (typically 5–10).

Existing supporting infrastructure:
- `Zensim::compute` (`zensim/src/metric.rs:954`) — single-pair score
- `Zensim::compute_with_ref` (`metric.rs:1079`) — reuse precomputed reference (saves ~50% on the iterative loop)
- `Zensim::precompute_reference` (`metric.rs:1052`) — once-per-source
- `PreviewV0_5Tuner` profile — the bake purpose-built for this (monotonic dial, JND=60, JOD=30 — see `SOTA_TRAILS.md`)

### What's missing

| Codec | Has `target_zensim`? | Notes |
|---|---|---|
| zenwebp | ✅ yes | shipped |
| zenjpeg | ❌ | needs the same outer-loop plumbing |
| zenjxl  | ❌ | needs plumbing |
| zenavif | ❌ | needs plumbing |
| zenpng (lossless) | n/a | no q dial |
| zengif | ❌ | optional; dither/palette decisions |

**Estimated cost per codec**: 1–2 days each. The outer loop is
~100 LOC. The main work is wiring the precompute-reference path
through the codec's existing API so we don't re-run the 372-feature
extraction every iteration. Tracker task — not in scope of the
metric work itself; should be opened against each codec's repo.

### Cross-codec consistency at the dial

Per `benchmarks/tuner_v10_cross_codec_baseline_2026-05-24.md`
(task #1 output): at matched-anchor pairs, **median |Δ| = 0.6–1.5
score units in the 60–90 band**. For "target score 70" use, the
expected cross-codec spread is < ±1.5 units across JPEG/WebP/AVIF/JXL.

**The dial is precise enough today.** What's NOT precise enough is
the score 0–55 region (floor saturation, task #1 finding #2); that's
the gap task #6 (Tuner v11 retrain) closes.

## Use case B — Picker training (already wired)

### What's wired

A "picker" is a model that takes `(image_features, target_score)`
and outputs `(codec, q)` — the right codec+quality combo to use.
This is the zenpicker/zenpredict crate stack:

- `zenpicker::CodecFamily {Jpeg, Webp, Jxl, Avif, Png, Gif}` + `AllowedFamilies` mask
- `zenpicker::MetaPicker` wraps a `zenpredict::Predictor` (the same
  Predictor that drives MLP bakes — type-symmetric with zensim's runtime)
- Training data: per-codec parquets at
  `/mnt/v/zen/picker-training/2026-05-19/butter/{zenjpeg,zenwebp,zenavif,zenjxl}.parquet`
  (1000 refs × 19 q-levels each, 380 cols = ref_basename + codec + q +
  butter_max + butter_pnorm3 + 372 features). 19,000 rows per codec.
- The training pipeline (`zenanalyze/zentrain/`) produces per-codec
  bakes; current production picks are at `zensim/weights/picker_zen{jpeg,webp,jxl,avif}_2026-05-19.bin`.

### What it asks of zensim

A picker training cycle queries: "given features X at codec C and q Q,
what does zensim say?" — answered by `bake_verdict` (or
`predict_features_with_bake` for direct feature input). This is
already a batch operation against pre-extracted feature parquets.
**No new architecture needed.** When the canonical codec-target bake
rotates (e.g., task #6's Tuner v11), the picker training command just
re-runs against the new bake.

### What to document

The "which bake do pickers train against?" decision: **PreviewV0_5Tuner**
(currently `v_tuner_v10_2026-05-20.bin`). Task #2 makes this official.

## Use case C — In-encoder RDO (the hard one)

### What "in-encoder RDO" means in a traditional codec

Codecs like JPEG, WebP-lossy, AVIF, JXL make thousands of per-block
or per-coefficient quality decisions inside the encode loop. Each
decision is "given this transform coefficient (or block / segment /
partition), what quant value minimizes `rate + λ · distortion`?"

The **distortion term** is per-block, recomputed for every candidate
decision. For a 1920×1080 JPEG with 8×8 DCT blocks, that's 32,400
blocks × ~10 candidate quant values = ~324k distortion calls per
image. Trellis quantization adds another 64 × per-coefficient
decisions per block. **The total budget per distortion call is
microseconds, not milliseconds.**

### Why zensim doesn't fit there today

zensim is fundamentally **per-image, multi-scale**:

1. Build 4-scale Gaussian pyramid on both source and distorted (full image).
2. Per scale, per channel: compute mean/var/cov maps, SSIM map,
   edge-diff map, optionally activity-masked + IW-weighted versions.
3. Pool features per scale (mean / max / variance / percentiles).
4. Concatenate 372 features.
5. MLP forward pass + (optional) PCHIP spline.

Steps 1–3 are inherently global — the multi-scale pyramid means
a 64×64 block's "scale 3" is 8×8, which is at the floor of what
the blur kernel handles. And step 5 (MLP) is a network trained
against per-image scores, not per-block.

**Even if you ran zensim on a per-block basis** (`Zensim::compute`
on each 64×64 block), the 3.85 ms floor times 32,400 blocks =
124 s per image. Times 10 candidate q values = 21 minutes. Times
the trellis dimension = days. Not viable.

### Three paths if (C) becomes a real requirement

**Path 1 — Differentiable end-to-end zensim.**
Re-implement the feature extraction in an autograd framework
(CubeCL has autograd primitives; PyTorch is the ML-ecosystem default).
This makes the full per-image zensim usable as a backprop target for
training a *codec's* parameters (e.g., quant tables, R-D-O lambdas)
via standard ML.

- Pro: principled, can train any differentiable codec component to
  optimize zensim end-to-end.
- Con: massive engineering effort. Every kernel in `zensim/src/`
  (color conversion, multi-scale blur, SSIM, edge-diff, IW pooling,
  MLP forward) gets a backward pass. Months of work. The codec being
  trained also needs to be differentiable; most production codecs
  aren't.
- Status: not started. Out of scope unless (C) becomes the gating
  use case.

**Path 2 — Fast per-block proxy (zensim-lite).**
Train a small CNN (or even a linear model) that takes codec-internal
quantities (transform coefficients, block statistics, edge maps)
and predicts the full zensim score. The training signal is
(codec_internal_features → full_zensim_score) pairs over the
existing safesyn corpus.

- Pro: per-block evaluation in microseconds; codec gets a usable
  distortion term inside the trellis. Analytical gradient is easy
  for a small net.
- Con: it's a proxy — accurate only on the training distribution.
  Drift between proxy and full zensim limits how much it can
  actually improve the codec output (the codec optimizes the proxy,
  but quality is gated by the full zensim's verdict at output).
- Status: not started. ~1 week per codec to design + train.

**Path 3 — Skip in-encoder RDO; output-only zensim (current SOTA).**
Codecs continue using simpler internal proxies (mozjpeg/jpegli use
PSNR / sum-of-squares + scale factors; libjxl uses butteraugli for
some decisions but a custom distortion for trellis). zensim runs
only at the **output level** as the fitness function for choosing
among candidate encodes.

- Pro: zero new architecture. This is what all production zen
  codecs already do (mozjpeg-rs, jpegli, libjxl, libavif).
  zenwebp's `target_zensim` is exactly this pattern.
- Con: codec internal decisions aren't directly optimizing zensim;
  the codec output is whatever its internal proxy plus the outer
  loop produces.
- Status: production. Improving this means improving the codec's
  internal proxy quality, not adding zensim to the inner loop.

### Recommendation

**Path 3 (output-only zensim) is the recommendation for v1 of "the
canonical codec-target metric"** — it's already shipped, it's how
every major codec actually works, and it lets the metric improve
independently of any codec internal.

**Path 2 (proxy net) is the upgrade path** when a specific codec
(probably zenjxl, which has the most flexibility in its variable-DCT
trellis) wants direct zensim feedback inside the inner loop. Defer
until a codec asks for it.

**Path 1 (differentiable end-to-end)** is research. Worth scoping
the engineering cost once Path 2 has empirical evidence about how
much accuracy the proxy approach leaves on the table.

## What changes operationally

Nothing in the metric crate changes for use cases A + B. The codec
crates need:

1. **Per-codec `target_zensim` plumbing** (use case A). Pattern is
   `zenwebp/src/encoder/api.rs`; copy it to zenjpeg/zenjxl/zenavif.
   File issues in each codec's repo. Not zensim's responsibility.
2. **One stable public re-export from zensim::profile** so codec
   crates have a single import:
   ```rust
   /// The canonical codec-target bake. Used by codec `target_zensim`
   /// iterative loops and picker training. See SOTA_TRAILS.md "Tuner trail".
   pub const CODEC_TARGET_PROFILE: ZensimProfile = ZensimProfile::PreviewV0_5Tuner;
   ```
   Land this in task #2 (the documentation task).

## Open questions for the user

1. **Which codecs actually want `target_zensim` plumbing soon?**
   zenjpeg, zenjxl, zenavif — pick one to seed the pattern; the
   others follow. (Not blocking; useful for sequencing.)
2. **Does any codec want to investigate Path 2 (proxy net) now?**
   If zenjxl's variable-DCT trellis is interested, a 1-week
   experiment is reasonable. Otherwise this stays deferred.

## See also

- `benchmarks/tuner_v10_cross_codec_baseline_2026-05-24.md` — task #1
  measured cross-codec consistency; informs A's precision floor.
- `SOTA_TRAILS.md` "Tuner trail" — the bake that codecs train against.
- `~/work/zen/zenwebp/src/encoder/api.rs` — reference implementation
  of the iterative `target_zensim` outer loop.
- `~/work/zen/zenanalyze/zenpicker/` — picker crate, use case B.
- `/mnt/v/zen/picker-training/2026-05-19/butter/` — picker training
  data, ready to re-run when the canonical bake rotates.
