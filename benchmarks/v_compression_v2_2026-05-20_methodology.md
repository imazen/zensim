# PreviewV0_5CompressionV2 — V9 PCHIP spline calibration on the Compression bake

Task #177 (2026-05-20).
Sibling work: task #176 / commit `5c5ca6b` (PreviewV0_5BalancedV2).
Spline mechanism upstream: EXP-CROSS-CODEC-V9 (commits `0829b51` +
`ddd85b9` + `5386d55`, PreviewV0_5TunerV3 ship).

## Summary

Ports the V9 PCHIP spline calibration mechanism onto the existing
Compression-trail ship without retraining the network. Adds
`zentrain.output_calibration_spline` metadata to the V_24-per-sample-α
s4 bake; ships as `ZensimProfile::PreviewV0_5CompressionV2` (opt-in
only — `PreviewV0_5Compression` remains the default for backward
compat per the task's ship-gate clause).

The mechanism is architecture-agnostic and was already proven on
both V_22-mix (plain MLP, BalancedV2) and V_24-per-sample-α
(TunerV3). The Compression bake is V_24-per-sample-α — the same
family as TunerV3 — so the spline fit is structurally clean.

## Why

The Compression bake's per-sample-α-mixed output is **distance-shaped**:
high raw value = low quality, range ≈ [-27, 20] across the V9 anchor
parquet's `target_score` bands (median raw_pred per band, see
"Calibration log" below).

The Compression profile's `ProfileParams` carries
`soft_clamp_score: true`, which routes the raw output through
`100 / (1 + exp(-(raw - 50) / 20))`. For raw ∈ [-27, +20], this
squashes the user-facing dial into the narrow band **[2.07, 18.24]**:

| raw    | soft_clamp(raw)             |
|-------:|----------------------------:|
| −27    | 2.07                        |
| −15 (med) | 3.71                     |
| 0      | 7.59                        |
| +20    | 18.24                       |

A user invoking `zensim-target` and typing "score 60" will never
get a hit — the dial maxes out at ~18. The `bake_verdict`
sign-tolerant SROCC of 0.864 (CID22) hides this because it reads
the raw post-α-mix `y_pre` before any clamp/squash.

**This is the same dial-bug pattern that the BalancedV2 ship
caught** (commit `5c5ca6b`). Two parallel findings now:

- Balanced base: clamp(0, 100) on distance-shaped raw → 96.8% of
  CID22 predictions pinned to 0.
- Compression base: soft_clamp on distance-shaped raw → dial
  collapsed to ≈ [2, 18].

Both are structurally broken user-facing dials that the spline fix
addresses without retraining.

## Mechanism

Same V_24-per-sample-α s4 bake bytes as
`PreviewV0_5Compression` (300 → 128 → 128 (identity passthrough)
i8 + zerobias + lz4 MLP carrying `zentrain.per_sample_alpha_head`
metadata, 44,109 bytes, md5 `f09a9abdce00805000c1d112c2421b2d`).

The only addition is a 7-knot **post-network PCHIP spline** baked
into the new `zentrain.output_calibration_spline` metadata key.
The Fritsch-Carlson monotone-preserving spline is structurally
rank-preserving so this is a pure calibration change — no
retraining, no rank quality change.

Runtime dispatch lives in `zensim::metric::forward_one_bake`
(unchanged from the BalancedV2 ship). Order of operations for the
Compression-V2 bake:

1. Forward pass through the 300 → 128 → 128 MLP, yielding the
   hidden vector `h` (128 floats).
2. Apply the per-sample-α head: `y_pre = α(h) · y_rank(h) +
   (1 − α(h)) · y_pool(h)`.
3. (No tanh-pin — this bake doesn't carry `tanh_output_head`.)
4. **Apply the PCHIP spline**: `score = pchip(y_pre, knots)`.
5. Hard clamp `score ∈ [0, 100]` (no-op for in-distribution
   inputs; catches OOD extrapolation tails only).

The `soft_clamp_score` flag is set to `false` for the V2 profile
(the spline already maps onto [0, 100]; the soft-clamp would
re-squash an already-calibrated dial).

## Calibration log

Per-band median predicted (raw, pre-spline) on the V9 anchor
parquet at `/mnt/v/zen/zensim-training/2026-05-20-v9-anchors/anchors_v9_372col.parquet`
(22,008 rows × 300 features, scored via `predict_features_with_bake
--bake-post raw` so the per-sample-α-mixed raw output is exposed
before any clamp/spline):

| target_score | n    | median raw | p25      | p75      | min      | max      |
|-------------:|-----:|-----------:|---------:|---------:|---------:|---------:|
| 0.0          | 1159 |  13.886    |   9.142  |  18.869  |  −7.299  |  37.257  |
| 10.0         |  296 |  17.547    |  10.299  |  20.321  |  −2.467  |  37.257  |
| 30.0         | 2055 |  10.313    |   5.970  |  13.156  | −13.943  |  27.513  |
| 50.0         | 3934 |   3.847    |   1.079  |   6.398  | −16.812  |  20.588  |
| 60.0         | 3933 |  −3.580    |  −5.996  |  −1.697  | −21.402  |  19.445  |
| 80.0         | 3714 | −16.927    | −18.269  | −15.384  | −25.053  |  −1.573  |
| 90.0         | 3406 | −20.207    | −21.817  | −18.319  | −27.146  |  −9.571  |
| 100.0        | 3511 | −26.888    | −27.078  | −21.990  | −27.361  | −10.547  |

Inferred bake shape: **distance-shaped** (target=high raw_med=−26.89,
target=low raw_med=+13.89).

The spline knots (xs sorted ascending, ys following the bake's
distance shape ⇒ ys monotone-decreasing):

```
  x=-26.8879 → y=100.0  (lossless ceiling)
  x=-20.2067 → y= 90.0
  x=-16.9266 → y= 80.0
  x= -3.5800 → y= 60.0  (JND / PJND anchor — bit-exact knot)
  x=  3.8475 → y= 50.0
  x= 10.3135 → y= 30.0  (JOD anchor — bit-exact knot)
  x= 13.8858 → y=  0.0  (worst-codec floor)
```

PCHIP derivatives at the knots: `[-0.456, -2.089, -2.158, -1.411,
-1.893, -4.731, -10.286]` — all negative, confirming monotone
decreasing across the full domain.

Verification on a 1000-point dense grid `xs[0] − 5 .. xs[-1] + 5`:
`max_drop = 0.0` (monotone bit-exact, direction=−1).

**Band 10 (target=10) was dropped at fit time** — the network puts
target=10 at raw=17.55, which is *above* target=0 at raw=13.89.
This is a direction violation the monotone spline cannot fix. The
spline linearly extrapolates the target=10 region using the
surrounding `target=0 → target=30` knot slope; the production
clamp pulls the worst-bucket bandwidth-floor predictions to 0
either way.

This is the **same band-10 direction violation** that BalancedV2
saw on the V_22-mix bake — it's a V9-anchor-parquet property at
the low-quality end (target=10 was built from `b7.0` anchors that
collide with the worst-codec floor's `worstfloor` anchors), not a
bake-architecture property.

## Cross-corpus SROCC (Mohammadi panel, `bake_verdict`)

Held-out evaluation on the canonical val parquets at
`/mnt/v/zen/zensim-training/2026-05-15-full-features/`:

| Corpus  | n    | Compression base | CompressionV2 | Δ SROCC |
|---------|-----:|-----------------:|--------------:|--------:|
| CID22   | 4292 |           0.8641 |        0.8641 | +0.0000 |
| KADID   | 10125 |          0.9316 |        0.9316 | +0.0000 |
| TID     | 3000 |           0.8893 |        0.8893 | +0.0000 |
| KonJND  | 1008 |           0.8080 |        0.8080 | +0.0000 |
| AIC-3   |  600 |           0.8183 |        0.8183 | +0.0000 |

SROCC bit-exact preserved on all 5 corpora — PCHIP is
rank-invariant by construction.

PLCC, KROCC, OR, PWRC, and Z-RMSE drift by ≤ 0.04 from refit (the
spline reshapes the calibration curve, but does not change the
ordering). Full panel diff in
`benchmarks/v_compression_v2_2026-05-20_verdict.md` and
`benchmarks/v_compression_base_2026-05-20_verdict.md`.

## Anchor-landing (V9 anchor parquet)

Median V2 score over each `target_score` band:

| target | n    | median V2 | abs err vs target  |
|-------:|-----:|----------:|-------------------:|
|     0  | 1159 |   0.000   |   0.000            |
|    30  | 2055 |  30.000   |   0.000            |
|    50  | 3934 |  50.000   |   0.000            |
|    60  | 3933 |  60.000   |   0.000 (JND)      |
|    80  | 3714 |  80.000   |   0.000            |
|    90  | 3406 |  90.000   |   0.000            |
|   100  | 3511 | 100.000   |   0.000            |
|    10  |  296 |   0.000   |  10.000 (band dropped — see above) |

By construction, every kept band lands at its integer knot.

## Cross-codec consistency (V9 anchor parquet)

Per `(ref_basename, target_score)` group with ≥ 2 codecs present
in the V9 anchor parquet, compute `std(prediction)` across codecs
in that group. Report mean and max `cc_std` per band.

Compression base scored through the **production runtime path**
(`predict_features_with_bake --bake-post clamp`), which mirrors
the production hard-clamp behavior (this is the dial the user
actually sees):

| target | Compression base mean/max cc_std | CompressionV2 mean/max cc_std |
|-------:|---------------------------------:|------------------------------:|
| 0      | 1.668 / 9.895                    | 4.309 / 31.237                |
| 10     | 2.375 / 6.813                    | 10.500 / 32.104               |
| 30     | 1.273 / 7.499                    | 4.922 / 29.606                |
| 50     | 0.851 / 6.680                    | 3.100 / 27.375                |
| **60 (JND)** | **0.130 / 4.260**          | **2.096 / 12.020**            |
| 80     | 0.000 / 0.000                    | 3.783 / 13.037                |
| 90     | 0.000 / 0.000                    | 4.847 / 13.814                |
| 100    | 0.000 / 0.000                    | 3.543 / 13.279                |

Compression base mean cc_std appears extremely tight at high
targets — but this is **the dial-bug talking**: 100% of target=80,
90, 100 predictions clamp to 0 under the production runtime,
producing cc_std=0 because all codecs land at the same flat
floor. The user-facing dial is broken; the tight cc_std is a
mirage of the saturation.

CompressionV2 cross-codec std at PJND (JND=60) mean=2.096 —
**passes the V9 ship's `cc_std ≤ 5` mean gate at JND**.

Max cc_std exceeds 5 across the board (JOD max 29.6, T60 max
12.0, T80 max 13.0) — wider than V9 TunerV3 on every band's MAX.
This is expected: the Compression bake was NOT trained with
cross-codec equivalence pairs (V9 TunerV3 was). The spline can
only land the per-band MEDIAN at the integer anchor; per-image
variance across codecs remains the network's responsibility.

For reference, the BalancedV2 ship saw mean cc_std = 2.17 at JND
(same V9 anchor parquet); CompressionV2's 2.096 is marginally
tighter — both pass the V9 ≤5 gate.

## Ship decision

Per the task's gate clauses:

1. **All Compression base SROCC numbers preserved within ±0.01**:
   YES (bit-exact, Δ=0.0000 on every corpus).
2. **JND/JOD bit-exact**: YES (medians 60.000 / 30.000 at the
   knots).
3. **Range ≥ 80**: YES (V9 anchor parquet predictions span the
   full [0, 100]).
4. **Mean cc_std at JND ≤ 5**: YES (2.096 < 5).

→ **Passes the ship gate.**

Per the task's "if passes BUT cross-codec is wider than V9
TunerV3: ship as opt-in, NOT default" clause:

- Max cc_std is wider than V9 TunerV3 at every band (the
  Compression bake was not cross-codec-trained).
- → **CompressionV2 ships as opt-in profile only**
  (`ZensimProfile::PreviewV0_5CompressionV2`).
- `PreviewV0_5Compression` still routes to the unchanged
  `PROFILE_PREVIEW_V0_5_COMPRESSION` for backward compat.

## Parallel finding to BalancedV2

This is the **second instance** of the production-dial-bug pattern
that was first surfaced by BalancedV2 (task #176):

- BalancedV2 caught: **Balanced** bake's distance-shaped raw +
  hard `clamp(0, 100)` → 96.8 % of CID22 predictions pinned to 0.
- CompressionV2 catches: **Compression** bake's distance-shaped
  raw + `soft_clamp_score` → dial squashed into ≈ [2, 18].

In both cases, `bake_verdict` reported strong SROCC because it
reads the raw `y_pre` before any clamp. The user-facing dial via
`zensim::Zensim::compute` was structurally broken.

The spline mechanism fixes both bakes identically via metadata
injection. This pattern likely affects **every distance-shaped
RankNet-trained bake** in the zensim slot — a `bake_verdict`-only
audit cannot catch it. A new audit pass would check the dial
range under `predict_features_with_bake --bake-post clamp` (or
the equivalent production runtime) for every bake in
`zensim/weights/` to surface remaining instances.

(Audit not run in this task — out of scope. Surfaced here for the
followup queue.)

## Bake artifacts

- **Spline calibrator (re-usable)**: `scripts/v_next/calibrate_balanced_v9_spline.py`
  (sibling script — works on any 300-input distance-shaped bake;
  the per-sample-α-head dispatch is handled inside
  `predict_features_with_bake`, so the calibrator is unaware of
  the architecture beyond input dim).
- **Output bake**: `zensim/weights/v_compression_v2_2026-05-20.bin`
  (44,208 bytes, +99 over the 44,109 base).
- **Spline knots CSV**: `benchmarks/v_compression_v2_2026-05-20_spline.csv`.
- **Calibration log**: `benchmarks/v_compression_v2_2026-05-20_calibration.log`.
- **Bake verdict (full Mohammadi panel)**:
  - Base: `benchmarks/v_compression_base_2026-05-20_verdict.md`
  - V2:   `benchmarks/v_compression_v2_2026-05-20_verdict.md`
- **Cross-codec log**: `benchmarks/v_compression_v2_2026-05-20_cross_codec.log`.

## Reproducibility

```sh
# 1. Build CLIs
cargo build --release --bin predict_features_with_bake --bin bake_verdict \
  -p zensim-validate

# 2. Fit spline + write new bake
python3 scripts/v_next/calibrate_balanced_v9_spline.py \
  --bake zensim/weights/v_compression_persample_2026-05-18.bin \
  --out  zensim/weights/v_compression_v2_2026-05-20.bin \
  --predict-bin ./target/release/predict_features_with_bake \
  --n-features 300 \
  --spline-csv benchmarks/v_compression_v2_2026-05-20_spline.csv

# 3. Verdict on both bakes
./target/release/bake_verdict \
  --bake zensim/weights/v_compression_persample_2026-05-18.bin \
  --output benchmarks/v_compression_base_2026-05-20_verdict.md
./target/release/bake_verdict \
  --bake zensim/weights/v_compression_v2_2026-05-20.bin \
  --output benchmarks/v_compression_v2_2026-05-20_verdict.md

# 4. Smoke tests
cargo test --release -p zensim --test compression_v2_profile
```
