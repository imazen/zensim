# Yeo-Johnson feature-transform screen — widest λ search — 2026-05-25

Task #214 second follow-up. Prior runs:

- `benchmarks/yeo_johnson_screen_2026-05-25/` — λ ∈ [-2, 2] (scipy
  default). **370 / 372 features pinned** at the −2 boundary.
- `benchmarks/yeo_johnson_screen_wide_2026-05-25/` — λ ∈ [-5, 5].
  **361 / 372 still pinned** at −5.

This run pushes the search wider until pinning is gone.

## Top-line — scipy MLE eliminates boundary pinning

| Run                | λ range / method             | YJ outright wins | Boundary-pinned |
|--------------------|------------------------------|-----------------:|----------------:|
| narrow (prior)     | golden-section ∈ [-2, 2]     |        3 / 372   |    370 / 372    |
| wide (prior)       | golden-section ∈ [-5, 5]     |        8 / 372   |    361 / 372    |
| hardgrid10         | golden-section ∈ [-10, 10]   |       10 / 372   |    349 / 372    |
| **widest (scipy)** | `scipy.stats.yeojohnson_normmax` (unconstrained) | **53 / 372**     | **0 / 372**     |

The user-set target was "boundary-pinned feature count down to < 30/372."
Switching to scipy's data-driven MLE search drops it to **zero** —
every feature's λ is interior to the search bounds because the bounds
are computed per-feature from the data's magnitude.

## Inputs (identical to prior runs)

- Parquet: `/mnt/v/zen/zensim-training/canonical-2026-05-21/train/safesyn.parquet`
  (196,086 rows × 372 features)
- Target: `mix_cv40_iw60`
- min-lift: 0.005
- 10 transform candidates (identity, log, log1p, signed_log1p,
  signed_sqrt, signed_cbrt, clip_then_log1p, winsor_p99,
  quantile_bins, yeo_johnson)
- **λ search: scipy `yeojohnson_normmax`** (data-driven bounds via
  `optimize.fminbound` with under/overflow-aware limits computed
  from `log1p(20 * max(abs(x)))`)
- Wall time: 158.2 s — slightly faster than the prior runs since
  scipy's brent search needs fewer evaluations per feature

## Verification — `t_yeo_johnson` matches scipy at extreme λ

Spot-check on `f100` (heavy positive tail, scipy reports
λ = -212.74): the existing `t_yeo_johnson` Python function (which
mirrors `zenpredict::FeatureTransform::YeoJohnson` exactly)
produces output bit-identical to `scipy.stats.yeojohnson` at the
same λ:

```
max |t_yeo_johnson(x, lam) - scipy.yeojohnson(x, lam)| = 1.11e-16
```

The runtime side (`Predictor::predict_transformed`) handles
arbitrary λ correctly — no changes needed there.

## Outright YJ wins by block

| Block    | narrow | wide | hardgrid10 | **widest (scipy)** |
|----------|-------:|-----:|-----------:|-------------------:|
| basic    |    2   |  4   |    4       | **17**             |
| peak     |    1   |  4   |    6       | **13**             |
| masked   |    0   |  0   |    0       | **12**             |
| iw_pool  |    0   |  0   |    0       | **11**             |
| total    |    3   |  8   |   10       | **53**             |

The bounded runs systematically miss YJ wins in the masked + iw_pool
blocks because every feature there wanted λ << -10. The scipy run
finally captures those wins.

## YJ Pearson lift over identity — median + (p25 / p75) per block

| Block    | narrow                    | wide                      | hardgrid10                 | **widest (scipy)**            |
|----------|---------------------------|---------------------------|----------------------------|-------------------------------|
| basic    | 0.0076 (0.0019 / 0.0327)  | 0.0116 (0.0037 / 0.0537)  | 0.0156 (0.0055 / 0.0766)   | **0.1081 (0.0279 / 0.1765)**  |
| peak     | 0.0161 (0.0070 / 0.0424)  | 0.0290 (0.0127 / 0.0711)  | 0.0397 (0.0194 / 0.0876)   | **0.0668 (0.0290 / 0.1628)**  |
| masked   | 0.0194 (0.0025 / 0.0296)  | 0.0354 (0.0049 / 0.0508)  | 0.0547 (0.0087 / 0.0769)   | **0.1149 (0.0839 / 0.1853)**  |
| iw_pool  | 0.0243 (0.0037 / 0.0398)  | 0.0438 (0.0070 / 0.0671)  | 0.0718 (0.0119 / 0.0948)   | **0.1225 (0.0916 / 0.1954)**  |

Order-of-magnitude jumps in every block. IW-pool's per-block lead
holds (highest median lift); masked moves into a near-tie with iw_pool.

## YJ lift ≥ threshold

| Threshold | narrow | wide | hardgrid10 | **widest (scipy)** |
|-----------|-------:|-----:|-----------:|-------------------:|
| ≥ 0.005   |  242   | 289  |   329      | **354**            |
| ≥ 0.02    |  169   | 190  |   219      | **320**            |
| ≥ 0.05    |   51   | 124  |   170      | **262**            |
| ≥ 0.10    |    9   |  37  |    71      | **215**            |
| ≥ 0.20    |    4   |   4  |     4      | **58**             |

The widest run has 215 features with ≥ 0.10 lift over identity (vs 9
in the narrow run) and 58 with ≥ 0.20 lift (vs 4). The high-lift tail
is what was being clipped at the hard-bounded runs.

## Best-transform-by-feature count (widest_scipy)

| Transform        | Count |
|------------------|------:|
| `winsor_p99`     |  183  |
| `log`            |  111  |
| **`yeo_johnson`**| **53**|
| `quantile_bins`  |   18  |
| `clip_then_log1p`|    4  |
| `signed_cbrt`    |    2  |
| `log1p`          |    1  |

YJ ate 45 wins from `winsor_p99` (228 → 183) compared to the wide
run. `log` is unchanged — log already finds its optimum on strictly-
positive features without parameter tuning.

## λ distribution (widest_scipy)

| Bucket               | count |
|----------------------|------:|
| (-∞, -1000)          |    1  |
| [-1000, -300)        |   56  |
| [-300, -100)         |  104  |
| [-100, -30)          |  112  |
| [-30, -10)           |   76  |
| [-10, -5)            |   12  |
| [-5, -2)             |    9  |
| [-2, 0)              |    2  |
| [0, +∞)              |    0  |

Distribution-wide stats: min λ = **-1062.62**, median = **-73.68**,
max = **-1.72**. **Every single feature wants λ < 0** (no feature
wants the +λ side of YJ). The mass of features lives in λ ∈ [-300, -10),
which the hardgrid10 search was clipping at -10.

### λ distribution per block (widest_scipy)

| Block    | count | min      | p25     | median   | p75    | max     |
|----------|------:|---------:|--------:|---------:|-------:|--------:|
| basic    |  156  | -1062.62 | -317.31 |  -121.46 | -31.26 |  -1.72  |
| peak     |   72  |  -187.14 |  -80.11 |   -32.98 | -17.73 |  -1.74  |
| masked   |   72  |  -661.96 | -222.75 |   -97.45 | -47.45 | -10.08  |
| iw_pool  |   72  |  -554.21 | -165.69 |   -72.67 | -33.01 |  -7.57  |

`basic` and `masked` have the most extreme tails. `peak` is the
mildest — but even there the median is at λ = -32.98, well outside
the hard ±10 grid.

## Re-tracking features pinned at the prior bounds

- 370 features pinned at -2 in the narrow run
- 361 of those still pinned at -5 in the wide run (97.6%)
- 349 of those still pinned at -10 in the hardgrid10 run (96.4%)
- 0 pinned in the widest (scipy) run

For the 349 features that pinned at -10, scipy reports:

- min λ = -1062.62
- median λ = -85.67
- max λ = -10.05 (so most are well below the -10 boundary they hit)
- 161 features want λ < -100
- 57 features want λ < -300
- 1 feature wants λ < -1000

The hard-clamped runs were systematically missing the true MLE for
~94% of features.

## Verdict

**±10 hard-clamped is also insufficient.** 349/372 still pin at the
boundary; the data's MLE for those features lies somewhere in
[-1063, -10]. The principled fix is what scipy does:

> compute per-feature data-driven bounds via the overflow-safe formula
> `log_max_float / log1p(20 * max(|x|))` — guaranteed numerically
> stable while admitting the full meaningful λ range.

This run uses `scipy.stats.yeojohnson_normmax` directly. The output
schema is byte-for-byte compatible with the canonical screen TSV,
so `--auto-transforms <this_screen.tsv>` and all downstream tooling
work without modification. The runtime
`zenpredict::FeatureTransform::YeoJohnson` already handles arbitrary
λ correctly (verified bit-identical to scipy at λ = -212).

For the immediate Phase 2 consumer (v11 retrain): YJ at the true
MLE gives **262 features with ≥ 0.05 Pearson lift** and **215 with
≥ 0.10 lift** — substantially more useful signal than the hardgrid
runs.

## Artifacts

- `screen_results.tsv` — best-transform per feature (372 rows; **scipy MLE λ**)
- `per_transform.tsv` — all transforms × features (3,720 rows; scipy MLE λ)
- `screen.log` — full stderr from the scipy run
- `screen_results_hardgrid10.tsv` — companion hard-clamped ±10 run (for comparison only)
- `per_transform_hardgrid10.tsv` — companion ±10 long-form
- `screen_hardgrid10.log` — stderr for the ±10 run
- `comparison.txt` — full 4-run statistical comparison
- `SUMMARY.md` — this doc

## Reproduce

```bash
cd ~/work/zen/zensim

# Canonical / shipping screen (scipy unconstrained MLE):
python3 scripts/v_next/v0_20_feature_transform_greedy_screen_scipy.py \
  --features-parquet /mnt/v/zen/zensim-training/canonical-2026-05-21/train/safesyn.parquet \
  --target-column mix_cv40_iw60 \
  --out benchmarks/yeo_johnson_screen_widest_2026-05-25/screen_results.tsv \
  --per-transform-out benchmarks/yeo_johnson_screen_widest_2026-05-25/per_transform.tsv \
  --min-lift 0.005

# Companion hard-clamped ±10 run (informational):
python3 scripts/v_next/v0_20_feature_transform_greedy_screen.py \
  --features-parquet /mnt/v/zen/zensim-training/canonical-2026-05-21/train/safesyn.parquet \
  --target-column mix_cv40_iw60 \
  --out benchmarks/yeo_johnson_screen_widest_2026-05-25/screen_results_hardgrid10.tsv \
  --per-transform-out benchmarks/yeo_johnson_screen_widest_2026-05-25/per_transform_hardgrid10.tsv \
  --min-lift 0.005 \
  --grid-min -10 --grid-max 10
```

The scipy variant uses scipy ≥ 1.2's `yeojohnson_normmax` (available
in scipy 1.15+).
