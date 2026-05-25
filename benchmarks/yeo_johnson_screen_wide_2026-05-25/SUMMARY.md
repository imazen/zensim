# Yeo-Johnson feature-transform screen — wider λ grid — 2026-05-25

Task #214 follow-up. The previous YJ screen (artifacts at
`benchmarks/yeo_johnson_screen_2026-05-25/`) hardcoded
λ ∈ [-2, 2] (scipy's documented default). 370 / 372 features pinned
at the −2 boundary, indicating under-counted YJ wins. This run
widens the search to **λ ∈ [-5, 5]** via two new flags
(`--grid-min` / `--grid-max`) on the screen script.

## Top-line

**With the wider grid YJ now wins outright on 8 / 372 features**
(vs 3 / 372 at λ ∈ [-2, 2]) — a 2.7 × increase. But the headline
masks a more important finding:

**361 / 372 features (97 %) STILL PIN at the new −5 boundary.** The
wider grid is *not sufficient*; the YJ MLE wants even more extreme
negative λ for the heavy-positive-tailed features that dominate
this corpus. The win count went up because the relaxed bound let
the search reach a higher-likelihood region for *every* feature,
not because most features found their true MLE.

## Inputs (identical to prior run)

- Parquet: `/mnt/v/zen/zensim-training/canonical-2026-05-21/train/safesyn.parquet`
  (196,086 rows × 372 features)
- Target: `mix_cv40_iw60`
- min-lift: 0.005
- 10 transform candidates (identity, log, log1p, signed_log1p,
  signed_sqrt, signed_cbrt, clip_then_log1p, winsor_p99,
  quantile_bins, yeo_johnson)
- **λ grid: [-5, 5]** (was [-2, 2])
- Wall time: 145.2 s (faster than prior's 183.7 s — golden-section
  iter count is bounded by log(range/tol), not by range)

## Best-transform-by-feature count (both runs)

| Transform        | Prior ([-2, 2]) | Wide ([-5, 5]) |
|------------------|----------------:|---------------:|
| `winsor_p99`     |             233 |        **228** |
| `log`            |             111 |            111 |
| `quantile_bins`  |              18 |             18 |
| `clip_then_log1p`|               4 |              4 |
| **`yeo_johnson`**|           **3** |          **8** |
| `signed_cbrt`    |               2 |              2 |
| `log1p`          |               1 |              1 |

YJ stole 5 wins from `winsor_p99`. `log` is unaffected — features
where `log` wins are strictly-positive heavy-tailed and YJ at any
λ struggles to beat the log-Jacobian-free closed form.

## YJ lift over identity — comparison

| Threshold | Prior (n) | Wide (n) | Δ      |
|-----------|----------:|---------:|-------:|
| ≥ 0.005   |       242 |      289 |   +47  |
| ≥ 0.020   |       169 |      190 |   +21  |
| ≥ 0.050   |        51 |      124 |  **+73** |
| ≥ 0.100   |         9 |       37 |  **+28** |
| ≥ 0.200   |         4 |        4 |    0   |

The mid-tier lift band (0.05..0.10) is where the wider grid pays
off most — 2.4 × more features now meaningful-lift, and 4 × at the
0.10 threshold. The bottom and top of the lift distribution are
relatively stable: the very-easy-win features (lift 0.005-0.02) were
already finding *some* signal at the −2 boundary, and the
extreme-tail features (lift ≥ 0.20) were already at the bound's
useful limit.

## YJ vs log (≥ 0.02 lift, or log N/A)

- Prior: 238 features
- Wide:  243 features (+5)

Only a small lift here because most "YJ beats log" features have
non-positive values where `log` is undefined — that's a
*structural* win, unaffected by λ search range.

## λ distribution

| bucket           | prior | wide  |
|------------------|------:|------:|
| `[-5, -4)`       |     0 |   364 |
| `[-4, -3)`       |     0 |     4 |
| `[-3, -2)`       |     0 |     2 |
| `[-2, -1)`       |   372 |     2 |
| `[-1, 0)` through `[+4, +5)` | 0 |   0 |

| Range              | Prior pinning | Wide pinning |
|--------------------|--------------:|-------------:|
| at lower bound     |  370 (−2.0)   | **361 (−5.0)** |
| interior           |             2 |     11        |
| at upper bound     |             0 |      0        |

**Re-tracking the 370 features pinned at −2.0 in the prior run**:

- 361 (97.6 %) re-pin at the new −5.0 boundary
- 7 land in (-5.0, -3.0]
- 2 land in (-3.0, -2.0]
- 0 escape above −2.0

The features that "needed only" λ ∈ (-3, -2] in the wider run
prove the prior bound was tight; the 97 % that re-pin at −5 prove
the new bound is *also* tight.

## Per-block summary

YJ Pearson lift over identity, median + p25 / p75 over each block's
features:

| Block    | Range        | Prior median | Prior p25 | Prior p75 | Wide median | Wide p25 | Wide p75 | Δ median |
|----------|--------------|-------------:|----------:|----------:|------------:|---------:|---------:|---------:|
| basic    | f0..f155     |       0.0076 |    0.0020 |    0.0335 |      0.0116 |   0.0038 |   0.0539 |   +0.0040 |
| peaks    | f156..f227   |       0.0161 |    0.0072 |    0.0441 |      0.0290 |   0.0130 |   0.0724 |   +0.0129 |
| masked   | f228..f299   |       0.0194 |    0.0025 |    0.0302 |      0.0354 |   0.0049 |   0.0520 |   +0.0161 |
| iw_pool  | f300..f371   |   **0.0243** |    0.0038 |    0.0402 |  **0.0438** |   0.0071 |   0.0673 |   +0.0195 |

Outright YJ wins per block:

| Block    | Prior | Wide |
|----------|------:|-----:|
| basic    |     2 |    4 |
| peaks    |     1 |    4 |
| masked   |     0 |    0 |
| iw_pool  |     0 |    0 |

**IW-pool's per-block lead held.** It still has the highest median
YJ lift (0.0438 vs prior's 0.0243 — **+0.0195 median improvement**,
the largest of any block). The relative ordering across blocks is
preserved: `iw_pool > masked > peaks > basic` for median YJ lift,
in both runs. The wider grid amplifies but does not invert the
2026-05-25 finding that IW-pool features respond best to
information-preserving non-linear monotone transforms.

Notable: outright YJ wins remain concentrated in the basic + peaks
blocks (4 + 4). The masked and iw_pool blocks still have zero
outright YJ wins because `winsor_p99` continues to dominate
boundary cases there — but the per-feature lift over identity
indicates YJ is a strong second-best across those blocks.

## Verdict on the λ grid

**±5 is NOT sufficient as a permanent ceiling.** 97 % of features
still pin at −5. Future options, in order of practical
preference:

1. **Use scipy's unconstrained `yeojohnson` MLE** — fully variance-
   stabilizing without artificial clamps. The screen would then
   report the true MLE λ even when it goes to −10 or below. The
   Rust runtime's `fit_yeo_johnson` would need a matching
   un-clamped variant (or a wider clamp like [-10, 10]).
2. **Bound at λ ∈ [-10, 10]** if a hard clamp is desirable for
   numerical stability. Re-run the screen — if pinning persists
   at −10, the underlying data needs a different transform family
   (e.g., a discriminator on the upper tail).
3. **Switch to feature-bucketed transforms** for the heaviest-
   tailed blocks. The fact that almost every feature wants the
   most-negative λ available indicates these features share a
   *qualitative* shape (extreme positive skew) that a single
   per-feature MLE-fit YJ may not be the right primitive for —
   e.g., `winsor_p99 → log → standardize` chain or `log(p99-x)`-
   style transforms.

For the immediate downstream consumer (next V_X training
experiment): YJ at λ = -5 boundary still provides 124 features
with ≥ 0.05 Pearson lift over identity — substantially more than
the 51 at the previous −2 boundary. That's enough lift to be
worth feeding into the trainer's auto-transforms TSV consumption.

## Artifacts

- `screen_results.tsv` — best-transform per feature (372 rows)
- `per_transform.tsv` — all transforms × features (3,720 rows)
- `screen.log` — full stderr (per-block summary, top 20, timings)
- `comparison.txt` — raw output of the prior↔wide diff analysis
- `SUMMARY.md` — this doc

## Reproduce

```bash
cd ~/work/zen/zensim
python3 scripts/v_next/v0_20_feature_transform_greedy_screen.py \
  --features-parquet /mnt/v/zen/zensim-training/canonical-2026-05-21/train/safesyn.parquet \
  --target-column mix_cv40_iw60 \
  --out benchmarks/yeo_johnson_screen_wide_2026-05-25/screen_results.tsv \
  --per-transform-out benchmarks/yeo_johnson_screen_wide_2026-05-25/per_transform.tsv \
  --min-lift 0.005 \
  --grid-min -5 \
  --grid-max 5
```

The new `--grid-min` / `--grid-max` flags default to [-2, 2] for
backwards compatibility with the prior 2026-05-25 screen.
