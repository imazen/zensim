# Eval panel requirement — MANDATORY for every ship-grade bake (2026-05-29)

Every ship/no-ship, A/B, or "is this bake better" decision MUST run the **full
two-panel eval**, not just the rank panel. SROCC-on-static-corpora alone hides
dial defects that break the codec-target use case (see the V0_5 wave: high rank
panel, collapsed dial). Both panels run against **stored feature sets** — no
re-encoding — so any model rescores against the identical features.

## The two panels

### 1. RANK panel — `bake_verdict`
Full Mohammadi 2025 statistics (SROCC + PLCC + KROCC + OR + PWRC + Z-RMSE) per
corpus on the 6 canonical val parquets (CID22, KADID, TID, KonJND, AIC-3, AIC-4),
aggregate + 10-band. **Caveat baked into interpretation:** KADID and TID are
100% train==val pair-overlap — a bake trained on them shows memorization there,
not held-out skill. The genuinely held-out corpora are **CID22 + AIC-3 + AIC-4**
(+ KonJND, semi-held-out). Rank a bake on those; treat KADID/TID as integrity
guards.

### 2. DIAL panel — densified multi-codec grid (native in `bake_verdict`)
Three distinct rates per codec curve, plus per-q dial span across codec
configurations — the codec-target axis (G1 dynamic range, G3 strict
monotonicity, G4 cross-codec reach). This is what the rank panel does NOT
capture. The three rates are reported **separately** because they are different
failures:
- **forward** — strict-increase rate (score goes up as quality goes up).
- **inversions** — adjacent steps where the score runs *backwards*
  (s₁ < s₀−1e-9). A real dial bug: targeting "score 70" lands on the wrong
  config. Reported as its own rate; **monotonicity = 1 − inversions**.
- **ties** — adjacent steps within ±1e-9 (flat dead-zones). A coarse
  precision/saturation problem, not a backwards dial — kept separate from
  inversions.

**Gates:** G3 monotonicity (1−inversions) ≥ 93%, tied ≤ 5%; G1 dial span
p5 ≤ 25 / p95 ≥ 85. The panel also prints each codec's min..max representable
param and score@worst→@best so cross-codec reach (G4) is visible.

The dial grid is **densified where dial precision matters most**:
- **q0** (dial floor)
- **step-1 across q90→q100** for q-parameterized codecs (near-lossless — where
  dials saturate; coarse grids hide tied dead-zones here)
- **JND zone densified** (q70→q90 step 2 — the visually-lossless band)
- **JXL swept in butteraugli distance** (its native axis), at a variable-density
  ladder finest near lossless: **0→0.3 step 0.025, 0.3→1 step 0.05, 1→3 step 0.2,
  mid 3.5..10, low-q tail 13→25 step 2** (49 distinct distances). Relabeled to a
  monotone q-equivalent **q = 100 − 4·distance** (unrounded), so d=0→100,
  d=0.025→99.9, d=25→0 — the dial axis sorts by quality and the full
  representable distance range maps onto [0,100].

Built across 4 codec families (JPEG/WebP/JXL/AVIF) at 372 features.

## How to run — `bake_verdict` does BOTH panels natively

The DIAL panel is **built into `bake_verdict`** (Rust): every run that computes
SROCCs of a bake also emits the dial panel — there is no separate step or
wrapper to remember.

```bash
bake_verdict --bake <bake.bin> [--output panel.md] [--dial-grid <parquet>]
```

The output `.md` carries the RANK panel (per-corpus Mohammadi), the
CODEC_TARGET_GOALS scorecard, AND a `## DIAL panel` section (strict
monotonicity, tied rate, per-q dial range, per-codec breakdown, G1/G3 gates).

The dial grid defaults to the canonical path
(`/mnt/v/output/zensim/eval_panels_2026-05-29/dial_grid_372col_2026-05-29.parquet`);
override with `--dial-grid <path>` or the `ZENSIM_DIAL_GRID` env var. If the
grid file is absent, `bake_verdict` emits a **loud SKIPPED note** (it is
pure-Rust and does not do network) — fetch the stored grid once:

```bash
aws s3 cp s3://zentrain/eval-grids/dial_grid_372col_2026-05-29.parquet \
  /mnt/v/output/zensim/eval_panels_2026-05-29/ \
  --endpoint-url https://338ad3b06716695d6e2c81c864e387d8.r2.cloudflarestorage.com
```

The standalone `qsweep_eval` binary still exists for multi-bake side-by-side
dial comparisons, but the mandatory single-bake panel is `bake_verdict` alone.

## Stored feature sets (R2 — download on demand)

Bucket `zentrain`, prefix `eval-grids/`, endpoint
`https://338ad3b06716695d6e2c81c864e387d8.r2.cloudflarestorage.com`:

| grid | rows | what | sha256 (prefix) |
|---|--:|---|---|
| `dial_grid_372col_2026-05-29.parquet` | 3,230 | densified multi-codec q-sweep, 372 feat, `(image_id, codec, q, f0..f371)` | `0fe14e82` |
| `corruption_grid_372col_2026-05-28.parquet` | 2,016 | codec-corpus#7 structural-corruption, 372 feat, `(entry, f0..f371)` — regression-gate rescoring | `cff99045` |

Local mirror: `/mnt/v/output/zensim/eval_panels_2026-05-29/`. The wrapper fetches
from R2 if the local copy is absent. To rescore a NEW bake against the stored
features: just run `scripts/eval_panel.sh <new_bake.bin>` — it pulls the grid and
forwards the bake over the stored 372-feature vectors (no encode, no extract).

These feature vectors are GPU-extracted (`zen-metrics sweep --metric zensim-gpu
--zensim-features-regime with-iw`); bit-equivalent to the CPU 372-feature path
within metric tolerance, which is irrelevant to rank/monotonicity.

## Adding to / refreshing a grid

Dial grid:
```bash
python3 scripts/v_next/build_qsweep_expanded.py     # rebuilds from on-disk codec variants
python3 - <<'PY'  # consolidate per-codec sweep parquets → one (image_id,codec,q,f0..f371)
# (see scripts/v_next/build_qsweep_expanded.py merge step)
PY
aws s3 cp dial_grid_372col_<date>.parquet s3://zentrain/eval-grids/ --endpoint-url <ep>
```
Bump the date suffix and update this doc's sha256 table when the grid changes.

## Why this is mandatory

A bake can win the rank panel and be a broken dial (V0_5 Balanced: panel-best by
meanG3, but 60% tied / collapses to 0 above q50 on the dial). A bake can have a
fine coarse dial and fail at near-lossless step-1 granularity (Cell5: 0.8% tied
on the coarse 16-q grid, 13.1% on the densified grid). Only the two-panel run
catches both. Single-panel verdicts are a regression — do not accept them.
