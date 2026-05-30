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
Per codec curve, every adjacent-quality step lands in ONE of **five
mutually-exclusive buckets** (they sum to 1) — so a real ranking error is
never conflated with sub-JND noise, dense-grid oversampling, or a *codec*
quality ceiling. The codec-target axis (G1 dynamic range, G3 monotonicity,
G4 cross-codec reach):
- **forward** — Δ > 0.5 score-pt: a clear, user-visible quality increase.
- **forward sub-resolution** — 1e-9 < \|Δ\| ≤ 0.5 pt (distinct features): the
  dial moved but by less than half a score-point. **EXPECTED on the densified
  near-lossless grid** (adjacent configs are sub-JND apart) — NOT gated.
- **inversions** — Δ < −0.5 pt: the dial ran *backwards* by a user-visible
  amount. A real ranking bug (targeting "score 70" lands on the wrong
  config). **The gated metric; monotonicity = 1 − inversions.**
- **codec-saturated** — adjacent features near-identical (L-inf < 1e-5): the
  *codec* emitted the same image at two different q (zenjpeg/webp quality
  ceiling — fractional q above ~q98 produces byte-identical encodes). The bake
  MUST score identical inputs identically — this is the codec's limit, NOT a
  bake defect, so it is **NOT gated against the bake**. Reported on its own line.
- **flat / clamp** — features DIFFER but \|Δ\| ≤ 1e-9: the bake collapsed
  distinct inputs to one score — a real metric dead-zone (what V0_5-Balanced
  suffered: 60% flat above q50). Gated separately. (v47=0.000, Cell5=0.074.)

**Why a 0.5pt material threshold:** the strict "any backwards > 1e-9" rate
runs ~5% on a clean bake, but the median backwards step is ~0.3 pt — sub-JND
noise from densely sampling near-lossless, not ranking errors. The panel
prints the strict rate + backwards-step magnitude med/p90 as a diagnostic so
the split is visible; the gate uses material (>0.5pt) inversions. v39 (the
defective prior ship) is the contrast: median backwards step 3.3 pt, 73%
material inversions — a genuine catastrophe the magnitude split exposes.

**Gates:** G3 monotonicity (1 − material inversions) ≥ 93% AND flat/clamp
≤ 5%; G1 dial span p5 ≤ 25 / p95 ≥ 85. The panel also prints each codec's
min..max representable param and score@worst→@best so cross-codec reach (G4)
is visible.

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
