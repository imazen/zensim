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

### 2. DIAL panel — `qsweep_eval` on the densified multi-codec grid
Monotonicity rate + tied rate + per-q dial span across codec configurations —
the codec-target axis (G1 dynamic range, G3 strict monotonicity, G4 cross-codec
reach). This is what `bake_verdict` does NOT capture. **Gates:** G3 strict
monotonicity ≥ 93%, tied ≤ 5%; G1 dial span p5 ≤ 25 / p95 ≥ 85.

The dial grid is **densified where dial precision matters most**:
- **q0** (dial floor)
- **step-1 across q90→q100** for q-parameterized codecs (near-lossless — where
  dials saturate; coarse grids hide tied dead-zones here)
- **JND zone densified** (q70→q90 step 2 — the visually-lossless band)
- **JXL swept in butteraugli distance** (its native near-lossless axis),
  relabeled to a monotone q-equivalent

Built across 4 codec families (JPEG/WebP/JXL/AVIF) at 372 features.

## How to run (one command)

```bash
scripts/eval_panel.sh <bake.bin> [label] [post_mode]
# post_mode: clamp (default) | mapped (distance bakes) | raw
```

It runs both panels, downloading the dial grid from R2 on demand, and prints
the headline rank-SROCC + dial-monotonicity/tied. Outputs land in
`$OUT_DIR/{rank_panel,dial_panel}.md` (default `/tmp/eval_panel_<label>/`).

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
