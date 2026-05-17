# V_20 three-directions experiment summary (2026-05-15)

After the V_20 input-shaping single-MLP seed=1 result showed
**+0.129 CID22 B3 [30,40)** lift but **−0.014 CID22 aggregate**
regression (commit `994da4aa`), three follow-up directions to convert
the B3-specialist mechanism into a shipping form:

1. **D1**: full V_18-recipe 3-way concat with V_20 transforms
2. **D2**: V_18 ship + V_20 IS ensemble mix (already shipped)
3. **D3**: tighter transform set (lift ≥ 0.10 → 60 features)

## D2 result (live as of commit `aea40049`)

Per-pair output ensemble of V_18 ship + V_20 IS seed=1 via the
`ensemble_mix` tool. **No new training**; uses existing committed
bakes.

The mix on CID22 (4292 pairs):

| α | agg | B3 [30,40) priority | Δ B3 vs V_18 | Δ agg vs V_18 |
|---:|---:|---:|---:|---:|
| 1.0 (V_18 alone) | 0.8934 | 0.0246 | — | — |
| 0.9 | 0.8937 | 0.0507 | +0.026 | +0.0003 |
| **0.8** | **0.8935** | **0.0765** | **+0.052** | **+0.0001** |
| **0.7** | **0.8890** | **0.1047** | **+0.080** | **−0.0044** |
| 0.0 (V_20 IS alone) | 0.8794 | 0.1554 | +0.131 | −0.014 |

**Best ship candidates** (cleaner than V_20a multi-output's −0.14 trade):

- **α = 0.8**: B3 +0.052 at aggregate match — minor B6/B7 wobble within noise
- **α = 0.7**: B3 +0.080 at aggregate −0.004 — favors priority bands

α = 0.8 looks like the obvious ship — it MATCHES V_18 aggregate
while lifting the priority B3 band by 0.052 SROCC. To productize
requires multi-bake runtime support in `Zensim::compute` (∼3 hr).

## D1 result — pending (training in flight)

Cycle-14 seed=1 and seed=42 with TV + V_20 transforms (using same
98 flags as V_20 IS at lift ≥ 0.05), then 3-way concat at
0.65/0.30/0.05 with the V_20 IS seed=1 base. Same recipe as V_18
ship, but with input-shaping applied.

Hypothesis: TV regularization on the cycle-14 components combined
with 3-way averaging may preserve the B3 lift V_20 IS provides while
stabilizing the B4-B8 mid-band regression (which is the V_20 IS
single-MLP downside).

Bakes will be at:
- `benchmarks/v0_20_d1_cycle14_s1_2026-05-15.bin`
- `benchmarks/v0_20_d1_cycle14_s42_2026-05-15.bin`
- `benchmarks/v0_20_d1_concat_3way_2026-05-15.bin`

ETA: ~50 min (2 × 17 min training + ~5 min concat + eval).

## D3 result — pending (training in flight)

Single MLP seed=1 with TIGHTER transform set: only features with
Pearson lift ≥ 0.10 from the greedy screen (60 flags vs V_20 IS's
98 at lift ≥ 0.05).

Hypothesis: the B4-B8 regression in V_20 IS may come from
"borderline" transforms (lift 0.05-0.10) introducing noise. The
tighter 60-flag subset may give cleaner mid-band behavior while
keeping most of the B3 lift.

Bake: `benchmarks/v0_20_input_shaping_lift10_seed1_2026-05-15.bin`.

ETA: ~17 min.

## Results table (will be filled in as D1 and D3 complete)

| Variant | KADID | TID | CID22 (agg) | CID22 B3 [30,40) | Notes |
|---|---:|---:|---:|---:|---|
| V_18 ship (3-way concat) | 0.9427 | 0.9526 | **0.8933** | 0.0246 | ship baseline |
| V_18 base seed=1 (single MLP) | 0.9464 | 0.9568 | 0.8880 | 0.0471 | apples-to-apples |
| V_20 IS seed=1 (98 transforms) | 0.9497 | 0.9616 | 0.8794 | **0.1534** | original single-MLP |
| V_20b distortion manifold | 0.9656 | 0.9793 | 0.8660 | 0.0270 | falsified for B3 |
| **D2: V_18 + V_20 IS α=0.8** | — | — | **0.8935** | **0.0765** | matches V_18 + B3 lift |
| **D2: V_18 + V_20 IS α=0.7** | — | — | 0.8890 | **0.1047** | priority-favoring |
| **D3: lift ≥ 0.10 (60 transforms)** | TBD | TBD | TBD | TBD | training in flight |
| **D1: 3-way concat with transforms** | TBD | TBD | TBD | TBD | training in flight |

## Verdict criteria

A V_20 ship CANDIDATE passes iff:
- CID22 aggregate SROCC ≥ 0.8933 (V_18 ship floor)
- CID22 B3 [30, 40) SROCC ≥ 0.13 (fast-ssim2 floor — the gap V_20 targets)
- KADID and TID within −0.005 of V_18 ship
- No CID22 band regression >0.02 vs V_18 ship

D2 at α=0.8 doesn't quite clear the B3 ≥ 0.13 bar (lands at 0.0765),
but it's a clean ship at zero aggregate cost. D2 at α=0.7 has B3 =
0.1047 (just under the 0.13 bar) at −0.004 aggregate cost.

D1 and D3 results will clarify whether the in-flight architectures
can clear both bars simultaneously.

## Next steps after D1 + D3 land

1. Compare D1, D2, D3 per-band against this table
2. Pick the best CID22-B3-lifting candidate that doesn't break the
   aggregate or other bands
3. If the best is D2 (ensemble), implement multi-bake runtime
   support in `Zensim::compute` (∼3 hr engineering)
4. If best is D1 or D3 (single bake), swap into the PreviewV0_3 slot
   per the standard ship procedure
