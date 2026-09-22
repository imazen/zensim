# DVIFM constants fit — full JSON (pointer)

The machine-readable output of the 2026-09-20 constants lane is 510988 bytes, too large for git
(>30 KB rule). Summary and verdict: [`dvifm_constants_2026-09-20.md`](dvifm_constants_2026-09-20.md).

| | |
|---|---|
| Path | `/mnt/v/output/zensim/dvifm-loss-2026-09-20/report/dvifm_constants_2026-09-20.json` |
| sha256 | `167520938059b42dbf7d4f430ba9121cd2be4e34584d5140036d5d5fefa3c65d` |
| Bytes | 510988 |
| Produced by | `research/2026-09-dvifm/loss-constants/fit_loss.py` (+ `run_ladder.sh`, `collect.py`, `report.py`) |
| Companion spec | `/mnt/v/output/zensim/dvifm-loss-2026-09-20/report/constants-v1.json` (15 cells + schema) |
| Arm comparison | `/mnt/v/output/zensim/dvifm-loss-2026-09-20/report/compare.json` (11 arms) |
| Per-arm artefacts | `/mnt/v/output/zensim/dvifm-loss-2026-09-20/report/fits/artefact_*.json` |

Headline it carries: per-cell β is identifiable at 1 of 15 cells (luma level 0, β = 0.635, profile
[0.287, 1.648]); a **shared** β is identified across 7 domains with intervals intersecting at
**[0.607, 0.779]**, which covers the Legge–Foley / Watson–Solomon band. The other 13 cells are
masking-off/gate/saturated and their fitted values (to 23.0) are flat-objective artefacts that must not ship.
