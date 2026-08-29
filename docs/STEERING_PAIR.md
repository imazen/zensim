# The steering pair — "anchor-lantern" (A scores, e060 steers)

Status 2026-08-28: **proposed configuration, adoption user-gated.** The
frozen SDR scorer stays `north-anchor` (`W10L9PH_s4004_packed`, sha
61ebc456…); this doc records the measured case for mounting
`river-lantern` (`PH_s4004_e060`, the stamped epoch-60 checkpoint) as its
STEERING-MAP COMPANION, the corruption-head pattern applied to the
diffmap. Codenames: `benchmarks/candidate_names.json` (append-only).
**Priority ruling (user, 2026-08-28): a UNIFIED ship candidate outranks
the pair as the target** — the lodestar track (W12-U, registered in
`benchmarks/balance_campaign_2026-08-28.md`) trains for one bake that
matches the pair; the pair is the fallback configuration and the bar the
unified candidate must meet.

## Why a pair at all

Scoring quality and map quality peak at different checkpoints of the same
training run. The final (A) wins rank/dial/identity; the epoch-60
checkpoint (e060) has the coherent attribution map (M3a 0.833 vs A's
0.763). With H3 magnitude steering ON, the map's allocation value is real
— and A's own map makes A WORSE:

| arm (jxl 27-cell, h3-mag, k3 emit-best) | med \|err\| | ±2 | dBytes vs A-scalar |
|---|---|---|---|
| A scalar (no map) | 0.343 | 25/27 | — |
| A + own map | 0.404 | 23/27 | −0.83% |
| **A + e060 map = anchor-lantern** | **0.300** | **25/27** | +0.32% |
| e060 + own map | 0.205 | 23/27 | (weaker scorer; fewer ±2) |

k2 cross 0.652 also beats A-scalar k2 0.971. Overhead measured: the map
roughly DOUBLES loop time (median loop_ms 289 vs 142 per cell, 576²-class
crops); byte-neutral. Footprint: 149 KB + 169 KB packed (both 667→128 f16
pruned 944-class).

## Mechanism

jxl-encoder `JXL_ZENSIM_MAP_BAKE=<path>` (commit fd2f4351): the second
bake's FD gradient drives the model map / H3 magnitude steering while
`JXL_ZENSIM_RD_PROFILE` keeps scoring. Unset = structurally identical
code path; loud width + probe asserts. Gain: `ZENSIM_H3_GAIN` default 10
— SWEPT 2026-08-28 (pair, k3-best): g5 0.281 (24/27), g10 0.300 (25/27),
g20 0.281 (23/27), g40 0.350 (25/27). Flat optimum 5-20; default
retained; the "unswept gain" caveat is CLOSED.

## Challenges run (nothing displaced e060)

- `deep-loom` (w11_s4014_final, M3a 0.872 — highest ever) as A's map:
  0.326 (24/27). Close, not better. Own-map: 0.477, +16.8% bytes (its
  top-zone under-reporting makes the loop over-encode).
- `clear-ember` (w11_s4014_e050, best dial calibration + cid22 champ)
  own-map: 0.406 (24/27), +4.1% bytes — worse than A scalar. **No
  existing single bake is near-unified.**
- Finding: M3a predicts map-on loop value COARSELY (map-strong ≫
  map-weak); above M3a ≈ 0.83 it DECOUPLES (0.872 steers no better than
  0.833). Selection inside the map-strong class needs the loop itself.

## Scope + registered follow-ups

Evidence scope: ONE codec (jxl), one instrument (27 cells = 9 nonphoto
crops × t∈{70,80,88}), h3-mag arm, substrate = the Aug-26 loop binary.
Follow-ups (registered, not started): avif/other codec loops have no map
steering wired (per-codec loop ownership — each codec owns that work);
the jxl loop's own-score path still pins deprecated ZensimProfile::A
(re-seed ZENSIM_DISTANCE_TARGETS); map-aware training loss is the next
lodestar lever if W12-U's data lever fails.
