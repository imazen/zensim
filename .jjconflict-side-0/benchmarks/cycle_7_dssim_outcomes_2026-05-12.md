# Cycle-7 dssim co-training experiments — outcomes (2026-05-12)

## Summary

Three V_X candidates trained today on 2026-05-12 evening:

| Variant | Recipe | CID22 SROCC (n=4292) | Δ vs V0_16 | Δ vs V0_25 control |
|---|---|---:|---:|---:|
| **V0_16** (ship) | unknown exact recipe (per CLAUDE.md: TV=20 seed=1 h128 KonJND-aligned) | **0.8919** | — | +0.041 |
| V0_24 v1 | --human-csv only, dssim_w=0.3, **TV=0 (bug)** | 0.8315 | -0.060 | -0.019 |
| V0_24 v2 | --human-csv real-codec+quality, **TV=104k pairs**, dssim_w=0.3 | 0.8254 | -0.067 | -0.025 |
| **V0_25 control** | V0_24 v2 recipe but **dssim_w=0.0** | **0.8505** | -0.041 | 0 (reference) |

## Decomposed regression

V0_25 vs V0_16 gap = 0.041 SROCC. This is the recipe-drift residual:
neither dssim weight nor TV-pair count is at fault. V0_25 had TV
active with 104,319 pairs at weight=20 (same as V0_16 spec).

V0_24 v2 vs V0_25 gap = 0.025 SROCC. This is the dssim cost at weight=0.3.

Total V0_24 v2 vs V0_16 gap = 0.067 SROCC:
- 0.041 from missing supervision (dominant, 61%)
- 0.025 from dssim_weight=0.3 too aggressive (39%)

## What V0_16 had that V0_25 doesn't

Per zensim CLAUDE.md ship-line for V0_16:
> "h=128, TV=20, seed=1, KonJND-aligned. Trained on safe-synthetic CSV"

V0_25 matched all of those EXCEPT KonJND alignment. Our trainer has
`--konjnd-anchor-csv` flag (line 727-741 of train_v_next_mlp.py); V0_16
must have used it. We don't have the canonical KonJND-anchor CSV path
on hand for this session — would need to find it in `/mnt/v/dataset/konjnd-1k/`
or elsewhere.

Also possibly missing:
- KADID/TID human-MOS mixed supervision (V0_4 recipe; V0_16 inherits arch)
- Concordance filter (`--concordance-filter ssim2_butter`)
- Specific batch/lr schedule

## Cycle-7 dssim verdict

**dssim co-training at weight=0.3 hurts CID22 by 0.025 SROCC** in
the V0_25-recipe context. To gain JPEG-AI coverage (the original
hypothesis), we'd need to:

1. Lower dssim weight (try 0.05 or 0.1)
2. AND first reproduce V0_16's exact recipe (KonJND-anchor missing)

For shipping: V0_16 stays. The dssim experiment was a useful
calibration but doesn't beat V0_16.

## Next sessions

Per the goal #1 "match-or-exceed fast-ssim2", V0_16 already meets
this on all 3 held-out corpora. No urgency to flip ship.

When user returns from vacation, decide:
1. **Drop cycle-7** — V0_16 is good enough; JPEG-AI is a small
   subset of any real-world deployment.
2. **Continue cycle-7 with KonJND + lower dssim weight** —
   reconstruct V0_16 recipe first, then add dssim=0.05.
3. **Direct JPEG-AI training-data acquisition** — bypass dssim
   co-training entirely; add JPEG-AI synth pairs.

V0_24 v1, v2, V0_25 bakes archived at `/tmp/zensim_loop/bakes/`.
