# dvifmish variant screen — round 2 (registered 2026-09-23, before any round-2 result)

## Why a second round

Round 1 varied one factor at a time from the suggested baseline (Y′CbCr three-plane,
two-state gate, our pyramid, shared-β prior). Under the prior protocol a gate has no β,
so its only constant, the knee, stays at the fit rows' 10th percentile of block contrast
and only the flattest ~10% of blocks count. Measured on the committed composite (mean
SROCC over CID22-A and KonFiG-val, three seeds): every gate arm scored 0.29–0.31, while
the same prior with no masking scored 0.7296 and with the smooth curve 0.6921. The
planes and pyramid arms all sat on that gate baseline (0.2894–0.3148), so round 1 cannot
separate those factor values.

## Round 2

* **Baseline** = the round-1 arm with the best mean composite once every round-1 arm is
  scored, including the three SafeSyn-fitted visibility arms (`const-safesyn`,
  `vis-curve-fit`, `vis-off-fit`, refit after the 4948b98d fitter fix).
* **Arms** = the round-1 planes values (Y′ only, XYB three-plane, XYB Y only) and pyramid
  values (talk pyramid, `[1 3 3 1]`) with that baseline's visibility form and constants
  protocol; three seeds each, the same 4,000-row SafeSyn subsets and fitter.
* **Decision** = the committed round-1 rule (`screen_decide.py`): survivors within 2σ of
  the best mean composite, baseline always carried; teacher legs reported only.
* Every round-1 and round-2 arm becomes a named preset (refit on all three subsets).

## Amendment 1 (2026-09-23T03:52Z, commit 581195be; before any round-2 fit started at 04:37Z)

The last bullet above said each arm's preset would be refit on all three subsets. Changed:
each arm's preset is its **seed-1 fit**, the constants the selection legs actually scored. A
union refit would ship constants that no selection leg had measured. The decision rule, the
arms, the seeds and the legs are unchanged; this affects only which constants ship as presets.
