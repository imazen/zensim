# A cheaper complete base model advances to peak-attribution work

The fixed three-seed `F_nonneg32` family plus D passes the canonical training
preference screen and the registered complete-model cost screen. It remains
unqualified: all 252 nonidentity maps omit active peak terms, corruption
protection is unattached, and human/HDR/floor/targeting/RD qualification remains.

## Choice, fit and evidence

Read the September 4 distillation wave, September 5 fast-class campaign and
September 6/7 corrections before choosing this family. Earlier cheap students
had important identity/floor failures; several distillation recipes also had
teacher and loss-owner defects. `F_nonneg32` was chosen before new scores for
its historical clean identity contract and smaller basic+peaks network. All
three existing seeds 4004/4005/4006 are retained equally; no best seed is chosen.
Their old mixed-era training and overlapping historical evaluation panels are
explicit limitations. No obsolete training recipe was rerun, and no historical
panel became fresh qualification evidence.

The existing preference analyzer now supports an explicitly registered
three-seed family against D. Its default nine-model mode and strict rule are
unchanged. The new screen explicitly registers noninferiority because strict
improvement cannot beat D's zero errors on this panel. It still requires all
three seeds, source/model/pixel identity checks, twelve exact identities, no
nonidentity above 100, and no content-class regression. All three F seeds fail:
63/68/78 unresolved independent-judge pairs, versus zero for D, out of 2471.

The subsequent registered fit uses equal F weights plus D. Search D weights
0.00 through 0.99 in hundredths and freeze the first resolving all 1,647 FIT
consensus pairs. The chosen ordered weights are:

```text
F_nonneg32_4004  0.11333333333333333
F_nonneg32_4005  0.11333333333333333
F_nonneg32_4006  0.11333333333333333
D               0.66
```

All 824 CAL pairs also resolve. FIT uses 2010/1054/6068/6610/7066/9380/8206/8384;
CAL uses 1214/6064/9066/8462. These are canonical training origins. CAL has been
seen in earlier engineering screens; it is not an untouched validation set.
No weight is selected on its composed CAL result. No encodes, independent
judge runs, neural-network fits or validation/terminal evaluations occurred.

`BakeScorer::ensemble` serves the complete calibrated composition. Three
actual image/cache/spatial audits (blend, F-only mean, D-only endpoint) cover
792 pairs/maps. Scores match independent weighted arithmetic to 1.43e-14;
pixel/cache/stored-f32/spatial scores agree exactly. Twelve identities remain
100 and nonidentity scores range from −28.5800 to 98.0405. The F-only mean has
49 FIT and 18 CAL unresolved pairs; the D endpoint reproduces its exact scores.

The packed F members each declare 228 inputs, with all 156 basic and 72 peak
inputs active; no masked/IW slots are consumed. Actual combined extraction
uses that cheaper regime. All 72 peak IDs f156..227 remain explicitly
unsupported in the blend/F maps; D-only coverage is complete. Scalar/map
score equality does not establish correct spatial attribution.

## Complete model cost

Same pinned post-activity-repair binary, same eight arms and synthetic inputs,
9950X3D CPU 8 with one worker, release without target-cpu=native:

| Geometry | Blend mean | D mean | SSIM2 mean | Blend observed range | Clean paired samples |
|---|---:|---:|---:|---:|---:|
| 1024², 40 rounds | 23.77 ms | 23.78 ms | 64.21 ms | 23.08–25.93 ms | ≥31 |
| 2048², 60-round repeat | 84.38 ms | 84.29 ms | 286.60 ms | 82.54–88.45 ms | ≥55 |

The first 40-round 2048 run had passing ratios/dispersion but only 29 clean
samples in its weakest comparison, below the registered 30 minimum. It remains
rejected measurement evidence. Repeat 2048 alone with 60 rounds and unchanged
bars; do not rerun the already admitted 1024 group. Both admitted groups meet
MAD/median ≤5%, mean ≤1.25×D and ≤SSIM2, and observed maxima ≤50/200 ms.
The repeat has no gate waits; the first run has one advisory. This is a cost
screen, not full performance qualification: no raw p95, cached/map/HDR latency
or incremental per-worker RSS evidence is inferred from these means.

## Next work and reproducibility

Advance this fixed composition into the existing Rust peak-attribution owner.
Implement/check L8 terms and define the max-term treatment explicitly,
including ties and finite block removal. A smooth surrogate must not be
presented as exact max-removal attribution. Require canonical feature/score
parity, per-feature reconstruction and finite-block checks before declaring
coverage complete or enabling native steering. Then carry the complete model
through independent JXL spatial RD and the other codec loops. The corruption
head's existing honest false positives remain a separate required repair.

The owner change passes eight new refusal controls and all 18 existing refusal
controls; both old/new valid reports reproduce byte-for-byte. Python compilation
and 605-script lint pass. No Rust inference/API changed. The model fit/export
uses existing packed member bytes, and every model evaluation uses the Rust
surface. The legacy preference screen's failed advancement results remain.

The [portable research manifest](../zensim-experimental/weights/manifests/f_nonneg32_d_blend_2026-09-08.json)
and all three exact student files are tracked in `zensim-experimental`; D
reuses its existing tracked bake. Member paths resolve from the repository
root. Load the ordered files/weights through `BakeScorer::ensemble` or pass
them to the existing extractor's `--audit-ensemble` and
`--audit-ensemble-weights` options. A repository-relative nonidentity smoke
check reproduces the score and map exactly. No named production profile changes.

Artifact: `/mnt/v/output/zensim/cheap-model-screen-2026-09-08/`; verified Windows
mirror: `~/work/zensim-validation-2026-09-08/cheap-model-screen/`. The packet
contains registrations, exact model files/weights, source/decoder identities,
fit trace, input keys, original judge rows, audits, refusal controls, measured
binaries, timing/process logs and explicit incomplete/failed dispositions.
