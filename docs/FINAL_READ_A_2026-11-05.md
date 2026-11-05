# Frozen A: registered final-set assessment — 2026-11-05

All eight registered populations are complete. CLIC 2022 retains all
57,300 questions over 57,243 reference crops.

**KADID TERMINAL failed both registered confirmation rules.** No weights,
calibration or acceptance limits were changed on these results.

This assesses the frozen model A on the 0.3 development line at
[`70469822`](https://github.com/imazen/zensim/commit/704698224dd75b55289d2f570bc20fed317c13b7).
The bake SHA256 is
`d5e2a56db35c32620d9f9d9c19ba3bc78bd8fea2b8cd933cde8447428d85e62a`.

## Signed results

Signed Spearman correlation with 95% whole-reference percentile intervals:
10,000 draws, seed `20261105`. Butteraugli quality is explicitly **−p3**.
Negative correlations against distortion-oriented targets retain their sign.

| Population | Pairs / references | A | SSIMULACRA2 | Butteraugli quality |
|---|---:|---|---|---|
| CID22-B (quality MCOS) | 2100 / 24 | 0.908927 [0.876867, 0.939159] | 0.913094 [0.883263, 0.940670] | 0.781058 [0.698243, 0.861064] |
| CSIQ (quality 1−DMOS) | 866 / 30 | 0.946058 [0.938142, 0.953728] | 0.904757 [0.893626, 0.915690] | 0.814450 [0.797829, 0.830706] |
| LIVE full (quality) | 779 / 29 | 0.967369 [0.961121, 0.972129] | 0.959926 [0.949925, 0.967667] | 0.916378 [0.899526, 0.931583] |
| LIVE clean10 (quality) | 267 / 10 | 0.966511 [0.960110, 0.972366] | 0.960528 [0.944990, 0.970799] | 0.910527 [0.894010, 0.924940] |
| KonJND JPEG TERMINAL (PJND threshold) | 100 / 100 | -0.368034 [-0.533383, -0.180964] | -0.400978 [-0.567517, -0.208665] | 0.052950 [-0.162601, 0.260998] |
| KADID TERMINAL (quality MOS) | 2000 / 16 | 0.892839 [0.883444, 0.901577] | 0.800193 [0.778929, 0.822535] | 0.571754 [0.536982, 0.605816] |
| AIC-4 (distortion q_jnd) | 300 / 5 | -0.921776 [-0.961733, -0.892516] | -0.912549 [-0.959556, -0.877436] | -0.896924 [-0.965041, -0.839287] |
| SDR25 / AIC-4 JPEG-AI subset | 50 / 5 | -0.855078 [-0.991147, -0.739966] | -0.841345 [-0.989892, -0.706615] | -0.841633 [-0.991706, -0.708303] |
| MCL-JCI full50 (distortion count) | 5000 / 50 | -0.896932 [-0.914725, -0.878509] | -0.884865 [-0.906228, -0.862672] | -0.850527 [-0.885807, -0.814715] |
| MCL-JCI clean40 (distortion count) | 4000 / 40 | -0.904456 [-0.921112, -0.886364] | -0.890713 [-0.912269, -0.867321] | -0.855605 [-0.892609, -0.815794] |

KonJND within-reference SROCC is undefined: 100 pairs over 100 references.
Its pooled signed result and reference interval above follow the registration.
SDR25 is a subset of AIC-4, not an independent collection.

## CLIC 2022

All 57,300 questions over 57,243 reference crops were retained. Majority
agreement is the registered primary; historical soft-vote agreement is the
approved supplement. The cached targets match on all 57,300 questions, so
both panels coincide. Tied majorities and tied predictions receive half credit.

| Metric | Majority primary, 95% CI | Soft-vote supplement, 95% CI |
|---|---|---|
| A | 0.746632 [0.743006, 0.750179] | 0.746632 [0.743006, 0.750179] |
| SSIMULACRA2 | 0.736230 [0.732504, 0.739863] | 0.736230 [0.732504, 0.739863] |
| Butteraugli quality = −p3 | 0.688342 [0.684479, 0.692035] | 0.688342 [0.684479, 0.692035] |

Intervals use 10,000 whole-reference draws, seed `20261105`, percentile/type7.
CLIC 2022 shares the protocol of the earlier CLIC 2021 confirmation read.
All 114,593 reference–candidate pairs were scored; all 229,186 feature RGB
checks matched the native decode map. All 1,303 pixel-identical pairs returned
exactly 100 through cached and native A scoring. All 64 serial/eight-thread
peer controls matched exactly. The preference file was parsed once, retaining
both targets together; statistics then used the cached targets.

## KADID confirmation

All 2,000 original TERMINAL rows were retained, including 64 pixel-identical
pairs. Cached A and native pixel scoring returned exactly 100 on all 64.

| Comparison | Signed SROCC difference | Required lower limit | Paired 95% interval | Result |
|---|---:|---:|---|---|
| A − research B | -0.048780075448 | -0.008181138444 (−2 SE) | [-0.057570507085, -0.041547035624] | Failed |
| A − research by_v2fy | -0.040488748392 | −0.005 | [-0.049274191331, -0.032347297759] | Failed |

A−B paired-reference bootstrap SE is `0.004090569222`. Both
paired intervals use the same 10,000 reference draws. B's SROCC is
`0.941619 [0.937374, 0.946677]`; by_v2fy's is
`0.933328 [0.925635, 0.941472]`.

Research B is the pinned `codec_target()` bake at formula revision 5,
SHA256 `89cb4861e2651ff8d4a98bb879feed83a0f9eb3f22b83e644e42eb2291d74b2e`.
The coordinator fixed by_v2fy to seed 0, last epoch 119 before labels. Its
original SHA256 is
`d34f53111fa8baff4fc6003947768facfedc123e5bb1fd97aef99302297bd6d5`.
The original lacked a formula-revision declaration. Before KADID labels, the
coordinator approved a separate metadata-only Rev4 copy, SHA256
`130ddf713c37551f8ec204b49d25cd4f61eb4cb858bf3c055ee07242dbad2017`.
All numerical model content and 256 cached-forward probes matched exactly;
the copy then served all 2,000 pairs natively at Rev4.

## Exposure and population limits

These are frozen-A assessments of previously exposed populations. The ledger
records October 4 R7/R7a comparisons on CID22-B, CSIQ, AIC-4, MCL-JCI and
KonJND JPEG, plus October 18 candidate reads on CID22-B, CSIQ and LIVE clean10.
KADID TERMINAL had no earlier zensim model scoring, but its labels had
third-party studies on October 24–26 before this registration.

CID22 uses the approved B24 population, including `3316926_opo25u`, rather
than the historical clean23 allowlist. Its reference set's other half
contributed training. LIVE's 19 overlapping Kodak references contain TID2013
training content; the fixed October 3 clean10 subset is reported separately.
MCL-JCI shares ten sources with KonFiG training images. Its clean40 excludes
sources 1, 3, 6, 7, 9, 17, 28, 31, 45 and 50, fixed before this read.

The historical MCL target table and builder were lost. The coordinator
registered its replacement before parsing: mean count of QF≥q chain points
across 30 subjects per source, q=1..100, with target 0 at QF100 by definition.
Each pair uses the pristine BMP and coded JPEG; the QF100 anchor does not
assert identical pixels. Reconstructed table SHA256:
`8d9568e9b897b82106843b88fa59c59c14f9f11ea080e084a827cf63db17be1e`.
Byte identity with the lost table cannot be verified. MCL ratings also had
third-party studies on October 24/25, without prior registration.

## Reproducibility

Every target parse followed a committed, pushed and verified exposure pin;
failed or partial reads would also spend their population. Features were
extracted serially, and prediction and core statistics ran through the Rust
owners. Comparators consumed the same native decoded RGB pixels, verified by
hashes and independent PNG readback. Public/research split parity matched
135,625 normalized contract records with zero differences. This assessment
did not fit, select or change A.

The [complete research record](https://github.com/imazen/zensimx/blob/6d639be157daf2d87be09eba9f445ebdf8eec137/benchmarks/final_read_a_2026-11-05.md) contains the registration, exact exposure
ledger references, signed KROCC, PLCC after 4PL, collection-scale RMSE,
within-reference intervals, all 25 KADID distortion panels, native scatter
diagnostics, raw-file and binary hashes, and resource receipts.
