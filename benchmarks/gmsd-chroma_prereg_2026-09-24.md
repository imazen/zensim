# C8 chroma revision preregistration — 2026-09-24

Lane `gmsd-chroma`, corrected parent `620a384e06e49422d501bdddc9c0439afc846a98`.
Commit this document before new pixel calibration or C8 measurements. The
separate Part A TRAIN reporting read labels after freezing its predictors;
those results do not select any definition, threshold, input or constant here.
No Part B human labels, fits or potential tests are authorized by this protocol.
All outputs remain quarantined and default OFF.

## Inputs and deterministic selection

Reuse the earlier pixel-only CID22 TRAIN/SafeSyn selection and decoded RGB8,
without editing that lane's files. The source index is
`/var/tmp/gmsbank/calibration/planes.tsv`, SHA256
`aa49fbe114acb464c9d88316c28b7bd1e0b1ee8a17b6def9a742b6d4d034a6d3`.
Selection JSON SHA256
`e01e1f09b4688a7a77e92b28a367549ddb5975191d0ae2656d700402f9d7a84f`;
classes TSV SHA256
`824c8e2b9be0e11fd1c8a37a6f447ee4195c39561a70ff97817dc2ee57209e5d`;
pairs TSV SHA256
`f6f201193100c537a633ee81b09587554e40ac8359011f1d452850bc4ff870bb`.
Verify every RGB file digest and pair key before use. Only keys/paths/dimensions
from bank metadata may be read to recover original reference groups:
CID22 TRAIN keys SHA `c99a0887705b17bff05bdbfc55b89f9b4f060bc94a009302543f15d9a0127219`,
SafeSyn keys SHA `12d48d7fc02afd5026067a348ea20a3896ea6de9a94bd99a25ec2db071c2b28f`.

Retain the 652 native pairs. Make two deterministic RGB8 derivatives per
eligible source pair with maximum dimension 128 and 384, respectively.
Dimensions are nearest integers to the original aspect ratio, ties upward;
reject a derivative if either resulting dimension is below 64. Each output
pixel averages the non-overlapping source rectangle whose boundaries are
`floor(i*source_length/output_length)` on each axis, using an integer sum and
round-to-nearest, ties upward. This box reduction uses every source pixel once,
does not upscale, and has no crop, edge extension or random seed. Apply exactly
the same geometry to reference and distortion. Record geometry and new digests.

Derivatives inherit the source reference's frozen content class and TRAIN role;
the class describes the source content, not a fresh classifier at each size.
The four classes remain photo/screen/line_art/mixed. Size uses the derivative's
maximum dimension: tiny <256, small 256–511, medium 512–1023, large >=1024.
For each corpus/content/derived-size stratum, retain at most 32 original
reference groups in SHA256(ref_group) order and at most two original pair keys
per group in ascending order. Native selection is unchanged. Report all 32
strata, rejected geometries and counts; do not manufacture unavailable native
large CID22 data. Tiny/small derivatives are not new independent references.

## Frozen units mapping and constants

Preserve the existing Y gradient literals. Calibrate X and B independently
from pixels at co-sited half-resolution interior sites. The research owner's
RGB8-to-XYB conversion and box pyramid produce the XYB planes. Independently
2x2-box the raw 0..255 RGB8 channels in f64, then apply MDSI's linear opponents:
`M=.34R-.60G+.17B` for X, `H=.30R+.04G-.35B` for B. This is a declared units
mapping between different colour spaces, not semantic/reference equivalence.
For odd dimensions the units instrument uses only the common floor-half grid.

For CS, center stored XYB X by `f64::from(0.42_f32)` and stored B by
`f64::from(0.55_f32)`, retaining the existing factor 14 in X. The value ratio
is `abs(opponent)/abs(centered_XYB)`. For gradient constants, use the magnitude
of the opponent's 3x3 Prewitt derivative divided by 3, divided by the XYB
central-difference magnitude with no 1/2 factor. Match the prior Y instrument:
the XYB subtraction is f32 before promotion; subsequent gradient arithmetic
is f64. Both ratio numerator and denominator must exceed 1e-6. Exclude borders
so different extensions cannot affect units calibration. No labels or metric
predictions participate.

For each of the four ratios (X/B × value/gradient), combine eligible reference
and distorted sites into one pair median, then take an equal-weight median
across admitted pair/geometry records, following the previous Y ratio method.
Empty eligible-site records are reported and omitted only for that ratio;
failure to obtain a finite positive pooled ratio blocks constants production.
Report site counts, empty counts and p25/p50/p75 by all strata and pooled.
These are descriptive quartiles; no bootstrap or inferential independence
claim is attached to the derivatives.

For channel j, `CS_Cmid_j = 550 / Rvalue_j^2`, and
`GRAD_Cmid_j = 140 / Rgradient_j^2` (MDSI C1 as the gradient analogue).
Emit five source literals at `Cmid_j * 4^(k-2)`, k=0..4. Reuse the same
calibrated channel constants at each retained scale. Do not tune them after
viewing gates or Part A correlations. No runtime transcendental builds a bank.

## Revised layout and arithmetic

Native scale 0 retains only the 15 Y gradient features. At scales 1,2,3
(half, quarter, eighth resolution), retain X/Y/B gradient loss/gain/deviation
(45 per scale), followed by joint X/B CS mean-loss/deviation (10 per scale).
Five constants each contribute adjacent CS mean/deviation slots.
The C8 widths at 1/2/3/4 scales are 15/70/125/180; four-scale output remains
f1322..f1501. This explicitly revises an unlanded family with no consumer.
No f0..f1321 definition, ordering or arithmetic may change.

The coarser chroma placement is motivated by measured low-pass chromatic
contrast sensitivity ([Mullen 1985](https://doi.org/10.1113/jphysiol.1985.sp015591)).
This source motivates the engineering choice; it does not establish that
these pixel scales represent particular retinal frequencies without display
geometry, or that dropping native chroma is perceptually optimal.

For centered values xr,xd,br,bd and per-channel Cx,Cb, use
`L = ((xr-xd)^2/Cx + (br-bd)^2/Cb) /
     (xr^2/Cx + xd^2/Cx + br^2/Cb + bd^2/Cb + 1)`.
Emit the mean of L and population deviation via per-row f64 Welford and
row-ordered Chan merges. With equal C and author opponent coordinates this
is algebraically 1-CS. Its difference numerator makes identity exactly zero.
Do not clip negative author CS or replace deviation with a raw second moment.
Fuse these accumulators into the existing coarse X gradient pass, with the B
planes co-sited; no extra image traversal. Use the existing arcane SIMD
dispatch/inline generic-body pattern; document any inability to place a
literal rite attribute on the generic backend helper.

## Gates and fixed decisions

1. On all 116 existing author oracle cases, pass unshifted f64 MDSI H/M planes
   through the actual chromaticity helper with Cx=Cb=550. CS maps must reach
   absolute error <=1e-12 and relative <=1e-9 where |reference|>1e-12. A factor
   ten wrong constant must fail on at least one nonidentity case. This tests
   the formula, separately from XYB's declared coordinate convention.
2. Dump the first eight native calibration TRAIN pairs in source TSV order,
   all three XYB channels and all four scales. An independent NumPy mirror
   must match every revised C8 slot at relative error <=1e-6, denominator
   `max(abs(reference),1e-12)`. Retain the inherited f32 complete-interior-V8
   gradient arithmetic and f64 boundary/trailing arithmetic in that mirror.
   Multiply all new constants by 16 as a negative control; it must fail.
3. Reuse the 144 TRAIN-pair matrix (64 CID22,64 SafeSyn,16 KADID) ×
   native/v3/scalar × 1/8 threads, on/off/corrected-parent. Require bit-identical
   f0..f1321 and exact-zero C8 identity. Input TSV SHAs respectively:
   `734e034ccdf03fa1fa76c519a92af296fefd08a7871c1716500d23d988187684`,
   `9eaa27e3729bfa6aeb5f2b21d14dabee0c287a03266c8e87dd7c3c9d480ae8c4`,
   `4d780829d691a2963a19474482a37428b9b20b370a6572f3f616e072b3a1befa`.
   Record each actually supported tier; a missing tier is not a pass.
4. No dead slot on that frozen nonidentity population; strict contrast
   reduction must route gradient change to loss, and constant chroma shifts
   must register CS loss. Registry mapping must round-trip all slots/scales.
5. Interleaved zenbench on/off cost at 256²,1024²,2048²,4096² and 1/8 threads;
   retain raw rounds, fit alpha+beta*pixels descriptively, mark contention.
   Report the inherited <=5% marginal-cost goal without retuning on failure.
6. Registered-producer servability census must refuse 0; run required focused
   tests, formatting and clippy. Failures/missing results remain explicit.

Update the design/calibration notes and potential proposal: P1 means this
revised 180-slot family; propose optional P2b = baseline bank + reference-exact
MDSI peer columns. The coordinator decides inclusion before the potential
owner reads any labels. This lane does not run that potential analysis.
