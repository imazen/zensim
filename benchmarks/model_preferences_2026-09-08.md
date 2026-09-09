# Base-model preferences on canonical JXL outputs — September 8 registration

Registered before candidate scoring. The decoder repair is complete. The July
#69/#70 study and September reuse audit report useful H3 steering for the tested
MLP, but no benefit for tested linear B. The newer fixed-D policy also fails
independent RD. Do not keep retuning that failed allocation rule while assuming
its base-model preferences are adequate.

Reuse the corrected native scalar ladder: twelve already admitted canonical
training origins, 21 distances each, plus one exact identity pair per origin.
No new encoding, source admission, training, validation or terminal scoring.
Use all 264 pairs, with immutable original bitstreams and canonical decoded
PNG hashes. Do not reuse old decoder-era scores. This is a training diagnostic,
not held-out model selection or a claim that nominal distance always ranks
perceptual quality correctly.

Compare current packed/declared-ID D, B and generation A, plus all three saved
A_plain and H_anchorlad control seeds 4004/4005/4006. Historical training recipes
retain their documented split/provenance limitations. They are candidate
mechanisms for this screen, not newly qualified training recipes. C's different
944 feature contract is outside this fixed 372-class comparison; no result
here rules out C or qualifies the whole product. Every score comes from the
existing canonical extractor's BakeScorer pixel/cache audit. No companion is
attached: this screen isolates base preferences; the D228 corruption head's
failed honest protection remains separate required work.

Independently score the verified decoded PNG pairs with the pinned native
zenmetrics SSIMULACRA2 and Butteraugli instruments. Use the existing RD analyzer
to report identity=100, distorted scores above identity, score range, distance
inversions, and same-source pair preferences where *both* judges agree.
Resolve judge differences only when SSIM2 differs by more than 0.1 and
Butteraugli pnorm3 by more than 0.005; these are the existing engineering
screen resolutions, not human JND claims. Candidate differences within 1e-6
are ties, reported separately. Count wrong-direction and tied predictions
as unresolved pairs. Report all ordered distance pairs, adjacent pairs,
near-lossless pairs (both distances <=0.1), and each content class. Compare
scores only within a source; no correlation against codec q is called human
perceptual accuracy.

Advancement rule, fixed before results: a candidate may proceed to a new native
steering screen only with complete serving parity, twelve exact identities,
zero distorted outputs above 100, strictly fewer unresolved consensus pairs
than D, and no content-class regression in that count. A saved three-seed recipe
advances only if all three satisfy those conditions; do not select its best
seed. This is an advancement screen, not a replacement release gate. If no
family advances, repair base-model training preferences using a new registered
recipe; do not soften the rule or spend validation on a failed candidate.

Owner reuse: `extract_features_372col --audit-bake`, the existing zenmetrics
batch judges, and `rd_probe_analyze_2026-07-18.py`. Pin models, binaries, pair
keys, source/decoded hashes and commands in the fresh artifact manifest.

## Results

All 2,376 candidate pixel audits (9 models × 264 pairs) pass the existing
pixel/cache score comparison. A separate 264-pair D audit proves the retained
PNG judge inputs decode to the same RGB hashes and scores as the bitstreams.
Both judges complete all 264 pairs. Every model returns exactly 100 on all
12 identities and none exceeds 100 on the 252 codec outputs.

The independent judges resolve 2,471 of 2,520 same-source codec-output pairs.
Unresolved model preferences below include ties (none occurred):

| Model | Wrong consensus pairs | Adjacent wrong | Near-lossless wrong |
|---|---:|---:|---:|
| D | 0 | 0 | 0 |
| B | 3 | 3 | 1 |
| Generation A | 3 | 2 | 3 |
| A_plain 4004 / 4005 / 4006 | 170 / 202 / 179 | 31 / 33 / 34 | 87 / 94 / 93 |
| H_anchorlad 4004 / 4005 / 4006 | 69 / 51 / 48 | 21 / 17 / 18 | 39 / 29 / 25 |

The newer MLP failures occur in all four content classes. Generation A's three
failures are small near-lossless inversions on training origins 9380 and 8384;
D orders those pairs correctly. Negative output tails remain present (A minimum
-30.8295, D minimum -38.4005). This does not validate the separate codec-floor
instrument or the full negative-tail release gate.

No candidate advances under the preregistered strict-improvement rule. Its
zero-error D baseline also exposes a limitation: strict improvement is
unattainable on this panel, even for another zero-error candidate. Preserve
that outcome; do not interpret it as proof D is optimal or that more training
must fix D's bulk scalar ordering. The evidence instead separates good scalar
preferences on this panel from D's already failed independent spatial RD.
No release threshold changes and no held-out candidate qualification follow.

Eighteen negative controls reject schema/threshold changes, missing/duplicate
models, wrong source roles/families, missing/keyed pairs, input hashes, audit
model/pixel identities, broken parity, incomplete/miskeyed/nonfinite judges,
and existing outputs. Missing-judge rejection also passes under Python -O.
The valid control reproduces the complete result exactly. Python compilation
and the 605-script lint pass. Rust inference is unchanged; the pinned existing
extractor and judge binaries are reused, with hashes in INPUTS.json.

Artifacts: `/mnt/v/output/zensim/model-preferences-2026-09-08/`, mirrored to
`~/work/zensim-validation-2026-09-08/model-preferences/`. The packed model copies,
264 pair keys, original and decoded file/pixel hashes, serving audits, full
judge rows, commands, controls and preferences.json retain all measurements.
No new encodes, network fitting, validation or terminal scoring occurred.

## Next registered model experiment

Test a convex blend of the existing generation-A MLP and D, retaining the MLP
while repairing its measured near-lossless preferences. The complete calibrated
blend already has a Rust surface owner: `BakeScorer::ensemble`. Extend the
existing pixel audit CLI to exercise that owner and bind both member hashes,
ordered weights, complete pixel/cache scores and actual feature reuse. Do not
use the legacy `ensemble_mix` raw Predictor path as a serving oracle: it does
not apply this complete calibrated-score composition.

Fit only on the eight existing fit origins (2010, 1054, 6068, 6610, 7066, 9380,
8206, 8384). The four probability-calibration origins (1214, 6064, 9066, 8462)
remain separate for this blend's training-calibration check. Search D weights
0.00, 0.01, ..., 1.00 in ascending order; select the smallest weight that gives
strict correct order (>1e-6) on every fit-origin consensus pair. Fit mathematics
may use Python, but all resulting candidate scores and spatial attribution
must execute through the Rust surface. Bind the final model files and weight
representation explicitly. No new features or repeated deterministic fits.

Before spatial experiments, require zero consensus disagreements/ties on both
fit and calibration origins, twelve exact identities, no output above 100,
and complete Rust serving parity. This is a replacement *experiment* rule for
a newly fitted composition: the previous strict-improvement screen cannot
resolve candidates against D's zero-error baseline. Its failed results remain
unchanged. Matching D's scalar ordering is necessary, not evidence of a spatial
improvement or a qualified model. The independent spatial RD, human ranking,
corruption, targeting and performance release bars all remain required. Stop
if the selected blend fails calibration; do not search against those outcomes.
