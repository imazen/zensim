# V40 registered SDR results — 2026-10-09

hc4 and uh4 are AS-GOOD under the registered SDR guards. hb4 is NOT AS-GOOD because KonFiG fails the source guard. palette is NOT AS-GOOD because the mean guard fails; every source delta is negative and `adopt=false`.

This records the completed decisions without recomputing them. Exact statistics, seed deltas, source deltas and admission pins are in [the JSON summary](v40_result_summary_2026-10-09.json); the complete artifacts and verified tower mirror are pinned in [the artifact pointer](v40_results_2026-10-09.pointer.md).

The shared fresh V40 control has 40 cells. Each of hb4, hc4, palette and uh4 has 40 cells: four source folds × ten seed indices (0–9), 200 unique fits including the shared control. All four jobsets completed postfit admission. E29 has 120 prediction panels (control plus two arms); E32 and E31 each have 80 (control plus one arm). The same control fits are reused.

The signed statistic equally averages four source deltas within each seed; SE uses ten paired seed units. W2 equally averages the KADID and TID2013 worst-three distortion-type deltas within each seed. Larger signed deltas are better. Guards require mean ≥ −0.002, each source ≥ −0.005, and W2 delta > −2 × its SE.

| Arm | Signed delta | SE | n | W2 delta | W2 SE | W2 n | Mean / source / W2 guards | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|---|
| hc4 | 0.00020195628009737343 | 0.0010308567420564759 | 10 | 0.002951793517868667 | 0.005408487307233505 | 10 | PASS / PASS / PASS | AS-GOOD |
| hb4 | -0.0015464923319287228 | 0.0015646258312876971 | 10 | 6.513828962660117e-05 | 0.00842498505664468 | 10 | PASS / FAIL / PASS | NOT AS-GOOD |
| palette | -0.002956562407006813 | 0.0012202495476900117 | 10 | -0.003848334032416806 | 0.004911433623787785 | 10 | FAIL / PASS / PASS | NOT AS-GOOD |
| uh4 | -0.0008230868590355184 | 0.0015426376681047054 | 10 | -0.00027629142432894217 | 0.010654672497864332 | 10 | PASS / PASS / PASS | AS-GOOD |

| Arm | KADID | TID2013 | KonFiG | CID22-A(25) |
|---|---:|---:|---:|---:|
| hc4 | 0.0007468464087687532 | 0.002570732165752121 | -0.003472356333740634 | 0.0009626028796092534 |
| hb4 | 0.0016775748335080888 | -0.0027080882987633448 | -0.008259966112729855 | 0.0031045102502702203 |
| palette | -0.0018372138504098023 | -0.004772501998705792 | -0.004065557193972025 | -0.0011509765849396336 |
| uh4 | -0.003935608211933827 | -0.0010582750978287315 | -0.0014346608468485567 | 0.003136196720469042 |

E32 palette: t = -2.4229162080852404, df = 9, one-sided p = 0.9807867935668866, `adopt=false`. Its registered adoption rule additionally requires signed delta > 0.002 and p < 0.05. The E29 and E31 decision objects contain no t/df/p or adoption field; none is inferred here.

Each source assessment uses KADID TRAIN+SELECT (7,869 observations), TID2013 (3,000), KonFiG TRAIN+VAL (756), and CID22-A(25) (2,192), totalling 13,817 observations per four-source rotation. These are D1 design-released human populations; each fit excludes its assessed source. They are already research-exposed populations, and these results do not establish untouched external generalization. KonFiG retains the design-grid label reconstruction caveat.

AS-GOOD establishes only the registered SDR retention tolerances against this matched control. It does not establish an HDR improvement, independent confirmatory performance, or a production composition change. The uh4 training arm includes admitted UPIQ TRAIN inputs, but this assessment does not read an optional UPIQ report. No optional HDR, external or UPIQ report was opened; those require a separate exposure authorization and freeze. KADID TERMINAL remains unopened by this lane.

Control pins SHA-256: `4d7acfc887603b123f9631f38435df5477b6e628a372596cb8beb6128bddc84c`.

Training program SHA-256: `de44ea9b1ff85031265dfa1d60731de47f177d6816fb0880f85d28f38c405517`.

Assessment program SHA-256: `9fc7ea638174305ffea142ba00e7e77f5dd62b95ee44d9ebed3cd1dad60db0c6`.
