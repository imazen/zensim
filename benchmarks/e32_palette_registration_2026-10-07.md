# E32 — PALETTE augmentation of the four-source by_v2fy recipe

Status: final registration proposal for coordinator review; **not registered
on main and not authorized to launch by this lane**. This file fixes the
scientific rule before any E32 fit or assessment. The original draft remains
unchanged. No human-label payload or E30 model/result bytes were opened here.

## Launch prerequisite and control

**E32 launches only after all 40 E30 nA3 cells are complete and their pins are
frozen.** E30 has not run yet according to the coordinator brief. This lane
does not certify cell existence from paths or the arm name. E30 registration:
[e30_four_source_registration_2026-10-07.md](e30_four_source_registration_2026-10-07.md),
original registration commit `d61eac116a52`.

Control is E30's complete nA3 four-source arm, **not its E24 comparison control**:
`sel:59f0bbc2f290@h32:H128:cv16:cf98`, head N, inherited formula revision 5,
seeds 0..9 × kadid/tid2013/konfig/cid22_a25, final zero-based epoch 119 of 120.
KADID TRAIN+SELECT, TID2013, KonFiG TRAIN+VAL and CID22-A25 retain D1 roles.
The arm adds all 42 palette_v2 slots; there is one primary arm, with no
post-result N/feature selection, pruning, sign-search or palette tuning.

Before the first E32 fit, freeze every source/seed's E30 result bytes,
final-119 bake, feature list, program/binary and fit/data receipt SHA-256.
Reuse requires exact labels/roles/ordered rows, inherited feature bits,
teacher and coverage inputs, scaler/transform procedure, loss weights,
optimizer, sampler/RNG/seed streams, epochs and fit program/binary identities.
A palette transport extension must prove its control projection unchanged.
A strict-admission or program change that alters baseline fitting invalidates
reuse. If exact reuse cannot be established, the coordinator registers one
complete fresh matched control before any E32 fit/assessment, after the E30
completion prerequisite. Freeze this decision and its parity audit first;
no outcome-based baseline choice or mixture of old/new cells is permitted.

AIC-3 remains excluded from fitting, preprocessing, packing, evaluation and
control discovery. Never enumerate AIC-family cells to discover controls.
The four LODO sources are source-disjoint; fit preprocessing on each fold's
training rows only. KonJND BPG is not an extra human label leg. Existing
teacher/coverage companions retain their authorized roles and must have
identical pins between control and arm. Historical five-source E24 controls
cannot stand in for this nA3 baseline.

## Frozen identity and transport contract

Canonical order is materialized in
[e32_palette_feature_ids_2026-10-07.json](e32_palette_feature_ids_2026-10-07.json):
420 pinned by_v2fy IDs in their existing order, then f1825..f1866 in N-major,
signal-major order, exactly **462 arm IDs**. Identity-list SHA-256:
`207d6d29ed86395166444c60fb4fc788856818801e6b64bcbc36ed7b81d1773d`.
Its source `costset2_2026-10-03.candidate_ids.json` SHA-256 is
`0a6a20dc356acef3bef9deffc411f03189813e8b924fddcf7b22f7efea6b9f17`.
The six palette signals per N=2..8 are mean centre shift, signed lightness,
signed chroma, signed hue, population EMD and largest matched centre shift.
The ordered ID list is fixed here; transport launch pins remain unfilled.

Record separately every dense transport position, source column, canonical
ID, arithmetic era and key domain. Project `palette_f1825`..`palette_f1866`;
never project legacy auxiliary f1825+ by number or overwrite them. Retain
those auxiliary columns/owners without adding them to the primary arm.
Join instrument rows by `(member_set,pair_key,row_id)` and preserve repeated
observations and physical order. Coverage keeps its separate path-derived
key and original-selection-index contract, distinct from pixel-pair keys.
NaN means absent, not measured zero. All 462 primary reads must be measured
and finite after the registered projection/cast; fail admission otherwise.

Reject palette_v1, unknown/relabelled eras, map/ID mismatch, positional
fallback, unauthorized members and AIC-family rows before any fit. The
PALETTE2 verifier checks exact pinned instrument bytes and its semantic
identity/map before value joins. Freeze the final consumer admission program
and its negative controls as well; a receipt flag alone does not qualify a
new trainer transport. No serving bake may read this research family.

Numerical producer: `e60a6ad74a47981f93f969d63b605c7a88ab09b0`.
Binary SHA-256: `575f39b4a0bb04e224ceca15eed5e878d53273511b884a895a3630e606d4ccc6`.
Palette set: `palette@w1867/palette_v2#30b09cd1`.
Inherited arithmetic revision: 5.
Bank manifest SHA-256:
`46587338cc74ba59e38fe96e637776bfe74135fde29f3f70e22b51605fc50917`.
Instrument manifest SHA-256:
`9f7523bf7d3aaa32418d40d83adb44edccff9e70cc75acc32eb8d5711fe89934`.
Canonical artifact root: `/mnt/v/output/zensim/palette-2026-10-07-v2/`.
Initial palette_v1 artifacts are superseded and cannot be mixed or repaired
by flipping cached signs. The numerical correction was made before any fit
or label read; final sign terms use each palette's own population weights.

Mandatory prelaunch freeze inventory: final transport archive/inventory and
per-table/key/contract hashes; ordered-ID/map manifest; numerical cast policy;
source/D1 role decision receipts; trainer/predictor/assessment program and
binary hashes; teacher/coverage/scaler provenance; sampler/RNG contract;
40-cell control pins and recipe-parity report; complete cell grid; extraction
and semantic-verification receipts. All entries must be populated with exact
hashes and frozen by the coordinator before launch. A path, width, schema
name or `_VERIFIED` flag does not establish identity. Missing pins block launch.

## Seed-paired statistic and fixed decision rule

There are exactly ten inferential seed units, each containing all four folds.
For seed s and source k, let delta(s,k) be the arm signed SROCC minus its
paired control signed SROCC using the existing registered score orientation.
Set d_s = (delta(s,kadid) + delta(s,tid2013) + delta(s,konfig) +
delta(s,cid22_a25))/4. Mean Delta = mean_s(d_s), and
SE = sample_sd(d_s, ddof=1)/sqrt(10). Equal source weights are fixed.
Compute covariance implicitly through these within-seed composites; never
treat 40 folds as independent seeds or use a quadrature of per-source SEs.
For the primary test, t = Mean Delta / SE, df=9;
p_one_sided = 1 - CDF_StudentT_9(t) for H0: Delta <= 0.
Seed variability describes optimization uncertainty under this fixed design,
not independent content-population sampling or product qualification.

E21 numerical guards remain unchanged: Mean Delta >= -0.002; every source's
ten-seed mean delta >= -0.005; W2 mean delta > -2 paired-seed SE.
For W2, within each KADID and TID source/seed panel take the arithmetic mean
of the three lowest signed distortion-type SROCC values, separately for arm
and control. Subtract the paired control W2; average the two source deltas
within seed; then compute W2 mean and sample_sd(ddof=1)/sqrt(10) over seeds.
KADID/TID are the fixed W2 source set. This fixes the seed-level reduction
before fitting and does not silently inherit E21's historical source-SE
quadrature convention. Per-source worst-three membership may differ between
arm and control because this metric measures each panel's lower tail.

Adopt only if all three as-good guards pass **and** Mean Delta > +0.002
**and** p_one_sided < 0.05. There is exactly one primary arm and one primary
improvement test. Nonfinite/undefined metrics, missing cells or zero-SE
degeneracy mean INCOMPLETE, never an automatic significance pass. Undefined
distortion-type SROCCs cannot be dropped to change the worst-three set.
Report all seeds and source deltas, raw prediction tails, W1/W2 and failures;
no best-N/seed/fold or post-result assessment-rule choice.

## External panels and interpretation limits

External NITS/LIVE/MCIQA are frozen assessment-only and cannot train, select
a model or reverse the primary rule. Report seed-paired NITS full and
per-distortion panels; LIVE overall and the pre-fixed ten content-disjoint
references (nineteen overlap TID/SafeSyn). Report MCIQA global naturalness,
colour smearing and semantic/colour misalignment separately, with the existing
orientation/aggregation owner and exact schema dimension names. MCIQA uses
COCO originals as an exploratory fidelity proxy for colourization plausibility;
it is not full-reference human ground truth. External labels were not opened
by this lane. KADID TERMINAL, JPEG-AIC family, CID22-B and protected data remain
unread under this registration proposal.

The 32×32 centre lattice can miss systematic changes: an independent 64×64
probe changed every even-column pixel (2,048 pixels) while all sampled signals
stayed zero because this lattice selected odd columns. Hue sign can reverse
after repartition/rematching, including small rotations. Population-weighted
lightness is effectively N-invariant; chroma uses centroid chroma, so clustering
changes its interpretation. Global distributions do not locate preserved-
population spatial smearing/misalignment or resolve the weak CHROMAQ HF
response. RGB8 sRGB/BT.709-D65 only; native HDR is refused. No
sampling/assignment/N/sign-search change is permitted after results; changing
arithmetic requires a new version and re-extraction.

Before later serving adoption, register a deliberate serving implementation,
API/revision contracts, old-vector/tier gates, measured runtime cost,
independent colour judgments and product qualification gates. E32 research
improvement alone cannot authorize a serving bake or historical-table admission.
