# E28 registered SSIM2-recipe preparation — 2026-10-07

Registration: `e28_ssim2recipe_registration_2026-10-07.md`, committed at
`82c9af81`, including its coordinator amendments. This is preparation evidence.
The 100-cell experiment is not launched, and no adoption verdict is available.

## Fixed choices and implementation

The teacher/source membership pin was committed at `67228a12` before either new
arm fit. KonFiG pooling for s2m was recorded at `02fa4990` before fitting.
The grouped NM mapping was committed at `a1028fbc` before any NM objective call.
The reviewed E27 owner dependencies were restored from `29592719`; the binding
registration base lacked those Python owner updates. This does not change the
registered E28 population or its parent control.

The 420-column selection is `59f0bbc2f290`; head N, hidden 128, seed indices 0–9,
120 epochs × 50,000 pair draws, and fixed final epoch 119. The prepared manifest
contains both arms across five registered source-held-out folds and ten seeds.

The existing CPU RankNet owner replaces within-reference endpoints by uniform
same-dataset endpoints with probability 0.5 on opted-in legs. Independent same-leg
32-row Pearson batches add a differentiable loss of weight 0.5. The batch schedule
and independent RNG were fixed in the pre-fit pin. Teacher legs retain their MSE
term, and human legs retain rank supervision. The registered plain CPU head is
required; unsupported combinations refuse rather than ignore new switches.

s2o retains both teachers and inherited coverage, isolates the four nonheldout
human source legs, and pools only SafeSyn/CID22/KADID/TID. s2m uses the exact four
registered populations minus a corresponding heldout source: CID22 fit,
KADID TRAIN, TID2013 and KonFiG TRAIN. It drops SafeSyn, coverage, KADID SELECT,
KonFiG VAL, CID22_a25 and AIC3 from training. The latter two remain heldout folds.
KonFiG uses design-grid labels, not a flicker-boosted reconstruction. This stated
limitation applies to both arms; s2m pools within that dataset as its four-leg
recipe requests. No comparison crosses datasets.

## Diagnostic and interpretation

NM uses 42 grouped linear weights plus four monotone cubic parameters (46 total).
Grouping follows the declared basic/v2 signal structure across retained scales
and channels; standardization uses included fit rows only. The objective uses
CID22 MSE in /100 units, signed tau-b and raw Pearson. That unit choice and the
optimizer budgets were fixed before fitting; this is not a published-coefficient
reproduction. Fit weights are persisted before reading the exploratory heldout
panel. Correlations run in process, with the final panel checked by the Rust owner.

The KADID-held-out smoke exhausted the registered NM iteration budget, then the
registered Powell evaluation budget: NM 2,000 iterations / 2,729 evaluations,
objective 0.49034840009512326; Powell 7 iterations / 12,000 evaluations,
objective 0.4001255973976253. `converged=false` is retained. No budgets or thresholds
were changed after observing the fit. KADID signed SROCC was 0.5352471314724624,
signed KROCC 0.40566294099197636 and raw PLCC -0.0775797514824902.
It remains **POTENTIAL — ceiling, not a model score**, produces no bake, and cannot
support an adoption claim. The one-fold comparison to the ten existing E24
KADID controls is report-only.

The existing E13/E24 assessment owner now checks the E21 signed guard and W2,
then signed raw KROCC/raw PLCC with the registered two-SE rule over ten paired
seed means across the five fixed folds. It requires all 100 new cells and 50
parent control cells. If both arms pass, s2o has registered preference.

## Corrections before artifact freeze

The first prepared view manifests incorrectly named the registration commit as
their producer. `be40ab35` binds them to their actual committed preparation owner
and records registration separately. All 37 numerical table hashes and 25 key
hashes matched the original preparation; both NM parameter arrays matched exactly.
The corrected data pack is used in the final image smokes.

Final review found that E28 coverage metadata replay omitted the live pooled
endpoint draws. `5b2cc589` fixes the replay and verifies the live sampler digest.
It also closes a GPU refusal gap when a nonzero pooling knob has no leg names.
Neither correction changes the registered model computation or experiment design.
The final binary and image are rebuilt and both full smokes rerun after this fix.

An initial smoke failed because the inherited image lacks `/usr/bin/time`.
The smoke now binds the host measurement executable read-only; nothing is installed
at worker boot. A later derived-image build rejected a stale program SHA before
fitting; its driver was corrected. Failed attempts and superseded artifacts are
preserved alongside final evidence.

## Exposure and launch boundary

Only admitted E27-stage exploratory tables were used. No protected bank payload,
sealed labels, HDR VAL or confirmation labels were read. No external SDR panel
was opened; a gated report-only command is prepared for after complete harvest.
The new recipes are historical research replay, not strict product qualification.

No source, container or data was published, and no cell was enqueued. The existing
zenfleet declaration/executor/harvest owners prepare and verify every cell. The
launcher requires a coordinator authorization receipt tying landed source and
published profile pins to exact program/data/manifest hashes and the image ID.
Existing placement and controller owners are retained; no filler or worker restart
is performed. The absence-of-authorization check leaves queue bytes unchanged.
