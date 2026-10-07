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

## Final verification and artifact pins

The final derived image uses the corrected data pack and Rust producer
`5b2cc589429546b8c569509bc4fea3e04c5ad2af`; Python preparation producer
`be40ab351adc92542cc74e23ee532b7d487ca80d`. The local zenmetrics profile producer
is `4e4db22fed6b984d3e42399db50225fce0209343`
(change `kplzuzlklxslosxqnyoovosutxwusntz`). These are local commits for
coordinator landing; they have not been pushed by this lane.

- program_sha: `3d21e77fbfa9948479261578c5ff885e0ed0c78e58089ce9128f3bf764c2930e`.
- data_sha: `ac8a52ceaa57b936ddd920595618d90f5d24a9d4910ab23d055879a83c738c51`.
- manifest_sha: `f5274de3da22c6bcfb93526461cc3b643cabe85a7ef9132ceeaf2cecd618d75e`.
- Prepared local image ID: `sha256:a4dbeffa54a35d5a8396050222309353c84c970fcda305ea5e3642e66de8df0c`.

Full smokes used the existing fit executor in one-CPU, 6 GiB containers with
forced v3 dispatch. Both ran 120 × 50,000 pair draws and selected final epoch 119,
then scored all 7,869 admitted exploratory KADID rows. Receipt verification used
`harvest_fit_cells.verify_blob` and `zenpredict inspect` on the actual bake.
Both embedded coverage digests match the live trainer's six-million-draw digest.
Both final prediction vectors and panel statistics exactly match their initial
full smoke vectors; the provenance/coverage corrections did not alter weights.

| arm | training seconds | max RSS, KiB (`time -v`) | cgroup peak, bytes | signed SROCC | signed KROCC | raw PLCC | live digest |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| s2o | 277.3 | 981848 | 2158080000 | 0.8782367801082288 | 0.6915927236200436 | 0.8268446744355169 | `d17561d29635e0b1` |
| s2m | 347.8 | 336376 | 1224232960 | 0.7691091556080535 | 0.5871011369513537 | 0.7354113225457796 | `0c88baf2d917f341` |

The outer Docker-client wrappers reported `run-heavy: done rc=0` with
peak-RSS 0.03 GiB; that number measures the supervisor. The table above reports
the measured fit process RSS and complete container peak. Both cgroups recorded
zero OOM kills. The cap is 6 GiB, with existing placement entries unchanged and
no worker/filler restart. These times are local smoke measurements, not a speed
comparison or a fleet throughput estimate.

The final driver exited 2 after both successful smokes and their cleanup because
its lint comments were edited while the shell was suspended inside Docker.
The completed executor's `run-heavy rc=0`, output hashes, full smoke receipt,
and Docker `die exitCode=0` plus `destroy` events independently verify completion.
A followup lookup found the container already removed and overwrote its original
inspect capture; the preserved Docker event receipt replaces that capture.
The corrected driver passes `bash -n` and shellcheck. No training output was
changed or test expectation relaxed to handle this wrapper error.

Short parity covers the five requested path families on one real-table fixture:
historical, strict, hd, hp and ha, each two epochs × 128 draws. It compares all
non-repro ZNPR bytes and all reproduction fields after normalizing only timestamp
and trainer/output argv paths. HDR legs use their actual owner-resolved effective
weight 4.287408376091277 and registered supervision modes. This bounded check
satisfies the requested short fit; it is not an exhaustive parity claim across
all legacy configurations. Both nonzero pooling knobs with a GPU runtime refuse
before an intentionally missing group payload is opened.

Final Rust library regression: 285 passed, none failed, one preexisting ignored
external-assessment admission test requiring its pinned feature-only manifest.
No ignore was added. CI-equivalent all-target/all-feature workspace clippy passed
with warnings denied. Scoped formatting and runnable-script lint passed.
The full Python regression passed all 175 tests (70.319 seconds; run-heavy
76 seconds, peak-RSS 0.42 GiB). Archived artifact receipts are recorded in the
adjacent preparation pointer and readiness record.

The permanent local evidence archive contains 460 payload files (3,017,418,659
bytes), including the saved final image, all program/data versions, fit receipts,
parity fixtures and failed-attempt logs. The NAS transfer verified all 460
SHA-256 checksums; `run-heavy: done rc=0 93s | peak-RSS 0.02GiB` measures the
local transfer supervisor. A first transfer failed on SSH argument quoting before
extraction; the retry used an explicitly quoted remote command and succeeded.
Supplemental transfer receipts preserve that failure and the final documentation
snapshot. No source, image, data or manifest was published as part of preparation.
