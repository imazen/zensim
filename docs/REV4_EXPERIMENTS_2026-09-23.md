# zensim Rev4 — the experiments that decide what it should be (2026-09-23)

**Why this exists.** The first Rev4 design brief ([ZENSIM_REV4_REPORT_2026-09-23.md](ZENSIM_REV4_REPORT_2026-09-23.md))
failed its mission. It consolidated known fixes and accepted a tie with SSIMULACRA2 without testing *why*
zensim falls short. This document replaces its conclusions with experiments. Each experiment answers one
question, is cheap before it is expensive, and carries a decision rule fixed before it runs. The brief's
evidence section (§1) remains the background. The [controlling production plan](PRODUCTION_PRIORITIES_2026-09-15.md)
and its split and owner rules still bind.

## User direction (2026-09-23) this program follows

- zensim is an **innovation platform**, not one metric. Fast and rich variants are fine.
- Two specialised metrics — one for **regression testing** of rendering engines, one for **codec control** —
  are fine *if the specialisation shows a real, measured benefit*.
- **Speed below 512×512 does not matter.** Speed gates cover 512² and up.
- Multimetric agreement did not help as a **training signal** in the past. It may still work as a **gate**.
- CVVDP may have been a poor teacher because of the display conditions chosen. Find out, and test better
  ones.
- Add **artificial rendering corruptions**: downsampling in sRGB instead of linear light, premultiplication
  errors during scaling, and the like.

## What we know, and what we do not

**Known** (records in the brief §1):
- zensim ties SSIMULACRA2 on CID22 and wins on CSIQ.
- The served default trails SSIMULACRA2 within-image, on AIC-3 and on fine-grained forced choice.
- CVVDP, IW-SSIM and MS-SSIM lead on the AIC-4 sample.
- KonJND resisted nine mechanisms.
- A non-negative-distance head makes the dial contract hold. Codec floors respond only to floor-reaching
  data. Capacity trades against KonJND and floors.

**Not known — the questions this program answers:**

| # | Question | Why it decides Rev4 |
|---|---|---|
| Q1 | Is zensim's deficit concentrated at **near-threshold / high-fidelity** quality? | If yes, the core bet is visibility modelling. If no, the deficit lies elsewhere and the bet changes. |
| Q2 | Was CVVDP a poor teacher only because of its **display model**? | It decides whether a physically grounded teacher exists for the high-quality band before any human data arrives. |
| Q3 | Does a **visibility front end** (contrast sensitivity and masking in cycles/degree at a declared viewing geometry) help *when the teacher can reward it*? | Past CSF and band-visibility trials added small feature families under an SSIMULACRA2-dominated fit and judged them by no-refit ablation, so they cannot distinguish "useless" from "unrewarded". |
| Q4 | Does multimetric agreement work as a **selection gate**? | A cheap guard against selecting a candidate that games one axis. |
| Q5 | Does a **testing-specialised** metric beat general metrics at catching rendering regressions at a fixed false-alarm rate? | It decides whether the platform ships two metrics. |
| Q6 | How **hackable** is each metric under AI-driven tuning at an equal attack budget? | Codec control with a learned tuner is an optimisation against the metric. |

## Rules for every experiment

- **Preregister first.** Each experiment gets a preregistration record in `benchmarks/` naming the hypothesis,
  arms, controls, data roles, statistic, decision rule and budget, committed before any result is read.
- **TRAIN-only selection.** Held-out human sets (CID22-49 excluding sealed CID22-B, AIC, KonJND, CSIQ,
  UPIQ) are read once per frozen arm, and every read goes into the exposure ledger.
- **Statistics.** k ≥ 3 seed means and reference-clustered bootstrap CIs, through the single statistics
  owner. No best-of-k. One era per table. Global cross-source correlation is reported first, then per-codec,
  per-source and within-image.
- **Judge ≠ steer.** Anything tuned against a metric is judged only by metrics that were not in the loop, and
  by humans.
- **Delegated runs are quarantined.** They go on local quarantine branches and must show their work (commands,
  outputs with sha256, and where each number came from). They are audited by Opus before any value is used.
  A measured negative is a result.

## E1 — Where does zensim lose? (answers Q1; analysis only, no new compute)

- **Data:** the existing per-pair outputs of the board's full evaluations. This covers B, D, Rev3 fast and
  rich, and the 944 flagship, next to SSIMULACRA2, butteraugli (3-norm and max), IW-SSIM, MS-SSIM, GMSD and,
  where rows exist, CVVDP at the matching display. Corpora: CID22 (49-reference board rows), CSIQ, AIC-3,
  the AIC-4 sample (crop and full resolution, separately), SDR25, KonJND-504 and the JPEG-AIC forced
  choices.
- **Method:** bin every corpus by human quality: JND units where the labels are JND-scaled, quality quantiles
  otherwise. Within each band, report pairwise ordering accuracy and the range-restriction-safe band
  statistics (tail SROCC) from the statistics owner, plus the paired zensim−peer difference with
  reference-clustered CIs. For forced choice, split by question type (within-codec vs cross-codec) and by
  fidelity level.
- **Decision rule:**
  - *Near-threshold concentration* is **confirmed** if zensim's deficit to the best peer in the top-quality
    band(s) excludes zero on at least two corpora, while its mid and low bands tie or win.
  - It is **refuted** if the deficits are uniform or sit in the low-quality bands.
  - Anything else is **unresolved**, and the next experiment names the missing data.
- **What it changes:** confirmed → E3 is the core bet. Refuted → E3 drops to a side experiment and we look at
  what the losing bands share.
- **Executor:** Opus, because it settles the program's premise. Existing tables only; no new compute.

## E2 — Is CVVDP a good teacher at the right display? (answers Q2)

- **E2a — inventory (cheap).** Name the display model behind every CVVDP value ever used as a training
  target, teacher column or evaluation row.
  - Known so far: every CVVDP value on the board used `standard_4k`. The CPU path ignored the display flag
    until 2026-09-22.
  - The v12/v13/v47 falsification records do not state a display. The zenmetrics runners default to
    `standard_4k`.
  - The GPU binary accepts a display flag but writes the same column for every display, so its column name
    cannot certify which display produced it.
  - Output: a table of column → display → evidence (manifest, command line or code path), with "unknown"
    written where it is unknown.
- **E2b — accuracy by display (moderate).** Score CVVDP with the parity-verified port at `standard_4k`,
  `sdr_4k_30`, `standard_fhd`, `standard_phone`, `iphone_14_pro` and the local `modern_oled_phone_indoor`
  preset.
  - *Selection data (the only data used to choose):* TRAIN-role human legs (KADID-10k, TID2013, KonFiG
    TRAIN).
  - *Report, not select:* each held-out set at its **documented** viewing geometry, taken from the dataset
    paper in the zenpapers corpus. For example, AIC uses `standard_fhd`.
  - Also report each display's behaviour in the E1 quality bands.
  - **Decision rule:** a teacher display is chosen by TRAIN legs only. If no display beats `standard_4k` on
    TRAIN human legs by a preregistered margin, CVVDP's poor showing as a teacher stands and E2c does not
    run.
- **E2c — the teacher rerun (heavy, conditional on E2b).** Repeat the falsified CVVDP-teacher recipe on a
  fast head, k = 3, with the E2b display. Its control is the same head with the SSIMULACRA2 teacher.
  - **Decision rule:** adopt CVVDP as the high-quality-band teacher only if the E1 near-threshold band
    improves (CI excludes zero) with no suprathreshold regression.

## E3 — Visibility front end, crossed with teacher (answers Q3; heavy, after E1 and E2)

- **Arms:**
  - Front end: current Rev3 features, versus the same features plus a CSF and masking stage evaluated in
    cycles per degree at a declared viewing geometry (pixels per degree as an input).
  - Teacher: SSIMULACRA2, CVVDP at the E2 display, and human TRAIN legs.
  - Crossing the two gives six arms, on a fast head, k = 3, TRAIN-only selection.
- **Reading the matrix:**
  - If the front end helps only under the CVVDP or human teacher, the past CSF and band-visibility failures
    were an artefact of the teacher, and the front end becomes the core of the codec metric.
  - If it helps under no teacher, drop it.
  - If it helps even under SSIMULACRA2, it is a cheap win.
- **Also measured:** the viewing-geometry input, compared by scoring crop and full-resolution renderings.
  This tests whether declared geometry removes the rendering sensitivity the AIC-4 sample showed.

## E4 — Multimetric agreement as a gate (answers Q4; analysis only)

- **Method:** for every candidate already on the board, compute its rate of disagreement with the
  two-reference rule (SSIMULACRA2 and butteraugli agree on an order the candidate reverses) on TRAIN-side
  ladders. Test whether that rate predicts the candidate's held-out human rank and its dial-contract results
  better than the registered composite does.
- **Decision rule:** adopt it as a selection gate only if it predicts held-out outcomes the composite misses,
  with a CI. It never becomes a training signal again without new evidence.

## E5 — Rendering-regression corruptions and a testing metric (answers Q5)

- **E5a — the generator (moderate).** Add a rendering-corruption family set to the existing corruption owner
  or to zensim-regress. Each family has a correct reference implementation and a broken variant:
  - downsampling in gamma-encoded sRGB instead of linear light, across kernels and ratios;
  - scaling RGBA without premultiplication, which fringes colour at alpha edges;
  - premultiplying twice, and forgetting to un-premultiply;
  - gamma applied twice, or omitted;
  - half-pixel and one-pixel geometry shifts, and wrong edge handling;
  - the wrong resampling kernel;
  - swapped channels and wrong chroma siting;
  - an ICC profile or primaries dropped (Display P3 treated as sRGB);
  - EXIF orientation ignored;
  - 16→8-bit truncation instead of rounding, clamping overflow, and a changed dither.

  Alpha cases are scored after compositing over black, white and a checkerboard, and the protocol is stated.
- **Benign-drift set:** SIMD-tier rounding, float vs fixed-point versions of the same correct operation, and
  ±1 LSB rounding. All come from imazen implementations with distinct code paths.
- **Measured for every arm:** detection rate at 1% and 0.1% false alarm against the benign-drift set, the
  severity ordering, and localisation (does the diffmap overlap the damaged region).
  - General arms: max-abs/PSNR, SSIMULACRA2, butteraugli (max and 3-norm), DSSIM, GMSD, zensim B, D, and
    Rev3 fast and rich.
  - Specialised arm: a testing variant using linear-light, alpha-aware, max/tail-pooled features, calibrated
    on TRAIN corruption families only.
- **Decision rule:** the platform ships a separate testing metric only if it beats the best general arm on
  held-out corruption families by a preregistered margin at fixed false alarm.
- **Also:** re-label the existing canonical corruption families as rendering-class or codec-failure-class,
  checking each family's provenance first. Report the two classes separately from now on.

## E6 — Hackability under AI-driven tuning (answers Q6)

- **Attacks at an equal budget for every metric:**
  1. *Knob search.* The controller searches the encoder's knob space for the highest score at fixed bytes.
     This is the AI-tuning scenario. The result is judged by metrics outside the loop, and later by humans
     through Squintly's preregistered gMAD protocol.
  2. *Enhancement.* Sharpening and contrast boosts. A non-negative head should score these at or below the
     unenhanced image; the experiment checks it.
  3. *White-box gradient* attack on zensim, using the pixel adjoint. Only zensim has an adjoint, so peers get
     a black-box attack at a matched evaluation budget, and the asymmetry is stated.
- **Also:** score JPEG-AI decodes (SDR25) to measure fidelity metrics against neural-codec artefacts. Keep
  *fidelity* and *appeal* as separate claims.
- **Output:** score inflation against independent judges per metric. The claim is relative robustness, not
  immunity.

## Order, cost and what each outcome changes

| Step | Experiment | Cost | Runs after | Decides |
|---|---|---|---|---|
| 1 | E1 regime analysis | analysis | — | the core bet |
| 1 | E2a CVVDP display inventory | analysis | — | whether past CVVDP verdicts are valid |
| 1 | E4 agreement-as-gate retrospective | analysis | — | the selection gate |
| 2 | E2b CVVDP accuracy by display | moderate (scoring) | E2a | whether a physical teacher exists |
| 2 | E5a rendering corruptions + benign drift | moderate (generator + scoring) | — | whether there are two metrics |
| 3 | E2c CVVDP-teacher rerun | heavy (training) | E2b | the high-quality-band teacher |
| 3 | E3 front end × teacher | heavy (training) | E1, E2b | the codec metric's architecture |
| 4 | E6 hackability | moderate | a frozen candidate | robustness claims and the tuner rules |

The Rev4 codec metric is specified only after steps 1–3 report. The first brief's structural parts (the
non-negative head, the ladder hinge and floor-reaching data for the dial contract, integrity as a separate
flag) stay in force regardless of these outcomes; they are supported by earlier measurements. Speed and
qualification gates cover 512² and up.

## Step 1 results (2026-09-23, audited, landed at `c7355525`)

Three Opus lanes ran E1, E2a and E4 on existing data only. Each was preregistered and shows its work. The
coordinator re-ran their recompute commands, and E1's key cell reproduced exactly.

- **E1 — REFUTED.** zensim's deficit is *not* concentrated near threshold, for either the served default B
  or the best candidate (the 944 flagship C). Records: `benchmarks/rev4_e1_regime_2026-09-23.{md,json}`.
  - **B's deficits on JPEG-AIC are uniform across quality bands.** AIC-3 near-threshold / mid / low:
    −0.047 / −0.076 / −0.042, all with CIs excluding zero. B also collapses on the most distorted CSIQ band
    (−0.191 [−0.244, −0.137]; old bake era, to be rechecked on the served bytes).
  - **C shows no near-threshold deficit anywhere.** It trails SSIMULACRA2 in CID22-A's middle and lower
    bands (−0.012 to −0.019, CI excluding zero), and CVVDP at `standard_fhd` in the lowest AIC-4 band.
  - **Exploratory lead:** the losing cells are **cross-codec** ordering errors; same-codec ladder order ties
    in every band. E3 therefore drops to a side experiment, and the cross-codec lead gets its own
    confirmatory test (E1b).
- **E2a — facts established.**
  - Every CVVDP teacher and evaluation column used `standard_4k` (75.4 ppd). The CLI hard-coded it before
    zenmetrics `088f4bf5` (2026-05-25), and all the falsified CVVDP-teacher recipes drew on data scored
    before that commit.
  - Exceptions:
    - the `A_Phone` bake's teacher was CVVDP at `modern_oled_phone_indoor` (110 ppd, never
      conformance-checked);
    - the HDR teachers used a 1,000 cd/m² linear display at 4K geometry until 2026-08-06;
    - the AIC-4 row rescored at `standard_fhd` is unmerged.
  - **No human study we use sits at 75 ppd.** Stated or derivable human geometries are KonJND 24.3, LIVE
    26.8–33.5, CID22 46.9 and UPIQ-SDR about 51–57 ppd. KADID, TID, CSIQ, KonFiG, AIC-3/4 and SDR25 do not
    state a geometry.
  - The E2b design is proposed in `benchmarks/rev4_e2a_cvvdp_display_2026-09-23.md`:
    - **arms:** seven displays, including `sdr_fhd_24`, which separates geometry from peak luminance;
    - **selection:** on KADID/TID/KonFiG TRAIN only;
    - **margin:** pooled SROCC +0.010, with a Bonferroni 99% reference-clustered CI excluding zero on both
      KADID-train and TID-train;
    - **prerequisites:** landing the CVVDP display fix, rebuilding the pycvvdp environment, and pinning which
      AIC-3 study our labels come from.
- **E4 — NEGATIVE.** Agreement with the two-reference rule adds no held-out prediction beyond the registered
  composite on CID22-A, AIC-3, KonJND or CSIQ: every Bonferroni CI spans zero, over 19 lineages. Adding it
  *worsens* CID22-A prediction (−0.229 [−0.401, −0.012]). It predicts only the TRAIN-side A7r floors, which
  is SSIMULACRA2 consistency measured twice. **Agreement is not adopted as a gate.**
  - *Exploratory lead, not evidence:* butteraugli-only disagreement caught 45 of 46 measured contract
    failures, at a cost of 131 of 313 false rejects.
  - Record: `benchmarks/rev4_e4_agreement_gate_2026-09-23.{md,json}`.

## Step 2

- **E1b — cross-codec confirmation (analysis only, next).** Preregistered. On references scored under more
  than one codec (CID22-A, AIC-3, the AIC-4 sample, the forced-choice cross-codec questions), split
  same-reference pairs into same-codec and cross-codec. Compare pairwise ordering accuracy per metric, with
  reference-clustered CIs.
  - **Confirmed** if zensim's deficit to the best peer is CI-excluding-zero on cross-codec pairs, and ties
    or wins on same-codec pairs, on at least two corpora.
  - **What it changes:** if confirmed, cross-codec supervision becomes the core Rev4 codec-metric
    experiment (E7, to be designed): matched-quality cross-codec TRAIN pairs, plus E2b's teacher.
- **E2b** runs after its prerequisites land (the CVVDP display fix belongs to another lane's unmerged
  workspace).
- **E5a** (rendering corruptions and the testing metric) is independent and can start any time.

The paper is held until the program reports and a Rev4 candidate qualifies.
