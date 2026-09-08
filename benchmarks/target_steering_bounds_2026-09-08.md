# Attainable-target steering, September 8, 2026

The revised experiment measures bounds first, calibrates only on training
families, then compares the existing scalar controller with and without those
seeds at budgets 1, 2 and 3. **No native diffmap or product qualification is
claimed.** The [protocol](../docs/TARGET_STEERING_PROTOCOL_2026-09-08.md) owns the
method and the [structured result](target_steering_bounds_2026-09-08.json)
contains every model/policy/budget, class, endpoint/interior slice and paired
family delta.

## Inputs and chronology

- Canonical corpus: `imazen/imazen-26` at
  `187fbf338ce08e8e6654db7f04ddae58d5263da2`. The August 23 repository move,
  August 27 family split and August 30 measured URL index supersede older
  codec-corpus and digit-only status paragraphs. Sources satisfy both the
  digit and family split; no family crosses calibration/evaluation.
- 12 training origins (3 each photo/document/graphic/screen), 8 validation
  origins (2 each class), one existing 256-longest-side rendition per origin.
  Selection: SHA256 ordering of `target-steering-4004:<origin>` within class,
  unique families, no metric-based selection. This is a bounded instrument
  check, below the source/size coverage needed to ship a calibration.
- Canonical `cleanpicker-ladder11@2026-08-23` bytes are verified against its
  manifest. Existing rendering lineage is retained; no images were resized or
  re-decoded to construct a new dataset. The Rust reader interprets the PNGs
  as opaque RGB8 sRGB without ICC conversion.
- Existing Rust `check_holdout_overlap`: train-versus-validation nearest
  dHash distance **17**; train-versus-CID22-49 minimum **18**; no d<=10 flags.
  dHash is a screen, not proof against crop/content leakage. The historical
  metric models themselves do not acquire clean training provenance here;
  this tests the generalization of the seed calibration, not a new metric's
  held-out rank. In particular, old sharing-provenance annotations still apply.
- Scorers: named B/D and the complete Rust `BakeScorer` surface for
  `H_anchorlad_s4004_byid.bin`, SHA256
  `07f1e35874a5f3b253d8ac26ac4c9045afd2d4a12a413a0392ae7df968d20a3e`.
  This exact by-ID artifact is the identity; do not substitute another H file.
  `ZENSIM_FORMULA_REV=1`; `RAYON_NUM_THREADS=8`.
- Frozen release instrument SHA256
  `d7384c255e691e484c5f93afe6efb0f2d2280a24aa1f57fd7e1e9bcb9c0cd720`.
  Source and dependency heads, lockfile and binary are preserved with the run.
  Later changes are Clippy control-flow/type-alias cleanup and analyzer/docs;
  the frozen binary remains the reproduction authority.

## What was actually measured

Each of 20 sources × 3 codecs has a 21-probe ladder (1,260 full encodes).
JPEG q=0..100 uses 4:2:0; WebP q=0..100 uses method 4; JXL uses the wrapper's
fixed defaults and log-spaced distance 0.01..25. All three models see the same
reconstructions. Lossless and alternative chroma/effort/resampling settings
are separate, unmeasured families.

The calibration takes median score at each knob across the 12 training images,
records any monotone-envelope adjustments, then inverts a nonflat segment to
predict q and local slope. Shot 2 uses that training slope; shot 3 can use the
two measured points. Neither the bound scores nor the target's witness knob
enter the controller. There is no Python inference path.

For each validation image/codec/model, retain fixed requests
`[-10,30,70,90,99]` only when a reconstruction witnesses its ±1 band, and add
five attained ladder-score quantiles without rescaling. Endpoint and interior
statistics are separate. These model-specific target sets test policies on
the same model; they do **not** support comparing different models' aggregate
errors as if their target populations were identical.

The 360 fixed requests comprise **95 witnessed, 151 outside the measured
envelope and 114 unwitnessed inside it**. Sparse-sweep noncoverage is not proof
of impossibility. No excluded request is counted as a steering failure.
There are **29/72** validation image/codec/model ladders with at least one
adjacent score inversion above 1e-5: endpoint monotonicity cannot simply be
assumed. Saturated bitstreams are counted by SHA256, and every bound is scoped
to the measured configuration and grid.

The resulting **2,730 steering cells** include 162 negative-target cells.
They perform 4,503 full encode/decode/score probes, plus 2,730 independently
accounted re-encodes verifying emitted byte identity and achieved score.
Independent CPU SSIMULACRA2 and Butteraugli pnorm3 judge each emitted output.
Their values are retained; no matched-quality RD improvement is inferred.

## Results on the current default B

Training-calibrated policy; error units are the original zensim score:

| Codec | Requests/budget | 1-shot median | 2-shot median | 3-shot median | 3-shot p95 | 3-shot hits ±1 |
|---|---:|---:|---:|---:|---:|---:|
| JXL | 50 | 1.218 | 0.368 | 0.321 | 2.918 | 45/50 |
| JPEG | 50 | 3.202 | 0.595 | 0.496 | 2.527 | 40/50 |
| WebP | 47 | 2.905 | 0.618 | 0.223 | 4.353 | 40/47 |

Same-source/target three-shot midpoint controls: JXL median **21.878**, p95
34.310, hits 7/50; JPEG median **0.973**, p95 15.021, hits 25/50; WebP median
**0.068**, p95 31.136, hits 25/47. WebP's median therefore gets **worse** while
its hit rate and tail improve substantially. The witness grid includes native
q=50, favoring exact midpoint hits; a median alone misses the failures elsewhere.
The train-curve policy reduces mean absolute error within **all eight
validation families**, for each codec and each budget. This is descriptive
paired evidence, not a significance or generalization guarantee from eight
families.

Endpoint gains do not explain the whole result: B's three-shot **interior**
median/p95 errors are JXL **0.368/1.890**, JPEG **0.565/2.527**, WebP
**0.543/2.855**. D and H have their own full tables in the JSON. Across all
trained-policy three-shot cells, the worst error is still **13.719**.
No universal perceptual tolerance has been established; ±1 is an explicit
instrument stopping band, and ±0.5/±1/±2 hit rates are descriptive.

Calibration ran in 6 seconds; validation including bound sweeps, verification
and independent judges ran in 67 seconds. Process peak RSS was about 40 MiB
under the capped runner. These are single-run operating observations at small
geometry, not quiet/repeated production performance claims. Per-cell timings
and separate oracle/verification costs are in the raw rows.

## Native diffmaps: existing code and remaining work

The refreshed branch inventory is in the protocol. **These measurements use
the generic outer scalar loop, including JXL; no candidate map drives these
encodes.** Actual native owners already exist:

1. JXL: quant-field redistribution in `zensim_loop.rs`. Before a new native
   experiment, incorporate the September 7 decoder/reconstruction/DC-precision
   fixes (`df297ea1`, `bff0a5c8`, `61dccb75`) from current main. The local
   encoder used here predates those fixes; it is only externally decoded in
   this scalar experiment. Rejected issue103 feedback/coordinate branches are
   preserved negative research, not ready integration candidates.
2. JPEG: `target_quality.rs` outer secant and `encode/zq.rs` block-AQ loop.
   The latter's zensim dependency is pinned to `9d8f73a5`, not this working
   tree's new `BakeScorer`. Reconcile model identity and total pass count;
   `max_passes` and initial/correction passes cannot be assumed synonymous.
3. WebP: `encoder/zensim_target.rs` segment correction behind `target-zensim`,
   currently depending on zensim **0.2**. Correction is conditional on the
   score gap and existing secant state; require engagement traces. One-pass
   output carries NaN score/optimistic success, so final output must be judged
   independently rather than trusting that flag.
4. AVIF: `two_pass_zensim.rs` already documents global plus spatial correction,
   quantizer-lattice corrections and negative two-pass studies. Follow after
   JXL, with JPEG/WebP map work included. Fractional quality accesses more
   quantizers than an integer-quality sweep; preserve that correction.

The complete composed-model map API is still a serving gap: `BakeScorer`
currently serves complete scalars, whereas old native gradient mounts do not
establish support for every head/spline/ensemble/corruption composition. Close
that gap through a zensim Rust surface, verify gradient/map/scalar parity and
map engagement, then run native active-versus-neutral maps with honest inner
compare/outer encode counts and independent judges. This experiment supplies
the feasibility/calibration control for that work; it does not finish it.

## Validation and reproduction

Six Rust tests pass, including a real JPEG seed-consumption test observed to
fail when the first probe was forced back to midpoint. Four analyzer tests
pass, including refusal of missing/duplicate cells and an unwitnessed target
inside a measured envelope. Standalone all-feature/all-target Clippy, root
CI-exact Clippy, supported API snapshot checks, formatting and script lint pass.
No metric/training implementation changed or model default was promoted.

Full artifacts: `/mnt/v/output/zensim/target-steering-2026-09-08/`.
`SOURCE_PROVENANCE.json`, `BUILD.json`, `canonical/`, `train.json`,
`validate.json`, both dHash TSVs, `bin/`, `calibration/` and `validation/`
retain identities and results. `COMPLETE` is written only after every requested
cell succeeds. Use the frozen binary and exact arguments from INPUTS/BUILD;
the README documents the same two-stage fit/evaluate commands. The final
manifest binds all artifacts by SHA256.
