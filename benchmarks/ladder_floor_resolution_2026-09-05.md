# Is the ladder instrument's bottom triplet resolvable, or is A7r's jpeg/rav1e bar noise? (2026-09-05)

**Report-only lane.** No rule, registry, or default changed. Reads
[`ladder_instrument_2026-09-05.md`](ladder_instrument_2026-09-05.md) §4–§9 and
[`dial_addressability_gate_2026-09-04.md`](dial_addressability_gate_2026-09-04.md) §16
(the `A7r` floor-representability rule this all runs on), then asks one question about
the new instrument's own bars: **jpeg 0.5385 and `avif-rav1e` 0.5385 are the two lowest
mentor bars on the grid, and shipped Profile D misses jpeg by exactly one ladder
(0.5128 vs 0.5385) — is that gap a codec genuinely failing to resolve its own lowest
settings, or an artifact of grading three positionally-lowest steps that happen to sit on
a near-flat part of the curve?**

Short answer: **mostly real, not noise — but D's specific one-ladder jpeg miss is exactly
a boundary artifact, and fixing the boundary trades it for a previously-invisible
`avif-rav1e` defect.** Details below.

## 1. Instrument and method

Instrument: `/mnt/v/output/zensim/ladder-2026-09-05/instruments/` —
`dial_grid_372col_ladder.parquet` (9,593 distinct-setting rows × 372 features, sha256
`4c3874a78…`), `dialcells_ssim2_ladder.tsv` (the mentor `peer_ssim2`'s own per-cell score,
same 9,593 rows), `ladder_grid_ladder_full.parquet` (11,466 rows = every step incl.
saturated duplicates, with `encoded_bytes` / `score_ssim2` / `encode_sha`). 39 ladders per
codec (jpeg, webp, `avif-svt`, `avif-rav1e`), 26 for jxl (13 excluded as truncated-floor
per `imazen/jxl-encoder#101`, ladder-instrument doc §8.1).

**Owner tools, not a re-implementation of the scored numbers.** `bake_verdict` (rebuilt at
HEAD `927d9292` — the checked-out `target/release` copy predated `--gaddr-grid-truth` and
had to be rebuilt in this lane's own sibling workspace) via `scripts/dialgate_arms.sh score
<label> <bake.bin>` under `ZL_ERA=ladder`, with `ZENSIM_DIAL_PRED_OUT=<tsv>` added to dump
the exact `image_id\tcodec\tq\tpred` values `FloorMeasure::from_grid` scores — the SAME
scoring units `bake_verdict`'s own A7r uses (`score_row`, post-spline). **Validated before
trusting anything downstream**: re-running `bake_verdict` this way reproduces every
published A7r fraction in `ladder_instrument_2026-09-05.md` §9/§9.0 exactly (mentor jpeg
0.5385, `avif-rav1e` 0.5385, jxl 0.9231, `avif-svt`/webp 1.0; shipped D jpeg 20/39=0.5128,
D's other four codecs at or above the mentor). A1/A3 checks were also bit-identical to the
doc (`99.99996372112122` / `93.88421311743264`).

**Why a Python port of the rule was necessary at all.** The two REPORT variants below
(task items 2a/2b) each choose a *different* per-ladder step window than the pinned
`bottom_k=3` rule, and no CLI flag re-windows an existing bake's grading — grading a new
window is a new measurement, not a re-read. `zensim-validate/src/dial_addressability.rs::
FloorMeasure::from_grid` was read from source (not guessed) and ported line-for-line in
Python: `grid_min`/`n_ladders_at_min` computed ONCE per scorer over the whole instrument
(all 5 codecs pooled, matching the Rust struct's own field), `ordered` = strict increase
across the chosen window, `clamped` = any of the window's bottom-K values within
`clamp_eps=1e-9` of `grid_min` unless this ladder is the sole holder. The port reproduced
every current-rule bar and every shipped-D fraction to the same 4th decimal before it was
used on anything new — see §5.

Data + scripts: `/mnt/v/output/zensim/ladder-2026-09-05/floorres/` (`dumps/` — 6 per-cell
`bake_verdict` dumps; `tables/` — every TSV below; `verdicts/` + `verdicts_repeat/` — the
raw `bake_verdict` JSON, including the determinism-check repeat run; `logs/`). Analysis
script: `~/tmp/floorres_analysis.py` (reproduction command in §7; not committed — scratch,
per policy — but its exact steps are in this file).

## 2. Finding 1 — adjacent-step deltas at the floor: mostly real motion, not noise

For every ladder's four lowest DISTINCT settings (positions 0,1,2,3 — the window
`bottom_k=3`'s "ordered" check actually spans), the three consecutive deltas in
`peer_ssim2` and in shipped D's own dial, plus the encoded-byte ratio at the same steps:

| scorer | codec | n deltas | median Δ | p10 Δ | p90 Δ | frac(abs Δ < 0.25) | frac(abs Δ < 0.5) | frac(abs Δ < 1.0) | bytes ratio (median) |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| peer_ssim2 | avif-rav1e | 117 | 1.025 | −1.081 | 2.945 | 0.111 | 0.171 | 0.385 | 1.016 |
| peer_ssim2 | avif-svt | 117 | 6.443 | 3.250 | 11.715 | 0.009 | 0.009 | 0.034 | 1.158 |
| peer_ssim2 | jpeg | 117 | 0.426 | −1.084 | 2.428 | 0.179 | 0.393 | 0.632 | 1.007 |
| peer_ssim2 | jxl | 78 | 1.454 | 0.533 | 2.217 | 0.013 | 0.064 | 0.295 | 1.032 |
| peer_ssim2 | webp | 117 | 5.609 | 2.477 | 23.415 | 0.000 | 0.000 | 0.000 | 1.095 |
| D_shipped | avif-rav1e | 117 | 1.100 | −0.658 | 2.770 | 0.085 | 0.188 | 0.402 | 1.016 |
| D_shipped | jpeg | 117 | 0.379 | −0.342 | 1.943 | 0.222 | 0.521 | 0.761 | 1.007 |

Full table (all 5 codecs × 2 scorers): `tables/adjacent_step_deltas_summary.tsv`.

**Reading.** jpeg and `avif-rav1e` are the two codecs whose per-step ssim2 motion is
smallest (median 0.43 / 1.03 points, vs jxl 1.45, webp 5.6, `avif-svt` 6.4) **and** whose
per-step encoded-byte growth is smallest (median +0.7% / +1.6%, vs jxl +3%, webp +9.5%,
`avif-svt` +15.8%). That correlation is the mechanism, not a coincidence: at the very
bottom of jpeg's and `avif-rav1e`'s quality dials the encoder's own RD curve is nearly
flat in both bytes and quality, so a "next configurable setting" barely moves either
axis — which is exactly the regime where a strict monotonicity test is most exposed to
real (if small) non-monotonicity. It is NOT the regime where the settings collapse to
byte-identical duplicates — those are already removed as `saturated`; every row here is a
genuinely distinct bitstream (different `encode_sha`).

## 3. Finding 2 — classifying the mentor's own bottom-triplet failures: 78%/67% are genuine inversions, not ties

For each of `peer_ssim2`'s own failing ladders (the ones that make its bar `<1.0`), the
worst (most negative) of its three floor-window deltas, and whether `|Δ| < 0.5` (tie-
within-noise, per the task's own threshold) or `Δ ≤ −0.5` (genuine inversion):

| codec | n fail | genuine inversion (>0.5 wrong way) | tie-within-noise (abs Δ < 0.5) | worst-Δ median | worst-Δ min | bytes ratio at worst step (median) |
|---|--:|--:|--:|--:|--:|--:|
| avif-rav1e | 18 | **12 (67%)** | 6 (33%) | −1.430 | −6.204 | 1.008 |
| jpeg | 18 | **14 (78%)** | 4 (22%) | −1.289 | −12.132 | 1.006 |
| jxl | 2 | 1 (50%) | 1 (50%) | −0.889 | −1.509 | 1.024 |

(`avif-svt`, webp: 0 failures — mentor is 1.0 on both, nothing to classify.) Full rows:
`tables/mentor_failure_classification.tsv`; shipped-D's own classification on the same
ladders is `tables/D_failure_classification.tsv` (jpeg 8 genuine/11 noise of 19 fails,
`avif-rav1e` 11 genuine/7 noise of 18, jxl 0/1 of 1).

**Reading.** This is the direct answer to "noise or real": **on the mentor's own bottom
triplet, most jpeg and `avif-rav1e` failures are real, non-trivial ssim2 decreases** —
median magnitude 1.29 / 1.43 points, more than 2.5× the task's own 0.5-point noise
threshold, at essentially flat bitrate (median +0.6%/+0.8% bytes at the failing step
itself). ssim2 genuinely goes DOWN as the codec's nominal quality setting goes up, on
most of these ladders, not just wobbles within measurement precision. The minority class
(22–33% of failures, tie-within-noise) is real too, just smaller. jxl's n=2 is too small
to read a rate from.

## 4. Finding 2 — A7r re-graded under two windowing variants: D's jpeg gap dissolves, a new `avif-rav1e` gap appears

**Variant (a)** — walk each ladder from its lowest setting, skipping forward past any
step whose `|Δssim2|` from the last SELECTED step is `< 0.5`, until 4 mentor-resolvable
steps are collected (or the ladder is too short — none were, on this instrument).
**Variant (b)** — the lowest setting, plus the step whose ssim2 is closest to `+2.0`
above it, plus the step whose ssim2 is closest to `+5.0` above it (both drawn from
settings above the floor, then the 3 re-sorted by `q` for the ordering test). Both reuse
the SAME `ordered` + `off-clamp` test as the pinned rule, and the SAME per-scorer
`grid_min`/`n_ladders_at_min` (an instrument-wide property, unaffected by which window a
variant tests).

Represented fraction, all 7 scorers, all 3 rules:

| scorer | rule | avif-rav1e | avif-svt | jpeg | jxl | webp |
|---|---|--:|--:|--:|--:|--:|
| **peer_ssim2 (bar)** | current | 0.5385 | 1.0000 | 0.5385 | 0.9231 | 1.0000 |
| Profile D (shipped) | current | 0.5385 ✓ | 1.0000 ✓ | **0.5128 ✗** | 0.9615 ✓ | 1.0000 ✓ |
| **peer_ssim2 (bar)** | variant (a) | 0.6410 | 1.0000 | 0.6667 | 0.9615 | 1.0000 |
| Profile D (shipped) | variant (a) | **0.6667 ✓** | 1.0000 ✓ | **0.6667 ✓** | 1.0000 ✓ | 1.0000 ✓ |
| **peer_ssim2 (bar)** | variant (b) | 0.9744 | 1.0000 | 0.8974 | 0.9615 | 1.0000 |
| Profile D (shipped) | variant (b) | **0.9487 ✗** | 1.0000 ✓ | **0.9231 ✓** | 1.0000 ✓ | 1.0000 ✓ |

Full 7-scorer × 3-rule table (incl. D-previous, Profile A/B, `lam1em3`, `minus_f162`):
`tables/a7r_all_variants.tsv`; PASS/FAIL matrix: `tables/a7r_passfail_all_variants.tsv`.

Codecs passing (of 5), per scorer, current → (a) → (b):

| scorer | current | variant (a) | variant (b) |
|---|--:|--:|--:|
| peer_ssim2 (bar, trivially 5/5) | 5 | 5 | 5 |
| **Profile D (shipped)** | **4** | **5** | **4** |
| Profile D (previous) | 4 | 5 | 4 |
| Profile A | 1 | 1 | 2 |
| Profile B | 0 | 0 | 1 |
| lam1em3 | 2 | 2 | 4 |
| minus_f162 | 1 | 2 | 3 |

**Reading.** Variant (a) — grade only steps the mentor can itself tell apart by ≥0.5
points — makes shipped D **PASS all 5 codecs**, jpeg included (0.6667 vs the mentor's own
0.6667 at that window, exact parity), and jxl/`avif-rav1e` improve too. **D's one-ladder
jpeg miss under the pinned rule is exactly a boundary artifact of grading the literal
lowest 3 positions rather than 3 the mentor can resolve** — it is not a broader ordering
defect: every ladder that failed under the pinned rule failed inside the near-flat zone
Finding 1/2 identified, and skipping that zone cures it completely. Variant (b) — force
genuine separation (+2/+5 ssim2 points) rather than skip-to-next-resolvable — also cures
jpeg (D now *exceeds* the mentor, 0.9231 vs 0.8974) but **newly fails `avif-rav1e`**
(0.9487 vs the mentor's 0.9744, one ladder short) — a gap the pinned rule and variant (a)
both hide, because under both of those D and the mentor happen to land on the exact same
represented-count on `avif-rav1e` (they are not failing the *same* ladders — Finding 2
shows the mentor's own `avif-rav1e` failures are 67% genuine inversions, so at least some
of "D matches the mentor's count" is two different codecs of noise landing on the same
tally, not D actually tracking the mentor's ranking). **So: the jpeg gap is provably a
boundary artifact and disappears under either alternative window; the `avif-rav1e`
picture is murkier — mentor's own floor is mostly-real non-monotonicity, and testing a
cleanly-separated window shows D's ordering there is measurably worse than the mentor's,
not merely equally noisy.** `minus_f162`, built to cure a JXL inversion measured on the
OLDER canonical/preC/postC dial grid (`dial_addressability_gate_2026-09-04.md` §16, per
`d_peaks_slot_ablation_2026-09-05.md`'s ablation), does **not** generalize to this newer,
denser instrument — it fails jxl under all three rules here (0.846/0.846/1.000
current/a/b vs the mentor's 0.923/0.962/0.962) — the failure moved rather than closed.

## 5. Finding 3 — D's A1/A3 misses: deterministic calibration residuals, not noise

**A1 (pooled dial max, 99.99996372112122 vs bar 100.0, gap 3.6×10⁻⁵).** Exactly **9** of
the 9,593 cells carry `ssim2 = 100.0` to the stored precision, **all `avif-rav1e` at
`q=99.9`** (one per source image; `q=100` is a `saturated` duplicate of `q=99.9` on every
one of them — same `encode_sha`, confirmed in the full-archive table). Verified pixel
identity directly (decoded PNG vs reference PNG, `numpy.array_equal`, not inferred from
the score): **all 9 are byte-for-byte pixel-identical to their reference**, 0 differing
pixels out of up to 835,996 per image; the PNG *container* bytes differ (different
codec/compression) but the decoded RGB arrays do not. Their 372-feature vectors are
**exactly all-zero** — every one of 372 slots, all 9 rows — which is why D scores all 9
images identically (`99.999964...`, D's own reported pooled max): an additive model at an
all-zero input returns a pure constant (its bias/spline-at-zero), independent of which
image produced the zero vector. **D's A1 gap is therefore a single scalar property of the
bake — its own output-spline value at the identity point — not an image-specific or
extraction-specific event.** Confirmed deterministic: an independent repeat
`bake_verdict` run (fresh process, same binary) reproduced `99.99996372112122` to every
printed digit (`verdicts_repeat/gaddr_D_shipped.json`).

**A3 (robust ceiling, dial p95 93.88421311743264 vs bar 93.9743354, gap 0.090).** This is
a genuine two-distribution percentile gap, not a single-cell event: the rows neighboring
`peer_ssim2`'s p95 rank (9112/9593, interpolated) span jxl/jpeg/`avif-rav1e` at q
96.5–99.8 with real (non-zero) feature content, and D's dial values AT those specific
cells are scattered 92.4–96.9 — because D is a *different* model from ssim2, not a
rescaling of it, two independently-sorted distributions' 95th percentiles need not
coincide even when the underlying rankings mostly agree. Of the 481 cells at/above
`peer_ssim2`'s p95 threshold, only 9 (the same all-zero avif-rav1e cells from A1) are
degenerate; the rest are ordinary near-ceiling cells across all 5 codecs (avif-rav1e 293,
jxl 133, jpeg 38, avif-svt 14, webp 3). A3's 0.09-point gap is a real, small,
distribution-level calibration shortfall in the near-ceiling region — also confirmed
bit-identical across the repeat run.

**Is either gap within the instrument's own noise floor?** No, in the sense that matters:
there IS no floor to be within. The ladder-instrument's reproducibility gate
(§8.0 of `ladder_instrument_2026-09-05.md`; the task brief's "§6" does not match this
doc's numbering — the gate is §8.0) re-ran the jpeg leg from scratch and found
**2,574/2,574 identical `encoded_bytes`, 2,574/2,574 identical ssim2 to the exact printed
string, 300/300 identical bitstream sha256** — i.e., the pipeline that produced this
instrument has ZERO measured stochastic component at printed-string precision. A1/A3's
`3.6×10⁻⁵` and `0.090` gaps are not fluctuations that a re-run could land differently —
this report's own repeat run proved that directly, bit-for-bit — they are fixed
properties of D's spline evaluated on this instrument's fixed cells. A1 is best read as
"D's identity calibration is correct to 5 significant figures, not to the exact double";
A3 is best read as "D's overall dial distribution's 95th percentile sits 0.09 points
below ssim2's, a real if small shape mismatch near the ceiling."

## 6. Answer to the posed question

**Predominantly real, not noise — with a caveat that cuts against a simple "it's fine"
reading.** The mentor's own bottom-triplet failures on jpeg (78% genuine inversions,
median 1.29 ssim2 points) and `avif-rav1e` (67%, median 1.43 points) are mostly real,
non-trivial RD-curve non-monotonicity at flat bitrate, not measurement noise — confirmed
by a deterministic, bit-exact instrument. **D's specific one-ladder jpeg miss under the
pinned bottom-3 rule, however, IS a boundary artifact**: it evaporates completely
(5/5 codecs pass, jpeg exact parity with the mentor) once the window is moved to steps
that are actually resolvable, by either of two independent constructions (skip-to-
resolvable, or force-fixed-separation). **The same fix does not clear D on `avif-rav1e`
under the fixed-separation window** — it newly fails there by one ladder, a defect the
pinned rule's shared noise floor between D and the mentor was hiding. Net: the pinned
rule's jpeg bar is measuring something largely real but grading D on the wrong ladder for
the wrong reason; its `avif-rav1e` bar is measuring something largely real AND correctly
flagging D as under-performing it, just via a noisier route than necessary.

## 7. Reproduction

```sh
# rebuild bake_verdict at HEAD (the checked-out target/release predated --gaddr-grid-truth)
cd ~/work/zen/zensim   # or any sibling workspace at the same commit
cargo build --release -p zensim-validate --bin bake_verdict --bin bake_dial_refit

# dump per-cell dial predictions for a bake on the ladder instrument (score units,
# the same the pinned A7r reads)
ZL_ERA=ladder ZL_BV=$PWD/target/release/bake_verdict ZL_BDR=$PWD/target/release/bake_dial_refit \
  ZENSIM_DIAL_PRED_OUT=/tmp/dshipped.tsv \
  scripts/dialgate_arms.sh score D_shipped zensim/weights/d_sdr_add156_id100_negrich_dial_2026-09-05.bin

# the mentor's own per-cell scores are already at:
#   /mnt/v/output/zensim/ladder-2026-09-05/instruments/dialcells_ssim2_ladder.tsv

# re-run the full analysis (variants, classification, A1/A3 cells):
python3 ~/tmp/floorres_analysis.py
```

Bake paths used: D shipped = `zensim/weights/d_sdr_add156_id100_negrich_dial_2026-09-05.bin`
(sha256 `921a8f67…`); D previous = `zensim/weights/d_sdr_add156_dense_dial_2026-08-31.bin`;
Profile A = `zensim/weights/v47_strict_qat_native_2026-05-27.bin`; Profile B =
`zensim/weights/b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin`; `lam1em3` =
`/mnt/v/output/zensim/dpeaks372-2026-09-05/sweep/bakes/Dsweep_lam1em3_dial.bin`;
`minus_f162` = `/mnt/v/output/zensim/dpeaks372-2026-09-05/slots/bakes/minus_f162_dial.bin`.

Artifacts: `/mnt/v/output/zensim/ladder-2026-09-05/floorres/` (`dumps/`, `tables/`,
`verdicts/`, `verdicts_repeat/`, `logs/`). Nothing installed; `zensim/weights/` untouched;
`benchmarks/eval_annotations.json` and `benchmarks/dial_addressability_floor_2026-09-04.json`
untouched (no registry write — this is a report, not a bar change).

## 8. Floor-rule variants, owner-computed (2026-09-06)

**Owner-extension, opt-in.** The two windows this report derived by porting
`FloorMeasure::from_grid` into Python (§4) are now a `--floor-rule` flag on
`bake_verdict` itself — `zensim_validate::dial_addressability::FloorRule` —
so the comparison is a measurement through the owner, not a one-off script
kept in sync by hand. **Nothing installed changes**: the registry
(`benchmarks/dial_addressability_floor_2026-09-04.json`), `zensim/weights/`,
and every existing board row are byte-untouched.

### 8.1 The flag

```
--floor-rule {distinct|resolvable|spaced}   default: distinct
--floor-margin <f64>                        default: 0.5 (resolvable only)
```

- **`distinct`** (default) — the PINNED rule: literal positions `0..=K` by
  quality. Bar = the registry-pinned mentor fraction, exactly as every prior
  `bake_verdict` invocation reads it.
- **`resolvable`** — variant (a): walk forward from the lowest setting,
  skipping any step whose `|Δ mentor|` from the last SELECTED step is below
  `--floor-margin`, until `K+1` mentor-resolvable steps are collected.
  REQUIRES `--gaddr-grid-truth` — the window AND the bar are both computed
  from the mentor's own per-cell scores on the SAME instrument.
- **`spaced`** — variant (b): the lowest setting, plus the steps nearest
  `+2.0` and `+5.0` mentor points above it, re-sorted by quality. Also
  REQUIRES `--gaddr-grid-truth`.

`resolvable`/`spaced` have no registry entry — they are report-derived
windows, never pinned like `distinct` — so their bar is **ALWAYS
LIVE-computed**: the mentor is graded against its own per-cell truth through
the identical `FloorMeasure::from_grid_with_rule` call the candidate went
through (`per_codec_floor_rows_live`, never `per_codec_floor_rows`'s registry
lookup). Every row and note is stamped `rule=<tag>` in both the markdown and
the JSON (`"floor_rule"` / `"floor_rule_params"` fields) so a `resolvable`
fraction can never be silently compared against a `distinct` or `spaced` one.
Omitting `--gaddr-grid-truth` under `resolvable`/`spaced` is a REFUSAL at
argument-parse time (`bake_verdict: --floor-rule resolvable requires
--gaddr-grid-truth: …`), never a silent fall-back to `distinct`'s window.

### 8.2 Golden-identity proof for `distinct` (the default)

Three independent checks, all clean:

1. **By construction.** `FloorMeasure::from_grid` (the legacy,
   unparameterized entry point every pre-2026-09-06 caller uses) now
   DELEGATES to `from_grid_with_rule(..., FloorRule::Distinct)` — it is the
   same code, not a fork kept in sync by hand. `Distinct`'s branch reduces
   algebraically to the pre-existing body: window = `[0, 1, .., K]`, `ordered`
   = strictly increasing across all `K+1` points (the same `K` pairs the old
   code checked), `clamped` = the bottom `K` of that same window against the
   same `grid_min`/`n_ladders_at_min` (computed identically, over the whole
   ladder, unchanged).
2. **In-suite regression test.** `dial_addressability::tests::
   distinct_rule_matches_legacy_from_grid_on_every_existing_fixture` calls
   both entry points on six scenarios already exercised elsewhere in the
   suite (clean, tied, inverted-into-next-step, collapsed floor, sole holder,
   too-short) and asserts FIELD-IDENTICAL `FloorMeasure`s (`to_bits()`
   equality on every float). All 40 `dial_addressability` tests pass,
   including 12 new ones; the FULL `zensim-validate` suite (234 lib tests +
   every integration test file) passes unchanged — `cargo test -p
   zensim-validate` reports 0 failures anywhere in the crate.
3. **CLI-level, against a PRISTINE pre-change binary.** A throwaway sibling
   workspace was built at the unmodified base commit (`29c198af`, this
   report's own commit, BEFORE any of §8's code existed) and run with the
   identical invocation used below (shipped D, the registered CANONICAL
   372 grid, `--corpora cid22`, `--gaddr-json`). Diffing the JSON
   programmatically (key-by-key, not textually):

   ```
   keys only in the post-change run (additive):  ['floor_rule', 'floor_rule_params']
   keys only in the pre-change run (a removal):  []
   keys in both whose VALUE differs:             []
   ```

   Every existing field — every `A1`-`A7r` value, every note, every SROCC,
   the whole rank panel — is **byte-identical** to the pristine baseline.
   The only difference is the two new, purely informational fields this
   flag adds. The markdown report differs only in the wall-time line (the
   same non-determinism this repo's other "byte-identical" claims already
   carry — e.g. the 2026-08-30 default-root flip verification). The
   throwaway workspace was `jj workspace forget`-ten and deleted immediately
   after use.

### 8.3 The owner-computed per-codec table

Run on the `ladder` instrument (`dial_grid_372col_ladder.parquet`, sha256
`4c3874a78…` — the SAME instrument and cells §1-§7 measured), via
`scripts/dialgate_arms.sh score <label> <bake> 372` under `ZL_ERA=ladder
ZL_FLOORRULE=<rule>` for the six bakes, and `bake_verdict --dial-peer-scores
peer_ssim2=… --floor-rule <rule>` (peer mode, THROUGH THE OWNER) for the
mentor itself. `--floor-margin`/the `spaced` offsets are the flag's own
defaults (0.5 / +2.0 / +5.0), matching §4 exactly.

| scorer | rule | avif-rav1e | avif-svt | jpeg | jxl | webp | pass/5 |
|---|---|--:|--:|--:|--:|--:|--:|
| **peer\_ssim2 (bar)** | distinct | 0.5385 | 1.0000 | 0.5385 | 0.9231 | 1.0000 | 5/5 |
| **peer\_ssim2 (bar)** | resolvable | 0.6410 | 1.0000 | 0.6667 | 0.9615 | 1.0000 | 5/5 |
| **peer\_ssim2 (bar)** | spaced | 0.9744 | 1.0000 | 0.8974 | 0.9615 | 1.0000 | 5/5 |
| **Profile D (shipped)** | distinct | **0.5385** ✓ | **1.0000** ✓ | 0.5128 ✗ | **0.9615** ✓ | **1.0000** ✓ | 4/5 |
| **Profile D (shipped)** | resolvable | **0.6667** ✓ | **1.0000** ✓ | **0.6667** ✓ | **1.0000** ✓ | **1.0000** ✓ | 5/5 |
| **Profile D (shipped)** | spaced | 0.9487 ✗ | **1.0000** ✓ | **0.9231** ✓ | **1.0000** ✓ | **1.0000** ✓ | 4/5 |
| Profile D (previous) | distinct | **0.5385** ✓ | **1.0000** ✓ | 0.5128 ✗ | **0.9615** ✓ | **1.0000** ✓ | 4/5 |
| Profile D (previous) | resolvable | **0.6667** ✓ | **1.0000** ✓ | **0.6667** ✓ | **1.0000** ✓ | **1.0000** ✓ | 5/5 |
| Profile D (previous) | spaced | 0.9487 ✗ | **1.0000** ✓ | **0.9231** ✓ | **1.0000** ✓ | **1.0000** ✓ | 4/5 |
| Profile A | distinct | 0.3333 ✗ | 0.8462 ✗ | 0.3846 ✗ | 0.8077 ✗ | **1.0000** ✓ | 1/5 |
| Profile A | resolvable | 0.3590 ✗ | 0.8462 ✗ | 0.5128 ✗ | 0.8462 ✗ | **1.0000** ✓ | 1/5 |
| Profile A | spaced | 0.7179 ✗ | 0.9487 ✗ | 0.8462 ✗ | **0.9615** ✓ | **1.0000** ✓ | 2/5 |
| Profile B | distinct | 0.2051 ✗ | 0.4359 ✗ | 0.3333 ✗ | 0.3846 ✗ | 0.9487 ✗ | 0/5 |
| Profile B | resolvable | 0.1795 ✗ | 0.4359 ✗ | 0.5641 ✗ | 0.4231 ✗ | 0.9487 ✗ | 0/5 |
| Profile B | spaced | 0.4359 ✗ | 0.5897 ✗ | 0.8205 ✗ | 0.6923 ✗ | **1.0000** ✓ | 1/5 |
| lam1em3 | distinct | 0.2564 ✗ | **1.0000** ✓ | 0.4615 ✗ | 0.7308 ✗ | **1.0000** ✓ | 2/5 |
| lam1em3 | resolvable | 0.2564 ✗ | **1.0000** ✓ | 0.5641 ✗ | 0.8077 ✗ | **1.0000** ✓ | 2/5 |
| lam1em3 | spaced | 0.7692 ✗ | **1.0000** ✓ | **0.9231** ✓ | **1.0000** ✓ | **1.0000** ✓ | 4/5 |
| **minus\_f162** | distinct | 0.4615 ✗ | 0.9744 ✗ | 0.4615 ✗ | 0.8462 ✗ | **1.0000** ✓ | 1/5 |
| **minus\_f162** | resolvable | 0.5385 ✗ | 0.9744 ✗ | **0.6667** ✓ | 0.8462 ✗ | **1.0000** ✓ | 2/5 |
| **minus\_f162** | spaced | 0.8718 ✗ | **1.0000** ✓ | 0.8718 ✗ | **1.0000** ✓ | **1.0000** ✓ | 3/5 |

*(✓/✗ mark PASS/FAIL against the mentor's own LIVE fraction on that rule;
peer\_ssim2 is the bar itself, trivially 5/5 on every rule; bold = the row the
task asked to highlight — mentor bar, shipped D, `minus_f162`.)*

**Reproduces the report's own §4 numbers exactly.** The analysis lane's raw
TSV (`tables/a7r_all_variants.tsv`, still on disk) carries all 105 cells
(7 scorers × 3 rules × 5 codecs) at full `f64` precision. Diffed
programmatically against this run's owner-computed values:

```
compared 105 cells, 0 mismatches, max|diff| = 0.000e+00
```

Every digit of §4's peer\_ssim2 and shipped-D rows, and every "codecs
passing" count for all seven scorers (including Profile A/B, `lam1em3`,
`minus_f162`, which §4's inline markdown only reported as pass-counts) —
BIT-IDENTICAL. This is independent confirmation, through the compiled owner
tool rather than the hand-rolled Python port, of §4's finding: `resolvable`
clears shipped D's jpeg miss (4/5 → 5/5) at the cost of nothing, `spaced`
clears jpeg but opens a NEW `avif-rav1e` miss (4/5, but a DIFFERENT failing
codec than `distinct`'s), and `minus_f162` — built to cure a JXL inversion
on a DIFFERENT, older grid — does not generalize here (1/5 → 2/5 → 3/5,
improving under both alternatives but never catching up to shipped D).

### 8.4 Reproduction

```sh
cargo build --release -p zensim-validate --bin bake_verdict

L=/mnt/v/output/zensim/ladder-2026-09-05/instruments
export ZL_BV=$PWD/target/release/bake_verdict ZL_ERA=ladder

# the mentor itself, through the owner (peer mode) — --bake is a required
# but UNUSED placeholder in peer mode; the DIAL/G-ADDR block is entirely
# overridden by --dial-peer-scores
for RULE in distinct resolvable spaced; do
  "$ZL_BV" --bake zensim/weights/d_sdr_add156_id100_negrich_dial_2026-09-05.bin \
    --dial-peer-scores "peer_ssim2=$L/dialcells_ssim2_ladder.tsv" \
    --dial-grid "$L/dial_grid_372col_ladder.parquet" \
    --gaddr-grid-truth "$L/dialcells_ssim2_ladder.tsv" \
    --floor-rule "$RULE" --corpora cid22 \
    --gaddr-json /tmp/gaddr_peer_ssim2_$RULE.json
done

# the six bakes, through dialgate_arms.sh (now threads --floor-rule)
for RULE in distinct resolvable spaced; do
  ZL_FLOORRULE=$RULE scripts/dialgate_arms.sh score D_shipped \
    zensim/weights/d_sdr_add156_id100_negrich_dial_2026-09-05.bin 372
done

# the fair board, under a non-default rule (opt-in; default stays `distinct`)
scripts/gaddr_board_regrade.sh grade   # honors ZL_FLOORRULE / ZL_FLOORMARGIN
```

### 8.5 What landed

- `zensim-validate/src/dial_addressability.rs` — `FloorRule` (`Distinct` /
  `Resolvable { margin }` / `Spaced { near_lo, near_hi }`, `Default =
  Distinct`), `FloorRuleContext`, `FloorMeasure::from_grid_with_rule` (the one
  implementation; `from_grid` delegates), `resolvable_window` /
  `spaced_window`, `per_codec_floor_rows_live` (the LIVE-bar sibling of
  `per_codec_floor_rows`, never registry-backed), `Verdict.floor_rule` +
  `to_json`/`render_markdown` stamping. 12 new tests, including the two the
  task named explicitly:
  `resolvable_skips_a_mentor_near_tie_that_distinct_fails_on` (a synthetic
  ladder where the mentor inverts by 0.2 at the bottom — `distinct` fails the
  ordered dial, `resolvable` skips the unresolvable step and passes) and
  `collapsed_floor_fails_under_every_rule` (two ladders sharing the
  instrument minimum fail under all three rules, because position 0 — where
  both touch the floor — is in every rule's window).
- `zensim-validate/src/bin/bake_verdict.rs` — `--floor-rule` / `--floor-margin`
  CLI flags (validated at parse time, refused without `--gaddr-grid-truth`
  when the rule needs it), wired through `dial_panel`, `floor_rule_params`
  stamped into `--gaddr-json`.
- `scripts/dialgate_arms.sh` — `ZL_FLOORRULE` / `ZL_FLOORMARGIN` (always
  passed explicitly to `bake_verdict`, same discipline as `ZL_TAILPINS`).
- `scripts/gaddr_board_regrade.{sh,py}` — `--floor-rule`/`--floor-margin` on
  the `grade` subcommand (default `distinct`; the board's existing `product`/
  `retired` tail-pin re-grade is unaffected).

No registry write, no board graft, no default change. `zensim/weights/` and
`benchmarks/dial_addressability_floor_2026-09-04.json` remain untouched.
