# Frozen Rev3 codec-family evaluation — September 14, 2026

Preregistered extension of `rev3_qualification_2026-09-14.md`. Frozen scalar
assessment complete; all nine candidates remain NO-SHIP. The original protocol
is preserved as `PROTOCOL.md` in the artifact directory. This evaluation changes
no models, thresholds, splines, ensembles or training decisions.

## Admitted population

Use the keyed D1 validation views in `r1b-pools944-2026-08-30`:
imazen26 (6,953 rows), nonphoto (6,142), and hfnlproxy (7,717).
The per-slice URI manifests declare `split=validate` and pin the key-table
hashes; the local-pair manifests pin the TSV hashes. Before pixel access,
verify every source ends in {1,3,5}, every row says validate, and paths/targets
join exactly to their keyed rows. Never open the older terminal slices or
mixed picker tables. Exclude historical internal test origins as required by
the current DATA_SPLITS override. Deduplicate extraction by exact source and
distorted file identities while preserving every registered slice membership.

These three views overlap and are **one codec-family instrument**, not three
independent human studies. Targets are historical SSIMULACRA2 scores. The
imazen26 TSV uses raw scores; the other two use scores/100. Preserve original
values and units in admission, use score/100 consistently in the new verdict
tables, and never infer units from observed extrema. Hfnlproxy membership is
frozen by the historical >=91 teacher band, regardless of the fresh peer score.

## Measurement

- Extract native Rev3 full944 features through the existing Rust owner from
  original bitstreams. No reuse of the numerically incompatible old944 tables.
- Record file hashes, decoded RGB8 identities, geometry, extraction revision,
  source/codec/configuration keys, model and tool identities. Refuse missing
  or unscoreable rows; do not quietly restrict to convenient geometries.
- Compute optional fast-ssim2 in the existing extraction audit using the very
  same decoded RGB8 buffers. This reuses its crate and adds no kernel or
  independent decoder. Record historical-versus-current peer differences.
- Run all nine complete frozen Rust compositions on every admitted row.
  Repeat matched B/D through native Rev1 extraction. Retain the previous human
  panels unchanged. Report source/codec/content coverage, rank, within-source
  behavior, outlier envelopes, raw tails/clumping/saturation and above-identity
  emissions. Teacher agreement cannot establish independent human superiority.
- Preserve the existing selected619 spatial failure and full944 unsupported
  refinement disposition; native targeting and spatial RD remain separate gates.
- Use the existing `bake_verdict`, `panel`, peer promotion and gauntlet owners.
  No partial composite may masquerade as a complete product assessment.

Artifacts: `~/work/zensim-validation-2026-09-14/rev3-codec-eval/`.
This is evaluation of fixed candidates, not an adaptive model-selection loop.

## Completed assessment

All nine frozen candidates and matched native B/D now have six populated panels,
24,777 panel memberships each. The prior 3,965 human-study memberships are
unchanged in rank and predictions. New extraction succeeded for all 20,655
distinct pairs, with 19,498,320 finite Rev3 feature cells and no failed rows.
B/D independently scored every added pair from original image bytes; all
20,812 membership predictions per baseline match the verdict exactly.
No training, calibration, feature selection or test/terminal access occurred.

**All nine candidates remain NO-SHIP.** The proxy table below uses **signed**
SROCC: negative quality correlation must not be mistaken for a positive result.
The existing gauntlet magnitude column also retains `srocc_signed` in its source.

| Model | imazen26 | nonphoto | near-lossless proxy | Unique new scores >100 |
|---|---:|---:|---:|---:|
| full944_h128_ens5 | 0.6776 | 0.6202 | -0.0607 | 1221 |
| full944_h256_ens5 | 0.7014 | 0.6455 | -0.0414 | 1032 |
| local120_h128_ens5 | 0.7749 | 0.7860 | -0.0504 | 0 |
| selected619_h128_ens5 | 0.6554 | 0.5955 | -0.0716 | 1329 |
| y40_h32_ens5 | 0.8263 | 0.8115 | 0.2475 | 0 |
| y40_h128_ens5 | 0.8115 | 0.7945 | 0.1664 | 0 |
| y60_h32_ens5 | 0.8174 | 0.8086 | 0.1707 | 0 |
| y60_h128_ens5 | 0.8127 | 0.8050 | 0.2496 | 0 |
| linear60 | 0.8534 | 0.8421 | 0.2108 | 0 |
| matched_B | 0.8603 | 0.8498 | 0.3500 | 0 |
| matched_D | 0.8546 | 0.8453 | 0.2947 | 0 |

These are fixed candidate assessments, not a new model-selection round.
The four wide/local high-capacity candidates have negative aggregate signed
correlation in the near-lossless proxy. Wide models also emit scores above
identity on distorted images. Conversely, y40/y60 ensembles top out around
83–87 in that proxy, although its historical labels are all at least 91.
The linear control is close to B/D on the broad proxy but is not qualified.
None of these results establishes the ceiling of its feature set: the frozen
training recipe used only KADID/TID and lacks the full product training mix.

The 1,000 paired source-cluster bootstrap draws retain all renditions and codec
configurations for each drawn origin (seed 914, 87/61/87 sources). On imazen26,
full944/H128 minus D has a 95% interval [-0.2134, -0.1410];
linear60 minus D has [-0.0086, 0.0065].
Intervals are exploratory, with no multiplicity-adjusted superiority claim.
Full source intervals and 165 per-codec/aggregate panels are retained in the
artifact directory. Raw errors supplement the rank, outlier and scatter panels.

Historical-versus-current fast-ssim2 absolute differences have p95 values
0.289 / 0.328 / 1.220 points for imazen26/nonphoto/hfnlproxy respectively;
maxima are 5.145 / 15.113 / 4.004. Teacher agreement SROCC is 0.99991 /
0.99988 / 0.95233. Exactly 273 historical high-band rows now fall below 91;
their registered membership is preserved. Drift outliers retain row IDs and
source/codec keys. A teacher self-row is deliberately absent from the board.

The latest discussion is **Rev3: six measured panels, human + codec validation**.
Its SSIMULACRA2 reference row retains the prior three human panels. The fair
view continues to exclude unqualified candidates. Missing CID22/AIC3/AIC4
leave canonical composite coverage at 3/6 and its value null.

Native canonical Rev3 extraction and public cached candidate scoring are
measured here; complete per-candidate image/map audits on all added rows are
not claimed. Earlier selected619 spatial failures and full944 unsupported
refinement remain. B/D legacy training-era metadata still cannot qualify
their provenance despite exact native prediction checks. Current decoding is
RGB8 SDR; this is not HDR qualification.

Reproduction evidence: `PROTOCOL.md`, `ADMISSION.json`, `EXTRACTION_VERIFIED.json`,
`features-rev{1,3}/_MANIFEST.json`, `verdicts/RESULT.json`,
`baseline-audits/RESULT.json`, `baseline-verdicts/RESULT.json`,
`TEACHER_DRIFT.json`, `CODEC_PANELS.json`, `BASELINE_CODEC_PANELS.json`, `UNCERTAINTY_PROTOCOL.json`,
`UNCERTAINTY.json`, and the producing scripts/logs. The exact extraction binary
is retained separately from the rebuilt source binary. Rebuild smoke compares
3,776 feature cells and four audit records byte-for-byte; missing audit output
is refused before input decoding. All actual scores/statistics use existing
Rust owners. The legacy 372 imazen26 filename contains `test` because it is a
fixed loader slot: only fresh admitted validation rows are written in this
new isolated root; no historical test table was opened or relabeled.

Validation: root CI-exact Clippy, standalone extractor Clippy/release build,
formatting, script lint, both gauntlet render/data gate suites, and the live
HTTP payload check pass. The live page carries all six exact row counts for
the eleven candidate/baseline rows and only three human panels for SSIMULACRA2.

Remaining product work includes representative train-side feature/model
development, complete composition pixel/spatial audits, corruption specificity,
witnessed native targeting and spatial RD, and controlled latency/memory gates.
Keep eval outputs frozen; develop those steps using the train-side contract.
The complete assigned goal remains active.
