# September 14: color-correct corruption fixtures and a preserved mobile miss

The existing corruption generator now accepts an explicit native SDR input
contract. It converts tagged SDR through the existing Rust color owner, clips
into sRGB, then quantizes to opaque RGB8 before applying the unchanged corruption
operators. This restores the originally admitted mobile source 8014. It is a
**TRAIN engineering diagnostic, not a qualified corruption model or HDR result**.

The summer gauntlet links this study under TRAIN development. Existing frozen
human panels, composites, scalar targeting, native JXL results and product
qualification statuses are unchanged. Published TEST can assess frozen candidates
where EVAL is absent under the latest user directive; this study reads neither.

## What changed and what was checked

Build the existing example with `--features m3-fixtures,zen-decode`, then run:

```sh
zensim-bench/target/release/examples/m3_fixture_gen corruption \
  --in source.png --out fresh-output --ref-id TRAIN_ORIGIN --class screen \
  --seed 1 --input-contract sdr-native-clip-v1
```

The opt-in writes `native-corruption-fixtures-v2`, binding original file hash,
ICC/CICP interpretation, conversion receipt, delivered pixel hash and generated
PNG hashes. The corruption math revision remains unchanged. The default retains
the strict v1 input contract. The new route still requires opaque RGB8/RGBA8
source PNG, refuses orientation/animation and HDR metadata, and uses full CMS
when required. It does not strip a profile and reinterpret its codes as sRGB.
The delivered fixtures are explicitly RGB8; this does not test native HBD/HDR
preservation or extend the product HDR qualification.

For TRAIN calibration source 1200, all **718 PNG/JPEG files** match the earlier
sRGB catalog byte for byte, both with and without the opt-in. Source 8014 still
refuses under v1 and succeeds under v2. Its embedded profile requires full CMS;
normalization changes 19,035 of 28,160 pixels, with maximum change 74 code values.
The original source remains intact. Invalid ICC, HDR CICP, unknown contract and
duplicate contract arguments all refuse before creating an output directory.

Seven native input tests pass, including independent f64 P3 colorimetry and u8
delivery within one code value, all 256 sRGB codes preserved, native precision
retained and nonopaque delivery refused. Native/default generator Clippy,
CI-exact root Clippy and script lint pass. These are bounded correctness checks,
not proof for every possible profile.

## Frozen head: canonical assessment

The existing PP914 basic228 H32 seed7101 base and Rev3 corruption head remain
byte-identical; activation remains probability **greater than 0.9**, followed by
`min(base, corruption)`. No fitting, relabelling or threshold adjustment occurs.
The original TRAIN severe-proxy cause/extent rules were applied before reading
model output. The public Rust scorer supplies all scores, feature audits and
prepared-map checks. `corruption_gate_eval.py --integrity-admission` owns the
assessment, using independently bound normalized-PNG samples and interpretation.

| Measurement | Result |
| --- | --- |
| Catalog audit rows / distinct pixel pairs | 716 / 705 |
| Severe proxy activation | 30 / 31 |
| Channel-operation severe proxies | 8 / 8 |
| Real-bug severe proxies | 19 / 20 |
| Geometric severe proxies | 3 / 3 |
| Known-valid activation (JPEG q20, q10 and identity) | 0 / 3 |
| Complete D228 consumed-feature audit | 716 / 716, exact |
| Pixel/cache/stored-f32 score and activation parity | 716 / 716, exact |
| Accepted prepared-map queries | 14,336 |

Recoverable/ambiguous operations stay unlabelled; their activation inventory is
not a false-positive rate. This one-source TRAIN diagnostic cannot establish
broad specificity or held-out detection. Identity returns exactly 100 via the
public identity path; the literal feature-only score without identity evidence
is 99.800827. That expected distinction is retained in the audit.

The miss is the whole-image `zenjpeg_progressive_ac_truncation` proxy on source
8014: lower screenshot rows become mosaic-like, RGB RMSE is 0.11139 and 19.98%
of pixels change materially. Its base/composed score is **−19.56253**, but head
probability is zero and prepared steering remains accepted. A negative perceptual
score does not replace explicit bug activation. Preserve this failure for
TRAIN-side development; do not weaken its label or threshold to erase it.

## Historical AVIF q85: still quarantined

The already admitted TRAIN-fit origin 8184 q85 file is AV1 4:2:0, full range,
BT.709 primaries, sRGB transfer and SMPTE170M matrix. Independent FFmpeg and
current Rust decoding both reproduce its pink cast. On 2,274 originally white
pixels, mean decoded RGB is approximately (255,235,255) versus (255,236,255).
This is not a defect unique to the current Rust decoder. Different chroma
reconstruction means the two decoders are not pixel-identical.

Original metadata records normal `s4-420`, q85, `modes_full`, encoded June 27
with zenavif `a5697e0a8b0d`, zenmetrics `acd6829a6a97-dirty` and zenrav1e
`22a58d58db1d`. The recorded zenmetrics tree's PNG-to-RGB8 path does not pass ICC
bytes into conversion. The producer was dirty: those committed sources cannot
prove its exact executed behavior or establish the q85 root cause. The original
181 unresolved native nonidentity pairs remain unresolved. Neither a head score
nor agreement between decoders establishes whether each original encoder input
was color-correct. No old packet is silently relabelled or mixed with v2 fixtures.

## Evidence and next step

The served report includes the mobile A/B miss, original/independent q85 decodes,
canonical assessment, commands, source snapshots, all generated fixtures,
negative controls and a hashed replay bundle. Work directory:
`/var/tmp/zensim-validation-2026-09-14/integrity-color-resolution`.

Next is a separately frozen, color-correct TRAIN packet with representative
honest low-quality and spatial encodes, reviewed catastrophic labels and source
family separation. Only then refit/calibrate the bug head and qualify the frozen
composition. Broader all-purpose models still have recorded human/dial failures;
AVIF/native spatial expansion, HDR qualification and production latency remain
unfinished. No model is promoted by this study.
