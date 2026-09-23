# AVIF decode comparison preregistration — 2026-09-23

## Question and identities

Determine why the September 14 SafeSyn RGB8 decode and the September 23 CVVDP smoke decode disagree on AVIF stimuli. The September 14 producer is `extract-native-admission` (SHA recorded in `SAFESYN_ADMISSION.json`); its zensim-bench lockfile records local zenavif 0.2.0, rav1d-safe `e73811f5d4dad81b75195ca18554fd8a5df19515`, and zenavif-parse 0.7.0. The working zenavif checkout's HEAD is `a7c56be9e77dc708f379eeaa288ab607f6d3f6e1`; this commit identity is provisional until the producer/source binding is checked. The alternative decoder is the CVVDP smoke binary from the September 23 lane. A tagged 0.1.7 arm is diagnostic only, not presumed to be the September 14 producer.

## Inputs and fixed selection

All stimuli are TRAIN-role SafeSyn data. Include all eleven AVIF paths in `/var/tmp/cvvdp-safesyn/smoke_out.jsonl` with `metric=ssim2`. Add twenty AVIF bitstreams from distinct other references: parse `/var/tmp/zensim-validation-2026-09-14/baseline-recovery/SAFESYN_ADMISSION.json`, keep `.avif` paths excluding the smoke reference, sort by `(SHA256(path UTF-8), path)`, then greedily select the first path for each new source basename until twenty are selected. Record SHA256 of each file and each input manifest in the result. No human labels are read. Comparison is paired on exact bitstream bytes.

## Measurements and decision rules

For each arm save width, height, descriptor, packed RGB8 SHA256, AV1 Y/U/V plane SHA256 where the API permits, count of changed pixels and channels, per-channel maximum and mean absolute difference, bounding box of changed pixels, and whether changed pixels cluster at alpha or image boundaries. The first stage with different output is the cause: AV1 planes, chroma upsampling, matrix/range/CICP, depth reduction, or alpha. Inspect film-grain signaling. A change of legal chroma filter is a decode choice; mismatched normative AV1 planes under identical settings is a defect requiring an independent AV1 reference. A metadata, matrix, range, alpha or rounding defect requires an isolated fixture and its normative rule. Do not infer correctness from a metric score alone.

For impact, recompute SSIMULACRA2 and zensim B for the 31 fixed pairs under both decodes where the existing scoring owner permits; report paired deltas and raw values, not an extrapolation. If a reference-clustered CI is used, use the existing `zen_stats`/`panel`/`bake_verdict` owner, seed 20260923, 10,000 bootstrap replicates and report tie handling from that owner. Pixel differences are deterministic and require no CI. If any measurement or normative cross-check is unavailable, report it as MISSING.

No CID22-B or other human holdout is read. No changes are applied to zenavif or sibling codec repositories.
