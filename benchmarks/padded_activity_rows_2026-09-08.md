# Reduce full-pool activity blur cost without changing scores

After the first padded SSIM repair, equal 400-call native profiles of the
existing `extract_paths_bench` identify the activity horizontal box blur as
about 63% of the full-versus-peaks sampled cycle increment. The two regimes
use the same deterministic 1024-square textured pair and pinned CPU 8, one
worker on the 9950X3D. Source inspection ties the extra blur to
`fold_v1_one_band`'s activity-map construction for masked/IW pools.

These are producer profiles, excluding final model composition and the
serving mean-offset pass. They are not complete-model latency measurements.
Sampled cycle estimates per call, rounded to millions:

| Work | Peaks | Full |
|---|---:|---:|
| All user cycles | 136.07 | 240.11 |
| Fused horizontal SSIM | 74.32 | 74.75 |
| Fused vertical SSIM | 37.26 | 38.08 |
| Horizontal activity box blur | below reporting threshold | 65.98 |

Each run is a diagnostic, not a repeated causal estimate. Some caller stacks
unwind to unresolved addresses; symbol self costs and actual producer source
identify the next owner. The first report waited on inherited debuginfod;
its process was stopped and reporting resumed offline on the same captured
samples. No successful encoding/extraction run was repeated for that repair.

## Intervention and evidence

The private v4x horizontal box-blur body now separates physical row stride
from logical width. Full sixteen-row groups use two reusable padded planes
at widths divisible by 256, and the old contiguous body handles all other
widths and remainder rows. At width 1024 the added arena is 130 KiB per
thread. Mirror indices, tile boundaries, ring reuse and sum order are unchanged.
The entire old arithmetic body reproduces after normalizing stride to width.

Direct tests pass all 175 signed-input width/height/radius combinations.
The library and selected golden/fold/SIMD/attribution/allocation/invariant
checks pass 466 tests with six ignored; sixteen complete BakeScorer surface
tests, including HDR and ensemble/corruption composition, also pass.
Root CI-exact Clippy, scoped formatting and the 605-script lint pass. All 1,320 pixel audits and 792 spatial audits
reproduce byte-for-byte, including all saved feature CSVs and the explicit
unsupported map terms. The first extractor build command selected an invalid
target/feature combination and was corrected to the existing canonical example;
the refusal and successful build logs remain retained.

Forty single-call rounds per size, complete Rust A/D blend and member/control
arms, show the following observed means:

| Surface | 1024² before → after ms | 2048² before → after ms |
|---|---:|---:|
| Complete A/D blend | 42.10 → 35.33 | 141.81 → 133.88 |
| D member | 25.31 → 23.73 | 90.08 → 87.58 |
| fast-ssim2 control | 70.57 → 70.27 | 300.81 → 301.47 |

Both blend improvements exceed the preregistered 5% threshold after
normalizing to fast-ssim2 (15.72% and 5.80%). The 576/1152 non-trigger controls
also pass: each has 40 rounds with nine base calls, at least 30 clean paired
rounds and MAD/median below 5%. Maximum normalized regression is 1.58%.
Retain the optimization. Batched controls do not establish latency tails.

Strict quiet admission remains incomplete (resource advisories/background
activity). No p95 or per-worker RSS qualification follows from these means.
The complete blend still exceeds 1.25 times D in the same build; cached/map,
HDR and memory costs remain unqualified. Missing pooled attribution,
corruption honest protection and independently judged spatial targeting remain
required. No fit, validation selection or terminal evaluation occurred.

Artifact: `/mnt/v/output/zensim/pool-profile-2026-09-08/`. It preserves the
registrations, both sources and binaries, native profiles, timing/process logs,
model identities, exact replay and checks. The verified Windows artifact
mirror is `~/work/zensim-validation-2026-09-08/pool-profile/`.

The two retained storage changes close large avoidable extraction costs while
preserving all quality evidence. The remaining blend cost is 1.49/1.53 times
D on these means. Next assess the existing cheaper training recipes and
training-data coverage before choosing another kernel repair or a competitive
D-regime model. Preserve the required spatial/corruption work and release bars;
a narrower feature vector alone is not an improvement.
