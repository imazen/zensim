# Ship gates — 2026-10-16

Status: **INCOMPLETE**. Missing: matched-quality native RD statistics, complete codec steering-field cost, and JPEG/WebP/AVIF targeting. JXL targeting covers all three frozen candidates; every candidate fails a three-shot bar. W-C/C corruption companions fail native honesty and swap recall; diff-410 lacks an authorized matching companion. G-ADDR fails for all three. The four-provider JXL TRAIN intervention screen is complete. The protected human-ranking composite has not run. No model is promoted.

These are standalone Rev5 packed candidates with D12 exact-gradient maps. W-C steering evidence is the frozen E34 seed-0 panel; the two original candidates have fresh r7900x E34 collector-cost/RSS measurements.

| Candidate | Packed bytes | SHA-256 |
|---|---:|---|
| w-c | 69065 | `f73059ff8f67ff177265d35dfd5ff5db61d8fc825a6b519cf6925dc8fc1d5e2a` |
| diff-frag-820 | 214904 | `89cb4861e2651ff8d4a98bb879feed83a0f9eb3f22b83e644e42eb2291d74b2e` |
| diff-410 | 108287 | `7cf4cfd92bb8824405db3ed27287bbec973a01eaeadf09b7ad037db92ab9a275` |

## W-C steering misses

Exactly 133/135 pass the joint M2 >=0.99 / M3f >=0.70 rule. Both failures are M2 failures.

| Case | M2 | M3f |
|---|---:|---:|
| broad-13-b64 | 0.8676470588235268 | 0.8676470588235268 |
| broad-202-b64 | 0.9790209790209722 | 0.9860139860139792 |

Engineering per-case artifact SHA-256 `ca8bb71dcdd78096aba23a5f683bb89f76bedbdf896fb0ff9411652a440f37f5`.

## Cached score plus map cost

The E34 collector times prepared-result construction without rectangle/refinement queries. These verdicts apply to that method; they do not qualify complete codec steering-field cost.

r7900x, one pinned worker, release build, no target-cpu=native. Every geometry/tier cell retained 32 accepted paired rounds under the unchanged quiet and clean-round gates. Same candidate uncached scalar p95 is the denominator; the bar is <=3. These timings do not establish the scorecard’s dev absolute scalar-performance bars. W-C reuses the E34 measurement; the other two candidates were measured on October 16.

| Candidate | Geometry | Uncached p95 ns | Cached score+map p95 ns | Ratio | <=3 |
|---|---|---:|---:|---:|---|
| diff-410 | v4x-t1-1024x1024 | 27539646 | 84563562 | 3.070611801 | FAIL |
| diff-410 | v4x-t1-2048x2048 | 111793320 | 339931985 | 3.040718220 | FAIL |
| diff-frag-820 | v4x-t1-1024x1024 | 27328595 | 73153906 | 2.676826452 | PASS |
| diff-frag-820 | v4x-t1-2048x2048 | 111112147 | 301271450 | 2.711417771 | PASS |
| w-c | v4x-t1-1024x1024 | 31226460 | 88415764 | 2.831437313 | PASS |
| w-c | v4x-t1-2048x2048 | 131452433 | 357638607 | 2.720669362 | PASS |

Peak incremental RSS uses independent fresh-process `/usr/bin/time -v` maxima: score+map minus a process retaining only the same inputs/model/scorer. Bar: 128 bytes/pixel +64 MiB. Codec memory is excluded.

| Candidate | Geometry | Baseline KiB | Score+map peak KiB | Increment KiB | Bar KiB | Verdict |
|---|---|---:|---:|---:|---:|---|
| diff-410 | 1024x1024 | 11644 | 168700 | 157056 | 196608 | PASS |
| diff-410 | 2048x2048 | 30168 | 606432 | 576264 | 589824 | PASS |
| diff-frag-820 | 1024x1024 | 12132 | 169388 | 157256 | 196608 | PASS |
| diff-frag-820 | 2048x2048 | 30668 | 607884 | 577216 | 589824 | PASS |
| w-c | 1024x1024 | 11980 | 169068 | 157088 | 196608 | PASS |
| w-c | 2048x2048 | 30584 | 606340 | 575756 | 589824 | PASS |

The canonical `e34_results.runtime` owner computed these verdicts from retained collector evidence. Instrument SHA-256 `c8abc298bb3f636f4aab2d350455db9bb97281160051d2eeb37c3c3ae3ecf2b7`; paired analyzer SHA-256 `798d938626e31c54e524da56687c468f7178d1629b4ee4f268692a290ccbba82`.

Collector resource receipts (these describe collection including quiet waits; retained-round admission is checked separately):

```text
k-a rss: run-heavy: done rc=0 1s | peak-RSS 0.58GiB | min-avail 23694MiB | peak-load 0.02
k-a timing: run-heavy: done rc=0 83s | peak-RSS 0.58GiB | min-avail 23083MiB | peak-load 1.07
k-c rss: run-heavy: done rc=0 1s | peak-RSS 0.58GiB | min-avail 23720MiB | peak-load 0.99
k-c timing: run-heavy: done rc=0 81s | peak-RSS 0.58GiB | min-avail 23078MiB | peak-load 1.48
w-c existing rss: run-heavy: done rc=0 1s | peak-RSS 0.58GiB | min-avail 25508MiB | peak-load 0.26
w-c existing timing: run-heavy: done rc=0 702s | peak-RSS 0.58GiB | min-avail 20215MiB | peak-load 8.15
```

Full private execution evidence on dev: `/mnt/v/output/zensim/shipgates-2026-10-16/`. Model weights and generated maps/bitstreams remain outside git.

## G-ADDR regression and contract

Both frozen instruments are bound to their original SSIMULACRA2 mentor rows by exact `(image_id, codec, q)` key equality: standard 4,424/4,424; ladder 9,593/9,593. The unchanged Rust owner uses the operative `resolvable` rule, K=3 and margin=0.5, and computes the mentor bar live on the same cells. Every candidate passes C1–C6 and fails A7r on both instruments.

A1–A6 and A8r are report-only; these new grid hashes have no legacy value-pin row, so their numeric bar is unavailable and the owner records NOT MEASURED. Their measured values are still shown. A7r is fully measured for all four standard-grid and five ladder-grid codecs. The A7r summary value is the number of failing codec floors; its bar is zero.

### standard — every row

| Row | Bar | W-C | diff+frag-820 | diff-410 |
|---|---|---:|---:|---:|
| A1 | report-only / unavailable | 98.2763089853 (not_measured) | 98.5982017639 (not_measured) | 98.5622455659 (not_measured) |
| A2 | report-only / unavailable | -51.1986943353 (not_measured) | -56.7155043365 (not_measured) | -60.9408228617 (not_measured) |
| A3 | report-only / unavailable | 95.9769167108 (not_measured) | 96.5009903122 (not_measured) | 96.0190790312 (not_measured) |
| A4 | report-only / unavailable | 12.4115763267 (not_measured) | 11.1082676971 (not_measured) | 12.0943899135 (not_measured) |
| A5 | report-only / unavailable | 149.475003321 (not_measured) | 155.3137061 (not_measured) | 159.503068428 (not_measured) |
| A6 | report-only / unavailable | 83.5653403841 (not_measured) | 85.392722615 (not_measured) | 83.9246891178 (not_measured) |
| A7r | ≤ 0 | 2 (fail) | 2 (fail) | 2 (fail) |
| A8r | report-only / unavailable | -65.7559382106 (not_measured) | -79.123065277 (not_measured) | -77.7259847099 (not_measured) |
| C1 | ≥ 0.93 | 0.995136637332 (pass) | 0.995368226031 (pass) | 0.994210282538 (pass) |
| C2 | ≤ 0.05 | 0 (pass) | 0 (pass) | 0 (pass) |
| C3 | > 0 | 0.548 (pass) | 0.654 (pass) | 0.6095 (pass) |
| C4 | < 0 | -82.3140573487 (pass) | -122.570029603 (pass) | -103.939595506 (pass) |
| C5 | ≤ 0 | 0 (pass) | 0 (pass) | 0 (pass) |
| C6 | ≤ 0 | 0 (pass) | 0 (pass) | 0 (pass) |

| Codec | Ladders | Mentor fraction | W-C | diff+frag-820 | diff-410 |
|---|---:|---:|---:|---:|---:|
| avif | 35 | 1 | 0.914285714286 (fail) | 0.914285714286 (fail) | 0.914285714286 (fail) |
| jpeg | 22 | 0.954545454545 | 0.909090909091 (fail) | 0.909090909091 (fail) | 0.909090909091 (fail) |
| jxl | 33 | 1 | 1 (pass) | 1 (pass) | 1 (pass) |
| webp | 16 | 1 | 1 (pass) | 1 (pass) | 1 (pass) |

### ladder — every row

| Row | Bar | W-C | diff+frag-820 | diff-410 |
|---|---|---:|---:|---:|
| A1 | report-only / unavailable | 100 (not_measured) | 100 (not_measured) | 100 (not_measured) |
| A2 | report-only / unavailable | -82.6541340235 (not_measured) | -91.3225761438 (not_measured) | -109.403687058 (not_measured) |
| A3 | report-only / unavailable | 94.46305344 (not_measured) | 94.8298500688 (not_measured) | 94.5966453154 (not_measured) |
| A4 | report-only / unavailable | -11.0971226887 (not_measured) | -14.0470997267 (not_measured) | -14.1206368738 (not_measured) |
| A5 | report-only / unavailable | 182.654134023 (not_measured) | 191.322576144 (not_measured) | 209.403687058 (not_measured) |
| A6 | report-only / unavailable | 105.560176129 (not_measured) | 108.876949795 (not_measured) | 108.717282189 (not_measured) |
| A7r | ≤ 0 | 1 (fail) | 1 (fail) | 3 (fail) |
| A8r | report-only / unavailable | -65.7559382106 (not_measured) | -79.123065277 (not_measured) | -77.7259847099 (not_measured) |
| C1 | ≥ 0.93 | 0.99001168845 (pass) | 0.98820529168 (pass) | 0.98639889491 (pass) |
| C2 | ≤ 0.05 | 0 (pass) | 0 (pass) | 0 (pass) |
| C3 | > 0 | 0.548 (pass) | 0.654 (pass) | 0.6095 (pass) |
| C4 | < 0 | -82.3140573487 (pass) | -122.570029603 (pass) | -103.939595506 (pass) |
| C5 | ≤ 0 | 0 (pass) | 0 (pass) | 0 (pass) |
| C6 | ≤ 0 | 0 (pass) | 0 (pass) | 0 (pass) |

| Codec | Ladders | Mentor fraction | W-C | diff+frag-820 | diff-410 |
|---|---:|---:|---:|---:|---:|
| avif-rav1e | 39 | 0.641025641026 | 0.487179487179 (fail) | 0.487179487179 (fail) | 0.461538461538 (fail) |
| avif-svt | 39 | 1 | 1 (pass) | 1 (pass) | 0.974358974359 (fail) |
| jpeg | 39 | 0.666666666667 | 0.717948717949 (pass) | 0.717948717949 (pass) | 0.641025641026 (fail) |
| jxl | 26 | 0.961538461538 | 0.961538461538 (pass) | 0.961538461538 (pass) | 0.961538461538 (pass) |
| webp | 39 | 1 | 1 (pass) | 1 (pass) | 1 (pass) |

On the floor-dense ladder, W-C and diff+frag-820 each fail AVIF-rav1e. diff-410 also fails AVIF-SVT and JPEG. A7r was previously missing, so these are additional measured failures beyond W-C’s steering misses. No bar was relaxed.

Binding receipt SHA-256 `a7be143213636ce46163374f4f47855a5f805107061074d249ea5958ffaca10d`. Full row evidence: `gaddr-attempt3/ALL_ROWS.json` under the evidence root above.

```text
run-heavy: done rc=0 1s | peak-RSS 0.15GiB | min-avail 42188MiB | peak-load 11.41
```

## Corruption — frozen companion composition

The complete registered 15,060-row synthetic/codec battery was audited separately for each candidate (9,036 TRAIN, 6,024 validation attempts). These are generator labels, with no human-label read. The fixed later release companion is ZCTH v4, Rev5, 420 declared inputs, SHA-256 `568380f1bd2fe6202ebcc5a194078d7885035f466abe93ada3a4b3ea8f4a26f2`, baked probability threshold 0.9. This explicitly composed diagnostic differs from the standalone perceptual candidates used in map/steering gates. It does not qualify the later severity-reviewed contract.

Only w-c and diff-frag-820 match the companion’s420 declared feature IDs exactly. Both fail zero honest native activation/lowering and100% non-inert RGB-swap detection; their perceptual branches produce different ordering counts. **diff-410 has no authorized corruption detector**: its410 declared IDs do not match. Its historical head-composed diagnostics do not qualify a corruption gate. Statistics below deduplicate exact pixel/input identities within each role.

| Shared measure | Bar | TRAIN | Validation | Verdict |
|---|---|---:|---:|---|
| Honest native activation | 0 | 0/456 | 1/304 | FAIL |
| Honest native score lowering | 0 | 0/456 | 1/304 | FAIL |
| Overall honest activation | <=1% | 0/488 | 1/326 | PASS |
| Overall honest score lowering | <=1% | 0/488 | 1/326 | PASS |
| Unique corruption detection | >=95% | 7713/7725 | 5288/5353 | PASS |
| Real-bug detection | >=90% | 466/473 | 289/317 | PASS |
| Non-inert swap GB | 100% | 154/154 | 116/117 | FAIL |
| Non-inert swap RB | 100% | 155/155 | 119/120 | FAIL |
| Non-inert swap RG | 100% | 139/139 | 120/120 | PASS |

Activation uses the actual identity guard: `!pixels_identical && probability > threshold`. Raw unguarded head-positive counts are 8/488 TRAIN and 7/326 validation; these include inert identities and are retained separately.

| Candidate | Standalone below-q20 TRAIN | Standalone below-q20 validation | Composed below-q20 TRAIN | Composed below-q20 validation | Composed >=99% |
|---|---:|---:|---:|---:|---|
| w-c | 2245/7725 | 1447/5353 | 7716/7725 | 5300/5353 | PASS |
| diff-frag-820 | 2285/7725 | 1577/5353 | 7715/7725 | 5303/5353 | PASS |
| diff-410 | 2450/7725 | 1723/5353 | not authorized | not authorized | MISSING DETECTOR |

The native failure is the validation document origin 5315, AVIF scalar CQ1: head probability 1.0, composed score 0.0. The missed swaps are the whole-image GB swap at opacity0.5 (6/7 cases detected) and whole-image opaque RB swap (6/7). Every origin/content/codec/family breakdown and each native/detection failure is retained in the evidence packet. No threshold was changed.

All15,060 rows per candidate passed native/cached/stored-f32 consumed-feature, score and diagnostic head-probability replay with maximum absolute deltas0. The diff-410 head replay is an unauthorized diagnostic only. Input packet SHA-256 `9d51d1e64d987465a5519a2cb9c1d5f19b271394a17f30e27e6b8054ad8b96bc`; extractor SHA-256 `6dad3b66c66ead7fb382c6ef46b17412cfedab6c53aaf0bf052dbecec348b496`.

Full owner reports are `corruption-attempt2/<candidate>/report.json`; authorized activation/failure bindings and the explicit missing-detector result are `corruption-authorized-bars/REGISTERED_BARS.json`, SHA-256 `5fe2140a325fbaad32801d609f8bd5b35b4e2d1255346f9788adde5aa2835e6b`.

```text
run-heavy: done rc=0 336s | peak-RSS 0.56GiB | min-avail 40497MiB | peak-load 13.96
```

## G-TARGET — W-C JXL

Frozen separate TRAIN calibration uses nine sources; evaluation uses eight disjoint validation families across photo, document, graphic and screen content, at max256side. The complete native owner validated every bound, coverage request, selected bitstream/pixel hash and encode/map count. Of80 requests,37 are jointly witnessed across scalar/neutral/active within±1; all80 receive a disposition. The same37 enter every arm/policy/budget. Sparse bounds do not prove the other interior requests impossible.

Active arm with the frozen train-curve seed; p95 uses the registered owner’s nearest-rank statistic:

| Shots | Median error | Bar | p95 error | Bar | Undershoot count | Fraction bar | Worst error | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.627105712891 | 2.0 | 2.58766174316 | 8.0 | 0/37 beyond8.0 | 0.05 | 2.88809204102 | PASS |
| 2 | 0.344909667969 | 1.0 | 1.27006530762 | 3.0 | 0/37 beyond3.0 | 0.05 | 1.48175048828 | PASS |
| 3 | 0.275978088379 | 0.5 | 0.942276000977 | 1.0 | 1/37 beyond1.0 | 0.01 | 1.25915527344 | FAIL |

The three-shot maximum1.25915527344 passes≤3, but the undershoot fraction1/37 fails≤1%. Failing case: origin5315, requested94.1147460938, achieved92.8555908203, signed error-1.25915527344. No bar was changed.

The1/2/3-shot active train-curve groups use37/48/50 complete bitstream encodes and111/144/150 native map evaluations. Each complete JXL encode has three internal reconstruction comparisons; these are counted separately from full encodes. Target error evidence does not qualify general size/mode/HDR coverage or independent-judge RD.

Owner analysis SHA-256 `ca9b73fad503ac446b8b8b33a3a0ed3b7125bc8af86603046f29b47701bf27d9`; full tables, per-class results and work receipts: `jxl-target-phased/w-c/eval/` under the evidence root.

## G-TARGET — original diff-frag-820 JXL

Same frozen source/protocol/driver as W-C, with its own untouched TRAIN calibration. All80 requests receive a disposition;38 are jointly witnessed. Every bound, selected bitstream/pixel identity and native work count passes the canonical owner’s validation.

| Shots | Median error | Bar | p95 error | Bar | Undershoot count | Fraction bar | Worst error | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.747051239014 | 2.0 | 5.74906921387 | 8.0 | 0/38 beyond8.0 | 0.05 | 5.85446166992 | PASS |
| 2 | 0.405738830566 | 1.0 | 1.37683105469 | 3.0 | 0/38 beyond3.0 | 0.05 | 1.80017089844 | PASS |
| 3 | 0.39595413208 | 0.5 | 0.999382019043 | 1.0 | 1/38 beyond1.0 | 0.01 | 1.57715606689 | FAIL |

Three-shot maximum passes≤3;1/38 undershoots beyond1 fails≤1%. Case: origin5315, requested90, achieved88.4228439331, signed error-1.57715606689. Owner analysis SHA-256 `8619aa8b72ee9db3bfadafa7fc60c6e422d2fc8cd75d8a2af749d88d0df8fd73`. Full per-class tables and work receipts: `jxl-target-phased/diff-frag-820/eval/`. No bar changed.

## Corruption companion ID admission

The coordinator requires an exact declared-ID match to the frozen release head.
Native model/head readers confirm the following ordered-ID comparisons at the
unchanged baked probability threshold0.9:

| Candidate | Candidate declared IDs | Head declared IDs | Exact match | Authorized detection companion |
|---|---:|---:|---|---|
| w-c |420|420|yes|release head568380f1|
| diff-frag-820 |420|420|yes|release head568380f1|
| diff-410 |410|420|no|none; lacks corruption detection|

The native API can form a union extraction plan for diff-410 plus this head, but
that does not satisfy the coordinator's exact-match restriction. Its composed
numbers above are retained diagnostic measurements and confer no corruption
ship-gate pass. The standalone validation below-q20 result remains1723/5353.
No companion was invented or retrained.

Binding receipt SHA-256 `38d6c697c725dc6c2f30d09561f01b1a1c1153e7b32f99b0e1ad057127c2512c`.

## G-TARGET — original diff-410 JXL

Same frozen source/protocol/driver, with separate untouched TRAIN calibration.
All80 requests receive a disposition;38 are jointly witnessed. The canonical
owner validates684target cells and504offline bound cells, retained stream/pixel
identities and work counts.

| Shots | Median error | Bar | p95 error | Bar | Undershoot count | Fraction bar | Worst error | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 1.04418945312 | 2.0 | 5.99224090576 | 8.0 | 0/38 beyond8 | 0.05 | 9.72933959961 | PASS |
| 2 | 0.277923583984 | 1.0 | 1.67393493652 | 3.0 | 0/38 beyond3 | 0.05 | 2.51805877686 | PASS |
| 3 | 0.260280609131 | 0.5 | 1.1628112793 | 1.0 | 1/38 beyond1 | 0.01 | 1.33506011963 | FAIL |

Three-shot p95 fails≤1;1/38 undershoots beyond1 fails≤1%. The maximum passes≤3. Case origin8113, requested70, achieved68.9324645996, signed error-1.06753540039.

Active train-curve full encodes38/57/65 and native map evaluations114/171/195 are counted separately. Owner analysis SHA-256 `478bce2479bd68efddf29ce285bb55792a5a9f1ac1ea45082e838377768a5d3d`. Full per-class/control/work evidence remains in `jxl-target-phased/diff-410/eval/`. This sparse development screen does not qualify other sizes, modes, codecs or HDR.

```text
run-heavy: done rc=0 640s | peak-RSS 0.05GiB | min-avail 37035MiB | peak-load 20.90
```

### Runtime scope of the registered collector

The exact E34 `e34_candidate_map` arm constructs `prepare_steering(source,1)` and times `session.compute(...).result().score()`, without rectangle/refinement queries. The reported ratios follow that unchanged method and qualify prepared-result construction. Requesting codec tiles adds work; native codec manifests include it in `encode_ms`. Full codec steering-field cost remains unqualified by the collector ratios.

## Native JXL intervention controls

All four providers complete the same four TRAIN families (photo, document, graphic and screen), distances1/3 and16 coarse regions with0.8/1.2 raw-quantizer interventions. Each provider has272 full encodes,256 interventions and eight exact neutral repeats. Every retained output passes libjxl0.12 decoding and all three fixed judges. The independent map oracle verifies float-map bits, hashes and input-pixel bindings.

All256 interventions change quantizers and pixels; none changes quantizers outside its selected region. Pixels also change outside every selected region; the responsible stage remains unresolved. Control bytes and quantizer fields match across providers; map predictions and candidate scores differ.

Representative primary/libjxl U8 comparisons differ by at most one channel level. Pixel parity remains an open finding; decoding success does not establish bit identity.

The table shows Spearman association between map density and central fixed-B quality improvement per added byte, using only positive byte derivatives. These are descriptive TRAIN mechanism measurements, with no acceptance bar or release qualification.

| TRAIN origin | Class | Distance | W-C | C | A | Butteraugli map peer |
|---|---|---:|---:|---:|---:|---:|
| 2010 | photo | 1 | 0.376471 | 0.702941 | 0.485294 | 0.617647 |
| 2010 | photo | 3 | 0.352941 | 0.467647 | 0.391176 | 0.064706 |
| 6610 | document | 1 | 0.638235 | 0.735294 | 0.732353 | 0.305882 |
| 6610 | document | 3 | 0.673529 | 0.697059 | 0.685294 | 0.800000 |
| 7066 | graphic | 1 | 0.120588 | -0.108824 | 0.223529 | 0.173529 |
| 7066 | graphic | 3 | 0.352941 | 0.394118 | 0.258824 | 0.326471 |
| 8206 | screen | 1 | 0.135714 | 0.128571 | 0.146429 | 0.221429 |
| 8206 | screen | 3 | 0.591176 | 0.567647 | 0.597059 | 0.408824 |

Full control summary SHA256 `ad305bf3236ec0f62f5880ad5a46441e118f2d7f3007073b2fd05411d711963d`; payloads and all per-intervention records remain in the evidence archive.

## RD input admission and PNG representation

All24 original sources passed native strict opaque/sRGB8 metadata admission and an independent whole-image RGB-byte oracle. Individual visual review subsequently identified nine photographs,12 screen images and three game renders; the original “12 photo” description was incorrect. The GUI source uses its deterministic largest opaque rectangle; the indexed screenshot is losslessly expanded to8-bit PNG through the first-party encoder. Their original bytes, source URLs, family identities and preparation receipts remain archived. No quality grid, candidate, codec or fixed-judge arithmetic changed.

Original24-source manifest SHA256 `6ddb0f655ddfe307a6c9702a2928eb7b47690fc1b0104811287dfb08af8cdf75`. The launcher requires the hash-bound24-source admission receipt before encoding. Existing344 complete cells were hash-verified and reused. Full4032-cell JXL RD coverage and source-bootstrap results remain pending.

Decoded PNGs and float-map containers may use lossless palette or grayscale storage. Explicit sample expansion preserves every delivered RGB byte and every logical float-map bit; physical container hashes, logical-map hashes, input-pixel bindings and deduplication keys are checked independently. SSIMULACRA2 remains scalar-only under the approved artifact exception.

```text
run-heavy: done rc=0 14s | peak-RSS 0.55GiB | min-avail 37893MiB | peak-load 20.63
run-heavy: done rc=0 1s | peak-RSS 0.04GiB | min-avail 40957MiB | peak-load 16.97
run-heavy: done rc=0 4s | peak-RSS 0.15GiB | min-avail 40038MiB | peak-load 17.80
```

Private source-preparation helper tests cover strided palette/gray expansion and exact opaque cropping; strict Clippy and release build pass. JXL native-intervention source CI succeeds on its complete required matrix, including Windows ARM, macOS ARM64 and i686; run38991498232. These checks do not establish RD gains or decoder pixel parity.

### Bounded native group scheduling

An eight-arm replay of an already completed576x576 source/quality group preserves all8 encoded hashes, every decoded RGB byte and all39 ordered steering-map fields exactly. All outputs pass the independent decoder; neutral controls remain byte-identical to no steering. Frozen model, encoder, decoder, source, judge and quality-grid pins are unchanged. Complete hash-verified groups remain reusable.

Subsequent scheduling may run up to8 independent local arms inside one16GiB/jobs8 shared-lock heavy job. Each codec and independent decoder remains single-threaded. This changes scheduling only; grouped native timings are contended diagnostics and confer no quiet latency or per-worker memory qualification.

```text
run-heavy: done rc=0 25s | peak-RSS 0.10GiB | min-avail 17858MiB | peak-load 34.93
```

Exact replay receipt: `GROUP_REPLAY_VERIFIED.json` under the evidence root. Full RD coverage and matched-quality bootstrap results remain pending.

### Consecutive quality groups

Four consecutive qualities can share one bounded heavy job, with a separate complete eight-arm receipt written after each quality. A32-cell replay of four already measured qualities matches every encoded hash, decoded RGB byte and all156 ordered steering fields. An interrupted-journal check recovers16 unappended complete cells beside16 journaled cells, then selects the next quality without executing a codec. The21-point grid,24 sources, models and native/judge binaries are unchanged; full RD results remain pending.

```text
run-heavy: done rc=0 79s | peak-RSS 0.10GiB | min-avail 35226MiB | peak-load 6.41
```

Evidence: `QUALITY_GROUP_REPLAY_VERIFIED.json` and `quality-group-recovery-check/VERIFIED.json`. The active continuation uses `rd-jxl-quality-groups/` and `rd-jxl-quality-groups-judges/`; historical roots and every completed artifact remain archived. Source bootstrap and numeric RD verdicts still require the complete matrix.

## Approved source-class correction — October17

The coordinator approved retaining the complete original mixed cohort, correcting the three game renders to the registered screen/nonphoto class, and appending the next three unused photographs in the frozen SHA256(path) ordering: girl, flowers and rain. Every one was individually reviewed before admission. The corrected cohort has12 photographs and15 screen/nonphoto images; the minimum remains12 per class. Its expected complete JXL grid is4,536 cells. All27 sources passed native metadata and exact full-RGB admission. The additional encodes and complete4,536-cell analysis remain pending; no RD verdict is claimed.

Original source files, manifests, complete cells and measurements remain archived. Reused rows carry a class-only derivation bound to their original cell metadata and the approved revision; byte counts, encoded/decoded hashes, steering maps, source families, model/codec/judge pins and the21-point quality grid are preserved.

The corrected source-manifest SHA256 is `e9b21e153a7d22c59e5e6341f9e7f35d0050a5b55882558f62c808c4dbae56df`. A replay over all1,344 complete records from the first eight original sources changes only the504 game-render class annotations; the other840 rows are unchanged. Every non-class field matches its immutable original cell. This is metadata verification, with no new encode, score or fitted value.

The six-source certified-packet replay rehashes the physical map artifacts and retains successful logical-map/input-pixel certificates for all1,008 rows. Their reference-pixel hashes also match the independent source admission. The reporter requires that admitted-pixel binding and verifies each class-only derivation against its original complete cell. No metric values are recomputed by certificate reuse.
