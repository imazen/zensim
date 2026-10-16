# Ship gates — 2026-10-16

Status: **INCOMPLETE**. Targeting and native codec RD are pending. Corruption fails for all three candidates with the frozen companion head. G-ADDR is complete and fails for all three candidates. The protected human-ranking composite is excluded from this execution. No model is promoted.

These are standalone Rev5 packed candidates with D12 exact-gradient maps. W-C steering evidence is the frozen E34 seed-0 panel; the two original candidates have fresh r7900x map-cost/RSS measurements.

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

Every candidate fails zero honest native activation/lowering and 100% non-inert RGB-swap detection. All three share the same head detection results; their perceptual branches produce different ordering counts. Statistics below deduplicate exact pixel/input identities within each role.

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
| diff-410 | 2450/7725 | 1723/5353 | 7717/7725 | 5304/5353 | PASS |

The native failure is the validation document origin 5315, AVIF scalar CQ1: head probability 1.0, composed score 0.0. The missed swaps are the whole-image GB swap at opacity0.5 (6/7 cases detected) and whole-image opaque RB swap (6/7). Every origin/content/codec/family breakdown and each native/detection failure is retained in the evidence packet. No threshold was changed.

All 15,060 rows per candidate passed native/cached/stored-f32 consumed-feature, score and head-probability replay with maximum absolute deltas 0. Input packet SHA-256 `9d51d1e64d987465a5519a2cb9c1d5f19b271394a17f30e27e6b8054ad8b96bc`; extractor SHA-256 `6dad3b66c66ead7fb382c6ef46b17412cfedab6c53aaf0bf052dbecec348b496`.

Full owner reports are `corruption-attempt2/<candidate>/report.json`; guarded activation/failure bindings are `corruption-bars-attempt2/REGISTERED_BARS.json`, SHA-256 `e8f9b77a7df7217c8ce1459cb04956c2faa59fb3068ec0933e573c6203e5452c`.

```text
run-heavy: done rc=0 336s | peak-RSS 0.56GiB | min-avail 40497MiB | peak-load 13.96
```
