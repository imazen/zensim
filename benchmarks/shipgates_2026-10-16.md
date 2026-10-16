# Ship gates — 2026-10-16

Status: **INCOMPLETE**. Corruption, complete G-ADDR regression, targeting and native codec RD are pending. The protected human-ranking composite is excluded from this execution. No model is promoted.

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
