# Rev4 C8 GMSBANK qualification (2026-09-24 UTC)

Candidate `gmsbank` appends f1322–f1501 to corrected Rev4 base `87db7d773198`; implementation commit `c1a61c4c8e324cb8764ebb8b188728c29817049b`. No human labels were read and no model was fit. C8 inputs are the existing unit-XYB central-difference gradient magnitudes at four scales and X/Y/B channels. The five constants are pinned source literals, with pixel-only TRAIN calibration recorded separately in `benchmarks/gmsbank_calibration_2026-09-23.md`.

## Frozen checks and current results

| Gate | Result | Evidence |
|---|---|---|
| f0–f1321 identity vs corrected base and toggle off/on, 144 TRAIN pairs × 6 tier/thread modes | **Pass on final rebuilt binary:** 0 differences in 3,426,624 compared cells across base/off, off/on and base/on. | `/var/tmp/gmsbank/identity/report.json` (SHA256 `3be23dd196762bf4b7be0303eb76eb3b96cb321bd4a014cadabdd7fe2e4ddeea`); final log SHA256 `366bfb38bd1171a170fea6d0a318f4bc9626418f5b2a749c4879e368fd264f22` |
| All 180 C8 slots nonzero on the 144 TRAIN-pair corpus | **Pass on final rebuilt binary:** 0 dead slots; each was nonzero on 142 of 144 native serial TRAIN pairs. | `/var/tmp/gmsbank/command_records/final_dead_slots_after_clippy.log` (SHA256 `5fdcf54f16567b917456d5c30c2819ddb02529af7102aa90b1abd730955c1ba6`) |
| Independent NumPy map and pooling, 8 TRAIN pairs | **Pass on corrected base:** 1,440 values, max relative error `3.0291868626664007e-15`; 16× wrong-c control rejected 1,440/1,440. | `/var/tmp/gmsbank/command_records/final_numpy_reference.log` (SHA256 `352a9152c21b4aaf830962fd472b63939b29c8c0cf4506aca4f4182c2285417f`) |
| Four selected imazen-26 TRAIN graphics refs: blur/noise and zenjpeg q95→q5 step 5 | **Miss on corrected base:** c-order algebra 336/336; noise gain-dominant 4/4, deterministic 3×3 box blur loss-dominant 0/4, so polarity 4/8. The selection rule for these four references was not recorded. The explicit `just gmsbank-corpus` gate exited 1 on this assertion. Ladder reported per constant and quality; no unsupported quality-order monotonicity claim. | `/var/tmp/gmsbank/corpus_probe/report.json` (SHA256 `95397c1d2eaae6400832e2fe0bee8f0ede89659b66323dfd5ec58eafd3e5aae8`); final gate log SHA256 `6fa75e620287498a3d86c00a05216cd60dd7ef46cf429a6a6b188fc21f04ebd5` |
| C8 unit, planner, tier, strip and stride checks on corrected base | **Pass:** 4 focused unit tests, 1 explicit planner regression, 1 tier/MT8 test. Ten permutations across three tiers at each of two geometries; maximum C8 relative drift vs scalar `5.952e-5`, reported under the reviewed base policy. | `/var/tmp/gmsbank/command_records/final_{gmsbank_unit,plan_unit,gmsbank_tier}.log` |
| Correction: strict loss/gain orientation | **Pass:** 257 synthetic pixels with `m_d < m_r` feed all five constants and yield positive loss, bitwise zero gain. The final monotone-ramp-reference/uniform-destination pair has `m_d < m_r` at every pixel and yielded bitwise zero C8 gain in every dispatch permutation; both focused tests passed 1/1, 0 ignored. | `/var/tmp/gmsbank/command_records/correction_orientation_unit.log` and `correction_orientation_tiers_after_clippy.log` (SHA256 `721564ade1cc63c6e28cbecc5567523570a8ccaa07be605a4327120326285a88`, `075c853c0a9386f8528c9119c9b8edc87060ad945c905bc267ec06846260a343`) |
| ST and MT8 marginal cost, 256² to 4096², interleaved zenbench | **CONTENDED:** ST misses the ≤5% goal (β `19.7327 ns/px`, α `−0.4602 ms`, R² `0.999969`, fitted 1024² +8.82%, raw +9.02%). MT8 is below the goal (β `5.0441 ns/px`, α `0.0349 ms`, R² `0.999612`, fitted +4.46%, raw +3.79%). Eight rounds per size; ST had 2 gate waits and MT8 had 31, with drift warnings. | `/var/tmp/gmsbank/cost/{st1,mt8}.zenbench` (SHA256 `ad10f74c…`, `4e3ee992…`) |
| Exact-pixel GMSD/GMSM peers for all 18 bank sets | **Pass:** 248,983 unique pixel keys checked, expanded to 249,227 stimulus rows. Independent parquet check found 0 key/score mismatches; CID22-B was pixel-only (2,100 keys), no labels read. | `/var/tmp/gmsbank/peer_gmsd/_MANIFEST.json` (SHA256 `8d2799c8a9ebaa3f25fa7518b3d7591ee7742b696d7f5592b9ace9974d0d27fc`); `verification.json` (SHA256 `5429a455120bd41a0c729514e61670c124b85be576dc2fc49a8a9cf5bf695f5f`) |
| Measure-first 2,000 SafeSyn TRAIN pairs, stored prefix parity and heaptrack | **Pass:** 0/1,810,000 f32 differences across 905 stored bank columns and 0/2,644,000 bit differences in f0–f1321 vs corrected-base extraction on the same paths; 45.01 s wall, 44.43 pairs/s; heaptrack peak heap **613.80M**. The bank omits 81 of f0–f985, so the corrected-base comparison covers those columns. | `/var/tmp/gmsbank/measure_first/{features.csv,base_features.csv,parity.json,time.txt,ht.zst.zst}`; full-prefix log SHA256 `d16bcfa3fd509dfc8afd8610d2a9d44e867e5afd7ca61e170fa3e0c3eb89ed82`; heap analysis log SHA256 `5f6e91315e4cf1e9d5b6b5d167e2b8f15cde24e15a25dc438327523199fb0a4c` |
| Full zensim release library suite, all features | **Pass:** 570 passed, 0 failed, 9 inherited ignored. The first run set a process-wide formula override and failed; the corrected default-environment rerun passed. | `/var/tmp/gmsbank/command_records/final_lib_suite_default_rev.log` (SHA256 `a273649490a2639a262e5e4d4f410cdfcbd693f09df6bbd0d4cf8218c6d563d4`) |
| Format, `just clippy`, `just lint-scripts`, public API snapshot | **Pass:** format clean; Clippy clean after fixing two C8 iteration warnings; 638 scripts runnable; API snapshot check 1/1. Exactly two approved public Rust items were added. | `/var/tmp/gmsbank/command_records/final_{fmt_check,clippy3,lint_scripts,api_check}.log` |

The independent reference initially used f64 for all gradient magnitudes and missed the 1e-6 bar on a near-zero slot (`2.2695334219725836e-06` relative). The production gradient pass uses f32 for complete V8 interior chunks, then f64 for borders and tails. The separate NumPy implementation now mirrors that precision boundary and passes without changing the Rust family or its constants. Both runs are retained in the command records.

Polarity 4/8 on four graphics references; the gate's expectation holds on photographic content and fails on graphics; the signed split is content-dependent. The four graphics references (5012 map brochure, 7000 line plot, 8416 web screenshot, 9012 clipart) had no recorded selection rule. The box blur spreads edge gradients into flat flanking pixels on these references. No constants, references or blur strength were changed after observing the miss.

The Opus review's **not-preregistered diagnostics** (`/home/lilith/tmp/zensim-paper/rev4/REVIEW_GMSBANK.md`, correction 3) found blur loss > gain 4/4 (by 4–14×) and noise gain > loss 4/4 on four imazen-26 TRAIN photos (1248, 1552, 3004, 3302). Its separate NumPy gamma-luma proxy found blur loss dominance on 5/5 photo and 5/5 mixed references, versus 2/5 line-art and 3/5 screen references. These are exploratory diagnostics, not a replacement gate or an acceptance pass. On screen and line content, C8 gain can rise under blur.

Calibration covered 12/32 content/size strata; 20/32 were empty, including every tiny and small stratum. The source-constant size-sweep rule remains unmet. The four graphics fixture references had no recorded selection rule. The first all-f64 NumPy reference missed the 1e-6 tolerance at `2.2695334219725836e-06` relative; the passing reference mirrors production f32 interior magnitudes. The four `feature_plan::servability_census` tests in the release suite passed, with **0 registered producer sets refused**.

The original calibration ratio/XYB producer was restored from the rescue archive as `zensim/src/gmsbank_calibration_instrument.rs`, with explicit runner `scripts/gmsbank/calibration_instrument.sh`. A full 652-pair replay exited 0 and reproduced the original `ratios.tsv` and `xyb8.tsv` SHA256 values exactly (`767197df…`, `58986200…`). The runner requires a fresh input directory and does not silently skip missing input. Exact command, UTC bounds and hashes are in `/var/tmp/gmsbank/command_records/correction_calibration_replay.json`.

## Recompute from raw evidence

From the lane workspace:

```sh
python3 -c 'import json; r=json.load(open("/var/tmp/gmsbank/identity/report.json")); print(len(r["modes"]),r["compared_cells"],r["differing_cells"])'
```

Actual output: `54 3426624 0` (18 tier/thread cells × three pairwise comparisons).

```sh
python3 scripts/gmsbank/dead_slots.py /var/tmp/gmsbank/identity/cid22_native_mt1_on.csv /var/tmp/gmsbank/identity/safesyn_native_mt1_on.csv /var/tmp/gmsbank/identity/kadid_native_mt1_on.csv
```

Actual final JSON line: `{"dead_slots": [], "maximum_nonzero_pairs": 142, "minimum_nonzero_pairs": 142, "rows": 144, "slots": 180}`.

```sh
python3 scripts/gmsbank/numpy_reference.py /var/tmp/gmsbank/calibration/xyb8.tsv /var/tmp/gmsbank/calibration/first8_features_final.csv /var/tmp/gmsbank/calibration/report.json
```

Actual output: `{"cells": 1440, "max_relative_error": 3.0291868626664007e-15, "pairs": 8, "worst_local_slot": 3, "worst_pair": 2, "wrong_c_rejections": 1440}`.

```sh
python3 -c 'import json; r=json.load(open("/var/tmp/gmsbank/corpus_probe/report.json")); print(r["pairs"],r["algebra_cells"],r["polarity_passes"])'
```

Actual output: `84 336 4`. The eight per-reference loss/gain comparisons and 19×5 zenjpeg ladder are in the report.

```sh
rg 'peak heap memory consumption:' /var/tmp/gmsbank/command_records/measure_first_heap_analyze.log
```

Actual output: `peak heap memory consumption: 613.80M`.

```sh
python3 scripts/gmsbank/identity_compare.py /var/tmp/gmsbank/measure_first/base_features.csv /var/tmp/gmsbank/measure_first/features.csv
```

Actual JSON: `{"cells": 2644000, "differing": 0, "first_differences": [], "left": "/var/tmp/gmsbank/measure_first/base_features.csv", "right": "/var/tmp/gmsbank/measure_first/features.csv", "rows": 2000}`.

```sh
python3 -c 'import json; r=json.load(open("/var/tmp/gmsbank/peer_gmsd/_MANIFEST.json")); print(r["set_count"],sum(v["unique_pair_keys"] for v in r["sets"].values()),r["total_rows"],sum(v["decoded_pair_key_checks"] for v in r["sets"].values()),all(v["key_check"] for v in r["sets"].values()),r["sets"]["cid22_b"]["rows"])'
```

Actual output: `18 248983 249227 248983 True 2100`. `python3 /var/tmp/gmsbank/verify_peer.py` returned `{"sets": 18, "unique_keys": 248983, "rows": 249227, "cid22_b_rows": 2100, "parquet_key_score_mismatches": 0, "binary_sha256": "2ed8f6765e2f9a7bca5a4ea4a3b8b7b266fb10c57b5b0c9f5563f978c2108ea6"}`. The peer scorer binary SHA256 is `2ed8f6765e2f9a7bca5a4ea4a3b8b7b266fb10c57b5b0c9f5563f978c2108ea6`; every set's UTC command, exit, TSV SHA256 and parquet SHA256 is in `/var/tmp/gmsbank/command_records/peer_*.json` and the peer manifest.

## Limits

This is an implementation and qualification record. The potential arms P0–P3 are proposals in `benchmarks/rev4_gmsbank_potential_prereg_proposal_2026-09-23.md`; fitting and human-label evaluation belong to the potential owner. The graphics-fixture polarity gate remains a 4/8 miss. Whether to preregister a content-stratified polarity gate is the coordinator's decision.
