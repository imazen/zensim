# Rev5 entry and exact-refinement audit — 2026-10-04

Owner directive: the standard 420-ID by_v2fy bake must serve Rev5 through every valid entry. The earlier blanket local_refine refusal is removed. Rev5 replay uses finite XY window halos, canonical production strip partials and ordered stable-moment merges. Historical replay remains at revisions 1–4.

Qualification limits first: the latency matrix and optimization stop rule remain **UNQUALIFIED** under strict zenbench because an unrelated exhaustive f32 process (PID 760429) continues using a CPU. It was not interrupted. This report contains engineering serving/refinement evidence, **not newly trained Rev5 model quality**. Existing sampled-HDR, legacy sampled-profile and SDR-only cache domain limits are recorded in `~/tmp/zensim-paper/rev4/REV5_decisions.md`.

## Entry checks

Each row names a real behavior test; all use a revision-5 child or explicit process environment. Feature tests compare consumed values/full vectors against canonical extraction; score tests use the complete Rust BakeScorer pipeline.

| Entry | Dedicated test |
|---|---|
| BakeScorer::compute, identity, persistent scratch, cached attribution | rev5_entry_compute_identity_and_cached_attribution |
| prepare_steering + refinement_gain, env off/on | rev5_entry_steering_env_on_and_off |
| local_refine production replay: JPEG/textured, multi-strip, edge/unaligned, non-reference candidate, no-op, missing coarse planes | rev5_exact_refinement_goldens |
| Zensim Custom by_v2fy: scalar, codec hint, training params, classification, reference, reference scratch, all four strip wrappers, generic diffmap, cached diffmap, linear-planar diffmap | rev5_entry_diffmaps_and_strip_variants |
| Declared HDR BakeScorer score/identity/steering + exact PU channel rebuild, env off/on | rev5_entry_hdr_score_and_steering |
| PU-linear interleaved, planar with row padding, descriptor auto-dispatch and computed identity | rev5_entry_raw_hdr_profiles |
| Folded supported extraction and explicit append/append2/CSFW/DVIFM refusal, including streaming append; Linear/PQ/HLG HDR bake/session parity | rev5_entry_hdr_encodings_and_wide_layout_prefixes |
| Full 576-slot extraction regardless of the bake's 420-slot serving subset; parity with independent research extraction | rev5_all_features_is_complete_independently_of_bake_reads |
| Explicit unsupported raw family selectors, v1_only on/off, SDR/PU, pair/cache/streaming, masked/IW config and retained 944 | rev5_raw_family_toggles_refuse_before_narrowing |
| Basic-only f22/basic/basic+peaks steering: zero metadata with scalar feature/score parity | rev5_basic_steering_uses_zero_mean_offset |
| Rev5 mean-offset path does not read any pixels | streaming::rev5_retained_mean_offset_does_not_read_pixels |
| SDR sampled by_v2fy: v1 Y triangle 3/2, v1 XYB Mitchell 2, v2 XYB triangle 1/2/4/8, v2 XYB RobidouxSharp 1/3/5/7; scalar/session/tier parity | rev5_entry_sampling_plans |
| Bounded-v2 pair/toggles/cache/cache-moments/scratch and folded-720 pair/streaming | rev5_entry_v2_pair_cached_and_streaming_extraction |
| Ensembles: complete scalar/steering score and vector | rev5_entry_ensemble |
| Supported corruption companion preserves all read IDs during spatial probes | rev5_entry_corruption_companion |
| Research Request::for_slots basic+peaks+v2: full layout, exactly 576 emit slots/producer scope | rev5_basic_peaks_v2_request_computes_only_the_three_families |
| Validate stamp, bake densify, verdict before/after, panel, Parquet densify | scripts/check_rev5_tool_entries.py (six CLI runs) |
| Dense bake ID map survives pruning; prefix/scattered dense regression gates | bake_dial_refit densify tests |
| Manifest/revision admission for Rev4/5, refusal of unknown 6 | feature_set revision admission tests |

Generic weighted diffmaps still mean basic-weight spatial error; their score/vector now come from the full Rev5 bake plan. Use the prepared BakeScorer session for the complete bake sensitivity map. Rev5 strips delegate to the canonical fixed 128-row tree. Raw planar/interleaved PU inputs materialize an opaque RGBA adapter while preserving absolute linear-sRGB nits and row strides; typed BakeScorer HDR stays row streamed.

The independent review found three errors in the original entry receipts.
The fix round rejects explicit unsupported extraction requests instead of
accepting zero-filled families, computes all 576 supported features for
`compute_all_features`, and removes the global mean-offset pass from
basic-only steering. SDR/PU `compute_extended_features` requests masked
features and therefore refuses at Rev5, even for a supported bake; use full
supported extraction or an explicit supported research request. This is
separate from a wider storage layout with only supported requested slots,
which remains valid. The original zero-tail assertions and scoring-subset
comparison are superseded by the rejection/completeness tests above.
Fresh fix-round gate receipts are recorded below; older measurements
retain their original binary hashes and qualification limits.

## KADID env-on 48-case panel

Sources: `/mnt/v/dataset/kadid10k/images`, I01/I21/I41/I61 reference PNGs and their distortion-10 JPEG-coded PNGs at levels 03/05. Three seeds per arm, six frozen prior-revision dense bakes restamped 5 for serving diagnostics, no fitting or model selection. Every case uses `ZENSIM_FORMULA_REV=5 ZENSIM_PREPARED_STEERING=1 ZENSIM_NEIGHBOUR_EXACT=1 RAYON_NUM_THREADS=1`, `--block 8`, taskset CPUs 8–15; eight independent case processes. Every row has complete refinement coverage and 3,072 rectangle queries.

Binary SHA-256: `92c2a088bb4b53f16d3be3eed4daabdad449f96904a0cd3329fbc1442f53cb5d`. Raw reports/results: `/var/tmp/rev5/owner-panel2`. Final halo-tightened results exactly match the earlier panel on score, M2, M3a, M3f and coverage; only timings/paths change.

| Arm | JPEG level | Cases | Median M3f | Minimum M3f | Median M2 |
|---|---:|---:|---:|---:|---:|
| by_v2fy | 03 | 12 | 0.982926851 | 0.970330905 | 0.999982194 |
| by_v2fy | 05 | 12 | 0.965433126 | 0.854778532 | 0.999697268 |
| v2basic | 03 | 12 | 0.931737923 | 0.892576610 | 0.999965019 |
| v2basic | 05 | 12 | 0.900805409 | 0.833646462 | 0.999992878 |

| Arm | Seed | Image | Level | Score | M2 | M3a | M3f |
|---|---:|---|---:|---:|---:|---:|---:|
| by_v2fy | 5101 | I01 | 03 | 69.1874619 | 0.999952584 | 0.386802766 | 0.985738623 |
| by_v2fy | 5101 | I01 | 05 | 13.8539886 | 0.999997498 | 0.488373165 | 0.972088710 |
| by_v2fy | 5101 | I21 | 03 | 64.6183701 | 0.999991006 | 0.483403186 | 0.983363415 |
| by_v2fy | 5101 | I21 | 05 | 12.0326014 | 0.982144149 | 0.499441358 | 0.854778532 |
| by_v2fy | 5101 | I41 | 03 | 71.6460037 | 0.999968981 | 0.490688282 | 0.978241497 |
| by_v2fy | 5101 | I41 | 05 | 5.1576405 | 0.999913342 | 0.385765012 | 0.954184221 |
| by_v2fy | 5101 | I61 | 03 | 62.7066536 | 0.999974934 | 0.438512137 | 0.982917102 |
| by_v2fy | 5101 | I61 | 05 | 28.9251461 | 0.999991440 | 0.247296874 | 0.941178155 |
| by_v2fy | 5103 | I01 | 03 | 69.8227921 | 0.999987566 | 0.312872335 | 0.985397996 |
| by_v2fy | 5103 | I01 | 05 | 16.0586872 | 0.999988985 | 0.409297406 | 0.985169499 |
| by_v2fy | 5103 | I21 | 03 | 63.1053314 | 0.999983212 | 0.426079997 | 0.983202905 |
| by_v2fy | 5103 | I21 | 05 | 6.8605361 | 0.998764657 | 0.381173933 | 0.980646628 |
| by_v2fy | 5103 | I41 | 03 | 73.2306442 | 0.999982589 | 0.398381184 | 0.970330905 |
| by_v2fy | 5103 | I41 | 05 | 10.1840181 | 0.999481193 | 0.484227458 | 0.963799803 |
| by_v2fy | 5103 | I61 | 03 | 66.8559570 | 0.999981800 | 0.504450707 | 0.981769862 |
| by_v2fy | 5103 | I61 | 05 | 20.6221504 | 0.998057286 | 0.141146185 | 0.956661600 |
| by_v2fy | 5107 | I01 | 03 | 68.5265732 | 0.999984575 | 0.291769663 | 0.982407514 |
| by_v2fy | 5107 | I01 | 05 | 8.6985836 | 0.999997550 | 0.331833353 | 0.976658069 |
| by_v2fy | 5107 | I21 | 03 | 63.7880096 | 0.999988842 | 0.250268165 | 0.982936601 |
| by_v2fy | 5107 | I21 | 05 | 16.1388016 | 0.998578188 | 0.401125567 | 0.982070049 |
| by_v2fy | 5107 | I41 | 03 | 68.7055054 | 0.999981007 | 0.353870918 | 0.971312204 |
| by_v2fy | 5107 | I41 | 05 | 9.3982944 | 0.999996478 | 0.447469184 | 0.964716085 |
| by_v2fy | 5107 | I61 | 03 | 64.0798569 | 0.999976077 | 0.372517069 | 0.982982425 |
| by_v2fy | 5107 | I61 | 05 | 22.4873772 | 0.999473067 | 0.283828959 | 0.966150166 |
| v2basic | 5101 | I01 | 03 | 72.8853531 | 0.999981050 | 0.306930870 | 0.950885935 |
| v2basic | 5101 | I01 | 05 | 7.4982347 | 0.999838350 | -0.004755431 | 0.878305871 |
| v2basic | 5101 | I21 | 03 | 65.4357910 | 0.999992925 | 0.384382696 | 0.931612108 |
| v2basic | 5101 | I21 | 05 | 12.0530558 | 0.999995781 | 0.313690410 | 0.949359395 |
| v2basic | 5101 | I41 | 03 | 72.0106735 | 0.999947601 | 0.218593856 | 0.913611174 |
| v2basic | 5101 | I41 | 05 | 11.7045956 | 0.999995020 | 0.343926452 | 0.846820102 |
| v2basic | 5101 | I61 | 03 | 61.3668175 | 0.994613814 | 0.418357919 | 0.897502936 |
| v2basic | 5101 | I61 | 05 | 22.6546173 | 0.999992460 | 0.337134651 | 0.926139585 |
| v2basic | 5103 | I01 | 03 | 72.6921539 | 0.999967169 | 0.240948621 | 0.952947903 |
| v2basic | 5103 | I01 | 05 | 12.2217169 | 0.999996478 | 0.167058148 | 0.896374453 |
| v2basic | 5103 | I21 | 03 | 67.2894897 | 0.999988406 | 0.539783275 | 0.941357626 |
| v2basic | 5103 | I21 | 05 | 8.0014830 | 0.999975998 | 0.211786929 | 0.833646462 |
| v2basic | 5103 | I41 | 03 | 74.3391190 | 0.999978991 | 0.279027677 | 0.892576610 |
| v2basic | 5103 | I41 | 05 | 9.9040432 | 0.999972947 | 0.413984832 | 0.870006785 |
| v2basic | 5103 | I61 | 03 | 61.7621574 | 0.999991851 | 0.334063905 | 0.931863739 |
| v2basic | 5103 | I61 | 05 | 15.9970360 | 0.998067284 | 0.219692516 | 0.838982363 |
| v2basic | 5107 | I01 | 03 | 73.5768814 | 0.999846653 | 0.258607353 | 0.960839287 |
| v2basic | 5107 | I01 | 05 | 8.5902863 | 0.999996291 | 0.072874064 | 0.959646533 |
| v2basic | 5107 | I21 | 03 | 68.9851685 | 0.999957585 | 0.400353329 | 0.924541645 |
| v2basic | 5107 | I21 | 05 | 11.2804527 | 0.999995239 | 0.433907475 | 0.955211391 |
| v2basic | 5107 | I41 | 03 | 73.0385132 | 0.999962870 | 0.304994477 | 0.920660798 |
| v2basic | 5107 | I41 | 05 | 16.3596859 | 0.998883618 | 0.324730284 | 0.905236366 |
| v2basic | 5107 | I61 | 03 | 64.1958923 | 0.999805144 | 0.362897138 | 0.958647441 |
| v2basic | 5107 | I61 | 05 | 26.6980457 | 0.999993296 | 0.365763528 | 0.907567134 |

## Query cost diagnostic

Pinned CPU 8, 1MP cost fixture, existing ignored `local_refine::tests::cost_per_query_1mp`; these runs overlap gates and the unrelated exhaustive job. They are measured diagnostics, not strict quiet-box certificates. Rev5 retains 16,464 extra bytes for production strip partials. Replay recalculates affected strip pools to preserve their canonical reduction order; only blur windows/candidate gathers are restricted to the finite cone.

| Revision | Snapshot heap bytes | Full fold ms | 8×8 query µs | 32×32 query µs |
|---:|---:|---:|---:|---:|
| 3 | 54,067,200 | 131.2 | 6646.7 | 8362.7 |
| 4 | 54,067,200 | 137.8 | 2278.9 | 2583.8 |
| 5 | 54,083,664 | 75.1 | 2023.8 | 2228.7 |

Test executable SHA-256: `899c452d3863e73b7dde2672a2acb1f8268a5cc2bd9bb57b7356ff1c5f931706`. Raw logs: `/var/tmp/rev5/owner-query-cost`.

## Data/model provenance

The CLI fixture uses independently audited Rev5 vectors with synthetic engineering targets. Its root manifest identifies that purpose and source paths; it is never a human label table, training admission or quality panel. The restamped bakes preserve the original feature-set metadata and frozen weights: that old producer identity is not a claim of newly trained Rev5 provenance. Research producer tests separately enforce the true basic+peaks+v2 scope.

| Model file | SHA-256 |
|---|---|
| human-by_v2fy-h128-full-s5101.bin | `1324bc8548f4a1f829f076ddb9a29d058167d5770f67f5e15c8c7e7604c974c2` |
| human-by_v2fy-h128-full-s5103.bin | `b7bfbaaa5a67ca1277b5efccb6d70ab9c38cda72a37b276f127c83eacba626d1` |
| human-by_v2fy-h128-full-s5107.bin | `c2279081330e4f2852200874c5f61901de25c2f9dd98635bf0d8e1ac39bedd99` |
| human-v2basic-h128-full-s5101.bin | `885e59ff3186ace1323bc402036de4a5294dc209a9624fa763ada6bed37ed054` |
| human-v2basic-h128-full-s5103.bin | `a52bc6f8e700f904c206661082d62045a4876bbcf5527601b5ca5f156c2a32bb` |
| human-v2basic-h128-full-s5107.bin | `103f41216e7ed4031c0ebc8be039812f1b817ddf6672072b377815f4da376927` |

| Input PNG | SHA-256 |
|---|---|
| I01.png | `2a1d70b387662023765bd19ad16c93a18547798b5add9c6935498eb91dfa2245` |
| I01_10_03.png | `4003d1fb3841ec32618bac7dfe699cd22dfa239723b4eed3faf158b190a0a2af` |
| I01_10_05.png | `7a6a8019b48a76b2c1a9824ad78fa8579d5ecf295cb79f3ba03dc60ae6dd8a3e` |
| I21.png | `aa22eea63123827574ef1357cf962aba829842ded700586f7d2d52226392a94b` |
| I21_10_03.png | `9926d158b61a6485668c0e0a3177a83e67a83ced115baa071df1f5a3396b5e09` |
| I21_10_05.png | `45a2b2333f9c19495d526b92c01a95f1eb0ad3eda7845cce2880a2f119916881` |
| I41.png | `de642e794e0576164b46048f7d44b7ec055dab8d62867c5b5e4b97fdfee27140` |
| I41_10_03.png | `6da85a206f919b34c859890455e4280dcd76e688c051bcee2291cdd1712939db` |
| I41_10_05.png | `a7381fdd2957b0ae035d892fe66ed3369d452d798b3dafd711be687d3fff95ff` |
| I61.png | `b83aa2a505de4dcec451e07126445111b260804d931698c13b92b3427dbeb701` |
| I61_10_03.png | `8b08328518958fafa37bea5baf248580e664b52dc4d713419dfc59fe5914da5c` |
| I61_10_05.png | `e73f0b586740dcdc1bd829a1c0c585581a7b5530964d231880cb3e829c8ec1a2` |

## Preserved pre-review correctness and provenance receipts

Full release suite passes (629 library tests, plus every release integration/doc target; 13 ignored diagnostic tests). The latest all-features library suite passes 645 tests, 13 ignored. CI-exact Clippy, formatting, script lint, rev4serve and public API checks pass. All 27 feature cells pass both Clippy and tests (54 checks). The API delta is only the authorized FormulaRevision::Rev5 variant.

Twenty-one foreign checks pass: WASM SIMD128, native i686 scalar and AArch64 via QEMU each run full-vector/native-file parity, identity, custom-profile cached/strip/diffmap entries, raw HDR, four sampling contracts, bounded/cached/streaming-v2 entries, and exact local refinement. The foreign library checks use no default features plus custom-profiles, feature-regime-v2 and training. WASM uses the pinned +1.98.1 toolchain and explicitly forwarded revision/parity-file environment; i686 uses the host 32-bit loader; AArch64 uses clang cross-linking and qemu-aarch64.

Fresh native tier audit: 48 vectors at each revision, Rev1–4 all 192 files unchanged versus the frozen historical archive; Rev5 unchanged versus its final arithmetic reference and identical across native tiers. Independent two-pass moments: 264 checks, maximum relative error 8.665431753e-14. The historical Rev3/4 owner and broad steering comparisons again pass with zero differing numeric JSON fields. Final by_v2fy work census passes all 16 score/map × 1/4MP × 1/8-thread × v4x/v3 configurations: no scale-0 X/B terms/storage, no peaks, one vertical/activity chain, and no warm zero fill.

Raw gate locations: `/var/tmp/rev5/owner-final-gates2`, `owner-final-extra`, `owner-permutations2`, `owner-cross`, `owner-historical-steering`, `owner-final-audits`, `owner-final-census`, and `owner-tool-tests2`. An initial evidence driver named a nonexistent densify_feature_tables binary; its preserved log ends rc=101 before executing a tool. The corrected rescore_parquet build and all six CLI runs pass in `owner-evidence-rest`; no failed application check is counted as a pass. Earlier custom-entry failures and a feature-specific Clippy failure are preserved as pre-fix evidence.

| Engineering fixture/artifact | SHA-256 |
|---|---|
| ext_kadid.parquet | `03edf9d08ce79e92339f7623653f6a9eb56ec533a2b297d03feeb68febf21135` |
| dense-table.parquet | `9773275519f7d6b21136dc533270b2abae1a28505c43600519da04b406ae7a6c` |
| _MANIFEST.json | `2a98f41dd90b9d6cb95281f03e10e6cb2aa4bc38c8fff76603799c83ec9b24e0` |
| before.tsv | `7d09fd10cd143fd3f28e51dd0647f39324eb722d87a23cbc2d3f3248ac2ab29f` |
| after.tsv | `7d09fd10cd143fd3f28e51dd0647f39324eb722d87a23cbc2d3f3248ac2ab29f` |

The final post-directive [speed matrix](rev5_speed_final_2026-10-04.md) contains all 192 measurements and 48 fixed/per-pixel fits. Every run remains strict-gate unreliable; it does not certify the stop rule. The earlier [engineering ladder](rev5_ladder_2026-10-04.md) is preserved with stage/binary identities.


## Independent review fix round — final receipts

| Gate | Fresh fix-round result |
|---|---|
| Full release suite | 634 library passes, 13 ignored; all integration/doc targets pass |
| All-features library | 649 passes, 13 ignored |
| CI-exact Clippy, fmt, script lint, Rev4 serving, API snapshot | All pass |
| Feature permutations | 27 cells, 54 Clippy/test passes |
| Foreign checks | 33 passes: 11 each on WASM SIMD128, i686 scalar, AArch64/QEMU |
| Historical vectors | 192 Rev1–Rev4 files unchanged; 48 per revision |
| Rev5 vectors/tier parity | 48 audit files match; frozen native reference preserved |
| Independent moments | 264 checks, max relative error 8.665431753e-14 |
| Local-refine goldens | Native and all three foreign targets pass; original 1e-10 + 1e-6 relative delta bar retained |
| Historical steering | Rev3/Rev4 each: 48 owner cases and 384 broad rows, zero differing results |
| Work census | All 16 configurations, 32 warm assertions, pass |
| Validation CLI entries | All six pass; verdict rows and retained table columns unchanged |

Fresh receipts are under `/var/tmp/rev5/fix-native-gates`,
`fix-permutations`, `fix-cross-gates`, `fix-vector-audits`,
`fix-historical-steering`, `fix-work-census`, `fix-tool-entries`, and
`fix-rest-gates`. The driver sources are `/var/tmp/rev5/fix-*.py`. Heavy
commands use run-heavy 16 GiB / 8 jobs and private targets; native runs pin
CPUs 16–23 and foreign runs CPUs 24–31. The foreign runners/toolchains and
frozen native parity file are unchanged. Original and failed receipts remain
preserved. Historical steering PASS here means numerical identity with the
frozen baseline; pre-existing individual quality FAIL cases remain unchanged.

All three findings were reproduced with the retained reviewer executable
before edits (`fix-reproduction/reviewer-probe-before.log`). Its 97×83 probe
returned a zero-filled 204-slot append family, omitted nonzero f0 from
all-features extraction, and reported different scalar/steering offsets.
The new tests reject all explicit unsupported selectors on the raw routes,
compare all 576 supported slots against an independent research request, and
check f22/basic/basic+peaks metadata and score parity across tiny, odd and
multi-strip inputs. The empty-plane mean-offset regression verifies that no
global image read occurs. Wider supported requests still compute through a
validated plan, including the 944-layout refinement goldens and full-width
research request. Unsupported raw extended extraction is an explicit error.

Fresh census binary SHA-256: `e0aea8cd9b840a987ea3f0f2552665543aeabc2d8d5ccae30bd4cf48ec3a2dfc`.
The fresh engineering fixture is `/var/tmp/rev5/fix-tool-entries`; its source
and dense Parquet bytes, before/after verdict TSVs, stamped bake and dense
bake are identical to the preserved fixture above. Manifest SHA-256
`5b50065cbefe4cfce2a23c21a8ef6d2de8f3a1ae681de39889740a69055e1ab2` records the new audit paths.
The preserved 48-case panel and latency matrices use their original
pre-review binaries; they are not final-fix performance measurements.
Quiet-box qualification remains missing. No model was fitted or promoted.
