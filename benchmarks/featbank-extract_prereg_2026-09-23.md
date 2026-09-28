# Preregistration — `featbank-extract` lane (2026-09-23)

Brief: `/home/lilith/tmp/zensim-paper/rev4/FEATBANK_EXTRACT_brief.md`, binding
`/home/lilith/tmp/zensim-paper/rev4/DEVIN_COMMON.md`. Design:
`docs/REV4_FEATURE_BANK_PLAN_2026-09-23.md` §2 (as committed in workspace
`zensim--rev4-featbank` change `skkovssn` / commit `2b5b4c20`).

## Scope of this preregistration

This lane **fits nothing and analyses no human label**. Part A converts existing
Rev3 `ceiling_rev3` f64 feature caches into the f32 bank layout and rebuilds
`pair_key`. Label columns are copied through **byte-identically as data** —
TRAIN-role labels into `labels__<source>.parquet`, and the KADID SELECT human
label column into `bank/_sealed/kadid_select/labels__human.parquet`, which this
lane never opens for analysis. No statistic, correlation, ordering or selection
is ever computed on a label column. The only operations on any label column are
copy / seal.

Pixel-hash note: the ceiling `human_*.parquet` caches carry no pixel hashes and
the ceiling extraction run did not pass `--audit-jsonl`, so no per-row pixel-hash
record exists for them. This lane re-derives `reference_pixels_sha256` /
`distorted_pixels_sha256` by running the **same pinned extractor binary**
(sha256 `7c7ffbbfa033e8ca1a8f103d472b61ccde061c2394b03d519af2852ee8eeda87`,
recorded in `CID22_ADMISSION.json` / `SAFESYN_ADMISSION.json`) in
`--audit-jsonl` mode over the declared input paths. Validity is established
before the run by a 25-pair probe against the ceiling audit probes
(`audits/human-full944-h128-full-s5101.jsonl`): all 25
reference/distorted pixel-hash pairs must match exactly, **and** the
re-extracted f64 `f0..f943` must equal the stored parquet columns bit-for-bit
on those rows. (Probe result: 25/25 hash pairs identical, 25/25 rows × 944
features bit-identical — recorded in the worklog before the full run.)

## Inputs (sha256)

| input | sha256 |
|---|---|
| `/var/tmp/zensim-validation-2026-09-14/baseline-recovery/cid22-train944.parquet` | `fb666c4255e9b6df6f4d6778829708f3bcbcd3bcc2127c6fb90d6e9b38f33a7a` |
| `…/cid22-train-audit.jsonl` | `9b8a1404930946979173b4e6ec31427207099e70e52b8d166e7ec1b013f1313f` |
| `…/cid22-train944.csv.manifest.json` | `351a6879435384bb91d339ec0b6fc3cfc1a7d3d2ce4232ba3a6e84691c2600e9` |
| `…/cid22-train944.csv.producer.bin` | `915cc4c96b54943785c9d3063c1137234959088280e7274dac9fe6913ef0b90a` |
| `…/CID22_ADMISSION.json` | `0a64ad74ce5946bff3ef8abb72137f26339d480541df598e38f818a65c4ae2b9` |
| `…/CID22_EXTRACT_STATUS.json` | `e3848aab9fee256a4da6424d2c3d873282688c165af8c9bf3f172236e7cd14f7` |
| `…/CID22_VERIFIED.json` | `2d435b815b620e5c969f2735773ea51867b07205e1ebc82f060f9220122f0d4f` |
| `…/cid22-train-pairs.tsv` | `0ec4576a41415424b99dcacd0f45ed5a543a72d32410f4e88f289b392da3bc2b` |
| `…/safesyn-train944.parquet` | `6044fdc8cf4f646cda6457a9e73e54edd7d9fed57dfd95f1c47e1f4091220c68` |
| `…/safesyn-train-audit.jsonl` | `a6766a5aa973c73c3e9c8d0e94a86bb1034bde691a6e21a7f4b2a7ef96064b22` |
| `…/safesyn-train944.csv.manifest.json` | `351a6879435384bb91d339ec0b6fc3cfc1a7d3d2ce4232ba3a6e84691c2600e9` |
| `…/safesyn-train944.csv.producer.bin` | `915cc4c96b54943785c9d3063c1137234959088280e7274dac9fe6913ef0b90a` |
| `…/SAFESYN_ADMISSION.json` | `9ced0f04adecfcabebf05c136c05f1c4849bfbc9f798734af642c4d4ee70fd4b` |
| `…/SAFESYN_EXTRACT_STATUS.json` | `6d982302c77f1a260c76d7db193ea14a387e81c144e9463db18c99c4d9e961d5` |
| `…/SAFESYN_VERIFIED.json` | `f8660d8df8a240a79d751475af726a20e1733b649198ba01795a9396c90365d8` |
| `…/safesyn-train-pairs.tsv` | `5a53976070a5e21b2bb7fe0d05f58b510b93e141dd9207dd3e2cd337fd15cd3b` |
| `~/work/zensim-validation-2026-09-13/ceiling/final/human_fit.parquet` | `da8fa67109527470f73152a46665b364f4418d5201083ad28c145b3b4ad51e58` |
| `…/human_half.parquet` (subset view; not separately banked) | `f2c54c3098e7b278b47b56f6aeccb59555352719297c548e039d0ab5e54afb75` |
| `…/human_dev.parquet` | `a0d4fe5dbd21062a13430505f096b4dd7f7780fc8f9306a24876c06a45c29282` |
| `…/human_test.parquet` | `dbe6a1e91043bc73c195b3604af6aedf523ca172323b4133628d2b0900c1b74b` |
| `…/_MANIFEST.json` | `593a68f6bc6570be583bc194d4a623858b3c464f0f768ab0a72a96f1e9994ecf` |
| `…/INPUTS.json` | `44a574b77538c1971a2a18508ef7b1ccd93f517e8cd9bd0cd72de14d3b056c9c` |
| `…/pairs.tsv` | `43c409ffbf16dea899f34bb19ef1392e9dc67f24c9ea7ce0f63bfec740bbc514` |
| `…/features.csv.manifest.json` | `351a6879435384bb91d339ec0b6fc3cfc1a7d3d2ce4232ba3a6e84691c2600e9` |
| `…/features.csv.producer.bin` | `915cc4c96b54943785c9d3063c1137234959088280e7274dac9fe6913ef0b90a` |
| extractor binary `…/native-integrity-admission/extract-native-admission` | `7c7ffbbfa033e8ca1a8f103d472b61ccde061c2394b03d519af2852ee8eeda87` |

## Sets and conversions (Part A)

| bank set | source | expected rows | label handling |
|---|---|---:|---|
| `cid22_train` | `cid22-train944.parquet` + `cid22-train-audit.jsonl` | 17,611 | `labels__ssim2_oracle.parquet` (stored `human_score` col = peer-SSIM2 oracle) |
| `safesyn` | `safesyn-train944.parquet` + `safesyn-train-audit.jsonl` | 196,086 | `labels__ssim2_oracle.parquet` (`human_score` + `original_oracle`) |
| `kadid_train` | `human_fit`+`human_dev` rows with `corpus=kadid` | 5,000 | `labels__human.parquet` (TRAIN-role; copy only) |
| `tid2013` | `human_fit` rows with `corpus=tid` | 3,000 | `labels__human.parquet` (TRAIN-role; copy only) |
| `kadid_select` | `human_test` (all `corpus=kadid, role=test`) | 3,125 | features+keys only; `human_score` sealed to `bank/_sealed/kadid_select/` |

`pair_key = sha256(ref_pixels_sha256 ‖ dist_pixels_sha256 ‖ input_contract)`
(hex), `input_contract = "legacy-rgb8"`. Sidecar: `pair_key` + one `f<ID>` f32
column per populated ID (905 columns; the 39 structural-zero IDs are listed in
each `_MANIFEST.json`, not stored), zstd level 3, `BYTE_STREAM_SPLIT` column
encoding, `use_dictionary=False`, row groups of 65,536 — the exact writer
configuration measured in `benchmarks/rev4_featbank_2026-09-23/f32_cast_probe.py`
(design lane).

## Statistics / decision rule

No inferential statistic exists in this lane. Acceptance is the brief's
Acceptance A checklist, evaluated mechanically:

1. row counts equal the source (17,611 / 196,086 / 5,000 / 3,000 / 3,125);
2. `pair_key` unique per set;
3. every stored f32 equals `np.float32(source f64)` exactly — **full check**,
   all cells, not a sample;
4. key join with the audit is complete (every row gains a distorted-pixel hash;
   any row without one is refused);
5. per-file sha256 recorded in each `_MANIFEST.json`;
6. measured bytes and B/row per set.

Any failure of 1–4 refuses the conversion (no output is declared valid); it is
reported as-is.

## Part B (out of scope of this prereg's data handling)

New-family sidecar extraction is gated on the `featbank-impl` lane landing and
on `FEATBANK_FLEET_GO.md` for any fleet run. Neither exists at preregistration
time. If it unblocks, the extraction runs under `~/tmp/devin/heavy`, measured on
2,000 SafeSyn rows first; held-out-set feature extraction is additionally gated
on explicit instruction (ruling D4 now permits pixels-only extraction, but this
lane still extracts only the TRAIN-role sets named in the brief plus any set the
coordinator explicitly names).

## Seeds

None — nothing is sampled, fitted or randomized. The 200-row re-extraction check
in Acceptance B, if reached, uses a recorded seed.

## Amendments (dated)

**Amendment 2026-09-23-A — acceptance rule 1 vs content-addressed dedup.**
Rule 1 above expects set row counts equal to the source (5,000 / 3,125 for the
KADID ceiling sets). The bank deduplicates by `pair_key`: pixel-identical
stimuli collapse to one key row (kadid_train 5,000→4,880, kadid_select
3,125→3,050, kadid_terminal 2,000→1,952, csiq 866→865). Stimulus counts are
unchanged; `keys.n_stimuli` records multiplicity and labels keep every source
row. Collapsed rows are verified to carry bit-identical features before dedup.

**Amendment 2026-09-23-B — KADID SELECT labels unsealed.**
The plan draft sealed eval-role labels under `bank/_sealed/`. Ruling D1
(2026-09-23) makes KADID SELECT potential-exposed, so its labels are usable
by fit lanes and are emitted unsealed in-set. The same applies to konfig_val,
cid22_a25 and aic3 (potential-exposed) and konjnd_bpg_val (LODO-exposed at
fold eval). Confirmation sets emit no labels file at all (D4 pixels only).

**Amendment 2026-09-23-C — sealed held-out label replicas (review corr. 4).**
Extraction pairs/raw files that replicated held-out `human_score` columns
were moved under `/var/tmp/rev4-featbank/_sealed/` (move, not regenerate;
sha256s unchanged). Future extraction of held-out pairs writes a placeholder
score, as the 25-row probe did.
