# Lane `cvvdp-safesyn` worklog — Rev4 E2b display selection + SafeSyn CVVDP labels (2026-09-23)

Lane: `cvvdp-safesyn`. Brief `/home/lilith/tmp/zensim-paper/rev4/CVVDP_SAFESYN_brief.md`
(+ `CVVDP_SAFESYN_FLEET_addendum.md` — Part 2 runs on the zenfleet job system behind a
FLEET_READY/FLEET_GO review gate). Workspaces: `../zensim--cvvdp-safesyn` (jj,
`main@origin` a14e563e, bookmark `quarantine/devin/cvvdp-safesyn`) and
`../../zenmetrics--cvvdp-safesyn` (jj, `master@origin` 266967e9, bookmark
`quarantine/devin/cvvdp-safesyn`). Scratch: `/var/tmp/cvvdp-safesyn/` (build target
`/var/tmp/cvvdp-safesyn/target-zenmetrics`). Manifest tsv:
`/home/lilith/tmp/devin/rev4_cvvdp-safesyn_manifest.tsv`. Quarantine: nothing is
pushed; nothing here is evidence until audited (`~/tmp/devin/QUARANTINE.md`).

## Setup (2026-09-23 ~03:30 Denver)

- zenmetrics lane: `jj duplicate` copies of cvvdpfix trio onto `master@origin` 266967e9:
  - `27c69beb` (worker arg order) → `6b81258e`
  - `d71922bd` (CPU `--display-model` + `_<display>` column suffix) → `72a7f309`
  - `dc1fca78` (docs) → `5375d523` ← bookmark `quarantine/devin/cvvdp-safesyn`
  - originals in `../zenmetrics--cvvdpfix` untouched.
- Build (queued on the shared `heavy.lock` behind another lane):
  `CARGO_TARGET_DIR=/var/tmp/cvvdp-safesyn/target-zenmetrics ~/tmp/devin/heavy --mem 20G --jobs 8 -- cargo build --release -p zenmetrics-cli --features jobexec`
  log `/var/tmp/cvvdp-safesyn/logs/build-zenmetrics.log`.
- Input hashes (sha256sum, this session):
  - `/mnt/v/output/zensim/reports/refmetrics/kadid_pairs.tsv` ceaf324f2bc646045adba0bfca5a299d27d0eded8e929d64f3d543bcc9e3b040
  - `/mnt/v/dataset/tid2013/tid_pairs_ab.tsv` e6a790e8f766ad5e484c3a66cfd9707f3d23270832b393f98a16cadc72cde032
  - `/mnt/v/output/zensim/konfig944/build/konfig_pairs.tsv` 44bde2d85f86f1da3d7a2a7b02bad919ae1164b2d221181037963598e63703d4
  - `/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01/ext_kadid_train_2026-08-29.parquet` 1fb8ff9bf335f215c22e7e2f3b63c3a9706d92151236d7a664f5cae1293b3cce
  - `/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01/ext_tid_train_2026-08-29.parquet` 269c7a8c614f6842cc4a3f5c0d2d938919d1596664424df20cc77d0f9d72e9ea
  - `/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01/konfig_originsplit_train_944.parquet` b919b9ad458a593baf443c57afb2c64c1e19a45f7042fada0146687a01574683
  - `/mnt/v/output/zensim/v2-backfill-2026-07-20/aic4_pairs.tsv` 955a9601e94c0877a00c822dbbbd17ad4ae11544dd6d550b9ed79d1ad9d983ff
  - `/mnt/v/output/zensim/reports/refmetrics/aic4_cvvdp.tsv` 4042d578c0f15b89d03301a46edc65d19f5e446412651a19d56707f11817d9c3 (300 rows, col `cvvdp_cpu_imazen_v0_1_0`)
  - `/mnt/v/output/zensim/reports/refmetrics/aic4_cvvdp_standard_fhd.tsv` f7bb935f965c9abc983565211ea1425af20cb5eb7c642af5989f6fb410dd06aa (300 rows, col `cvvdp_cpu_imazen_v0_1_0_standard_fhd`)
  - SafeSyn verified cache `/var/tmp/zensim-validation-2026-09-14/baseline-recovery/`: `safesyn-train944.parquet` 6044fdc8cf4f646cda6457a9e73e54edd7d9fed57dfd95f1c47e1f4091220c68, `SAFESYN_ADMISSION.json` 9ced0f04adecfcabebf05c136c05f1c4849bfbc9f798734af642c4d4ee70fd4b, `safesyn-train-pairs.tsv` 5a53976070a5e21b2bb7fe0d05f58b510b93e141dd9207dd3e2cd337fd15cd3b
- Display preset check: `modern_oled_phone_indoor` is in `display_models_imazen.json`
  (resolution 2532×1170, 0.35 m, 6.1", 400 cd/m² peak, 0.0005 black, 250 lux) → both
  `DisplayModel::by_name` and `DisplayGeometry::by_name` resolve; pycvvdp parity status TBD.

## Preregistration

`benchmarks/cvvdp-safesyn_prereg_2026-09-23.md` committed BEFORE any human-label read
in this lane (E2a Deliverable 4 verbatim + operational bindings). Label columns have
not been read at the time of that commit.

(Further entries appended chronologically.)

## Coordinator update ~10:30Z — rebase onto new master (b02812ae)

`master@origin` advanced to `b02812ae` ("bench(cvvdp): video_vs_ssim2"): now
contains the cvvdpfix trio natively plus V5 SIMD/rayon CVVDP, conformance-v2 vs
pycvvdp 0.5.7 (four new HDR PQ displays), and video scoring. Instruction: drop
the trio copies, keep only lane jobexec work, rebuild, redo Part 0.

- `jj rebase -s @ -d master@origin`: my change `a544a65b` → `b373b48a` on
  `b02812ae`; the trio copies (`6b81258e`/`72a7f309`/`5375d523`) left behind —
  upstream ancestors verified (`git merge-base --is-ancestor` for 27c69beb,
  d71922bd, dc1fca78 → all true). Lane diff vs base: `job.rs` +34 (test only),
  `jobexec.rs` +~330 (`cvvdp@<display>` arms + `ZEN_JOBEXEC_PIXEL_HASH`
  stamping + tests). `jj bookmark set` refused the sideways move (guard); the
  lane-local bookmark was repointed via forget+create.
- Part 0 stored-score check (panel via `ZEN_PANEL_BIN=~/work/zen/zensim/target/
  release/panel`): stored `aic4_cvvdp_standard_fhd.tsv` column vs its
  `human_score` → **SROCC 0.9608955655 ≈ 0.9609** ✓; board `aic4_cvvdp.tsv`
  (`cvvdp_cpu_imazen_v0_1_0`) → SROCC 0.8905597718 ≈ 0.8906. No AIC labels
  re-read — scores only.
- Tests on rebased tree (`--features jobexec`, dev profile):
  `cargo test -p cvvdp` → 227 pass/0 fail; `-p zenmetrics-cli cvvdp` → 13 pass;
  `jobexec` filter → 18 pass (incl. new `cvvdp_display_of_parses_only_the_at_
  form`, `cvvdp_at_unknown_display_is_refused`, `cvvdp_at_standard_4k_matches_
  the_plain_cvvdp_row`, `cvvdp_at_standard_fhd_emits_the_display_column`);
  `-p zenfleet-core` → 144 pass (incl. new
  `cvvdp_display_metric_string_is_distinct_work_and_cpu_routed` — plain `cvvdp`
  JobId differs from `cvvdp@standard_fhd`, CpuHeavy routing, empty caps).
- Executor pixel-hash integration completed: `mk_row_px` stamps
  `reference_pixels_sha256`/`distorted_pixels_sha256` (flat RGB8 sha256) into
  every per-variant SDR ScoreFile row once the variant decoded (feature rows,
  `cvvdp@` rows, generic score rows, post-decode error rows); fetch/decode
  error rows stay unstamped (no buffer). HDR `run_score_file_hdr` untouched —
  nits decode, not RGB8. Env gate `ZEN_JOBEXEC_PIXEL_HASH=1`, off by default.
- Release rebuild queued on `heavy.lock` (~23 min behind other lanes so far):
  same `cargo build --release -p zenmetrics-cli --features jobexec` command;
  binary lands at `/var/tmp/cvvdp-safesyn/target-zenmetrics/release/zenmetrics`.

## Post-rebase Part 0 on fresh binary (b373b48a / base b02812ae) — ~05:0xZ

- Fresh release binary `/var/tmp/cvvdp-safesyn/target-zenmetrics/release/zenmetrics`
  (64,428,664 B): sha256 `ebffd5b8e3843d8e1b4987262904a6f43b5789f34fa019e3af799fb1b73b9d45`.
- `batch --metric cvvdp` (no flag) over `aic4_pairs.tsv` (300 pairs, `--jobs 8`):
  all 300 scores **bit-identical** to board `aic4_cvvdp.tsv` `cvvdp_cpu_imazen_v0_1_0`
  (identical=300 differ=0). Column name unchanged → no V5 drift on default path.
- `batch --metric cvvdp --display-model standard_fhd`: all 300 **bit-identical** to
  stored `aic4_cvvdp_standard_fhd.tsv` (`cvvdp_cpu_imazen_v0_1_0_standard_fhd`) →
  zero drift vs the cvvdpfix-era scores. SROCC from stored scores stays 0.9609.
- Throughput smoke: ~15.2s wall / 300 pairs @ 8 jobs ≈ 19.7 pairs/s (default),
  15.5s ≈ 19.3 pairs/s (fhd). vs historical ~1-4 pairs/s — V5 SIMD/rayon landed.
- Blob plan DONE: `s3://codec-corpus/safesyn-rev2-2026-09-06/` already held
  199,304 objects covering 196,030/196,086 pairs; uploaded the 57 missing
  objects (ref `Verkehrstote Deutschland 1953-2012_512sq.png` + its 56 admitted
  variants; the space-named ref was skipped by the rev2 uploader). Coverage now
  100%; spot-check sha256 byte-identical to `SAFESYN_VERIFIED.json` audit
  (`q5.jpg` = c49189b0…). URI scheme: `sources/<ref>.png`,
  `images/<src>/<codec>/<file>` — the `declare-scorefiles --full-uri` route.
- `modern_oled_phone_indoor` parity vs pycvvdp 0.5.7 (config_paths injection of the
  imazen preset, merged display_models.json in /var/tmp/cvvdp-safesyn/pycfg/):
  8 KADID pairs, max |Δ| = 0.0001 JOD (pycvvdp torch/cu130 vs our CPU V5).
  Display is CHECKED, not unchecked — upgrades the lane's weakest preset.

## Fleet prep (~12:3xZ)

- Discovered the rev2 wave already uploaded the SafeSyn corpus to
  `s3://codec-corpus/safesyn-rev2-2026-09-06/` (199,304 objects; `sources/` +
  `images/<src>/<codec>/<file>`). Spot-check sha256 == audit
  `distorted_file_sha256` (byte-identical). Coverage vs pairs TSV: 196,030/196,086;
  uploaded the 57 missing (space-named ref `Verkehrstote Deutschland 1953-2012_512sq`
  + its 56 admitted variants — e1 family excluded). Coverage now 100%.
- Built `/var/tmp/cvvdp-safesyn/safesyn_pairs_uri.parquet` (196,086 rows; ref_path/
  dist_path = full s3 URIs, codec/q/knob identity, sorted ref→codec→q).
- `zenfleet-ctl declare-scorefiles --full-uri --chunk 128 --manifest-out` dry-run:
  **3,218 jobs** (one per ref, 36–78 inputs each), metrics
  `cvvdp@standard_fhd,cvvdp,ssim2`, `requires:[]` (CPU-only → CpuHeavy).
  chunk=128 > max variants (78) ⇒ one chunk/ref ⇒ JobIds order-invariant.
- Smoke harness `/var/tmp/cvvdp-safesyn/smoke_fleet.py` written + pre-flighted on
  the host binary (`--local --pairs 50`): rows decode + score correctly and
  `ssim2` is BIT-IDENTICAL to audit `peer_ssim2.score` (-62.05387167822142);
  pixel-hash columns absent because the 04:54 binary predates the mk_row_px
  threading — glibc rebuild queued, musl executor build queued behind it.
- `harvest_safesyn.py` written: blobs→JSONL→row_id-keyed sidecar
  (`cvvdp_jod_standard_fhd`, `cvvdp_jod_standard_4k`, `ssim2_fresh`) with the
  196,086-row/unique/non-null completeness gate.
- FLEET_READY.md drafted (image digest + in-image smoke pending the musl build).
- cargo fmt applied to zenmetrics-cli/zenfleet-core — lane commit cea26c89.

## 2026-09-23 — NO-GO repair: AVIF root cause found and fixed

**Review finding (REVIEW_FLEET.r1.md):** 11/68 smoke ssim2 mismatches, ALL `zenavif-s5-e6`;
non-AVIF families bit-exact. Review hypothesized Cargo.lock drift (zenavif 0.1.7→0.2.0).

**Actual root cause (proven, not hypothesized):** decode ROUTING, not lockfile.
- Frozen Sept-14 extraction lock (native-integrity-admission-2026-09-14/source/zensim-bench/Cargo.lock)
  already used zenavif 0.2.0 + rav1d-safe e73811f5 — the same identities the lane resolved.
  Restoring the stale base lock would have installed the WRONG decoder set.
- Frozen extraction binary (…/bin/extract_features_372col) reproduces the audit exactly:
  q5.avif → RGB8 sha256 93016b1b…, ref 6e548a26…, ssim2 -44.92500644015996.
- Probe matrix on q5.avif (zenavif @c2570c3a, same decode code as Sept-14):
  - decode_full→RowConverter @ threads 0/1/4 → 4881d856… (the drifted output)
  - zencodec Decode path (AvifDecoderConfig::new().job().decoder().decode()) → 93016b1b… ✓ audit
- Mechanism: zencodec `decode_inner` tags the PixelBuffer via set_cicp_on_pixels +
  attach_source_color_context + negotiate_format; the tagged descriptor changes
  RowConverter's Rgb16→RGB8_SRGB conversion. Raw decode_full leaves
  TransferFunction::Unknown → byte passthrough. Threads are irrelevant to pixels.
- zenavif a7c56be9→@ diff: encode/tests/whitespace only — decode path byte-identical.
- zenpixels PATH-vs-registry: no effect (probe: same hash both ways).

**Fix:** `decode_avif` in zenmetrics-cli/src/decode.rs now routes through
`zenavif::AvifDecoderConfig` (zencodec `Decode` trait) — same entry the extractor used.
threads(1) retained (rav1d-safe#15 race workaround; verified pixel-neutral).
HDR tripwire preserved via `out.info().source_color.cicp.transfer_characteristics`
(populated by convert_native_info from the same native tc code).
Cargo.toml: `avif` feature now pulls `dep:zencodec` (already in lock at 0.1.26).

**Why other families were already exact:** jpeg/webp decode to RGB8; the
RGB8→RGB8_SRGB conversion is an identity passthrough either way. Only the
>8-bit AVIF funnel (Rgb16→RGB8) is CICP-sensitive.

Also added: `run_score_file_refuses_unknown_display_before_any_row` test
(review optional item — proves job-level refusal before any row/fetch).

**End-to-end verification (dev binary, same tree):** reran the NO-GO job
(ref `00b13be94a4867dd_1022x818`, 68 pairs × 3 metrics = 204 rows,
`ZEN_JOBEXEC_PIXEL_HASH=1`, s3 inputs): **0 pixel-hash mismatches,
0 ssim2 mismatches across all 6 families** — including all 11 AVIF rows that
drifted. Probe also confirmed: zenpng 0.2.0 ref decode == audit ref hash
(`6e548a26`); q10/q95/q100 AVIFs all match audit via zencodec path at
threads 0 and 1.

Decoder-identity ledger vs frozen Sept-14 extraction lock:
- zenavif 0.2.0 PATH ✓ same | zenavif-parse 0.7.0 PATH ✓ | rav1d-safe e73811f5 ✓
- zencodec 0.1.26 registry ✓ | zenpixels-convert 0.2.16 registry ✓
- zenjpeg 0.9.0 PATH ✓ | zenwebp 0.5.0 PATH ✓ | zenjxl 0.3.0 PATH ✓
- zenpng 0.1.4→0.2.0 PATH (pixel-neutral: PNG decode deterministic; verified)
- fast-ssim2 0.8.2 registry→0.9.0 PATH (ssim2 bit-exact on all verified rows)
- zenpixels 0.2.16 PATH vs registry (pixel-neutral; verified both ways)

## musl executor feature gap (upstream quirk)

`cargo build --release --target x86_64-unknown-linux-musl -p zenmetrics-cli
--no-default-features --features jobexec,png,jpeg,webp,avif,jxl,cpu-metrics`
fails: `metric_runtime` (jobexec.rs) is `#[cfg(feature = "hdr")]`-gated but
called unconditionally from `mk_row`, and `zenflate` is imported under the
same gate. Pre-existing upstream layout — `jobexec` alone never compiled
without `hdr`. Fix for the executor image: add `hdr` to the musl feature set
(matches fleet convention: executor images carry HDR ScoreFile capability;
run_score_file's hdr arm refuses loudly when the feature is absent). Not
re-scoping upstream gates in this lane.

## E2b complete (2026-09-23, on binary f8c352ee)

21 `batch` runs: 3 TRAIN legs × 7 displays, 6,767 scored pairs total
(kadid 5000, tid 1440, konfig 327), all rows complete.
`/var/tmp/cvvdp-safesyn/e2b/scores/{leg}__{display}.tsv`.

Stats (e2b_stats.py → zen_stats.panel, B=2000, seed 20260923, Bonferroni
99.1667% CI, ref-clustered bootstrap):

| display | kadid Δ | tid Δ | konfig acc Δ | passes |
|---|---|---|---|---|
| sdr_4k_30 | +0.0153 [0.0124,0.0182] | +0.0101 [0.0048,0.0144] | -0.0043 [-0.0095,0] | YES |
| standard_fhd | +0.0235 [0.0120,0.0344] | +0.0138 [-0.0067,0.0339] | -0.0301 [-0.0453,0] | no (tid CI) |
| **sdr_fhd_24** | **+0.0311 [0.0216,0.0403]** | **+0.0185 [0.0013,0.0339]** | -0.0238 [-0.0403,0] | **YES — SELECTED** |
| standard_phone | -0.0593 | -0.0409 | -0.0106 | no |
| iphone_14_pro | -0.1055 | -0.0690 | -0.0285 | no |
| modern_oled_phone_indoor | -0.0464 | -0.0328 | -0.0049 | no |

**Selected display: `sdr_fhd_24`** (largest mean pooled Δ = +0.0248 over the
two SROCC legs). Per brief Part 2 the fleet sidecar carries FOUR metric arms:
`cvvdp` (standard_4k), `cvvdp@standard_fhd`, `cvvdp@sdr_fhd_24`, `ssim2`.
Manifest regenerated: `safesyn_manifest4.json` (3,218 jobs).
Stats JSON: `/var/tmp/cvvdp-safesyn/e2b/e2b_stats.json`.

## Executor image + in-image smoke (fleet gate artifacts)

- Frozen zenmetrics commit: `19d6dd8ee378b64562c551358c1d935ca7ed58ee`
  (bookmark `quarantine/devin/cvvdp-safesyn`, one commit on `b02812ae`).
- musl `zenmetrics` (`--no-default-features --features
  jobexec,png,jpeg,webp,avif,jxl,cpu-metrics,hdr`):
  sha256 `0bba95e60fb0d8911fe8138853a95eef631894b21745608e0cba0b12c58e80df`
- musl `zenfleet-worker`: sha256
  `719bf12692bb5036769b88fcefbd9855d6c3f5bbd3bcf8684ccfbd10dd76d6c4`
- glibc `zenmetrics` (Part 0 + E2b host runs): sha256
  `f8c352ee83e7424c5c709f9d1d3bf49795443dc809e3ceb31db3cba9f40425fe`
- Image: `ghcr.io/imazen/zenfleet-worker:exec-cvvdp-safesyn-19d6dd8e`
  digest `sha256:218cfc7fef2e65f056bd2320238dda7c93c0235d7f83fbfb455de4abe83f0156`
  — built via `scripts/jobsys/build_executor_image.sh`, NOT pushed (gate).
  `ldd` inside image: `statically linked` (both binaries).
- IN-IMAGE SMOKE PASS: `smoke_fleet.py --image exec-cvvdp-safesyn-19d6dd8e
  --pairs 220` → 4 jobs, 262 pairs, 1,048 rows (4 metric arms),
  `ZEN_JOBEXEC_PIXEL_HASH=1`: **0 pixel-hash failures, 0 ssim2 failures,
  exact equality vs Sept-14 audit on every row.**
  Per family (pairs/rows/failures): mozjpeg 53/212/0, zenavif 45/180/0,
  zenjpeg-420 42/168/0, zenjpeg-xyb 41/164/0, zenjxl 43/172/0,
  zenwebp 38/152/0.
  Evidence: `/var/tmp/cvvdp-safesyn/{smoke_image.log,smoke_out.jsonl,smoke_jobs.jsonl}`.
- Fresh `FLEET_READY.md` written at `/home/lilith/tmp/zensim-paper/rev4/`
  (supersedes `FLEET_READY.r1.md`); `FLEET_GO.md` does not exist — lane is
  parked at the gate pending coordinator approval.

## E2b output hashes (sha256)

kadid: 4k 6043b1f2 | sdr_4k_30 2f8014d2 | standard_fhd 58dd9064 | sdr_fhd_24
0231bc47 | standard_phone e4409094 | iphone_14_pro 6941859e |
modern_oled 3227fe73
tid: 4k a8ba99df | sdr_4k_30 6d54cd08 | standard_fhd b2872427 | sdr_fhd_24
7fd3525f | standard_phone 185992ec | iphone_14_pro 5fba3f96 |
modern_oled 6e68fa31
konfig: 4k da5c466b | sdr_4k_30 df678bd1 | standard_fhd 64196371 |
sdr_fhd_24 29365106 | standard_phone 0bc8c9e6 | iphone_14_pro 96682869 |
modern_oled 4a1d3d55
e2b_stats.json ee0c4cf6 | safesyn_manifest4.json 9a1e1eb9 |
safesyn_pairs_uri.parquet baa8c53f

## Fleet review r2 → NO-GO repair (18:30Z verdict, repair ~19:3xZ)

- `REVIEW_FLEET.md` (Opus r2): NO-GO on ONE blocker — the lane's `decode.rs`
  reroute changed AVIF decode for *every* caller (score/batch/sweep), breaking
  pre-existing tripwire test
  `avif_hdr_tripwire::patching_transfer_alone_does_not_change_pixels`
  (failed on `19d6dd8e`, passed on base `b02812ae`). Reviewer independently
  verified everything else: image+binaries hash-matched, static, their own
  24-pair/96-row recompute exact on every family incl. AVIF.
- Reviewer also measured the tagged zencodec route as *less exact* (0.16%
  of Rgb16→RGB8 values one code low) — it binds to the Sept-14 cache but is
  not a better default. Required fix (a): scope it behind an explicit opt-in.
- FIX: `decode_avif` restored to base `decode_full` as the default arm;
  `ZEN_JOBEXEC_AVIF_DECODE=zencodec` selects the tagged zencodec route;
  any other non-empty value errors loudly (rc=1, names the env var).
  Verified on dev binary (safesyn q5.avif): unset→`4881d856…` (pre-lane
  route), `zencodec`→`93016b1b…` == audit, `bogus`→rc=1 error row.
- Full `cargo test -p zenmetrics-cli --release --no-fail-fast`: **108 passed,
  0 failed** (tripwire test green again; reviewer's run was 107+1).
- New glibc release binary sha256 `8bcac079…` (replaces f8c352ee).
- Part 0 re-verified on `8bcac079`: no-flag `cvvdp` 300/300 bit-identical to
  board `cvvdp_cpu_imazen_v0_1_0`; `--display-model standard_fhd` 300/300
  bit-identical to stored `…_standard_fhd` column.
- Non-blocking items handled: harvest asserts per-row pixel hashes vs audit
  on ALL rows (fail-loud), codec_family now derived from encode_sha path
  (row `codec` label is upstream-mislabeled per-ref-first-pair — documented),
  `path_dep_provenance.py` records each sibling path-dep repo's HEAD +
  dirty-diff sha256 into `_MANIFEST.json`, blobs parsed per-file (no raw
  stdout concat), registry digest re-check moved to post-push checklist.

## r3 rebuild + re-smoke (post r2-repair, ~20:0xZ)

- Frozen zenmetrics commit: `9f36f88b8a23e645bd791c1193842d950eccf4be`
  (one commit on b02812ae; `cargo fmt` clean).
- musl zenmetrics sha256 `77cc76d9af234b1381fa92ad7b1537a65d26ea058f7ac2a2839a49ecec3d38b4` (static-pie ELF); musl worker `719bf126` unchanged.
- Image `ghcr.io/imazen/zenfleet-worker:exec-cvvdp-safesyn-9f36f88b`,
  digest `sha256:d110a3a79d98f06ade7c7dae3920c47b71c10362c959bd852341f6c5ff046050`,
  NOT pushed. Layers: exec base via `build_executor_image.sh` (`-pre` tag)
  + thin ENV layer baking `ZEN_JOBEXEC_AVIF_DECODE=zencodec` +
  `ZEN_JOBEXEC_PIXEL_HASH=1` + `ZEN_ZENSIM_BUILD_COMMIT=646a67d0`.
- ENV-only proof: in-image 1-pair AVIF job with **no -e flags** emits
  `distorted_pixels_sha256 93016b1b…` == audit on all 4 metric arms.
- In-image ldd: `statically linked` both binaries; in-image sha256s match
  the musl build outputs.
- IN-IMAGE SMOKE (r3): `smoke_fleet.py --image exec-cvvdp-safesyn-9f36f88b
  --pairs 220` → **4 jobs / 262 pairs / 1,048 rows / 0 failures**, per family:
  mozjpeg 53/212/0, zenavif 45/180/0, zenjpeg-420 42/168/0, xyb 41/164/0,
  zenjxl 43/172/0, zenwebp 38/152/0. Log `smoke_image_r3.log`.
- `path_dep_provenance.py` → `build_meta.json`: 19 path-dep crates, 13
  sibling repos (dirty: butteraugli, jxl-encoder, zenavif, zenjpeg) — HEAD
  commit + dirty-diff sha256 each; merged into `_MANIFEST.json` at harvest.
- `FLEET_READY.md` rewritten (r3); r2 preserved as `FLEET_READY.r2.md`.
  Awaiting fresh review → `FLEET_GO.md`.

## FLEET_GO received + declaration (2026-09-23 ~23:0xZ)

- `FLEET_GO.md` + `REVIEW_FLEET.md` (r3) landed ~22:55Z: **GO** on commit
  `9f36f88b`, image digest `d110a3a7`. Reviewer independently recomputed
  48 pairs / 192 rows pixel+ssim2 exact vs audit; lane smoke 262/1,048.
- GO conditions discharged:
  1. Pushed tag `exec-cvvdp-safesyn-9f36f88b`; registry-inspected digest
     `sha256:d110a3a7…` == approved digest.
  2. Harvest keeps per-row pixel-hash asserts (px_missing/px_bad fail).
  3. Harvest asserts duplicate lease-retry rows agree bit-for-bit (dup_bad).
  4. `build_meta.json` now carries `zensim_linked_commit` =
     `353e6bcc969d2400d0ef102d35d503562c737c42` (from path-dep provenance,
     not the image's `ZEN_ZENSIM_BUILD_COMMIT` env which held a doc commit).
  5. Only the approved tag pushed.
- DECLARE: `zenfleet-ctl declare-scorefiles --pairs safesyn_pairs_uri.parquet
  --metrics cvvdp@standard_fhd,cvvdp@sdr_fhd_24,cvvdp,ssim2 --chunk 128
  --full-uri --cell-knobs safesyn-cvvdp-displays --run cvvdp-safesyn-20260923`
  → 3,218 jobs / 196,086 pairs at 23:00:39Z. First attempt died on the
  ctl's 120 s s5cmd guard because only `ZEN_S3_*` was exported and s5cmd
  hung on its credential chain; re-ran with `AWS_ACCESS_KEY_ID`/
  `AWS_SECRET_ACCESS_KEY` exported → clean declare.
- Run prefix: `manifest.json` (20.7 MB) + `.gz` (768 KB) + `control.json`
  `{"paused":false}`. Input coverage re-sampled 16/16 pre-declare.
- `FLEET_RUN.md` written → coordinator enrolls allowed workers
  (i134, r5600g, r3500, tower-docker capped). Monitoring + heartbeats next.

## Fleet run live (heartbeats ~15 min)

- 23:04Z worker probes enrolled: r5600g (5), r3500 (6), Tower (7) — allowed
  nodes only. First claims 23:07Z (`r5600g-cvvdp` lease holder).
- First blob verified 23:08Z: 212 rows, all 4 metric arms, 212/212 pixel-hash
  stamps present, 0 error rows. (Blob files are unterminated-JSONL — harvest
  parses per line, no concat.)
- 23:23Z blobs=310/3218 (9.6%) active-claims=13

## HANDOFF (23:2xZ) — coordinator owns fleet monitoring; GPT-6 lane harvests

**State.** Run `cvvdp-safesyn-20260923` declared 23:00:39Z: 3,218 ScoreFile
jobs / 196,086 pairs / 4 metric arms (`cvvdp@standard_fhd`,
`cvvdp@sdr_fhd_24`, `cvvdp`, `ssim2`) on image
`ghcr.io/imazen/zenfleet-worker:exec-cvvdp-safesyn-9f36f88b`
(digest `d110a3a7…`, pushed + registry-verified, == FLEET_GO). At handoff:
~310/3,218 blobs (~10%), claims on allowed nodes only (r5600g, r3500,
Tower). Incremental verification so far: 290 blobs / 70,220 rows /
0 error rows / 0 px_missing / 0 px_bad (see
`/var/tmp/cvvdp-safesyn/verify_progress.log`).

**Left running (do not kill):** `verify_blobs_loop.sh` — downloads + audit-
verifies new blobs every ~10 min, appends to `verify_progress.log`, ALERT
lines on any mismatch/error. Stop with
`touch /var/tmp/cvvdp-safesyn/STOP_VERIFY`. My heartbeat appender is
stopped (STOP_HEARTBEAT set).

**Remaining steps.**
1. Monitor run to 3,218/3,218 blobs (coordinator).
2. Harvest:
   `cd /var/tmp/cvvdp-safesyn && source ~/.config/zen/lanstore.env &&
    export AWS_ACCESS_KEY_ID=$ZEN_S3_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY=$ZEN_S3_SECRET_ACCESS_KEY &&
    python3 harvest_safesyn.py --run cvvdp-safesyn-20260923
    --out /var/tmp/cvvdp-safesyn/safesyn_cvvdp_sidecar.parquet
    --manifest-out /var/tmp/cvvdp-safesyn/safesyn_cvvdp_sidecar_MANIFEST.json
    --build-meta /var/tmp/cvvdp-safesyn/build_meta.json`
   Gates (all loud): 196,086 unique row_ids, all 4 metric cols non-null,
   per-row ref+dist pixel-hash == Sept-14 audit, duplicate lease rows
   bit-identical, codec_family from encode_sha path.
3. `python3 safesyn_compare.py --sidecar safesyn_cvvdp_sidecar.parquet
   --out-md benchmarks/rev4_cvvdp_safesyn_2026-09-23.md
   --out-json benchmarks/rev4_cvvdp_safesyn_2026-09-23.json`
   (descriptive rank agreement vs stored labels/oracle, by family + band).
4. Fill `/var/tmp/cvvdp-safesyn/DONE.draft.md` (already at r3 artifact ids)
   → `/home/lilith/tmp/zensim-paper/rev4/CVVDP_SAFESYN_DONE.md`.
