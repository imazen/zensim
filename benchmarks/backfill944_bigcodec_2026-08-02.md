# bf944 — bigcodec 944 fleet extraction (PLAN_SOTA944 P1 leg 3) — build record (2026-08-02)

The bigcodec leg of the 944 backfill: the 21 split views
(`ext924-canonical-2026-07-27/bigcodec/<dataset>/<split>_924.parquet`, 5,742,660
rows, keyed `encode_sha` == `encoded_filename`) re-extracted at
`FeatureRegime::Folded720Append2` (944) on the household zenfleet, per the
recipe in `benchmarks/backfill944_2026-08-01.md` §"bigcodec decision" and the
P1.5 ADJUDICATION (toggle OFF — plain `Folded720Append2` default math).

## Extractor / executor

- zensim rev **`d061636262387b8746ffa0c883a73731ce9ab789`** (origin/main tip,
  carries the adjudicated default math; `ZENSIM_APPEND2_DSTACT` NOT set — the
  dst-activity toggle stays OFF, gated bit-exact-off in-repo).
- zenmetrics executor: metric string **`zensim-foldapp2`** = the `zensim-foldapp`
  handler with `V2NewFeatureToggles { append2_block: true, ..default() }` at 944
  (commit `57b7b9ad`; parity test `folded944_matches_driver_args` — bit-exact vs
  the `v2_ab_extract` foldapp2 driver args incl. sub-64 + odd dims, **f0..f923
  bitwise-identical to the 924 regime** (the G-BF1 premise at unit level),
  append2 tail finite + bounded [0,1], scratch-reuse determinism).
- Image `ghcr.io/imazen/zenfleet-worker:exec-zensim944-57b7b9ad` (digest
  `sha256:ebb4bf361486…`), built LOCALLY from zenmetrics `57b7b9ad` + zensim
  `d0616362` + today's codec siblings, per the bake-everything rules (binary
  COPY'd in; canonical package + new TAG). Mac (neon tier) runs the native
  arm64 binaries built ON the mac from the same revs (zenmetrics `5b00f6ea`,
  zensim `d0616362`; `mac-worker/build_mac.sh`).
- Declared by `zenmetrics scripts/jobsys/declare_bf944_tiered.py` (see below);
  the flat `declare_bf944.py` pool it supersedes was retired + its objects
  deleted the same session.

## ★ FINDING: the bf924 baseline is SIMD-TIER-MIXED — bitwise G-BF1 requires tier-matched extraction

The two-stage launch discipline caught this at the smoke stage, before any
scale-out.

1. **Smoke FAIL.** First worker (lianli, Zen4 = AVX-512) on the flat pool run
   `bf944-zjxll-t22`: G-BF1 spot gate vs `tbig_924_full` = **0 of 3,465 rows
   bitwise-identical** — f0/f1-class mismatches at ~1e-10..2e-10 relative,
   ~235/924 slots per row.
2. **Decoder drift FALSIFIED.** The bf924 image
   (`exec-zensim924-2095f80b`) and the new bf944 image run on the SAME cells
   (metric `zensim-foldapp`, same data plane) produce **bitwise-IDENTICAL 924
   vectors for all six codec runs** (zjxll/zjxlm/zpng/zwebp/zavif/zjl2 — the
   `ab_decode` instrument). Today's codec siblings decode identically to the
   2026-07-27 image; the new binary is byte-faithful to the old one.
3. **Worker attribution.** The bf924 ledger rows for the mismatching cells:
   all `worker=tower-unraid` (Threadripper 2950X = Zen1, **AVX2-only**), while
   the smoke ran on AVX-512.
4. **Tier proof (the `tier_proof` instrument).** The o_9736.png.scale192x192
   cell (tower-made in bf924), re-extracted with the NEW image:
   - on the **tower**: **BITWISE-IDENTICAL to tbig_924, 12/12 variants**;
   - on Zen4 boxes (wsl + lianli, AVX-512): 0/12, first mismatch f0
     rel=2.04e-10, ndiff=235/924 — and tower-vs-local shows the same deltas.
   The difference is purely the archmage SIMD tier's accumulation order
   (consistent with the workspace-level "AVX2-vs-AVX512 agree <1e-5 rel"
   mergeability note — the 924 wave accepted cross-tier merging row-wise).

**Consequence:** `tbig_924_full` (and the 21 views derived from it) is
row-wise tier-mixed; f0..f923 bitwise reproduction is achievable only by
re-extracting every cell on a box of the SAME SIMD tier as its bf924
extractor. Any uniform-tier re-extraction fails G-BF1 structurally.

### Ledger attribution of all 490,173 kept cells (0 missing)

| bf924 worker | cells | SIMD tier |
|---|--:|---|
| lilith-lianli (7900X Zen4) | 229,064 | v4x (AVX-512) |
| tower-unraid (TR 2950X Zen1) | 129,648 | v4 (AVX2) |
| zen-node-2 = jason (i5-13400F) | 97,170 | v4 |
| lilith-mac (M4 Pro) + mac-login-test + mac-debug | 31,317 + 170 + 32 | neon |
| wsl-smoke (7950X Zen4) | 2,772 | v4x |
| **totals** | **v4x 231,836 · v4 226,818 · neon 31,519** | = 490,173 |

Attribution = join of the July assembly's `matched_ledger.parquet` (the exact
kept rows of `tbig_924_full`, dedup job_id keep-first) back to the 54 bf924
ledgers on `(pool, job_id, output_sha)` — pins the winning attempt exactly.

## The tier-matched wave (what actually runs)

`declare_bf944_tiered.py` splits every bf924 run's cells by the attributed
tier into `bf944v4-*` / `bf944v4x-*` / `bf944neon-*` (manifest-index ↔
`zenfleet-ctl ids` — the owner's JobId hash), three pools:

| pool | runs | cells | boxes |
|---|--:|--:|---|
| `jobs/_pool944v4/runlist.tsv` | 31 | 226,818 | tower (24c capped) + i265 (20T) + ian (12T); jason SKIPPED (another session's training live on it) |
| `jobs/_pool944v4x/runlist.tsv` | 27 | 231,836 | lianli (24T); wsl excluded (P1 kadis extraction owns local compute) |
| `jobs/_pool944neon/runlist.tsv` | 8 | 31,519 | mac (M4 Pro, idle-only launchd, native arm64 build) |

Hard declare gates (all PASSED): per-run manifest-ids == attributed-ids set;
global tier totals == attribution histogram; source manifests verified to
carry exactly `["zensim-foldapp"]` before the metric rewrite.

**Fleet-stage spot gates (G-BF1, bitwise vs tbig_924 by encode_sha):**

| pool | first run gated | rows | verdict |
|---|---|--:|---|
| v4 (tower) | bf944v4-zjxll-t2 | 2,301 | **PASS — 2,301/2,301 bitwise-identical, 0 mismatches** (append2 range [0, 0.634]) |
| v4x (lianli) | bf944v4x-zavif-t6 | 2,400 | **PASS — 2,400/2,400 bitwise-identical, 0 mismatches** (append2 range [0, 0.696]) |
| neon (mac) | bf944neon-zjxll-t23 | 2,328 | **PASS — 2,328/2,328 bitwise-identical, 0 mismatches** (append2 range [0, 0.703]) |

All three SIMD tiers verified bitwise against the canonical 924 bytes before
scale-out completed — 7,029 fleet rows, 0 mismatches total.

## ★ SECOND FINDING: CPU-VENDOR nondeterminism in the MSCN append slots (G-BF1 round 2)

The first 21-view gate run FAILED G-BF1 on **exactly 22 columns in every
dataset incl. lossless PNG/WebP**: the `idx_append::MSCN_DIFF_MEAN`(+5) /
`MSCN_DIFF_L2`(+6) pair in 11 of 12 (channel×scale) append groups
(f725/726, 742/743, 776/777, 793/794, 810/811, 827/828, 844/845, 861/862,
878/879, 895/896, 912/913), deltas ~1e-8..1e-9 rel.

- Lossless codecs decode bit-exact on every CPU ⇒ pixels identical ⇒
  extractor-side. Only 22 slots move (a pixel diff moves ~235/924 — the
  tier study's signature) ⇒ localized to the MSCN divisive-normalization
  math (`sqrt(var + C)` reciprocal — approximate rsqrt/rcp instructions
  have VENDOR-SPECIFIC tables; one NR step leaves the observed ~1e-8).
  Filed: **imazen/zensim#56** (zero-tolerance bug; fix = exact sqrt+div or
  full-precision refine + a cross-vendor CI gate). Related infra bug found
  in the same diagnosis: **imazen/zenmetrics#38** (JobId::of depends on
  serde_json preserve_order via feature unification — the docker fleet's
  ledger ids are insertion-order-encoded, ctl/mac/bf924's sorted; all bf944
  joins handle both encodings).
- Direct proof: cell `o_7067.png.scale96x96_…_zenavif_q30_….avif`
  (bf944v4-zavif-t3) re-extracted on the tower = bitwise-identical to its
  bf924 row (0/924); on i265 = f725 off by 8.7e-9. In `zenavif_lossy/test`,
  85,392 of 271,488 rows diverge somewhere (only cross-vendor-re-extracted
  rows; the rest happen to refine identically).

**Consequence: G-BF1 requires CPU-VENDOR×tier-matched extraction.** The
measured equivalence classes (all verified by bitwise gates in this wave):

| class | boxes (bf924-era worker names) | cells attributed |
|---|---|--:|
| amdv4x (AMD Zen4, AVX-512) | lianli (`lilith-lianli`), wsl (`wsl-smoke`→`wsl-944`) | 231,836 |
| amdv4 (AMD Zen1/Zen3, AVX2) | tower (`tower-unraid`), ian (`zen-node-3`) | 129,648 |
| intelv4 (Intel RaptorLake/ArrowLake, AVX2) | jason (`zen-node-2`), i265 | 97,170 |
| neon (Apple M4 Pro) | mac (`lilith-mac` + test names) | 31,519 |

Zen1↔Zen3 and RaptorLake↔ArrowLake are interchangeable (measured: their
cross-extractions gate bitwise); AMD↔Intel are not. Future backfill waves
inherit this: partition by vendor-class, not SIMD tier alone.

**Repair wave (2026-08-02 ~18:0xZ):** census over all wave ledgers (both
JobId encodings): 406,203 / 490,173 cells already had a class-matched blob
(the wave's multi-worker re-scoring); **83,970 repair cells** declared as
`bf944amd-*` (44,679; `_pool944amdv4`; tower+ian) and `bf944int-*` (39,291;
`_pool944intelv4`; i265+jason) by `zenmetrics scripts/jobsys/
declare_bf944_repair.py`. Selection at assembly =
`bf944_classpref_select.py`: exact-bf924-worker first, then vendor class,
FAIL loud otherwise.

### Drain accounting for the original 173-cell gap (main wave)

The last 173 cells (bf944neon zavif-t0 170 / zjxll-t6 1+37 shard-collision
re-lost / zwebp-t7 2) were starved by a mac-worker pass pathology: the mac
run.sh passes carry no `--ledger-in`, so each pass re-walked all cells
against persistent chunk claims (fail-once ⇒ claim persists ⇒ skipped
forever) while pass timeouts + shard-name collisions across container
restarts (`mac-$WORKER-$cyc.chunk-*.parquet`) hid/re-lost rows. Cleared by:
stale-claim deletion + manual `zenfleet-worker` passes ON THE MAC (neon
class preserved) with the run's ledger snapshot as `--ledger-in`
(`done=170`, `done=1`, `done=37`; worker `lilith-mac-gapfix`). During
diagnosis two instrument passes with an incomplete exec env wrote 170
FAILED rows (`encoder_panic` = my missing `ZEN_R2_ENDPOINT`, not a real
panic); those shards were deleted from the run ledger before the fix pass.

## FINAL RESULTS (promoted 2026-08-03 ~02:5xZ)

**All 21 views: G-BF1 + G-BF2 + structural PASS** (join match_rate 1.0000,
5,742,660 rows exact; per-view JSON in `bigcodec/gates/`):

| dataset | train | validate | test |
|---|---|---|---|
| zenavif_lossy (775,152 / 464,352 / 271,488 rows) | PASS | PASS | PASS |
| zenjpeg_lossy | PASS | PASS | PASS |
| zenjxl_lossless | PASS | PASS | PASS |
| zenjxl_lossy | PASS | PASS | PASS |
| zenpng_lossless | PASS | PASS | PASS |
| zenwebp_lossless | PASS | PASS | PASS |
| zenwebp_lossy | PASS | PASS | PASS |

Canonical: `/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01/bigcodec/`
(+`_MANIFEST.json` w/ per-file sha256) · R2
`s3://zentrain/ext944-canonical-2026-08-01/bigcodec/` (50 objects) · Tower
`/mnt/tower/output/zensim-ext944-canonical-2026-08-01/bigcodec/` — sha
spot-verified (Tower 3/3, R2 1/1). Fleet table = part set
`tbig-944-2026-08-02/` (parts 000-012 + selection provenance; delta parts
011-012 shadow — read rule in DATA_PROVENANCE), triple-mirrored.

### Wall time / node utilization (all household, $0 paid)

| phase | wall | nodes |
|---|---|---|
| main tier-matched wave (490,173 cells / 5.74M rows) | **~8.6 h** (03:41→~12:15Z 99.96%) | v4: tower(24c cap)+i265+ian+jason(joined 08:20 after its other job) · v4x: lianli(+wsl container from 08:20) · neon: mac (idle-only) |
| neon straggler fix (173 cells: stale-claim starvation) | ~2 h diagnosis+fix | mac (manual gapfix passes) |
| vendor repair wave (83,970 cells) | **~1.9 h** (22:44→00:36Z) | amd: tower+ian · intel: i265+jason |
| assemble fetch (490k blobs) | 62+55 min (two full passes — see below) + **12.7 min delta** | wsl |
| 21-view join ×2 (initial + post-repair) | ~21 min each (per-dataset run-heavy, ext4-staged parts) | wsl |
| 21 gates ×2 | ~35 min each | wsl |
| promote + manifests | 99 s | wsl |
| mirrors (R2 26GB+25GB, Tower 51GB) | ~75 min chunked | wsl |

Throughput ~950-1,300 cells/min mid-wave across 5-7 boxes; the bf924
precedent (~7.5 h) matched despite the extra tier partitioning.

### Merge-path comparison (user directive: keyed union, not re-assembly)

The vendor repair merged via **keyed delta**: 94,242 corrected picks
(83,970 repair-extracted + 10,272 re-picks from existing multi-worker
blobs) fetched into delta parts 011-012 in **12.7 min (124 blobs/s)**, then
consumed by the part-set join delta-first (keep-first per encode_sha ⇒
repaired rows shadow stale ones; zero part rewrites). The full-assemble
alternative (re-fetch all 490,173 blobs) measured 55-62 min per pass — the
delta path is **~4.6× faster**, and the two full passes already spent (the
initial fetch + the post-reap restart) are exactly the cost the keyed
design avoids. (The restart also produced the part-rolling fix — a
harness-kill mid-write cost one footerless 23.9 GB file, quarantined .bak.)

### Honest accounting

- 2 fleet cells' blobs carry error records for some variants (9 error rows
  in the main fetch + 2 in the delta, bf924-precedent class: per-variant
  decode errors recorded honestly, never fabricated); every VIEW row is
  backed by a real feature row (match_rate 1.0000 gates that).
- The one genuinely-failed neon cell (o_1413.png.scale64x48, avif) was
  re-run on the mac (`done=1`) before final selection; final selection has
  0 missing.
- Superseded artifacts kept as .bak: the pre-vendor-fix staged views
  (`bigcodec_staged_prevendorfix.bak`) + the footerless first-fetch parquet.
- jason ran another session's training until ~08:20Z (excluded till idle);
  kids' boxes were used only while genuinely idle and their workers are
  now disabled again (enrolled-stopped); mac worker daemon left unloaded;
  tower container removed; all pool workers stood down at drain.

## Post-fleet pipeline (committed this session)

1. `scripts/v_next/fleet_blob_assemble_944.py` — three-pool ledger sync/scan +
   blob fetch → `tbig_944_full.parquet` (944 f64, keyed encode_sha; keeps only
   `regime=="folded720append2"` && len==944 rows; resumable).
2. `scripts/canonical_corpus/tbig_join_944.py` — emits the 21 views **in the
   924 view's row order** with every non-feature column byte-carried from the
   frozen 924 views and f0..f943 from the fleet table; coverage hard-gated
   (match_rate 1.0000 or FAIL). View order matters because
   `gate_backfill944.py` compares positionally; ID/target byte-carry makes
   G-BF2 exact by construction while G-BF1 stays a true re-extraction gate.
3. `gate_backfill944.py` per view (the P1 gate tool, unchanged) → per-view
   JSON reports in `gates/`.
4. Promote into `ext944-canonical-2026-08-01/bigcodec/` + `_MANIFEST.json`
   (build_commit = zensim `d0616362` + zenmetrics `57b7b9ad` + image digest) +
   triple-mirror (local + `s3://zentrain` + Tower) + DATA_PROVENANCE pointer.

## Honest gaps / caveats

- ~~The neon partition rides an unproven arm decode path~~ — CLOSED by
  measurement: the neon spot gate PASSED bitwise (2,328/2,328 vs tbig_924),
  so the mac's native arm64 build of today's siblings reproduces its own
  bf924 rows exactly. (The per-view gates remain the final arbiter.)
- jason carried 97,170 of the v4 cells in bf924 but is running another
  session's training this session — the v4 pool runs without it (tower + i265
  + ian cover the tier; wall-time impact only).
- The flat-pool smoke burned ~30 box-minutes on lianli (3,433 tier-mismatched
  blobs, deleted with the retired flat runs).
