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
| neon (mac) | (first blobs pending) | | pending |

RESULTS_PLACEHOLDER (per-view G-BF1/G-BF2 table + assembly stats + wall time
+ node utilization land here when the wave drains.)

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

- The neon partition (31,519 cells, 6.4%) rides the mac's native arm64 build
  of TODAY'S codec siblings; the July-vs-today decode identity was proven on
  x86 (ab_decode) but arm has no equivalent pre-proof — the neon pool's own
  G-BF1 gate is the arbiter (any NEON-path decode change since 2026-07-27
  would surface there as mismatches).
- jason carried 97,170 of the v4 cells in bf924 but is running another
  session's training this session — the v4 pool runs without it (tower + i265
  + ian cover the tier; wall-time impact only).
- The flat-pool smoke burned ~30 box-minutes on lianli (3,433 tier-mismatched
  blobs, deleted with the retired flat runs).
