# 720-feature backfill of the four zensim eval instruments (2026-07-22)

Context: `docs/V2_EXPERIMENT_PLAN_2026-07-20.md` "720 GAP AUDIT" — four
eval-side instruments lacked the 720-wide (v1-372 ++ v2-348) feature vectors
needed to run G-DIAL / corruption-gate / G-NP / G-IM26 against a 720 model.
This session **fully** closed dial_grid + corruption_grid (100% re-encoded/
re-extracted, verified) and **partially** closed nonphoto/imazen26 (~26%/23%
via a fleet ledger+blob join — see §3+4 for why it stopped short of 100%; not
fabricated, honestly incomplete, re-run recommended). Tooling:
`scripts/v_next/backfill_dial_grid_720.py`,
`scripts/v_next/backfill_corruption_grid_720.py`,
`scripts/v_next/fleet_blob_fetch_720.py`.

## Summary table

| corpus | rows produced | of original | method | verify | flags |
|---|--:|--:|---|---|---|
| dial_grid | 4,817 | 4,817 (100.0%) | re-encode (CPU) + re-extract | cross-backend drift, see below | 536 rows (11.1%) flagged L2>0.5 |
| corruption_grid | 2,016 | 2,016 (100.0%) | pure re-extract (pixels existed) | near-ULP (both CPU) | 0 flagged |
| nonphoto | 2,599 | 10,000 (26.0%) | fleet ledger+blob join, stopped early (see §3+4) | exact fingerprint match | 0 (join is exact-match-or-drop) |
| imazen26 | 2,297 | 10,025 (22.9%) | fleet ledger+blob join, stopped early (see §3+4) | exact fingerprint match | 0 (join is exact-match-or-drop) |

Output: `/mnt/v/output/zensim/v2-eval-720-2026-07-22/` + `_MANIFEST.json`
(per-corpus rows/method/verify-stats/sha256/zensim_main_commit).

## 1. dial_grid (4,817 cells) — re-encoded

**Why re-encode was needed, and why it can't be a literal rerun.** The
stored `dial_grid_372col_2026-05-29.parquet` carries feature vectors only
(pixels were never persisted). Its original build used
`zenmetrics sweep --metric zensim-gpu --zensim-features-regime with-iw`
(`scripts/v_next/build_qsweep_expanded.py`). As of 2026-07-19, GPU zensim
**scoring** is fully disabled (`MetricKind::ZensimGpu => panic!("zensim-gpu is
DISABLED...")` in `zenmetrics-cli/src/metrics/mod.rs`), so that exact command
now panics. Separately, CPU `sweep --feature-output` hardcodes 300 features
regardless of `--zensim-features-regime` (`sweep/run.rs`'s `want_features_cpu`
path calls `run_zensim_with_features` -> `zensim::score_with_features`, not
the regime-aware `run_zensim_features` that `jobexec`'s V2Ab path uses) — so
even a CPU `sweep --feature-output --zensim-features-regime v2-ab` would
silently produce the wrong width, not 720. This matches the plan's own
flagged footgun (I-3: "score-pairs --feature-output still emits 372/300").

**Method actually used** (avoids both issues by not asking `sweep` to compute
zensim features at all):
1. Reproduced the exact grid verbatim from `build_qsweep_expanded.py`: QGRID
   (40 points: 0,5,10,...,60, JND-zone 70-90 step2, near-lossless 90-96,
   fractional near-lossless 96.5-100) for jpeg/webp/avif; JXL_DISTANCES (50
   points: 0-0.3 step.025, 0.3-1 step.05, 1-3 step.2, tail to 25) for jxl, with
   `jxl_q_equiv(d) = clamp(100 - 4*d, 0, 100)`.
2. `zenmetrics sweep --codec {zenjpeg,zenwebp,zenavif,zenjxl} --sources <39 refs>
   --q-grid ... [--knob-grid '{"distance":[...]}'] --distorted-out-dir ...
   --pairs-tsv ...` — encode + decode-back ONLY (default CPU `zensim` metric,
   which is fine since only the GPU variant's scoring is disabled; no
   `--feature-output`). 39 source refs verified locally at
   `/mnt/v/input/zensim/sources/<image_id>.png` (39/39).
3. `zensim/examples/v2_ab_extract` (CPU, `--features feature-regime-v2,threads,training`)
   on the merged (ref, dist) pairs -> the full 720-wide vector (v1-372 ++
   v2-348) in one pass on identical decoded pixels.
4. Joined back onto the ORIGINAL 4,817-row identity (image_id, codec, q,
   codec_param) — 100% matched. f0..f371 in the OUTPUT are the ORIGINAL
   (GPU-extracted) values, unchanged; only f372..f719 are new, so any
   existing consumer of the 372 block sees zero shift.

**Cell-count bookkeeping.** The full cartesian re-encode (39 images x grid)
produced 6,487 cells — MORE than the original 4,817, since this run has no
GPU odd-dimension NaN issue (it never touches the GPU path). Two NEW failure
modes surfaced on this superset, neither reducing the 4,817-row backfill
(both combos were already absent from the original grid):
- Current `zenjxl` **rejects distance=0.0** ("invalid input: distance 0 out
  of valid range 0.0..=25.0"). Consistent, not a regression: the original
  grid already has **zero** rows at `codec_param==0.0`.
- Current `zenjxl` encode+decode-back returns **off-by-one dimensions** on
  13 odd-dimension source images (`513x769`/`769x513` -> decoded
  `514x770`/`770x514`), dropping 104 of 1,950 would-be jxl cells (8 of 50
  distances x 13 images). This is a genuine zenjxl/zenmetrics codec behavior
  discovered here, not introduced by this backfill — flagged for the owning
  team, not fixed (zenmetrics source is out of scope this session).

**372-feature verify** (per-row L2 of backfill-CPU vs original-GPU
extraction, flag threshold L2>0.5):

| codec | n | L2 median | L2 p90 | L2 p99 | L2 max | n flagged |
|---|--:|--:|--:|--:|--:|--:|
| jpeg | 920 | 0.0068 | 0.288 | 70.23 | 70.23 | 59 |
| webp | 960 | 0.098 | 405.6 | 795.8 | 795.8 | 348 |
| avif | 1,400 | 0.050 | 0.314 | 0.870 | 1.68 | 57 |
| jxl | 1,537 | 0.0056 | 0.251 | 2.21 | 80.4 | 72 |

The unflagged noise floor (median L2 ~0.005-0.05) is exactly what's expected
crossing GPU (original) vs CPU (backfill) zensim backends — CLAUDE.md's own
eval-grid pointer doc already documents this class as "bit-equivalent...
within metric tolerance, irrelevant to rank/monotonicity", not bit-exact.
The flagged tail is NOT noise — it cross-validates cleanly against
DOCUMENTED contamination:

- **jxl (72 flagged):** 20 of the 72 sit at `codec_param==0.025` (the exact
  distance the pre-fix jxl-encoder's `i16` DC-saturation bug hit —
  `benchmarks/jxl_nearlossless_contamination_2026-07-15.md`). **All 4** of
  that doc's "unambiguously broken" cells
  (`b2e6e2b5969eaf25_1022x818`, `85d6b54b6872b19b_512sq`,
  `7f7998c62e54398f_1024sq`, `3316926_opo25u_512sq`) are flagged here.
- **webp (348 flagged):** all **8** documented w11 GPU-garbage ladders
  (`eval_grids_2026-05-29.pointer.md`) are flagged, including
  `9059ec43b26aa167_769x513` — the one ladder that doc lists as corrupted for
  **both** webp AND jpeg. That is also the **only** jpeg-flagged image (all
  59 jpeg-flagged rows are that single image_id across its near-lossless
  q-range, L2 nearly constant ~70.23 regardless of q — the bit-constant-
  garbage signature, not a real encode difference). webp additionally flags
  8 images beyond the documented w11 list (NEW finding: `85d6b54b6872b19b_512sq`,
  `14ab4af28901fbeb1356b06d2d08ae06_512sq`, `7f7998c62e54398f_1024sq`,
  `090d19695a8b43c2_512sq`, `3316926_opo25u_512sq`, `5002b18aa50f70d9_512sq`,
  `00b13be94a4867dd_1022x818`, `96a0024c685ead3f_1024sq`) — several of these
  coincide with the JXL-contaminated source images, suggesting these specific
  refs may be generally fragile across codecs (high-DC / high-contrast
  content, unconfirmed). Worth a follow-up; out of scope here.
- **avif (57 flagged):** no single dominant ladder — diffuse drift spread
  across many images/params. Consistent with zenavif being actively reworked
  since 2026-05-29 (the task brief's own risk call: "AVIF is the risk since
  zenavif is in flux").

Flagged rows are **kept** in the output (`drift_flag=1`, `verify_l2_372`
column) rather than dropped — they're real cells with real (if noisier)
v2-348 features; downstream consumers can filter on `drift_flag` if they want
the conservative subset.

**One transient hiccup, not a bug:** the avif `zenmetrics sweep` foreground
call twice hit the Bash tool's 2-minute default timeout (AVIF encode is
slower than jpeg/webp) — relaunched with `run_in_background: true`, completed
cleanly (1560/1560, 547s).

## 2. corruption_grid (2,016 cells) — pure re-extraction

No re-encode needed: all 2,016 cells' pixels (672 base entries x 3 kind
variants `{corruption,q10,q20}`) plus the single shared reference
(`gb82_dog__reference.png`) already exist at
`/mnt/v/output/zensim/corruption_gate/*.png` (confirmed against that dir's
own `_MANIFEST.json`). Built a `(ref, dist)` pairs TSV and ran
`v2_ab_extract` directly. 2,016/2,016 matched (100%).

**Verify:** both the original 372 grid (`extract_features_372col --corpus
pairs`) and this backfill (`v2_ab_extract`) are CPU, on the same pixels, so
parity should be near-ULP — confirmed: L2 median **2.0e-8**, max **1.17e-5**.
Zero rows flagged. This also cross-validates the pipeline: the corruption
grid (CPU-only original) matches near-perfectly while the dial grid
(GPU-original) shows the expected cross-backend noise floor — the mechanism
producing drift in §1 really is backend difference, not a bug in this
session's extraction code.

**One transient retry, not a data problem:** first `v2_ab_extract` attempt
panicked mid-run — `Cannot allocate memory (os error 12)` reading
`gb82_dog__channel_swap_gb__sq8__op20__corruption.png`. That exact file
loads fine via PIL (576x576 RGB, valid PNG header) — a transient resource
hiccup while the concurrent T-big fleet's `jobexec` swarm was also running on
this box (peak-RSS only 0.48 GiB, far under the 24G run-heavy cap).
Immediate retry succeeded cleanly (2016/2016, 11s, rc=0).

## 3+4. nonphoto (10,000) + imazen26 (10,025) — fleet ledger+blob join

**These are NOT a separate corpus** — content-filtered subsets of the T-big
bigcodec (canonical-picker) validate/test cells the concurrent zenmetrics
session's fleet was filling. Verified: `ref_basename` (`o_NNNN.png.scaleWxH`)
is identical to canonical-picker `ref_filename`, so their 720 features are a
byproduct of the fleet run — a feature-space JOIN, not a re-extraction.

**The join was much harder than `docs/V2_EXPERIMENT_PLAN_2026-07-20.md`
assumed**, because the fleet exposes no consolidated `(ref, f0..f719)`
parquet:

- `s3://zentrain/jobs/bf-*/ledger/*.parquet` chunks are job-tracking metadata
  ONLY (`job_id/image_path/codec/q/knob_tuple_json/output_sha/status/...`)
  — and for `score_file` jobs, **`q` is always `-1` and `knob_tuple_json` is
  always `"scorefile"`**: the actual codec/q identity lives inside the
  RESULT blob (in `encode_sha`'s filename), not the ledger row. A "done"
  ledger status does not guarantee a valid feature blob either — two pools
  (`bf-avif`, `bf-zenjpeg-lossy`) were found to be **100% error blobs**
  (`{"kind":"metric","error":"...403 Forbidden..."}`) from a since-patched
  R2-permissions bug fetching `canonical/2026-06-27/.../encodes/...`. All
  other pools sampled (`bf-zjl2` and every `-tN` shard) were 100% valid.
- Blobs are **JSONL, not single JSON objects** — one job batches ~5-12
  variant results for the same source image, one JSON record per line. A
  naive `json.loads(body)` throws "Extra data" on every multi-variant blob
  (discovered when an initial full-scale launch showed 100% `json_error`
  across 30,000 blobs — caught before burning the whole budget on it).
- Blobs are addressed by content hash **per pool**, not globally
  (`s3://zentrain/blobs/` is a near-empty, unrelated prefix — a same-sha
  lookup there 404s).

**Method:** (1) bulk `aws s3 sync` of all 56 pools' `ledger/` dirs locally
(29,039 files, 304 MB — no reliable consolidated snapshot exists; a
`ledger_snapshot.parquet` glimpsed once in `bf-zjxll-t0` vanished on retry,
evidently a transient artifact of the fleet's own in-flight compaction, not
a stable API); (2) scanned + filtered to rows whose `image_path` ref-stem is
one of the 202 stems appearing in nonphoto/imazen26 (`scan-ledgers`),
yielding 657,750 matching ledger rows / 466,418 distinct `(pool, output_sha)`
after dedup; (3) excluded the 2 confirmed-broken pools (378,172 remaining
blobs); (4) `fetch-and-match`: fetch each blob (boto3, ~100-120 req/s
sustained at 96 threads; 256 threads regressed to ~110/s — R2-side
throttling), parse JSONL, keep `kind=="feature"` + `regime=="v2-ab"` +
720-length records, and match each against the eval sets' precomputed
fingerprint index `(refstem, round(f0..f371, 6))` **inline** — never
materializing the full ~4-5M candidate records (see next paragraph) — with
early exit once every eval row is matched.

**A real memory bug was caught before it could crash the box.** The first
full-scale launch used `{ex.submit(fn, x): x for x in items}` followed by
`as_completed(...)` over all 378,172 blobs at once. A `concurrent.futures
.Future` caches its result after completion, and the dict never removed
completed entries, so nothing was ever freed: RSS grew **~69 MB/s with no
plateau** (10.7 GB -> 13.3 GB in 37s), on track to OOM within minutes on this
58 GiB box (machine-safety mandate: never crash the box). Killed within ~3
minutes of launch, before any real damage; fixed with a bounded in-flight
window (`bounded_map()` in `fleet_blob_fetch_720.py`, at most
`workers*4` futures alive at once, `del`-ing each the instant its result is
consumed) and relaunched — confirmed flat/bounded on relaunch, and ~4x
faster besides (the leak had also been throttling throughput via memory
pressure).

**Final result: 2,599/10,000 nonphoto (26.0%) + 2,297/10,025 imazen26
(22.9%) — honestly incomplete, not fabricated, re-run recommended.** The
fetch was progressing steadily (rate climbed 71 -> 138 blobs/s as the thread
pool warmed up, matches climbing every checkpoint) when it **stopped
unexpectedly** after scanning 160,000/378,172 blobs (42.3%), with no Python
traceback in its log and no OOM-kill signature found in `dmesg` (kernel-log
access is otherwise unrestricted on this box, so the absence is a real
negative result, not a permissions gap). Checkpointed output at that point
is intact and is what's reported here — every `--report-every` checkpoint
rewrites the two output parquets in full, so nothing between checkpoints was
lost, only progress past the last one.

**Why this wasn't relaunched:** around the same time, a second, much larger
`fleet_blob_fetch_720.py fetch-all` invocation appeared on this same box —
a full T-big-corpus consolidation (949,343 blobs, ETA 90+ minutes),
apparently launched by the concurrent session that owns the T-big fleet
backfill (see "concurrent-edit collision" below — this is the same party).
It was still actively running 96 workers when the death was investigated.
Relaunching a second competing 96-worker fetch on top of that risks the
exact kind of resource contention the machine-safety mandate exists to
prevent, and may have been what killed the first run in the first place
(no direct evidence either way, but the timing lines up: their job started
~10 minutes before mine died). Given the task brief explicitly permits
partial results ("run the join against what exists... don't block on
100%"), this session accepted the checkpointed 26.0%/22.9% as final rather
than compound the load. **Recommended follow-up:** rerun
`fleet_blob_fetch_720.py fetch-and-match` once the box is free of concurrent
heavy jobs — the underlying ledger-scan + dedup (`fleet_ledger_matched_pooled
.parquet`, 466,418 distinct blobs after excluding the 2 broken pools) is
already built and reusable, so a rerun does not need to repeat the 56-pool
`aws s3 sync` or the ledger scan, only the blob-fetch+match stage.

## Concurrent-edit collision on `fleet_blob_fetch_720.py` (mid-session)

While `fetch-and-match` was running, `scripts/v_next/fleet_blob_fetch_720.py`
gained two functions (`_records_to_table_full`, `cmd_fetch_all`) that this
session did not write, between two of its own edits — discovered because a
routine re-read of the file showed unfamiliar content. `.workongoing` never
showed anything but this session's own claim throughout, so this was a
concurrent process editing the shared jj working copy directly (not through
the lock protocol) rather than a stale-marker handoff. The content turned
out to be a legitimate, well-built extension (a `fetch-all` streaming
full-merge mode + a `scan-ledgers --all` no-filter mode, for exactly the
T-big consolidation the concurrent zenmetrics session owns) — not
vandalism, just an unannounced shared-workspace collision (the class of
failure the CLAUDE.md's jj-workspace guidance exists to prevent).

Handling, in order: (1) froze the unreviewed diff into its own honestly-
labeled local commit (`wip(unreviewed): ...NOT authored by this session`)
rather than silently folding it into this session's own commit message or
discarding it; (2) the auto-mode classifier blocked the push of that
commit (reasonably — pushing unreviewed third-party code); (3) rather than
work around the block, forked this session's own next change off the last
verified-good commit instead, bypassing the frozen commit entirely, so nothing
of this session's authorship or the other party's work needed to be
destroyed; (4) shortly after, the other session independently `describe`d
and pushed its own clean version of the same content
(`feat(tbig): fleet_blob_fetch_720 fetch-all...`, landed on `main` on top of
this session's dial/corruption-grid commit) — resolving the situation
without further action needed. Verified after the fact: this session's
`e54a0c7d` commit is an ancestor of both the new `main` tip and
`main@{u}` — nothing was lost on either side.

## Tooling committed

- `scripts/v_next/backfill_dial_grid_720.py` — `build-pairs` (merge
  per-codec sweep `--pairs-tsv` outputs + a side identity map) / `finalize`
  (join to original identity, verify, write 720 parquet).
- `scripts/v_next/backfill_corruption_grid_720.py` — same shape for the
  pixels-already-exist case.
- `scripts/v_next/fleet_blob_fetch_720.py` — `scan-ledgers` / `fetch-blobs` /
  `join` / `fetch-and-match` (recommended, fused, bounded-memory) for the
  raw zenfleet job-system ledger+blob format.

## Known gaps / honest accounting

- dial_grid: 4,817/4,817 backfilled; 536 flagged as cross-backend drift
  (explained above, cross-validated against known contamination, not a
  pipeline defect); the two zenjxl encode limitations discovered (distance=0
  rejected, off-by-one on odd dims) are reported to the owning team, not
  fixed.
- corruption_grid: 2,016/2,016 backfilled, near-ULP verified, 0 flagged.
- nonphoto/imazen26: **26.0%/22.9% backfilled (2,599/10,000 + 2,297/10,025),
  honestly partial.** The fetch stopped at 42.3% blob-scan progress for a
  reason this session could not fully diagnose (no traceback, no OOM
  signature), and was not relaunched because a much larger concurrent
  T-big consolidation job (`fetch-all`, ETA 90+ min) was actively using this
  same box's resources at the time — see the dedicated section above.
  Never fabricated: any eval row with no exact fingerprint match in the
  scanned fleet output is dropped, not synthesized. imazen26 additionally has
  a hard ceiling of 99.6% (9,981/10,025) under this fingerprint scheme — 44
  rows share a duplicate `(refstem, rounded-f0..371)` key with another
  imazen26 row and can never be individually distinguished by it, regardless
  of fleet coverage. Rerun `fetch-and-match` (the ledger scan + dedup is
  already done and reusable) once the box is free of concurrent heavy jobs
  to improve coverage.
