# Dense JXL fleet sweep — prep + launch plan (2026-05-30)

Motivated by `jxl_training_data_investigation_2026-05-30.md`: the only viable
JXL-underscoring fix is a **dense, full-range, ssim2-targeted** JXL training
set (the existing LARGE is near-lossless-only and falsified twice). Per user:
build it dense (not the coarse q5–q100) on the **zenmetrics fleet**.

## Prep (done — all on /mnt/v, not git)

- **Source selection (k-means, per dense-axis rule):** clustered the **3,218
  safesyn source images** on zenanalyze native-size content features (33 feat),
  picked the **2,000 centroid-nearest** representatives.
  - features: `/mnt/v/output/zensim/jxl_dense_2026-05-30/source_features_native.tsv`
  - picks: `.../representative_sources.tsv`
- **Dense distance ladder (44):** near-lossless **0.025→0.5 step 0.025** (20) +
  0.5→1.5 step 0.1 (10) + 1.5→3 step 0.25 (6) + low-q tail 3.5..15 (8). Far
  denser than the old 16-level q-grid, dense where JXL underscored.
- **Chunks:** `.../jxl_dense_chunks.parquet` — **88,000 cells**
  (2,000 src × 44 dist), schema `image_path / codec=zenjxl / q=50(dummy) /
  knob_tuple_json={"distance":d}`.
- Builder: `scripts/v_next/build_jxl_dense_chunks.py` (k-means + ladder).

## Fleet launch path (omni chunk-sweep — the STABLE path, NOT the in-flux jobsys)

Worker `scripts/sweep/omni_backfill_chunk_worker.sh` (zenmetrics): one
`zen-metrics sweep` pass → all 6 metrics (`zensim-gpu,ssim2-gpu,
butteraugli-gpu,cvvdp,dssim-gpu,iwssim-gpu`) + 372 features + saved `.jxl`
bytes + diffmaps to R2.

1. Upload chunks parquet + `chunks.jsonl` manifest + worker to
   `s3://coefficient/jobs/<RUN_ID>/`.
2. **Smoke (1 box):** `launch_single_instance.sh --run-id <RUN_ID>
   --chunks <chunks.jsonl> --onstart scripts/sweep/onstart_omni_backfill.sh`
   → verify artifacts land in R2 (encoded bytes + diffmaps + scored parquet).
3. **Scale:** `launch_backfill.sh` for the rest.
4. Collect from R2 → assemble balanced `jxl.parquet` (ssim2 target) → add as a
   training group at MODERATE weight → retrain v47 recipe → two-panel eval.

Isolated from the `claude-jobdash-ui` agent's jobsys work by using own
`<RUN_ID>` namespace + the old chunk-sweep path. Cost ~$10–30 (88k cells,
6 metrics, GPU). Use scoped per-sweep R2 creds for the fleet workers
(`~/work/claudehints/topics/r2-credentials.md`), never the root key.
