# Pointer — rev4 cvvdp-safesyn Part-2 sidecar

Companion report: `rev4_cvvdp_safesyn_2026-09-23.{md,json}`.

## Sidecar (ScoreFile harvest → parquet)

- **Parquet:** `/var/tmp/cvvdp-safesyn/safesyn_cvvdp_sidecar.parquet`
  (196,086 rows — one per admitted SafeSyn pair; keyed by `row_id` exactly
  like `safesyn-train944.parquet`, plus `ref_basename`/`source_path`).
- **Manifest:** `/var/tmp/cvvdp-safesyn/safesyn_cvvdp_sidecar_MANIFEST.json`
  (build commit, binary/image sha256, display parameters, decoder
  identities, input sha256s — brief Part-2 spec).
- **Columns:** `cvvdp_jod_standard_4k` (historical teacher, plain `cvvdp`),
  `cvvdp_jod_standard_fhd` (200-nit FHD), `cvvdp_jod_sdr_fhd_24`
  (E2b-selected 100-nit FHD), `ssim2_fresh` (same-buffer binding control).

## Provenance chain

- Fleet run: `cvvdp-safesyn-20260923` on zenfleet
  (`s3://zentrain/jobs/cvvdp-safesyn-20260923/` — manifest/ledger/claims/
  blobs); 3,218 ScoreFile jobs × 36–78 variant inputs.
- Executor image:
  `ghcr.io/imazen/zenfleet-worker:exec-cvvdp-safesyn-19d6dd8e`
  (`sha256:218cfc7f…`), musl zenmetrics `0bba95e6…`.
- Code: zenmetrics `19d6dd8ee378b64562c551358c1d935ca7ed58ee` on
  `master@origin` `b02812ae` (bookmark `quarantine/devin/cvvdp-safesyn`).
- Pixels: decoded inside the executor via the same contracts as the
  2026-09-14 admission extraction — pixel-hash-verified on the gate smoke
  (262 pairs, all 6 codec families, 0 mismatches) and pinned per-row by
  `ZEN_JOBEXEC_PIXEL_HASH=1` stamps carried in the run's output rows.
- Inputs: `s3://codec-corpus/safesyn-rev2-2026-09-06/{sources|images}/…`
  (rev2 corpus; byte-verified against `SAFESYN_VERIFIED`).

## Regenerate

```bash
python3 benchmarks/cvvdp_safesyn/harvest_safesyn.py \
  --run cvvdp-safesyn-20260923 \
  --out /var/tmp/cvvdp-safesyn/safesyn_cvvdp_sidecar.parquet \
  --manifest-out /var/tmp/cvvdp-safesyn/safesyn_cvvdp_sidecar_MANIFEST.json
python3 benchmarks/cvvdp_safesyn/safesyn_compare.py \
  --sidecar /var/tmp/cvvdp-safesyn/safesyn_cvvdp_sidecar.parquet \
  --out-md benchmarks/rev4_cvvdp_safesyn_2026-09-23.md \
  --out-json benchmarks/rev4_cvvdp_safesyn_2026-09-23.json
```
