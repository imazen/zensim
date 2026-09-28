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

## Correction at landing (2026-09-26)

This pointer was written before the fleet run. The facts below come from the harvest
manifest (`safesyn_cvvdp_sidecar_MANIFEST.json`, sha256 `38ca9707f6347f2f…`, `build` block)
and supersede the provenance chain above:

- **Executor image:** `ghcr.io/imazen/zenfleet-worker:exec-cvvdp-safesyn-9f36f88b`
  (`sha256:d110a3a79d98f06ade7c7dae3920c47b71c10362c959bd852341f6c5ff046050`), musl
  zenmetrics sha256 `77cc76d9af234b1381fa92ad7b1537a65d26ea058f7ac2a2839a49ecec3d38b4`.
- **Code:** zenmetrics `9f36f88b8a23e645bd791c1193842d950eccf4be` on base `b02812ae`. The
  `19d6dd8e` image and commit named above are the round-2 build, superseded before the run was
  declared.
- **Companion report:** `rev4_cvvdp_safesyn_2026-09-23.{md,json}` was never written in zensim, and
  `safesyn_compare.py` was not run. The descriptive record is zenmetrics
  `benchmarks/cvvdp_safesyn_2026-09-23.md`, and the durable copies (LAN store and tower paths) are
  listed in zenmetrics `benchmarks/cvvdp_safesyn_2026-09-23.pointer.md`.
- **Sidecar:** sha256 `775bdb8f95dbb3d116faaa646ec8dcafbb0810bc11dc30264997641e28701bc9`,
  196,086 rows. The `/var/tmp/cvvdp-safesyn/` paths above are scratch copies.
