# HDRVID evidence pointer — 2026-10-10

Canonical: `/mnt/v/output/zensim/hdrvid-2026-10-10/` (frames 69 GB, frames-1080 8.5 GB, receipts, tables, extraction, e31-video, crosscheck, logs, bin with SHA256SUMS).
Mirror: `/mnt/tower/output/zensim-hdrvid-2026-10-10/` (rsync -rlt; SHA-256 verified: all 1,029 non-frame files + 60 sampled frames = 1,089 match; 4,741 files on each side).

| Artifact | SHA-256 |
|---|---|
| e31-video/report.json | `5ef1610940893ec5da193631f46723804c3c407ecb7bb781a31f019df350357f` |
| E31_VIDEO_REPORT_VERIFICATION.json (full per-study) | `0858440b2360b0a1ab7a6a844b4560c9817c17dfe6dbd5a11a8abc85d28dd28a` |
| exposure freeze | `bd6d283527290269ed96a7a39a3d1889cb5268e851f63b56614a9ec65f35fe29` |
| tables/hdrvdc.parquet / keys | `d1c9025ad573086220ec04b6e031df60916c61648c4d4316b233def004e351ef` / `ee372a497ec81ba788dabb25a92331ad490f2e1e83af0dbe50a71a4148be2c9d` |
| tables/avt.parquet / keys | `b58f5797ead736c2a3500b2c75ddd6efacb989f5ca662c72e47ae34fa7cc2081` / `a5782893c0691af3861657cfafa9583c165d59d41573bcb5c5c1cf3d9c82fe47` |
| bin/hdrvid_decode / hdrvid_extract (build 17f6bf6e) | `57117381…` / `bc7c39f1…` |
| bin/ffmpeg 8.1.3 / ffprobe | `95ab438e…` / `16ccbcc3…` (source tarball `7138d28c…`, signature verified) |

Correction: the table manifests' `decoder_era` string names zenpng `37c942ed`; the binary actually used zenpng 0.1.4 (crates.io), as every decode receipt records. Manifests are frozen; the source string is fixed.
Rust sources were rustfmt-reformatted after the binaries were built (layout only, no logic change).
