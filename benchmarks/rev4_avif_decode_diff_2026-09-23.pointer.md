# AVIF decode difference bulky artifacts

The compact result is `rev4_avif_decode_diff_2026-09-23.json`. All bulky outputs are in `/var/tmp/avif-decode-diff/` on the producing machine. They are scratch artifacts, not committed or durable archival storage. Rebuild with `bash compile.sh` and `bash replay.sh` from that directory, using the pinned TRAIN inputs listed in `selected.tsv` and the source locations in the scripts.

| Path in scratch directory | SHA256 |
|---|---|
| `selected.tsv` | `cbb2081f7c5e2f94d46bc6adc3577d34da9a0d231d5be7d02740699da5c4c435` |
| `pairs.tsv` | `d38f5711b46f3d51a0eae1ead5452af5c12ca64ebdf6dd0ef13733718d35eeb8` |
| `pixels.json` | `8fc21c9f4efe2032495783ce69920f16b5c5733c0cca0f3fc63f931c34d25aab` |
| `native.json` | `46cfa76a9ac84c236decfbfbb755fcffdb7894aed98e4f03005b99a8e10cfc14` |
| `scores.json` | `329f17444e1573caeb77fdfb44871ce6fb22f1e3ae020b244eec4c3dc781a90e` |
| `probe` | `cf4d86d507ae9d096122b5cdfa11cc9331b3e151bcd66941057b69a4c6761f1a` |
| `zenmetrics-avif-cicp-test.patch` | `5f22a9011878b5ae2974897ce3c12217baa1bf1e5528849979c1024f35b8e2eb` |

Do not apply `zenmetrics-avif-cicp-test.patch`: its expected RGB8 hash pins the inexact tagged route. Use the corrected test specification in `rev4_avif_decode_diff_2026-09-23.md`.

`cell*.rgb` and `cell*.native` are packed per-cell RGB8 and native RGB16 blobs; their individual SHA256 values are in `pixels.json` and `native.json`. `replay.tsv` records the final command timestamps and log hashes. Raw outputs occupy about 808 MB, mostly those pixel blobs.
