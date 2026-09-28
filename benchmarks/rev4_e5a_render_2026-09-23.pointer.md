# E5A render-regression bulky artifacts

The compact result is `rev4_e5a_render_2026-09-23.json` (26 KB). All bulky
outputs are in `/var/tmp/e5a-render/` on the producing machine. They are
scratch artifacts, not committed or durable archival storage. Rebuild with
`bash scripts/e5a_pipeline.sh` (gen/pairs/score/analyze stages) from the lane
workspace `~/work/zen/zensim--e5a-render`, using the pinned inputs listed in
`benchmarks/e5a-render_prereg_2026-09-23.md` §2-§3.

| Path in scratch directory | SHA256 |
|---|---|
| `results.json` (full analysis: 44 family×variant groups, 120 severity-SROCC cells, 40 localisation cells, per-family bootstrap) | `d41b53faa7127016041addf0189c2832fc81c785aef715d6f3e5b949010e926b` |
| `pairs.tsv` (993 scored pairs, key-joinable) | `9274bdd4b4a8e33e1fa7c719fad5b853707d116eb2c1e84e1c77e4cc66ee51ea` |
| `e5a_scores.tsv` (e5a_render_score: maxabs/psnr/t1-t5/gmsd/zensim_b + map sha256s) | `930848255e19f8ebf7e096ce3c1fb4440c618c45ffe15f0af34d6ec8c464d3a2` |
| `peer.tsv` (ssim2 + butteraugli, butter_map_sha256) | `225fae140defe9379d04795836151d8696c2665738f706a60cea22d092b7fc99` |
| `dssim.tsv` (dssim-core CPU via zenmetrics batch) | `ee043967d6e6d90762454150845c1d22a580bbcc2cf9be79b17c043c93128f59` |
| `tuner.parquet` (score_d, score_r915_fast, score_r915_rich) | `c346dc25d497201d696a342a3a83298be364d963d3b8e7d5c5df5f02ecfb5919` |
| `score.log` | `27db0e1f4efe7a0cca7f9fb389b803cb19dc54667fac3ee13e2d61f082454933` |

`fixtures/` holds ~2,900 PNGs with per-origin `_MANIFEST.json` (kind/family/
variant/severity/params/a/b files + pixel and PNG sha256 per item).
`maps/` holds per-item f32 error maps (`<key>__u8.f32`, `__lin.f32`,
`__gmsd.f32`, `__zensim_b.f32`) and `maps_peer/` the butteraugli diffmaps
(`<key>__butter.f32`); individual map sha256s are columns in `e5a_scores.tsv`
and `peer.tsv`. `target/` is the relocated cargo build directory (3.6 GB,
moved from `~/tmp/devin/target-e5a` per review correction 8; not needed for
recompute — rebuild with `cargo build --release --features e5a-render,
zen-decode` inside `zensim-bench/`).
