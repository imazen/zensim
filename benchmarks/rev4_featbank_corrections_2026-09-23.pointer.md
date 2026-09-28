# Rev4 feature-bank correction evidence (2026-09-23/24 UTC)

The large correction artifacts live under `/var/tmp/featbank-impl/`.
They are local audit evidence, not committed inputs. No file in this
pointer was written under `/mnt/v`.

| Artifact | Contents | SHA-256 anchor |
|---|---|---|
| `calibration/` (5.7 MB) | Old-definition TRAIN-pixel diagnostics and extraction command ledger | `calibration/commands.tsv`: `27988452a70ae8b3f78dd316d6ac147b1ff4c7564446f0a9fd140a972f922459` |
| `calibration_census.txt` | Old C1 hat occupancy, C3 saturation and C1 diagnostic quantiles | `9e719daa366ca448334c0d7de3815d00c0ed99ca7b174c05b57581684612deae` |
| `identity_revision/` (38 MB) | Revised 144-pair × 6-mode extractor CSVs, per-run manifests, stdout/stderr and commands | `summary.json`: `eec8f2c18fd818d140b82acb7f6c139eae2d5ef66797cd9abf2612ee13644099`; `commands.jsonl`: `c4b32dbc131aade34d53d4ce6976d0ba19b6986bf98127e09ae22f62601c8afd` |
| `identity.log` | Wrapper output for 18 revised old-slot comparisons | `b76649592594ad273eda99ea06aa136d2bf3e72d87444a270b16ca800b4f5021` |
| `post_census.txt` | Revised C1 emitted occupancy and C3 top-bin saturation | `d7e279db864558142145ec7040b0d3f17411d58a959b2bfb4f0c18ca75d5e6c1` |
| `family_final.log` | C1–C4 family test rerun, including JPEG ladder | `18a97b5fca0fb93613513da5c63a402a2cae5ccc7bd77ce790fbee3c5319cdfe` |
| `corpus_tests.log` | Explicit opt-in mounted corpus and synthetic CI gates | `4fe14ec609a1a95639dbdf49c209350a08e5ead7a2c95cbd8ffa39391d9a5919` |
| `parity_tests.log` | Revised release research and SIMD-tier parity tests | `885a4279d32af237716ba3f8f81917790d1b9921f1009b388c19ce6ca7066a11` |
| `clippy_final2.log` | Final CI-exact workspace Clippy pass | `e3d4a104064c0eb17cc978d1f3996d8c956348baa97ba63b5b88775cad2fc5c9` |
| `api_check_final.log` | Final public API snapshot check | `ff181372095868387ed7886538c346d362acfd088c6f6b0dac181af8e4198acf` |

`benchmarks/rev4_featbank_impl_WORKLOG.md` gives UTC start/end, exact
commands, exit codes and source lines. The extractor binary for the
revised gate has SHA-256
`18743b0391c0572c887c36c5aa8c9e787ead41532aecf659e94368b9f16c40e4`.
