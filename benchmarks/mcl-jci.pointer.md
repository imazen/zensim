# MCL-JCI working copy and label contract — 2026-09-23

The durable original is the Windows WSL `/mnt/v/datasets/MCL-JCI.zip`, 1,601,678,334 B, SHA256 `3d67a2a823b54e2102e89146a7c92837535b0e246a8e0531150434f35a604f7b` (Opus reviewer independently verified both copies). The extracted tree under `/var/tmp/datasets/mcl-jci/extracted/` is a disposable working copy; its 5,221 per-file SHA256 values are in `mcljci_extracted_sha256.txt` below. Do not make a new pipeline depend on the working copy until its chosen durable storage location is verified.

The 5,000-row `pairs_mcljci_src.tsv` has `ref_path` pointing to each **pristine BMP**, and `dist_path` to each coded QF JPEG. The *human label anchor* is the **QF=100 JPEG**, per the HVEI dataset paper. `jnd_dist` counts mean JND boundaries from that anchor and is **0 for (BMP, QF100) by definition**. This zero does not assert that the coded QF100 JPEG is pixel-identical to the BMP. Higher `jnd_dist` means more visible distortion. The DSSIM comparison used QF100 JPEG → QFq JPEG, q=1..99, not the BMP.

| Working-copy file | SHA256 |
|---|---|
| `MCL-JCI.zip` | `3d67a2a823b54e2102e89146a7c92837535b0e246a8e0531150434f35a604f7b` |
| `mcljci_extracted_sha256.txt` | `8dc6a76743391f5e0f510b3c8561e1e4e65f4b328d820fdf611452458de09daf` |
| `build_pairs.py` | `75c3f6024c10e99da414a3c1d671ef95838241607a26f664247d163401cfd29f` |
| `make_inventory.py` | `6539e0f0419bbe10a14d255528cd7e49291c0c594a4ddf764b246cbd0afd5daf` |
| `pairs_mcljci_src.tsv` | `5a7058f31f70f3cc0176a16e886a8c54832292de653a3a24bc04b84aa29c9c17` |
| `mcljci_labels.csv` | `36f17dd8bac2d085f718e388fd5db307d887338c9c2b296349bb2ef20f184802` |
| `dssim_pairs.tsv` | `12b1903d112f16fdd94a70274d41c79abe973eb2820ba12be6f0881ad93d4da2` |
| `dssim_pairs_sub.tsv` | `69d7f325292b4032e48f60969c9ff985beabf8779d6a1718b79d8a19aefc6510` |
| `dssim_scores.tsv` | `ca3229a6ede6e147808fda019089350cd33787dc0b2e442b9e0b2e2adcb39b8a` |
| `dssim_scores_sub.tsv` | `b62b605d3eaa85da6a004fc4e520c98dd8f515e7b75d802fca2892cb92832cad` |

The labels were read before preregistration, at about 12:42 UTC (06:42 -06:00). The exposure is recorded in `docs/DATA_SPLITS.md`. The full-grid DSSIM-vs-label result is n=4,950, SROCC 0.8664, PLCC 0.9048, KROCC 0.7144, PWRC 0.9880 (Opus review, `zen_stats.panel`). No zensim candidate was scored on this set.

The `build_pairs.py` file in this scratch directory still contains the original incorrect hand-rolled `signed_srocc` diagnostic (mixed ddof conventions). Do not use its printed correlations. The label table itself was reproduced independently by the reviewer; all correlation values in the committed record are the reviewer's corrected values from the statistics owner.
