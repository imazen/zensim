# Pointer — paper corruption lane bulk outputs (2026-09-22)

Bulk outputs live on block storage, not in git (`/mnt/v` was below the lab's
free-space floor during this run, so everything is on the root volume):

`/var/tmp/paper-corruption/`

| Path | What |
|---|---|
| `scores/` | per-pair scores: `cvvdp_standard_{4k,fhd}_{split}.tsv`, `gmsd_{split}.tsv`, `dssim_{split}.tsv`, `iwssim_{split}.tsv` (min side >= 176 origins only), `dvifmish_{split}/{preset}.tsv`, `profiles_{split}.parquet` (zensim PreviewV0_2/B/C/D + both Rev3 ensembles) |
| `pairs_iwssim/` | the IW-SSIM-eligible subset pair lists (origins with min(W,H) >= 176) |
| `analysis/` | owner JSONs (`owner_{set}_{split}.json`), breakdowns + clustered CIs (`extras_*.json`), `transfer_train_to_validate.json`, `crosscheck.json`, `tables.md`, `RUN.json` (every input path + sha256) |
| `BINARIES.sha256` | sha256 of every scoring binary and bake used |
| `logs/` | per-step logs, including the zenmetrics build log |
| `zm-target/` | cargo target of the zenmetrics build (cvvdpfix tip `d6e5ae96`) |

Inputs (not copied): pair lists `/mnt/v/output/zensim/dvifmish-eval-2026-09-22/pairs/corruption_{validate,train}.tsv`
(DVIFM-ish §6 run), decoded honest controls under `/var/tmp/dvifmish/decoded/corr_{split}/`, butteraugli and
fast-ssim2 per-pair scores from that run under `/var/tmp/dvifmish/peers/`.

No R2 or Tower mirror was made (`/mnt/tower` not mounted during this lane); the committed record carries the
numbers, and every per-pair file above is regenerable from the commands in the record.
