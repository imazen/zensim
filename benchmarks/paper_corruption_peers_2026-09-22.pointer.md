# Pointer — paper corruption lane bulk outputs (2026-09-22)

Bulk outputs live on block storage, not in git (`/mnt/v` was below the lab's
free-space floor during this run, so everything is on the root volume):

`/var/tmp/paper-corruption/`

| Path | What |
|---|---|
| `pairs_iwssim/` | the IW-SSIM-eligible subset pair lists (origins with min(W,H) >= 176) |
| `analysis/` | owner JSONs (`owner_{set}_{split}.json`), breakdowns + clustered CIs (`extras_*.json`), `transfer_train_to_validate.json`, `crosscheck.json`, `tables.md`, `RUN.json` (every input path + sha256) |
| `BINARIES.sha256` | sha256 of every scoring binary and bake used |
| `logs/` | per-step logs, including the zenmetrics build log |
| `zm-target/` | cargo target of the zenmetrics build (cvvdpfix tip `d6e5ae96`) |



No R2 or Tower mirror was made (`/mnt/tower` not mounted during this lane); the committed record carries the
numbers, and every per-pair file above is regenerable from the commands in the record.
