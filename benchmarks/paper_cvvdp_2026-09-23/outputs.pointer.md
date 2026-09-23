# Pointer: paper-cvvdp lane outputs (2026-09-23)

Bulky outputs live on the root filesystem, not in git and not on `/mnt/v`
(which was below the lab's free-space floor):

| path | contents |
|---|---|
| `/var/tmp/paper-cvvdp/pairs/` | pair tables built from the stored board tables (labels copied, metric column dropped). CID22 is split: `cid22A_pairs.tsv` (25 refs, with MCOS) and `cid22B_pixels_only_pairs.tsv` (24 sealed refs, **no label column**). `upiq_*` split by study. |
| `/var/tmp/paper-cvvdp/jobs.tsv` | the scored job list (parity jobs `par_*` + main jobs) |
| `/var/tmp/paper-cvvdp/scores/` | merged score tables per job, `manifest.json` (binary sha256s, input/output sha256s, loadavg), `shards/` (per-shard inputs, outputs and logs) |
| `/var/tmp/paper-cvvdp/refmetrics/` | the per-pair tables the peer-row builder reads (`--refmetrics-dir`) |
| `/var/tmp/paper-cvvdp/fulleval/` | the new peer rows (`peer_cvvdp_aicfhd`, `peer_cvvdp_studydisplay`, `peer_cvvdp_4k_cid22A`, `peer_cvvdp_upiq_display`) — new files; no board file was modified |
| `/var/tmp/paper-cvvdp/analysis.json` | parity, per-study UPIQ panels, paired bootstraps (copied into this directory as `analysis.json`) |
| `/var/tmp/paper-cvvdp/bin/` | `zenmetrics`, `upiq_hdr_score`, `panel` binaries used (sha256 in `analysis.json` / the record) |

sha256 of every table is in `scores/manifest.json` and `analysis.json`.
