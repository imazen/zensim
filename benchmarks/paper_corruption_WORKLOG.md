# paper-corruption WORKLOG — Devin quarantine line

cwd abbreviations: `C=/home/lilith/work/zen/zensim--paper-corruption`,
`A=/var/tmp/paper-corruption/analysis`. Times UTC.

## Timeline (from ~/tmp/devin/paper_corruption.log, verbatim timestamps)

- 05:04:34Z lane start (preamble: LANE_PREAMBLE.md + brief; no lane preamble file).
- 05:06:52Z workspace created at main@origin `ffc14647`; build-only zenmetrics
  workspace at cvvdpfix tip `d6e5ae96` (05:08:54Z).
- 05:10:43Z run1 queued under heavy lock: build cvvdp-display binary + score
  cvvdp 4k/fhd, gmsd, dssim, iwssim-subset, dvifmish ×3 presets, zensim
  v0_2/b/c/d + Rev3 rich/fast ens5 (validate + train). Log
  `~/tmp/devin/paper_corruption_run1.log`.
- 05:14:15Z analysis driver `research/2026-09-paper/corruption_peer_matrix.py`
  tested on existing fast-ssim2/butteraugli files: owner JSON reproduces
  DVIFM-ish cells exactly; exposure ledger entry registered in DATA_SPLITS
  before any new score read (commit `68ce6c10`).
- 05:37:44Z DEVIATION (supervisor patrol rule): withdrew queued lock job
  (never started), ran run1 outside the lock, `run-heavy --jobs 4 --mem 8G`
  on cpus 28-31 — accuracy-only scoring, no timed cells.
- 05:37:47Z–05:50:09Z run1 progress (per-step lines in lane log): build
  zenmetrics-cli @ d6e5ae965e6f; cvvdp 4k validate 05:40; fhd 05:41; gmsd
  05:41:43; dssim 05:41:46; iwssim 05:42:19; dvifmish 05:43:06; zensim+Rev3
  05:43:19; train legs 05:43:44–05:50:09.
- 05:50:12Z run1 complete, outside lock, 742 s, peak RSS 1.06 GiB; all score
  files written to `/var/tmp/paper-corruption/scores` (6024 validate rows,
  0 NaN, pair order verified; iwssim subset 3012 rows).
- 05:50:30Z analysis driver run rc=0 (owner `corruption_eval.py` sha
  `8e8e45d7…`, pixel-hash dedup revision): headline + CIs + transfer +
  breakdowns under `/var/tmp/paper-corruption/analysis`.
- 05:51:34Z checks: 0 exact ties anywhere; equal pass counts (dssim=gmsd
  validate, B=C iwssim176 train) verified as different pass sets
  (294/294 and 285/285 discordant) — coincidence, not a bug; DVIFM-ish
  feature-path B/D/Rev3 rows agree with pixel-API to ≤6 rows.
- 05:58:39Z record committed `c8d69a8e` (md + json + pointer);
  `~/tmp/devin/PAPER_corruption_DONE.md` written; lane DONE.
- ~10:5xZ Devin: appended per-number recompute commands+outputs to
  `PAPER_corruption_DONE.md` (standing rule); wrote this worklog.

## Verbatim commands (scoring + analysis, from lane log)

```
# run1 (accuracy-only, outside lock per patrol deviation):
run-heavy --jobs 4 --mem 8G  (script: /var/tmp/paper-corruption/run1.sh;
  cpus 28-31; build /var/tmp/paper-corruption/zm-target;
  outputs /var/tmp/paper-corruption/scores/*.tsv + logs/*)
# analysis:
python3 research/2026-09-paper/corruption_peer_matrix.py \
  --scores /var/tmp/paper-corruption/scores \
  --out /var/tmp/paper-corruption/analysis     (rc=0)
```

## Recompute checks for the DONE file (all exit 0, read-only)

See `~/tmp/devin/PAPER_corruption_DONE.md` "Recompute commands" — each
carries the verbatim command and its actual output against
`$A/owner_full_validate.json`, `$A/extras_full_validate.json`,
`$A/paired_vs_reference.json`, `$A/owner_iwssim176_validate.json`.

## Outputs / sha256

- Record: `benchmarks/paper_corruption_peers_2026-09-22.{md,json}` +
  `.pointer.md` — commit `c8d69a8e` (child commits `68ce6c10` exposure+driver,
  `527ac623` transfer+render, `5e333548` tie accounting, `88b51e05`
  owner-dedup follow, `e7619d7b` paired diffs).
- Owner sha256 `8e8e45d7e12000fe31ffbe4e8d273b50cb3968e1b70449e69dff8987cc891007`
  (in `$A/RUN.json`); binaries `BINARIES.sha256` in `$A/../`.
- Quarantine line: `quarantine/devin/paper-corruption` — head was `8afa0298`
  ("post-record scratch", empty); this worklog commits on top.
- Opus handoff `paper/corruption/opus-handoff` preserved (divergent, untouched).
- Manifest lines appended for `PAPER_corruption_DONE.md` (modified) and this file (created).
