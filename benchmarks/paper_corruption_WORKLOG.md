# paper-corruption WORKLOG — Devin quarantine line

cwd abbreviations: `C=/home/lilith/work/zen/zensim--paper-corruption`,
`A=/var/tmp/paper-corruption/analysis`. Times UTC.

## Timeline (from ~/tmp/devin/paper_corruption.log, verbatim timestamps)



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
