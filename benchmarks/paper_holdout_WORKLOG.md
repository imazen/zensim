# paper_holdout WORKLOG — Devin takeover (2026-09-23)

QUARANTINED pending audit. Times UTC. cwd = /home/lilith/work/zen/zensim--paper-holdout
unless noted. A = /var/tmp/paper-holdout/a. Pre-takeover steps (score_all.sh legs,
prep, most of assemble) are Opus's — see ~/tmp/devin/paper_holdout.log and
paper/<lane>/opus-handoff (kxnrqmxn).

## Commands

| start | end | exit | command | outputs |
|---|---|---|---|---|
| 05:55:1x | 05:59:4x | 0 | `taskset -c 16-31 env OUT=/var/tmp/paper-holdout/a bash research/2026-09-paper-holdout/score_all.sh` (relaunch; idempotent — skipped completed legs; untimed accuracy, coordinator-approved outside the heavy lock) | `$A/scores/dvifmish_<corpus>/<preset>.tsv`, `$A/scores/gmsd_konjnd_rawjpg.parquet`; log `~/tmp/devin/paper_holdout_score.log` — last line `score_all done fails=0` |
| ~06:01 | 06:02 | 0 | `env OUT=$A ZEN_PANEL_BIN=$A/bin/panel python3 research/2026-09-paper-holdout/analyze.py prep` | `$A/vec/comparators.json` sha256 b39dc637…; printed `prep: comparators written {'cid22a': 2192, 'csiq': 866, 'aic3': 600, 'aic4': 300, 'konjnd': 504, 'aic4full': 300, 'sdr25': 50}` |
| ~06:03 | — | 1 | same + `assemble` | FileNotFoundError: panel binary not found → set ZEN_PANEL_BIN |
| ~06:03 | — | 1 | same + `assemble` (with ZEN_PANEL_BIN) | TypeError: dict() got multiple values for keyword argument 'label' → one-line fix in analyze.py (`res = {x["label"]: x for x in panel_batch(jobs, stats="full")}`) |
| ~06:04 | 06:05 | 0 | same + `assemble` | `$A/stats.json` sha256 3f332423…; per-corpus SROCC lines (see record Table H4); checks printed: B-era srocc_between=0.9999239991555462, gmsd_konjnd png/rawjpg both -0.7841837326395679 |
| 06:05:02 | 06:05:54 | 1 | `env OUT=$A ZEN_PANEL_BIN=$A/bin/panel bash research/2026-09-paper-holdout/run_boot.sh` | `$A/boot/*.txt`; `boot done fails=1` — `git -C` fails inside a jj workspace (no colocated .git) → `regression_csiq_orig.txt` empty |
| ~06:10 | — | — | edit run_boot.sh: `jj file show -r ffc14647 …` instead of `git -C … show` | run_boot.sh |
| 06:18:24 | 06:19:31 | 0 | `env OUT=$A bash research/2026-09-paper-holdout/run_boot.sh` | `$A/boot/*.txt` regenerated; log line `REGRESSION OK: byte-identical`, `boot done fails=0`; regression_csiq_{orig,edited}.txt both sha256 cde7ad27… |
| ~06:20 | — | 1 | `env OUT=$A python3 research/2026-09-paper-holdout/build_fulleval.py` | panel binary not found in subprocess env → rerun with ZEN_PANEL_BIN |
| ~06:20 | ~06:21 | 0 | `env OUT=$A ZEN_PANEL_BIN=$A/bin/panel python3 research/2026-09-paper-holdout/build_fulleval.py` | `$A/EXPOSURE.json` sha256 d67c06e9…; `$A/fulleval/peer_{gmsd,dvifmish_*}_paper.{manifest,fulleval}.json` — fulleval sha256s: gmsd 3aa48a5c…, talk f2dbc535…, ours 41dd85ae…, gate 85754bc7… |
| ~06:21 | ~06:22 | 0 | `env OUT=$A python3 research/2026-09-paper-holdout/report.py` | `benchmarks/paper_holdout_2026-09-23.json` 30,189 B → over 30 KB cap → slimmed (drop per-arm plcc/krocc/signed/source → keep srocc/pooled_ci/within_boot/deltas) → 24,881 B; full JSON `$A/paper_holdout_2026-09-23.full.json` sha256 44d3b1b3…; tables `$A/tables_md.txt` sha256 c6f62b2a… |
| ~06:23 | — | 0 | assemble `benchmarks/paper_holdout_2026-09-23.md` = protocol header + tables_md.txt | record .md 20,159 B |
| ~06:24 | — | 0 | `jj describe` (this lane's first quarantine-line commit) | commit aa8ea6a1 on `quarantine/devin/paper-holdout` |

## jj operations (repository state, not files)

- `jj new` + `jj squash -f kxnrqmxn -t @ --keep-emptied`: moved my one-line
  analyze.py fix out of the empty opus-handoff change into my own change.
- `jj bookmark create quarantine/devin/paper-holdout -r @` → currently aa8ea6a1.

## Number provenance (record → raw output)

- Every SROCC/PLCC/KROCC in Table H4: `$A/stats.json` (written by
  `analyze.py assemble` via `zen_stats.panel_batch`, stats="full").
- Every pooled/within CI and paired Δ in Tables H1–H3: `$A/boot/<corpus>_vs_{ssim2,B}.txt`
  (paired_perref_boot.py, 10,000 ref-clustered draws, seed 20260901).
- Report layer computes nothing: report.py parses the two sources above.

## sha256 index (key outputs)

stats.json 3f3324239c49… · comparators.json b39dc63702cb… · EXPOSURE.json
d67c06e99a82… · boot/{cid22a,csiq,aic3,aic4,aic4full,sdr25,konjnd}_vs_{ssim2,B}.txt
in /tmp list (see manifest) · fulleval peer rows listed above ·
scores/gmsd_*.parquet: cid22 01a6de88, csiq e3aab835, aic3 ba3040cc,
aic4 9037c570, aic4full 773f627a, sdr25 68354a35, konjnd 17e1ebc1,
konjnd_rawjpg fca22d2c.

## Landing correction (2026-09-23 UTC)

Opus independently recomputed the CID22-A(25), CSIQ and KonJND numbers in `REVIEW_PAPER_MEASURE.md`. The prior “CID22-B labels never read” wording was false: `pairs/cid22.tsv` carries all 49 references' MCOS, but only A(25) rows were correlated; no CID22-B statistic was computed. Corrected the committed record and JSON, `/var/tmp/paper-holdout/a/EXPOSURE.json`, and the external DONE note. The record also corrected an overclaim beyond the reviewer's requested list: DVours−B on AIC-4 full resolution is not CI-separated in the pooled result, because its interval [−0.001, +0.041] crosses zero; the crop interval [+0.020, +0.046] excludes zero. The original code edits in `analyze.py`, `report.py` and `run_boot.sh` were carried over unchanged from `quarantine/devin/paper-holdout`.
