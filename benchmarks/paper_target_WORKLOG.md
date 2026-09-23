# paper-target WORKLOG — Devin quarantine line

cwd abbreviations: `T=/home/lilith/work/zen/zensim--paper-target`,
`R=/var/tmp/paper-target/runs/peer-20260923T055844Z`. Times UTC.

## Context (pre-takeover + earlier Devin work this session)

- 05:58–06:0xZ: `build_and_run.sh` ran under the shared heavy lock (run-heavy
  `--mem 40G --jobs 16`); produced `$R/` (measurements.jsonl 5796 cells,
  bounds.jsonl, REPRODUCTION.json, COMPLETE.json). exit 0.
- 06:0x–06:3xZ: Devin wrote `benchmarks/peer_target_steering_2026-09-22.{md,json,pointer.md}`;
  record commit landed on the Opus handoff change — later split so the Opus
  change `wsprnvzt` stayed pristine and the record is its own commit.
- Coordinator correction (`~/tmp/devin/PAPER_target_CORRECTIONS_coordinator.md`):
  the record's sign convention was inverted and the JXL D/GMSD-dial sign was
  wrong. Devin verified the analyzer semantics directly.

## Devin commands

### Sign-convention verification (light)

```
start ~06:4xZ  end ~06:4xZ  cwd $T
sed -n '290,310p' scripts/v_next/rd_probe_analyze_2026-07-18.py
exit 0
```
Output lines (the convention the record now states):
```python
delta = abs(one[key]["error"]) - abs(two[key]["error"])
...
"mean_family_delta": mean(family_means.values()),
"families_model_better": sum(1 for v in family_means.values() if v < 0),
"families_versus_better": sum(1 for v in family_means.values() if v > 0),
```
→ positive `mean_family_delta` = first model WORSE. Analyzer fix commit
`9593faa1`; corrected record + JSON commit `8201e967` (quarantine head).

### Recompute checks for the DONE file (light, all exit 0)

cwd $T; every command reads only `$R/eval/analysis_summary.json` /
`$R/REPRODUCTION.json`:

```
python3 -c "...cross_model_fixed_requests jxl b3 train_curve matched_D vs gmsd..."
0.15958805084228517 1 4 6
python3 -c "...webp matched_D vs ssim2..."
0.2549464362008231 2 5 11
python3 -c "...webp preview-v0.2 versus rows..."
zensim-c -0.2616 6 1
ssim2 -0.1807 5 1
gmsd -0.0464 2 4
python3 -c "...zensim-vs-ssim2 12 cells..."
positive: 11 / 12   (sole negative: webp preview-v0.2 −0.1807)
python3 -c "...undershoots by_target_kind b3 train_curve..."
jpeg matched_B=3 matched_D=5 zensim-c=6 preview-v0.2=2 ssim2=2 gmsd=2
jxl matched_B=1 matched_D=1 zensim-c=4 preview-v0.2=1 ssim2=0 gmsd=1
webp matched_B=5 matched_D=3 zensim-c=7 preview-v0.2=2 ssim2=2 gmsd=6
python3 -c "...summary b3 train_curve med/p95/worst..."
(18 rows — see DONE §6 / record tables)
python3 -c "...REPRODUCTION.json keys..."
1134 0 1008 {'rows_new_matched_models': 1860, 'rows_old': 1860, 'keys_matched': 1860, 'identical_all_fields_except_timing': 1860, 'missing_from_new': 0} {'probes': 504, 'identical': 504}
```

Full verbatim command lines are in `~/tmp/devin/PAPER_target_DONE.md` (they are
the recompute commands themselves).

### Outputs touched (sha256)

- `benchmarks/peer_target_steering_2026-09-22.md` — committed at `8201e967`
- `benchmarks/peer_target_steering_2026-09-22.json` — committed at `8201e967`
- `benchmarks/peer_target_steering_2026-09-22.pointer.md` — committed at `8201e967`
- `benchmarks/paper_target_WORKLOG.md` — this file
- `~/tmp/devin/PAPER_target_DONE.md` — DONE with verbatim recompute commands+outputs
- Raw run `$R/` unchanged since `COMPLETE.json` (read-only for analysis).

### Quarantine bookkeeping

- Bookmark `quarantine/devin/paper-target` at `8201e967`; this worklog commits
  on top of it.
- `paper/target/opus-handoff` = `wsprnvzt` preserved (divergent; untouched).
- Manifest lines appended to `~/tmp/devin/paper_measure_manifest.tsv` for
  `PAPER_target_DONE.md` (created) and `benchmarks/paper_target_WORKLOG.md`
  (created).

## Landing correction (2026-09-23 UTC)

The Opus reviewer independently recomputed target cells from the 5,796-row measurements.jsonl and identified five false or incomplete statements. The committed record now corrects endpoint medians, per-codec fixed-request coverage, hit rates, CONTENDED timing, and zenjpeg dirty=2 disclosure. No statistic or source code was changed. Source: `REVIEW_PAPER_MEASURE.md` target section.
