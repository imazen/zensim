# Part B C1–C4 extraction worklog (2026-09-24, quarantine)

Scope: new-family f986–f1321 sidecars for the promoted 18-set Rev4 bank.
All input lists derive from `keys.parquet`, with no label columns opened.
Source pixels use the bank's stored paths. `_sealed/` is never opened.

## Preparation

| UTC start–end | Cwd | Command | Exit | Output and exact source line |
|---|---|---|---:|---|
| 12:17:31–12:17:31 | zensim--partb | `df -BG --output=avail /home /var/tmp` | 0 | `/home` 68G; `/var/tmp` 1173G (command output). |
| 12:17:31–12:17:31 | zensim--partb | `jj status` | 0 | Clean, parent `6a7b88e5`; `6adb0c28` is an ancestor (jj log). |
| 12:20:24–12:20:24 | zensim--partb | `python3 -m py_compile scripts/rev4_featbank/convert_cache.py` | 0 | No output. |
| 12:20:24–12:20:24 | zensim--partb | `python3 scripts/rev4_featbank/convert_cache.py --set safesyn --partb-pairs /var/tmp/partb/pairs/safesyn_sample.tsv --partb-limit 2000` | 0 | `PARTB_PAIRS set=safesyn rows=2000 sha256=191af41be9570b21326b7d4c76e30306f3e39310595706c5cc8b3dd08e22f389 path=/var/tmp/partb/pairs/safesyn_sample.tsv`. |

The bank stores only populated f0–f943 old-family slots. It does not store
f944–f985, so the requested f0–f985 bank comparison is impossible as worded.
Parity below checks all 905 stored old-family slots on the measured 2,000 rows,
then a deterministic 1% of each full set (seed 20260924). The absence of the
42 slots is reported as MISSING, not inferred from width.

## Runs

The first extractor build request (12:18–12:22 UTC, `cargo build --release
-p zensim-bench --example extract_features_372col --features
training,zen-decode,verify-all` under `~/tmp/devin/heavy`) was canceled with
exit 130 while queued on the lock. It produced no binary. The user then
required every local dependency to come from fetched clean main/master.

`jj git fetch` succeeded for zensim and 13 sibling repositories (12:24–12:28
UTC; logs in `/var/tmp/partb/logs/fetch_*.log`, exit 0 each). `git archive`
of each fetched main/master commit produced clean source snapshots under
`/var/tmp/partb/src/`. zenavif's dirty checkout was never used. An initial
zenavif archive attempt used a mistyped hash and exited 2 with zero files;
the subsequent archive from the fetched ref succeeded (exit 0). No sibling
working copy was changed. Build metadata is `/var/tmp/partb/build_meta.json`
(sha256 `d0fa20ec38fc8419e33fca7cf4704d988dac1c473cf1f2b189b55533d62ec93e`):
20 local packages from 14 clean snapshots; 220 registry packages pinned by
the snapshot Cargo.lock. The build overlay redirects all resolved git
dependencies to those snapshots; Cargo metadata reports `git_sources []`
and `local_outside_snapshot []`. The overlay touches only scratch files.

Pair generation command (12:22 UTC, cwd zensim--partb, exit 0):
`for set in safesyn cid22_train konjnd_bpg_train mcljci kadid_train kadid_select tid2013 cid22_a25 cid22_b konjnd_bpg_val kadid_terminal csiq aic3 konfig_val konjnd_jpeg_select konfig_train aic4 konjnd_jpeg_terminal; do python3 scripts/rev4_featbank/convert_cache.py --set "$set" --partb-pairs "/var/tmp/partb/pairs/$set.tsv" || exit; done`.
Exact row counts and sha256s are in `/var/tmp/partb/logs/pairs.log`.
An independent PyArrow metadata/TSV-header and row-count check (12:43 UTC,
cwd zensim--partb, exit 0) emitted exactly
`PARTB_PAIR_INPUTS sets=18 rows=248983 key_count_mismatches=0 label_columns=0`;
raw line: `/var/tmp/partb/logs/pairs_check.log`. No `_sealed/` path was opened.

### Process defect found 12:53 UTC

The pre-existing `featbank_sets.py` registry called `_ceiling_sets()` at
import time. Part B pair generation and early import checks therefore parsed
the TRAIN ceiling `INPUTS.json`, whose rows contain a `target` field. No
target was analyzed, selected on, or printed; no held-out or `_sealed/`
source was opened. The TRAIN-target read was not preregistered. This is a
process defect and is not backdated as a preregistration. The registry now
loads that file lazily only for Part A conversions. A guarded import that
raises on `INPUTS.json` access passed with exact output
`PARTB_IMPORT_CEILING_INPUTS_OPEN=0 sample_keys=2000` (cwd zensim--partb,
exit 0). `python3 -m py_compile` passed for all three Part B Python files.

The first clean-source build request was also canceled with exit 130 while
queued, with an empty compiler log: Cargo's default cache was under `/home`.
The replacement queued at 12:44 UTC from
`/var/tmp/partb/src/zensim/zensim-bench` is one heavy-lock command:

```bash
/home/lilith/tmp/devin/heavy --mem 16G --jobs 8 -- bash -c 'set -euo pipefail; mkdir -p /var/tmp/partb/cargo-home/registry; cp -a /home/lilith/.cargo/registry/. /var/tmp/partb/cargo-home/registry/; if test -f /home/lilith/.cargo/config.toml; then cp /home/lilith/.cargo/config.toml /var/tmp/partb/cargo-home/config.toml; fi; echo CACHE_COPY_DONE; env CARGO_HOME=/var/tmp/partb/cargo-home CARGO_TARGET_DIR=/var/tmp/partb/target cargo build --release --locked --offline -p zensim-bench --example extract_features_372col --features training,zen-decode,verify-all'
```

Log: `/var/tmp/partb/logs/build_clean.log`; start/end markers are adjacent.
All cache writes and compiler outputs are under `/var/tmp/partb/`.

At 13:00 UTC, before compilation, a second `jj git fetch` of every sibling
found zenmetrics `master@origin` had advanced from `318a20f1` to
`43abef94`; the other 13 refs were unchanged. The queued build was canceled
with exit 130, compiler log empty. The zenmetrics snapshot was refreshed
from that commit with `git archive`; the prior snapshot was preserved. Cargo
metadata again reported 20 local packages, 0 Git sources and 0 paths outside
the snapshot. Revised `/var/tmp/partb/build_meta.json` sha256:
`5ce5df9a8626665eced02a62febf50499c2e391b17f576558a1aa7665d7b35b0`.

The new 13:02 UTC heavy-lock request runs
`/var/tmp/partb/prepare_sources.sh` **after** acquiring the lock. Command:

```bash
/home/lilith/tmp/devin/heavy --mem 16G --jobs 8 -- bash -c 'set -euo pipefail; mkdir -p /var/tmp/partb/cargo-home/registry; cp -a /home/lilith/.cargo/registry/. /var/tmp/partb/cargo-home/registry/; if test -f /home/lilith/.cargo/config.toml; then cp /home/lilith/.cargo/config.toml /var/tmp/partb/cargo-home/config.toml; fi; export CARGO_HOME=/var/tmp/partb/cargo-home CARGO_TARGET_DIR=/var/tmp/partb/target; bash /var/tmp/partb/prepare_sources.sh; cd /var/tmp/partb/src/zensim/zensim-bench; cargo build --release --locked --offline -p zensim-bench --example extract_features_372col --features training,zen-decode,verify-all'
```

It fetches
the current main/master refs, refreshes changed clean archives, regenerates
Cargo metadata/build_meta, then builds offline with `CARGO_HOME` and target
under `/var/tmp/partb/`. Cwd `/var/tmp/partb`; log
`/var/tmp/partb/logs/build_dynamic.log`; start/end
markers are adjacent. This avoids building a source snapshot that grew stale
in the lock queue.

## HANDOFF

Pending only if the quota stop file appears.

At 2026-09-24T13:44Z (cwd `/home/lilith/work/zen/zensim--partb`),
`PYTHONPYCACHEPREFIX=/var/tmp/partb/pycache python3 -m py_compile scripts/rev4_featbank/bank_report.py && python3 scripts/rev4_featbank/bank_report.py`
exited 0. Exact output: `n_sets: 18`, `problems: []`,
`total_unique_keys: 248983`, `total_stimuli: 249227` (JSON).
The report now inspects every sidecar, preserving the original
`sidecar_cols` and adding `sidecar_cols_by_file` to the global manifest.
Current bank `_MANIFEST.json` sha256:
`0e8490f3723afb75a98a8dcd4229b590fffcb9cd6927c38a839256f2aa9a9a57`.
This preliminary global manifest will be regenerated after all 18 new
sidecars are written.

At 2026-09-24T13:50Z, added `scripts/rev4_featbank/partb_reextract.py`
(sha256 `f3c6c1087980c5b69fdfa0a0569995b351ac8a5ac8e9b92ac5bb134b62fbe00e`).
It draws 200 seeded rows uniformly across all 18 bank sets from keys only and
compares a fresh extraction's 67,200 new f32 cells with the sidecars, with a
second pixel-hash audit. `PYTHONPYCACHEPREFIX=/var/tmp/partb/pycache python3
-m py_compile scripts/rev4_featbank/partb_reextract.py` exited 0 in the lane
workspace. It has not yet drawn or extracted any rows; those run after the
full sidecars exist.

At 2026-09-24T13:52Z, checked the preliminary bank report against the
existing tower mirror. Its JSON was identical except `created_utc` and the
new `sidecar_cols_by_file` field. The tower copy had sha256
`b208bc084abaa4ac67794f5a22bb47977d0f3880b1e1918cabfc5e7b226956d9`.
The preliminary report was preserved at
`/var/tmp/partb/preliminary_global_manifest.json` (sha256
`0e8490f3723afb75a98a8dcd4229b590fffcb9cd6927c38a839256f2aa9a9a57`),
and the pre-Part-B global manifest was restored from the mirror to avoid
changing the shared bank report while extraction is queued. It will be
regenerated only after all sidecars exist. No set file was changed.

At 2026-09-24T14:00Z (cwd `/home/lilith/work/zen/zensim--partb`),
`python3 scripts/rev4_featbank/partb_reextract.py draw` exited 0.
Exact line: `PARTB_REEXTRACT_DRAW seed=20260924 rows=200
pairs_sha256=7f7ba15c7387cd65bd3e52bc4687fbf7c3db7f2dc81197a001d100007500bcef
mapping_sha256=4aaf92d954010c092dd1c4ef2b701f1464d774616927a0800ea482b1dead1100`.
The TSV carries only pixel paths and row ids; it reads keys, not labels.

At 2026-09-24T14:05Z, added a quota-stop and `/home` ≥20G guard at
the start of `/var/tmp/partb/prepare_sources.sh`, which the queued build
invokes after taking the shared lock. `bash -n` exited 0. Script sha256:
`a64db974bbcd7167951d3b4dfa007c1ee958731bce73b7276aa4a134ecfb46ea`.
`df -BG --output=avail /home /var/tmp` printed `48G` and `1162G`.
The build log remained 0 bytes; no compilation started.

## HANDOFF — quota stop, 2026-09-24T14:12Z

The user issued a quota stop at 68.0% weekly usage. The stop file appeared.
At 14:12:09 UTC the queued build process `2772350` was still blocked on
`heavy.lock`, and `/var/tmp/partb/logs/build_dynamic.log` had 0 bytes.
`kill -TERM 2772350` canceled that specific waiter at 14:12:14 UTC. Its
parent and waiter exited; the log remains empty (sha256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`).
No binary was built, no sample gate or heaptrack ran, and no Part B sidecar
exists. The bank global manifest is back to its pre-Part-B sha256
`b208bc084abaa4ac67794f5a22bb47977d0f3880b1e1918cabfc5e7b226956d9`.
The local clean-source metadata is only a preparation record, not a build
claim: `/var/tmp/partb/build_meta.json` sha256
`5ce5df9a8626665eced02a62febf50499c2e391b17f576558a1aa7665d7b35b0`.
zenmetrics was last fetched at master `43abef94`; all 14 refs must be
re-fetched at the next build. The prepared 200-pair re-extraction TSV/JSON
are ready, but no fresh extraction has run. `run_set.sh` and
`run_sample.sh` were updated at 14:08Z with stop checks and passed `bash -n`;
their sha256s are respectively
`cd0d9caf8f8fa05098881d7ec01e2a989fe047a635914a4371f074dd09cb1d31`
and `543e86186da933bd92ce897ddb8acad58092d2c5a175dbe63ea1b626ca9d5782`.

Exact next steps after a new authorization that lifts the stop:

1. Check `/home` has at least 20G and the stop file is gone. In the lane
   workspace, run the prepared clean-source build through `~/tmp/devin/heavy`;
   `/var/tmp/partb/prepare_sources.sh` fetches all 14 current refs and
   refreshes archives after taking the lock. Confirm Cargo metadata has zero
   Git packages and zero local packages outside `/var/tmp/partb/src/`, then
   record the actual binary and build_meta hashes in the pointer.
2. Under one heavy job, run `/var/tmp/partb/run_sample.sh`; require 2,000
   SafeSyn rows, full stored-old f32 parity, measured wall/pairs per second,
   and heaptrack peak. The 42 f944–f985 slots have no bank-stored baseline
   and must remain listed as MISSING for that requested comparison.
3. If the timing gate supports local extraction, run
   `/var/tmp/partb/run_all.sh` (one locked job per set). Then run the
   independent `partb_verify.py`, fresh 200-pair re-extraction check,
   `bank_report.py`, mirror with niced rsync after tower `df`, and check
   three file hashes. Finish the held-out pixels-only ledger draft and DONE
   report. Do not read `_sealed/` or human labels, and do not push.

One process defect remains documented above: before the lazy-load fix,
importing the original set registry parsed TRAIN `INPUTS.json` with a target
field, although no target was analyzed. Do not retroactively claim that read
was preregistered. No held-out label file was opened.

## Continued by Claude Sonnet lane partb-c1c4, 2026-09-24T20:55:00Z

Takeover of the stopped Codex lane (user directive: Sonnet = iteration lane). The Codex quota stop does not bind this lane. Removed every `CODEX_QUOTA_STOP.md` guard from the scratch scripts `/var/tmp/partb/{prepare_sources,run_sample,run_set,run_all}.sh` (originals kept as `/var/tmp/partb/old_scripts/*.codex`); the `.workongoing` writer now targets this workspace, not the primary checkout (that marker belongs to another lane). Queued the prepared clean-source build under the heavy lock at 20:54:53Z (log `/var/tmp/partb/logs/build_dynamic.log`, done marker `build_dynamic.end`).

### Sonnet lane, 2026-09-24T21:41:50Z — build + sample gate

- First build (offline, --locked) failed: the refreshed zenavif main pins zenravif via git (cavif-rs), and the zensim-bench optional `corruption-corpus` git dep was also unresolved offline. Fix: dropped `--offline`/`--locked` (git pins fetched into `/var/tmp/partb/cargo-home`) and patched `corruption-corpus` to a `git archive` of codec-corpus at 8e10d4d7 (= origin/main). Final `cargo metadata`: 0 git-source packages, 20 local (`logs/build_gitpins.txt`). Cargo.lock sha256 `bf1130b4d9d50e7ee90772c2a8c9e0f264901c52b60a2b558fe8970d40e53080`.
- Codex's pair TSVs had no `human_score` column, which the extractor's `--audit-jsonl` requires (panic "audit input missing human_score"). `convert_cache.py`/`partb_reextract.py` now write a constant `human_score=0` placeholder (no label read); all 18 TSVs regenerated (sum 248,983 rows, sha256s in `logs/pairs2.log`).
- **Extractor defect on main (6a7b88e5):** `audit.rs` gate accepts widths 372/944/986 only, so `--full-rev4 --audit-jsonl` fails every pair ("audit requires 372, 944 or 986 finite canonical features"). Scratch-snapshot-only patch adds 1322 (`benchmarks/partb-c1c4_audit1322.patch`, 2 lines). The gate is a width/finite check; it does not touch feature computation. Needs an upstream fix in zensim main; coordinator decision.
- Binary (patched) sha256 `dd4db2eea31a27ab54315ea841793aa23bf76f6f4c871c22d8f5c8f9ee4434d3`.
- Sample gate (2,000 SafeSyn rows, 8 threads, under heavy): `scored 2000/2000 pairs in 43.5s (0 failed)`, `WALL_SECONDS=43.65`; `PARTB_CHECK set=safesyn rows=2000 old_checked=2000 old_f32_mismatches=0 finite=1 unique=1 identity_bad=0 c1_live=[21994, 18160, 18005, 17029, 15835, 13758] c3_p99_saturated=[0, 29, 24, 0]`. Heaptrack (separate run, 93.3 s under heaptrack): `peak heap memory consumption: 631.30M`, `peak RSS (including heaptrack overhead): 806.76M` (`logs/safesyn_sample.heaptrack.txt`).

### Sonnet lane, 2026-09-24T22:05:29Z — coordinator addition: csfw/DVIFM sidecar
COST_CUTS_AUDIT Ambiguous 1: also write `bank/<set>/features__csfw_dvifm.parquet` (pair_key + f944-f985, f32, keys order) from the same `--full-rev4` pass. `convert_cache.py` now builds it beside the C1-C4 sidecar (shared `_write_sidecar` helper, same manifest fields, finite/unique/coverage checks, per-feature live-cell and identity-nonzero counts; no old-f32 parity because the bank stores no f944-f985). `partb_verify.py` and `partb_reextract.py` extended to both sidecars (verify also re-reads both from the raw CSV on the 1% sample). Tested on the 2,000-row sample CSV: `PARTB_CHECK_CSFW set=safesyn rows=2000 finite=1 unique=1 identity_nonzero=0 dead_columns=[]`. The SafeSyn full run started 21:59:50Z before this edit; its bind step runs the edited converter, so no set needs a csfw-only pass.

### Sonnet lane, 2026-09-25T02:20:37Z — C8 gmsbank pass prepared (separate root /var/tmp/partb2)
Coordinator: C8 gmsbank (f1322-f1501) is on zensim main 26494c8a. Pass 2 runs AFTER pass 1 (chain script `/var/tmp/partb2/chain.sh`) from a clean main snapshot with `--full-gmsbank`, root `/var/tmp/partb2` (own target, build_meta, raw, logs; pass-1 binary/meta untouched). New `bind_gmsbank` in `convert_cache.py` writes `features__gmsbank.parquet` (pair_key + f1322-f1501) and re-checks f944-f1321 bit-for-bit on EVERY row against the two pass-1 sidecars and f0-f943 on the seeded 1% against the bank. Two scratch-snapshot patches were needed on main 26494c8a (both are defects on main for the coordinator): (1) `audit.rs` width gate lacks 1322/1502 (patch `/var/tmp/partb2/audit1502.patch`); (2) `zensim-bench/Cargo.toml` defines `gmsd` twice (lines 114 and 121, TOML duplicate key, introduced with 5cdcf70a), so cargo cannot even parse the bench crate; the second line is dropped in the scratch snapshot (`/var/tmp/partb2/dupgmsd.patch`). A 2,000-row SafeSyn gate (`gms_sample_check.py`) runs before the full pass.

### Sonnet lane, 2026-09-25T04:00:12Z — old-slot parity finding on identical pairs; user ruling
mcljci passed; **kadid_train FAILED the strict f0-f943 gate**: `kadid_train: coverage=4880/4880 old_f32_mismatches=81/44345` (`logs/kadid_train.log`). Diagnosis (`/var/tmp/partb/diag_old.py`, `diag_ident.py`): the single mismatching sampled row is a pixels-identical pair (dist_path is the reference image I42_01_01.png); across ALL 39 identical rows of the set every row differs (3,096 cells, ~80 old-family slots, bank stores 0.0, fresh extractor gives e.g. f393=0.538) while 242 non-identical rows differ in 0 cells (`kadid_train n 4880 identical_total 39 {ident_rows 39, ident_rows_mismatch 39, ident_cells 3096, other_rows 242, other_rows_mismatch 0, other_cells 0}`). Root cause is identity handling between the bank's extraction era and this extractor's research path (not investigated further; needs reviewer). C1/C2/C4 and csfw/DVIFM are zero on identical pairs as expected (`identity_bad=0`, `identity_nonzero=0`).
User ruling (AskUserQuestion, 2026-09-25): "Record, don't fail". Implemented: gate stays bit-exact for non-identical rows (hard fail); ALL pixels-identical rows are additionally checked and their divergence recorded in the manifest (`old_f32_identical_rows_checked`, `old_f32_identical_cell_mismatches`) and the PARTB_CHECK line. Applied in `bind_partb`, `bind_gmsbank`, and `partb_verify.py`. kadid_train re-bound from its existing raw CSV (no re-extraction; `logs/kadid_train_rebind.log`): `PARTB_CHECK set=kadid_train ... old_checked=48 old_f32_mismatches=0 identical_rows_checked=39 identical_old_cell_mismatches=3096`. `run_all.sh` now continues past a failing set and prints `PARTB_ALL_DONE failed=...`; remaining sets kadid_select..konjnd_jpeg_terminal re-launched.

### Sonnet lane, 2026-09-25T15:57:49Z — completion
Pass 1 (C1-C4 + csfw/DVIFM, binary dd4db2ee..., zensim 6a7b88e5) and pass 2 (C8 gmsbank, binary 730741b9..., zensim 390c99a3 = main at build time, a descendant of 26494c8a incl. 8a0850be "repair the C8 landing regressions") finished for all 18 sets. Correction to the earlier note: main 390c99a3 no longer has the duplicate `gmsd` key (fixed by 8a0850be), so the dupgmsd scratch patch was a no-op; the only scratch patch in the final pass-2 binary is the audit width gate (`benchmarks/partb-c8_audit1502.patch`). Verifiers: `partb_verify.py` (pass 1) and `partb_gms_verify.py` (pass 2) both PASS; fresh 200-row re-extractions 0 mismatches (75,600 and 111,600 cells). Bank global manifest regenerated (`bank_report.py`, problems: []); everything mirrored to tower with 3+5 sha256 checks. Report: PARTB_C1C4_DONE.md. Per-set table: `benchmarks/partb-c1c4_sidecar_table_2026-09-25.md`.

Per-set result blocks and heartbeats: moved to `benchmarks/partb-c1c4_perset_runlog.pointer.md` (full original: `/var/tmp/partb/WORKLOG_full_2026-09-25.md`).

### Sonnet lane, 2026-09-25T16:32:51Z — REVIEW_PARTB corrections (data unchanged)
Coordinator directive after Opus review (PROMOTE WITH CORRECTIONS). Applied: (1) pass-1 build meta regenerated as v2 (`/var/tmp/partb/write_build_meta_pass1_corrected.py`; sha256 `407e5f05...`; v1 `5ce5df9a...` kept as `build_meta_STALE_5ce5df9a_codex_13z.json`), 36 pass-1 manifest entries repointed (backups `/var/tmp/partb/manifest_backup_2026-09-25/`), `local_patch`/`era_note` fields added to all 54 entries, `bank_report.py` re-run (global manifest `38c8d435...`), verifiers re-run (PASS), manifests re-mirrored to tower; (2) build_meta pointer rewritten; (3)-(4),(7) DONE fixes; (5) this worklog trimmed from 88,229 to under 30 KB, per-set blocks behind `partb-c1c4_perset_runlog.pointer.md`; (6) file manifest rebuilt; audit-gate patch files removed from the tree (scratch copies stay under /var/tmp). `partb_verify.py` now asserts the v2 build meta's `binary_sha256`.
