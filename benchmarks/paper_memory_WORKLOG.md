# paper-memory lane WORKLOG (Devin quarantine line)

Quarantine bookmarks: `quarantine/devin/paper-memory` in BOTH workspaces —
`zensim--paper-memory` (record + pointer + this log) and
`zenmetrics--paper-memory` (driver scripts). Protected handoffs
`paper/memory/opus-handoff` untouched in both repos.
All times UTC. Bulky outputs under `/var/tmp/paper-memory/`; hashes below.

## Acquisition 1 — 2026-09-23 06:19:41Z–09:07:34Z (shared heavy lock)

`~/tmp/devin/paper_memory_acq1.sh` via `flock heavy.lock run-heavy --mem 40G
--jobs 16` (requeued pid 3441312 after a 1 s lockfile-collision failure of the
first attempt, logged 06:04:05Z). cwd `/home/lilith/work/zen/zenmetrics--paper-memory`.

1. `cargo build --release -p cpu-profile` → binary
   `/var/tmp/paper-memory/bin/cpu-profile-61d0124b56fe`
   sha256 `61d0124b56fec8361fe9aab200e7af21dd844c1d895e35cae9ec6fa713feb54c`
   (BUILD-DONE 06:20:09Z).
2. pair-fixtures → `/var/tmp/paper-memory/pairs/` 16 files + `SHA256SUMS`
   (centre crops 256²/1024²/4096²; 7000×5728 reflect-101 mirror extension).
3. smoke 256² every arm/mode (06:21:09Z).
4. `python3 benchmarks/heaptrack/paper_memory/memory_matrix.py --bin $BIN
   --pairs /var/tmp/paper-memory/pairs --out /var/tmp/paper-memory/heaptrack
   --tsv /var/tmp/paper-memory/memory.tsv --chunk full-small` (06:21:10Z).
5. `python3 benchmarks/heaptrack/paper_memory/nproc_throughput.py --bin $BIN
   --pairs /var/tmp/paper-memory/pairs --out /var/tmp/paper-memory/tput.tsv`
   (06:22:44Z–07:18Z) — N=1/4/8/16 single-threaded processes.
6. `python3 benchmarks/heaptrack/paper_memory/cold_warm.py --bin $BIN --pairs
   /var/tmp/paper-memory/pairs --out /var/tmp/paper-memory/cold --threads 1
   --cpus 2` (07:18:13Z–09:07:34Z).

Outputs after acq1: `memory.tsv` (65 rows), `tput.tsv` (49 rows),
`cold/cold_warm_1t.json`. All timing/throughput cells carry the
`CONTENDED` flag where foreign load crossed the 0.5-core gate — reported,
never hidden, never used for uncontended claims.

## Acquisition 2 — 2026-09-23 11:28:41Z–… (shared heavy lock)

`bash ~/tmp/devin/paper_memory_acq2.sh` (queued 10:23:32Z, pid 3731663;
acquired the lock at 11:28:41Z). Same binary/pairs as acq1.

```
for chunk in ref synth map strip full-40:
  python3 $PM/memory_matrix.py --bin $BIN --pairs $P \
    --out /var/tmp/paper-memory/heaptrack --tsv /var/tmp/paper-memory/memory.tsv --chunk $chunk
python3 $PM/cold_warm.py --bin $BIN --pairs $P --out /var/tmp/paper-memory/cold --threads 8 --cpus 0-7
```

- `ref` done ~11:30Z, `synth` ~11:30:54Z, `map` ~11:35:43Z, `strip` ~11:39:17Z,
  `full-40` ~11:42:53Z (40 MP heaptrack cells, the slow leg).
- `cold/warm 8T` started 11:43:00Z — IN PROGRESS at log time.

## summarize.py fixes (Devin, committed in zenmetrics workspace)

1. `IsADirectoryError`: pass the cold JSON file, not the `cold/` dir.
2. `TypeError: unsupported format string passed to NoneType.__format__` at
   `summarize.py:143` — this build emits no per-phase cold split
   (`cold_phase_median_ms = {}`), so `t_synth_ms/t_setup_ms/t_first_ms` are
   absent. Fix renders missing phase fields as `—`; no value invented.
   Commit `a6355cde` on `quarantine/devin/paper-memory` (zenmetrics).

## Post-acq2 — 2026-09-23 13:3xZ

`acq2: done` logged 13:30:56Z; `cold_warm_8t.json` complete (sizes 256/1024/4096,
all rounds CONTENDED).

```
python3 benchmarks/heaptrack/paper_memory/summarize.py \
  --memory /var/tmp/paper-memory/memory.tsv --tput /var/tmp/paper-memory/tput.tsv \
  --cold /var/tmp/paper-memory/cold/cold_warm_1t.json \
       /var/tmp/paper-memory/cold/cold_warm_8t.json \
  --out-md /var/tmp/paper-memory/tables_2026-09-23.md \
  --out-json /var/tmp/paper-memory/tables_2026-09-23.json
```

- exit 0, cwd `/home/lilith/work/zen/zenmetrics--paper-memory`, 170-line md.
- outputs: `tables_2026-09-23.md` sha256 `a9ed60814f1eb1443b2a472e32c9473437fa06f8aae1f2327fdf60a49dc176ec`;
  `tables_2026-09-23.json` sha256 `8b0ad5b209adbdb2b52740321ff2c4928afc2344e23053ecb5d0dc068b99258a`.
- inputs (sha256): `memory.tsv` `60b6ed9d41b7d2172c782e201e7241da3b5d201a0c224f0da82709c722c4e09c`;
  `tput.tsv` `bc48f62c58775b37dd094542e21b6f0731d5e65e67e4400963830fcf053d91da`;
  `cold_warm_1t.json` `fe832e90107cec143b39d61dd07efa856cebc30fe64439fd7b4ca4c1dfb7e6c2`;
  `cold_warm_8t.json` `cf70b504bd98e784f937f0cd483e525536d9cb0308de8a075b021825fd39dcd4`.
- record `<!-- RESULTS -->` filled with the md tables verbatim;
  `paper_memory_2026-09-23.json` = a copy of `tables_2026-09-23.json`.
- NOT-RUN cells kept: ssim2 + butteraugli at 16 MP × N=16 (projected
  38.8 / 47.1 GiB > 36 GiB budget).

## Landing correction (2026-09-23 UTC)

Opus independently recomputed the 16 MP heap shares and 1 MP N=16 throughput in `REVIEW_PAPER_MEASURE.md`. The original 53,776 B JSON exceeded the 30 KB commit cap; it was copied byte-for-byte to `/var/tmp/paper-memory/landing/` and SHA-pinned in the pointer. The committed JSON now contains the independently confirmed headline values, with the full original detail retained outside git. No measurement was rerun.
