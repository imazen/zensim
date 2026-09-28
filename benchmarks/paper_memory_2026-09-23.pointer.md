# paper_memory_2026-09-23 — raw-output pointer

All bulky artifacts live under `/var/tmp/paper-memory/` (not committed — they
are heaptrack `.zst` archives and per-cell logs). Everything below was produced
by the two heavy-lock acquisitions logged in `paper_memory_WORKLOG.md`.

## Driver binary

- `/var/tmp/paper-memory/bin/cpu-profile-61d0124b56fe`
  sha256 `61d0124b56fec8361fe9aab200e7af21dd844c1d895e35cae9ec6fa713feb54c`
  — zenmetrics `benchmarks/heaptrack/drivers/cpu_profile` built at zenmetrics
  commit `61d0124b` (merge of the cpu-profile driver + GMSD integer lane),
  release profile; `/var/tmp/paper-memory/bin/CURRENT` names it.

## Input fixtures (`/var/tmp/paper-memory/pairs/`)

`SHA256SUMS` in that directory covers all 16 files; key rows:

```
7e7742106c55d4d2e1b4f65e46331efa7006f789ff1ea97215c0e595d1f84fbd  ref_256x256.png
563537c411ffb6c345894d3aef49804fc03efae463ff277595b87194c64b59b4  ref_1024x1024.png
3c7f6c197996188cc9d95d8037ca2d4d95701fd80f901f74b31efe2287be05e9  ref_4096x4096.png
b4af6c28c6bb3384254e24a68f06a72041027be93dccddca4359fe6b02f3283c  ref_7000x5728.png
9b7b468958a0022a8820851fc4c7b89ff326f867cc18bf91dce7df2343d8bf81  ref_7000x5728.rgb
187b33f8fcb5d5d4ba3b62edf35bb9ea40a6ab1e3690aa83b72f293d3193a6f8  dist_7000x5728.rgb
```

The 7000×5728 set is the reflect-101 mirror extension described in the record
(§Inputs); all smaller sizes are centre crops, no resampling.

## Tables

- `/var/tmp/paper-memory/memory.tsv` — one row per matrix cell:
  UTC, arm, metric, preset, mode, W, H, threads, input, inputs_bytes,
  heap_peak_bytes, heap_print, peak_minus_inputs, time -v max RSS (KiB),
  wall/user/sys s, status, driver stdout tail, loadavg before/after.
- `/var/tmp/paper-memory/tput.tsv` — one row per N-process cell:
  arm, size, N, pairs_per_s, overlap, rss_sum_kib, foreign_avg_cores, status,
  per-process median loop ms.
- `/var/tmp/paper-memory/cold/cold_warm_1t.json`,
  `/var/tmp/paper-memory/cold/cold_warm_8t.json` — cold-launch raw ms per arm
  and round, phase splits where the binary emits them, warm-launch medians,
  contended-round accounting.
- `/var/tmp/paper-memory/heaptrack/*.zst|*.log` — per-cell heaptrack archives
  and `/usr/bin/time -v` logs (regenerate any cell with the commands in the
  WORKLOG).
- `/var/tmp/paper-memory/logs/` — driver stdout logs.

## Provenance notes

- Memory cells measure allocations, not speed; their `time -v` columns are
  never read as timings (Quiet-box rule in the record).
- Throughput + cold/warm cells carry the `CONTENDED` flag where foreign load
  crossed the gate; contended cells are reported, not hidden, and are never
  used for uncontended claims.

## Complete JSON moved at landing

The original 53,776 B `paper_memory_2026-09-23.json` is preserved byte-for-byte at `/var/tmp/paper-memory/landing/paper_memory_2026-09-23.full.json`; SHA256 `8b0ad5b209adbdb2b52740321ff2c4928afc2344e23053ecb5d0dc068b99258a`. The committed JSON is a compact headline summary; this full file carries all cells.
