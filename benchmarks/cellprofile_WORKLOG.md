# cellprofile WORKLOG

Lane: cellprofile. Base: `83a205ad` (optmlp tip). Bookmark `quarantine/devin/cellprofile`.
Objective: measured whole-cell profile of one MLP fit cell (H32 + H128, alone +
loaded), perf counters, byte-touch model, bit-identical-only proposals.
No repo source changes; instrumentation lives only under `/var/tmp/cellprofile/instr/`.

## Setup

- v7 image `ghcr.io/imazen/zenfleet-worker:fit-p0-v7`; executor `/usr/local/bin/fit-cell-exec`
  → `mlp_probe.py` → 4 inner parquet+train + refit + predict/panel → `mlp_importance.py`
  → pack tar.gz. Data: `data-full.tar.gz` (154 MB, sha 5db24…, content-addressed
  under `/scratch/fit-cell`, cached across cells on same scratch).
- Instrumented copies: `instr/fit-cell-exec`, `instr/mlp_probe.py`,
  `instr/mlp_importance.py`; JSONL phase records → `/scratch/phase_times.jsonl`.
  Zero output bytes changed: verified by untar+diff vs gate7-dev local blob
  `f8e548a5…` — all `.bin` weights and `.parquet` byte-identical; remaining diffs
  are embedded `hostname`, `timestamp_epoch`, and `t=` timing inside
  `fleet_stdout.log`/`train.log` (which propagate into `log_sha256`,
  `source_result_sha256`, `fleet_receipt.json` digests only). Cell is
  deterministic modulo clock/host strings.
- Cell commands: H32 `kadid_train/r0 hidden=32 outer=0 rep=0`; H128
  `aic3/minus_basic hidden=128 outer=0 rep=0`. `ZENSIM_MAX_TIER=v3` on this
  AVX-512 host (AVX2/v3 kernels inside).
- "Loaded" = normal dev-d* fleet running (≈16 containers on cpus 16-31 +
  gate lanes on 4-7). "Alone" = `~/tmp/devin/DEVBOX_PAUSE` flag → dev_guard
  pauses all dev-* (hysteresis 6/9 GiB guard unchanged); my cp-* containers
  (not dev-*) unaffected.

## H32 loaded run (manual exec wrapper, contended)

Executor total 930.20 s; probe subprocess 924.39 s; importance 1.63 s; pack 2.80 s
(90,636,261 B out). Fetch first-time 1.24 s / cached 0.14 s.
- inner_table: 0.35/0.34/0.32/0.27 s; inner_train: 190.40/187.07/164.53/187.27 s
  (inner2 dip = contention noise); refit_table 0.34; refit_train 190.90 s;
  predict 0.049; panel 0.050.
- importance: data_load 0.45; per family ~0.11 s (perm_table ~0.058 +
  perm_predict ~0.046 + panel ~0.003); 6 families, 6 forward passes total.
- Training ≈ 99.2% of cell. All non-training phases ≈ 8 s.

## Trainer structure (K=1 path, `mlp_train/mod.rs` ~L2720)

Per pair: 2 forward (`simd_mlp::forward_avx2`), 2 accumulating backprop into
`adam.g*`, `add_l2_grad_layer1` (full w1+gw1 sweep, l2λ=1e-5 active), gw2 L2,
`adam.step(w1,b1,w2,b2)` (adam_simd v3: load w,g,m,v; store w,m,v,g=0 →
8 accesses × 4 tensors), `nonneg_project` (w2/b1/b2 only), post-Adam
penalties no-op (COARSE_DECAY/GROUP_L1 = 0). f64 throughout.

Traffic model H32 (n=30,208 w1): fw ~0.50 MB, bp g accum ~0.97, L2 ~0.73,
adam ~1.93 → ≈4.1 MB/pair → ~12.4 GB cache traffic per 3M-pair inner fit.
H128 (n=120,832): ≈16.4 MB/pair.

## perf stat H32 inner1, LOADED (cpu 9, 213.8 s, user 212.9 s)

cycles 913.7G, instr 1,439.8G → IPC 1.576. backend_stalls 3,199.7G slot-cycles
(≈3.5/cycle, dominant); frontend no-ops 664.3G. L1d loads 768.5G, misses
115.7G (15.1%). L2: ls_rd_blk_l_hit_x 107.6G hits; ic_dc_miss_in_l2 4.94G;
ls_rd_blk_c 4.36G. DRAM: dram_io_all 221.1M fills ≈14.1 GB total (~66 MB/s);
demand near-DRAM 28.6M ≈1.8 GB. cache-misses 2.5%. LLC events not supported.
Counters muxed (~29.4% of runtime each) — ratios usable.
Verdict: single cell is L1/L2-bandwidth+dependency bound, NOT DRAM bound.
DRAM only becomes the wall when many cells share L3 (w1 set ~5.3 MB f64
weights+adam + ~30 MB feature tables per cell > L3 at ≥2 cells).

## Trainer phase split (from train.log t= checkpoints)

Loaded inner fit: 60 epochs ≈ 209.5 s (≈3.45 s/epoch ≈ 69 µs/pair); first
checkpoint t=3.8 s ⇒ load+init ≈0.4 s (0.2%); 13 evals (every 5 ep) inside
the flat 17.2 s/block cadence ⇒ eval ≲0.3 s each (~2% total); parquet read
inside startup.

## Pending

- h32l official run (worker+container timing, ledger, blob) in flight.
- h128l loaded in flight (~est 45-60 min).
- Alone: pause dev-*, h32a + h128a + perf alone + RSS sampler.
- Report: MISSING-first, ranked gated proposals.

## Refined byte-touch model (K=1, all f64)

H32 (w1 = 944×32 = 30,208 el = 236 KiB per array):
- 2× forward: w1 + x-row + biases ≈ 0.50 MB
- 2× backprop (gw1 RMW accum) ≈ 0.98 MB
- add_l2_grad_layer1: gw1 += λ·w1 ≈ 0.73 MB
- adam.step w1+b1+w2+b2: 4 loads + 4 stores ≈ 1.94 MB
- nonneg/penalties: ~KB. **≈4.14 MB/pair → ~12.4 TB cache traffic/fit (3M pairs)**
H128: per-pair 16.5 MB → ~49.5 TB/fit. Adam alone = 47% of pair traffic.
Feature rows are only ~15 KB/pair but over a ~30 MB f64 table → L3-resident
alone, DRAM-bound when N_cells×~35 MB working set exceeds shared L3.
Single-cell DRAM fills: ~14 GB/fit (~66 MB/s) → DRAM idle alone; saturation
is aggregate cache-capacity, not per-core bandwidth.

## Identity check (instrumented H32 vs gate7-dev local blob)

`diff -r` over untarred outputs: every `.bin` weight + `.parquet` byte-identical.
Residual diffs = `hostname`, `timestamp_epoch`, train.log `t=` fields, and
hashes derived from them (`log_sha256`, `source_result_sha256`, receipt).
bitexact.sh vs image binary: PASS (18 s, epoch lines + weights identical).

## Incidents

- First h128l attempt ran wrong cell (kadid_train/r0/128 manifest copied from
  h32). Killed ~10 min in; corrected to gate7h128-dev manifest (aic3/minus_basic).
  Corrected run also lost data-full.tar.gz on cleanup → fetch failed once;
  restored hardlink, relaunched ~15:50Z.
- Perf loaded ran outside `heavy` (single-core pinned measurement). Remaining
  heavy work goes through the lock.

## Final results (all 2026-09-25/26 UTC)

Alone (dev-* paused via DEVBOX_PAUSE, residual: other lanes' containers cpus 4-7):
- h32a: exec_total 574.40 s; inners 115.5/108.0/113.5/112.2; refit 120.0;
  fetch cold 0.78; importance 0.68; pack 1.73; container start 0.24; tail 0.21.
- h128a: exec_total 1965.73; inners 403.0/401.7/389.2/383.3; refit 385.0;
  fetch cold 0.74; importance 0.39; pack 1.03; container start 0.22; tail 0.13.
Loaded:
- h32l official: exec_total 1045.86; inners 206.6/201.8/211.5/201.1; refit
  217.1; fetch cached 0.14; importance 1.63; pack 3.06; start 0.44; tail 0.74.
- h32l manual: exec_total 930.20; fetch cold 1.24; inners 190.4/187.1/164.5/187.3.
- h128l: exec_total 3852.61; inners 856.5/741.7/789.4/739.4; refit 718.4;
  fetch cold 1.17; importance 0.90; pack 1.69; start 0.51; tail 0.60.
Ratios: H32 loaded/alone ≈1.82×; H128 ≈1.96×.

## perf stat alone vs loaded (H32 inner1, cpu 9)

alone 112.28 s wall: cycles 601.3G, instr 1438.6G, IPC 2.39,
backend_stalls 2966.9G slot-cyc (4.93/cyc), frontend no-ops 294.3G (0.49/cyc),
L1d 768.3G ld / 114.1G miss (14.8%), ls_rd_blk_c 1.145G, ic_dc_miss_in_l2 1.47G,
dram_io_all 163.4M (~10.5 GB), dram_io_near 12.7M, dTLB miss 13.0M,
cache-misses 1.47G (0.97%).
loaded 213.81 s wall: cycles 913.7G, instr 1439.8G, IPC 1.58,
backend 3199.7G (3.50/cyc), frontend 664.3G (0.73/cyc), L1d 768.5G/115.7G
(15.1%), ls_rd_blk_c 4.365G (3.8×), ic_dc 4.94G, dram_io_all 221.1M (~14.1 GB),
dram_io_near 28.6M, dTLB 14.0M, cache-misses 4.95G (2.5%).
Same instruction count both ways: degradation is 100% memory hierarchy
(L2→L3+ misses 3.8×). Alone cell is L1/L2-bound at IPC 2.39, DRAM ~66 MB/s;
loaded is L3/DRAM-latency bound. Confirms minibatch-1 full-matrix Adam
(47% of pair traffic) + 30 MB feature table per cell saturate shared
L3→DRAM at N×~100 MB working sets.

## RSS

- standalone H32 trainer alone (inner1, 3000+1000 rows): peak 126 MB.
- h128l loaded trainer (aic3): 63–69 MB steady; second phase ~129 MB.
- live dev-d* trainers (loaded fleet): 64–126 MB per cell.
- h32a/h128a RSS csvs empty (monitor raced container start) — bounded by above.

## Importance

6 families × (shuffle ~1ms + parquet ~12–66ms + predict ~13–51ms + panel ~3ms);
7 forward passes/cell (1 baseline + 6 permuted); subprocess 0.39–1.63 s =
0.02–0.16% of cell. Not Python-bound (parquet serialize + Rust predict
dominate). No action.

## Identity gates

- bitexact.sh on image binary: PASS (epoch lines + weights identical).
- h32a/h32l-manual/h32l-official blobs vs gate7-dev blob f8e548a5: file sets
  equal; all .bin + .parquet byte-identical modulo masked metadata fields;
  residual diffs = hostname, timestamp_epoch, t= in logs, derived shas.
- h128a blob vs gate7h128 verified c54b7e52: same; importance/result JSON
  equal minus sha fields.
- No repo source changed; no prototypes (owned files cover <1.5% of wall time).
  Proposals for kernel lanes ranked in CELLPROFILE_DONE.md.

## Proposals handed to kernel lanes (see report)

1. Fuse backprop+L2+Adam single-pass (~50% of w1 traffic) — optmlp/optadam.
2. Drop g zero-store via write-then-accumulate (~6%) — optadam.
3. f32 feature storage + exact widen — mine to land, crosses unowned plumbing.
4. Adam AoSoA operand interleave (0–5% speculative).
5. Faster pack compression (~1–2 s/cell; needs receipt-contract check).
6. Fleet scheduling by working set, not cpu count (ops lever).
