# Restored-cut cost — 2026-09-25 (quarantined)

**CONTENDED, descriptive, not a gate.** Measured on the shared dev box while the potential lane's MLP batches ran (zenbench: mt1 4 gate waits, 4096² 6 of 8 rounds; mt8 25 gate waits/noisy rounds). Rev3/sqrt, no `target-cpu=native`, interleaved arms, 8 requested rounds, 180 s cap per group. Baseline arm = C8-on `fold1502_gmsbank` in the same groups. The chain is nested: each row is the median difference to the arm before it (`scripts/restore_cuts/cost_fit.py`; alpha + beta*pixels over 256/1024/2048/4096).

**Read the z1max row with the mapdev row:** `mapdev` and `z1max` share ONE band-kernel side pass (V-blur + kernel + eight maps per cell). The first arm that turns the pass on (`mapdev`) pays for it; `z1max` then adds only the block-max reduction. `gmsnative` reuses the C8 gradient accumulators on native X/B; `dvifmgate` is one extra sum per DVIFM level (noise-level).

Marginal cost as % of the C8-on median at the same size (fit R2 in brackets):

| Threads | Family | 256² | 1024² | 2048² | 4096² | beta ns/px | R2 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | mapdev (+side pass) | 22.5 | 26.2 | 24.7 | 23.5 | 57.79 | 0.9999 |
| 1 | z1max | 4.1 | 2.3 | 2.1 | 2.2 | 5.54 | 0.9992 |
| 1 | gmsnative | 4.6 | 4.9 | 4.5 | 4.2 | 10.33 | 0.9999 |
| 1 | dvifmgate | -2.1 | -0.4 | 0.3 | 0.2 | 0.58 | 0.96 |
| 1 | **all four** | 29.2 | 33.0 | 31.5 | 30.2 | | |
| 8 | mapdev (+side pass) | 8.9 | 16.1 | 13.3 | 13.2 | 17.26 | 0.9997 |
| 8 | z1max | 8.2 | -0.3 | 1.5 | 1.7 | 2.34 | 0.994 |
| 8 | gmsnative | 2.6 | 0.8 | 0.3 | 1.0 | 1.28 | 0.97 |
| 8 | dvifmgate | -4.6 | 0.8 | -0.3 | -0.3 | -0.36 | 0.93 |
| 8 | **all four** | 15.1 | 17.4 | 14.9 | 15.7 | | |

C8-on medians (ms): mt1 15.06 / 256.65 / 1014.73 / 4135.17; mt8 8.25 / 138.93 / 544.99 / 2200.25. Negative percentages and the mt8 256² z1max value are noise (the sign of a sub-1% difference under contention), not negative cost.

Independent confirmation, extractor wall time (SafeSyn 2,000 pairs, 8 threads, `gate_sample2.log`): `--full-gmsbank` 52.24 s, all four families + prefix 67.05 s (+28%), max RSS 750 MB -> 888 MB.

Peak heap (heaptrack, `scripts/restore_cuts/memory.sh`, synthetic pairs, identical at 1 and 8 threads): baseline `--full-gmsbank` vs candidate all-on: 256² 17.08M -> 17.18M; 1024² 81.58M -> 115.90M; 2048² 247.20M -> 382.25M; 4096² 805.49M -> 1.34G. The side pass materializes both XYB pyramids (~32 B/px). Raw: `/var/tmp/restore-cuts/logs/memory.tsv` (sha256 `42f71785...`); cost raw `cost_v1_mt{1,8}.zenbench` (sha256 `03cbda27...`, `ce7c9e2b...`), fit `logs/cost_fit_v1.jsonl` (`93f98fc9...`). Bench binary sha256 `51545e76...` (clean-snapshot build, crates on main). No optimization was attempted; per the user ruling cost is reported, never a gate.
