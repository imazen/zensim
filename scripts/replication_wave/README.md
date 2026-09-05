# replication_wave — the 2026-09-05 seed-arm wave

Registration + results: [`benchmarks/replication_wave_2026-09-05.md`](../../benchmarks/replication_wave_2026-09-05.md).

Committed rather than left in scratch because the pieces are reusable for any k-arm seed
study, and because `/tmp` and `~/tmp` are not durable.

| file | what |
|---|---|
| `make_fits.py` | builds `fits.json` — each fit's argv is the recipe's embedded `zentrain.repro.argv` VERBATIM with only `argv[0]`, `--out`, `--dump-checkpoints-dir` and the seed flags changed |
| `run_wave.sh` | serial runner (one fit at a time — machine-safety rule), `run-heavy`, `PROGRESS.txt` streaming, a `.done` marker on every exit path so it is resumable |
| `postprocess.sh` | packs + harvests each fit AS IT LANDS (LATENCY discipline: a late wake-up costs nothing). Fails LOUD — marker + `POSTPROCESS_FAILURES.txt` + nonzero exit |
| `analyze_wave.py` | arm decomposition. **Re-derives no correlation**: every SROCC and CI is read from a `bake_verdict --full-json` fulleval; arm mean/spread are summaries of those measured values, compared against the owner's own bootstrap CI half-width |

Arms: **S** = `--init-seed S₀ --sample-seed {new}` (ORDER varies, init fixed);
**I** = `--init-seed {new} --sample-seed S₀` (init varies, order fixed). The legacy
diagonal draw is a member of both, which CTL-B proved by measurement
(`--seed X` ≡ `--init-seed X --sample-seed X`, 0/12 corpora differ).

Paths are wave-dated on purpose; point `ROOT`/`W` at a new dated dir to reuse.
