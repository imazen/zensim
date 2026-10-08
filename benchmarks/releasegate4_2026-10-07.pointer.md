# RELEASEGATE4 synthetic verification evidence

Date: 2026-10-07. Base: `8e81659b2f9e3673edd81e29047cffc8dae8d9cd`.
Implementation: `674830a5999e5f81230075f34ea6ef246689c973`.
Regressions: `acabdb2862483dc8098941f0ec637572e1670791`.
Local only; no push or real protected payload read.

Canonical local evidence: `/mnt/v/output/zensim/releasegate4-2026-10-07/`.
`VERIFY.json` indexes SHA-256s, source pins, commands, checks and limitations.
`probe-events.json` records the kernel sentinel events for the fixed code and
the same regression module run against a source-only export of the base.
The source snapshots and raw logs are retained beside the index. No R2/tower
mirror was made in this lane; the local evidence remains retained.

Authoritative logs: `tests-final-isolated.log` (43/43 pass),
`tests-reviewed-tip-final.log` (expected negative control: 18 failures, one
error across 13 methods), `panel-parity.log` (36 synthetic cases),
`clippy.log`, `fmt.log`, `ruff.log`, and `lint.log`. Earlier exploratory logs
remain under `~/tmp/releasegate4/`; their failed fixture versions are not the
final evidence.

Commands run from the repository root, with `TMPDIR=$HOME/tmp/releasegate4`:

```sh
ZEN_PANEL_BIN=/mnt/v/output/zensim/releasegate2-2026-10-07/bin/panel \
  ../scripts/run-heavy --mem 4G --jobs 1 -- \
  python3 -m unittest discover -s scripts/tests -p 'test_kadid_terminal*.py' -v
# Negative control: same test file copied into a source-only 8e81659b export;
# run there with the same environment and -p test_kadid_terminal_bound_payloads.py.
../scripts/run-heavy --mem 4G --jobs 1 -- python3 scripts/verify_panel_parity.py \
  --bin /mnt/v/output/zensim/releasegate2-2026-10-07/bin/panel
../scripts/run-heavy --mem 16G --jobs 8 -- cargo clippy --workspace \
  --all-targets --all-features --exclude zensim-wasm-tests -- -D warnings
../scripts/run-heavy --mem 4G --jobs 1 -- cargo fmt --all --check
GIT_INDEX_FILE=$HOME/tmp/releasegate4/lint.index \
  ../scripts/run-heavy --mem 4G --jobs 1 -- python3 scripts/lint_scripts.py
```

Script lint checked854 scripts. Its tracked-file conflict/hygiene scans used
a scratch index restricted to source/docs/config files, excluding1,305 payload
or other paths without reading their contents. The normal index was untouched;
the exact exclusion names are retained. This is not a full payload-file lint.
Scoped Ruff F and `git diff --check` also pass. No Rust/panel source changed;
the previous round's17 Rust panel tests were not rerun here.

Serial `run-heavy` records:

| Check | rc / seconds | peak RSS | min available | peak load |
|---|---:|---:|---:|---:|
| Final synthetic suite | 0 /72 | 0.41GiB | 40044MiB | 9.19 |
| Reviewed-tip negative control | 1 /26 | 0.37GiB | 44694MiB | 6.91 |
| Panel parity | 0 /8 | 0.10GiB | 48075MiB | 9.39 |
| CI-exact Clippy | 0 /58 | 0.93GiB | 39036MiB | 14.11 |
| Formatting check | 0 /1 | 0.12GiB | 43160MiB | 10.69 |

These are verification resource records on a shared machine, not performance
qualification. [The work log](releasegate4_WORKLOG_2026-10-07.md) describes
the object-binding contract and remaining P3 findings.
