# RELEASEGATE5 synthetic evidence

Base: `4b8f28b6924626613c28893a2ea87622d49d7737` (`main@origin` fetched
at lane start; contains `1cd8a888`). Implementation:
`9353b8b201c2293ab6108e5c475ece82e3403e67`. Regression/admission record:
`988b6d4e83d78f6901e01fddf50ad37ab6ca3b0c`. Protected-submount completion:
`a044c61afd7906dda5e94750ac07ebcd87632077`. Local only; no push.

Canonical evidence: `/mnt/v/output/zensim/releasegate5-2026-10-07/VERIFY.json`.
The index records final source/tip identities, source hashes, commands,
checks, probe events, resource lines and file SHA-256s. Raw logs and source
snapshots live beside it. A byte-verified mirror is retained at
`/mnt/tower/output/zensim-releasegate5-2026-10-07/`.

Authoritative logs: `tests-mount-final.log` (56/56), `hardening-final.log` (12/12),
`negative-control-final.log` (expected 7/7 failures), `filesystem-proof.log`,
`clippy.log`, `fmt.log`, `ruff.log`, and `lint.log`. `code-pins-final.json` is the
six-entry runtime source inventory. `assertion-identity.json` records unchanged
prior assertions. `probe-events.json` captures the logged positive/negative
inode events. `lint-excluded.txt` lists 1,312 excluded payload/other paths;
lint's conflict/hygiene scan is source/docs/config-only, not a payload scan.

The actual-device proof keeps its synthetic files at
`/mnt/tower/output/zensim-releasegate5-synthetic-2026-10-08/preparation/`
and `~/tmp/releasegate5/device-proof-corpus/`. No real corpus, ledger or
authorization was involved. Earlier exploratory logs remain under
`~/tmp/releasegate5/`; the authoritative files above own the conclusions.

From the repository root, set `TMPDIR=$HOME/tmp/releasegate5`,
`OPENBLAS_NUM_THREADS=1` and
`ZEN_PANEL_BIN=/mnt/v/output/zensim/releasegate2-2026-10-07/bin/panel`:

```sh
../scripts/run-heavy --mem 4G --jobs 1 -- just releasegate-tests
../scripts/run-heavy --mem 4G --jobs 1 -- python3 -m unittest discover \
  -s scripts/tests -p 'test_kadid_terminal_hardening.py' -v
../scripts/run-heavy --mem 16G --jobs 8 -- cargo clippy --workspace \
  --all-targets --all-features --exclude zensim-wasm-tests -- -D warnings
../scripts/run-heavy --mem 4G --jobs 1 -- cargo fmt --all --check
GIT_INDEX_FILE=$HOME/tmp/releasegate5/lint.index \
  ../scripts/run-heavy --mem 4G --jobs 1 -- python3 scripts/lint_scripts.py
```

The full run used the identical unittest command underlying the existing just
recipe. Negative controls copy the final fixture and hardening modules into a
source-only base export, then run the seven methods named in the raw log.
For a new actual-device proof, choose two fresh synthetic directories on
different filesystems and call the committed
`test_kadid_terminal_hardening.prove_filesystem_separation(preparation, corpus)`.
Do not overwrite the retained proof files. The shared justfile is unchanged.

[Admission](releasegate5_admission_2026-10-07.md) explains why the current
shared-filesystem layout must refuse and how future receipt/destination
preparation changes. [Work log](releasegate5_WORKLOG_2026-10-07.md) records
the tested boundaries. These are software checks, not a real D2 qualification.
