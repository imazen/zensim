# paper-gates lane WORKLOG (Devin quarantine line)

Quarantine bookmark: `quarantine/devin/paper-gates`. Protected bookmark
`paper/gates/opus-handoff` (pqotkmky, divergent — left for auditor).
All times UTC. Lane outputs live under `/var/tmp/paper-gates/` (too bulky to
commit); committed artifacts are the record `.md`, `.json`, and this log.

## 2026-09-23 ~10:5xZ — m3_fixture_gen build (outside lock, untimed)

cwd `/home/lilith/work/zen/zensim--paper-gates`

```
CARGO_TARGET_DIR=/var/tmp/paper-gates/target taskset -c 16-23 \
  cargo build --release --manifest-path zensim-bench/Cargo.toml \
  --example m3_fixture_gen --features m3-fixtures
```

- start ~10:5xZ, end 11:0xZ, exit 0. `zensim-bench` is in the workspace
  `exclude` list, so the earlier `-p zensim-bench` invocation failed with
  "cannot specify features for packages outside of workspace"; the
  `--manifest-path` form builds it standalone. Log `/var/tmp/paper-gates/m3_build.log`.
- output `/var/tmp/paper-gates/target/release/examples/m3_fixture_gen`
  sha256 `b642914c3a4acb6775c24ba223ecf76454ccaae8bf78ae3b3b13bb2c5267fcc3`.

## 2026-09-23 10:54:55Z–11:28:40Z — score_all.sh under the shared heavy lock

The queued acquisition `flock ~/tmp/devin/heavy.lock scripts/run-heavy --mem
16G --jobs 16 -- bash /var/tmp/paper-gates/build1.sh` won the lock at ~10:54Z;
its `score_all.sh` child (pid 51583) scored every arm end-to-end, exit 0
(`end loadavg` logged 11:28:40Z). Idempotent `[ -f ]/[ -d ]` guards skip
existing outputs.



## 2026-09-23 11:3xZ — run_gates.py (first attempt) — FAILED

```
python3 scripts/paper_gates/run_gates.py --bv /var/tmp/paper-gates/target/release/bake_verdict --out /var/tmp/paper-gates/bv
```

- exit 1: `dial_peer_cells.py` could not parse `np.float64(-8.03…)` — numpy
  2.x `repr` leaked into the `.src.tsv`. Fixed at
  `scripts/paper_gates/run_gates.py:179` (`{v!r}` → `{float(v)!r}`).

## 2026-09-23 11:4xZ–11:5xZ — run_gates.py (rerun) — exit 0

```
taskset -c 16-31 python3 scripts/paper_gates/run_gates.py \
  --bv /var/tmp/paper-gates/target/release/bake_verdict --out /var/tmp/paper-gates/bv
```

- bake_verdict sha256 `f8523c6e4bcae2215e33e98e5f81424bdd407486bd6f100b5f453cfcf6f13eb9`.
- 42 `cells/*.tsv` written with "grid coverage 100%"; 40 `gaddr/*.json`
  verdicts (19 scorers × {native,s100} + bake_B + bake_D).
- output `bv/summary.json` sha256 `c54dcafaa719df26f92532ba986f18fe375415c0892109f511b1a9e9d1172b6a`.
- every number in the record came from this file (checks/states/A7r floors)
  or the raw score tables (probe min/med/max, spans).

## 2026-09-23 11:5xZ — aic2026_addendum.py — exit 0

```
python3 scripts/paper_gates/aic2026_addendum.py \
  --out-json /var/tmp/paper-gates/bv/aic2026_addendum.json
```



## 2026-09-23 11:5xZ — render.py — exit 0

```
python3 scripts/paper_gates/render.py --summary /var/tmp/paper-gates/bv/summary.json \
  --aic /var/tmp/paper-gates/bv/aic2026_addendum.json > /var/tmp/paper-gates/bv/tables_md.txt
```

- output sha256 `4d168feb65a502cfab9963c064ea43719414c2a0a5a0d8001b0eb471624612b7`.
- Record tables §2–§7 are this file's sections verbatim (§0/§1/§6 text +
  §8 provenance written by hand).
- headline values cited in the DONE file were re-derived from
  `bv/summary.json` with the exact commands listed there.

## Landing correction (2026-09-23 UTC)

Opus review accepted the measured values and identified two committed files above 30 KB. Their original bytes were copied to `/var/tmp/paper-gates/landing/` and SHA-pinned in the committed pointer. The compact record quotes C1 from the fair s100 reading. This landing commit also brought over the entire new 266-line `scripts/paper_gates/run_gates.py` and the Opus hand-off's 79-line `bake_verdict.rs` per-codec scale-free step table and unit test, along with the qualified `BTreeMap` fix. These additions report step counts; they do not change a gate statistic. The Opus landing reviewer ran `cargo test -p zensim-validate --bin bake_verdict per_codec_strict` (1 passed, exit 0) and independently confirmed strict-backwards 0.2415 = 2,273/9,411.

### Landing compile check, 2026-09-23 17:59:05–17:59:28 UTC

- cwd `/home/lilith/work/zen/zensim--landing-fixes`
- command: `/home/lilith/tmp/devin/heavy --mem 16G --jobs 8 -- env CARGO_TARGET_DIR=/var/tmp/landing-fixes/target cargo check -p zensim-validate > /var/tmp/landing-fixes/cargo-check-zensim-validate.log 2>&1`
- exit 0; output `/var/tmp/landing-fixes/cargo-check-zensim-validate.log` SHA256 `07323976e909754f3162167bfc3323be15ba18cdc4c6736c5607c87da86f775b`
- exact result line: `Finished dev profile [optimized + debuginfo] target(s) in 21.94s`; wrapper: `run-heavy: done rc=0 22s | peak-RSS 0.46GiB | min-avail 29426MiB | peak-load 5.94`.
