# Subset-quality study — computed tables (block storage)

The committed artifacts of the study are
[`subset_quality_study_2026-09-04.md`](subset_quality_study_2026-09-04.md) (the
pre-registration + results) and
[`good_subsets_registry.json`](good_subsets_registry.json) (per-cell subset
fingerprints + held-out scores). The computed descriptor tables are larger than
the 30 KB git limit and live in block storage:

**`/mnt/v/output/zensim/subset-study-2026-09-04/`**

| file | what |
|---|---|
| `board_repro.json` | every board fulleval's `zentrain.repro` summary + the canonicalised arm key (argv with the seed value and `--out` elided) |
| `cov/arm_NNN.json` | `subset_sim` output per arm: full + early-window coverage per seed, per group, plus the sample-sequence digests |
| `phase2_rows.json` | the joined (arm, seed) table — 202 cells, 66 arms — descriptors beside held-out scores |
| `phase2_corr.json` | within-arm-rank Spearman per (descriptor, target), pooled and per group-structure |
| `phase2_null.json` | the permutation null (R=500) and the pure-luck control comparison |
| `h1_spreads.json` | per-descriptor between-seed relative spread within arm (the H1 degeneracy table) |
| `pilot/` | the split-seed decomposition pilot: 5 bakes, logs with `ZENSIM_SAMPLE_DIGEST`, verdicts |
| `gate/`, `gate2/`, `splitgate/` | the byte-identity and split-seed gate artifacts |
| `survey_siblings.py`, `run_phase2.py`, `analyze_phase2.py`, `seed_effect.py`, `build_registry.py`, `run_pilot.py` | the drivers |

**Everything here is regenerable.** The subsets are a deterministic function of
the recorded repro blocks, so `subset_sim` reproduces any of it, and each row's
`sample_sequence_digest` proves the reproduction is the same subset. Nothing in
this directory is a primary measurement that could be lost — the primary
measurements are the board fullevals, which live in
`/mnt/v/output/zensim/reports/fulleval/`.

Regenerate:

```sh
cd /mnt/v/output/zensim/subset-study-2026-09-04
python3 survey_siblings.py          # board -> board_repro.json
python3 run_phase2.py               # replay every sibling arm -> cov/, phase2_rows.json
ZEN_PANEL_BIN=<panel> python3 analyze_phase2.py
python3 seed_effect.py
python3 build_registry.py           # -> benchmarks/good_subsets_registry.json
```
