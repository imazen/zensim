# Pointer: dvifm-screen2b-2026-09-19 evidence (not in git)

Record: `benchmarks/dvifm_screen2b_2026-09-19.{md,json}`
Protocol: `benchmarks/dvifm_screen2b_prereg_2026-09-19.md`

All evidence >30 KB lives under:

    /mnt/v/output/zensim/dvifm-screen2b-2026-09-19/

Key artifacts:

- `recipe.json` — the executed recipe (sha256 `d95ec9ea…`, full hash in the
  report JSON). Arms: basic228 / basic228dvifm / basic228perm30 / y60;
  seeds 9201..9219 step 2; E∈{50,100}; N∈{2000,8000,8327}.
- `run/RESULT.json` — stage log + per-cell bake sha256s + argv +
  post-hoc stamp record + tool binary hashes.
- `run/SUMMARY.json` — the complete aggregation: `per_cell` (240 cells:
  dev pooled + per-group + per-reference SROCCs, dev2 leg, fit SROCC,
  final loss), `decision`, `bootstrap_refs`, `learning_curve`, `dev2`,
  `bounded_pixel_audit`.
- `run/cells/*.json` — one RESULT JSON per fit (240); per-leg side
  artifacts (`.scores`, `.jobs.tsv`, `.raw.tsv`, `.tsv`, `.per_group.json`).
- `run/human_*.parquet`, `run/*.features.bin` — fit/eval tables incl.
  permuted variants; `run/_MANIFEST.json` records every sha256, the
  permutation definition + leg seeds, and scale-subset row counts.
- `run/audits/` — bounded pixel-parity audit (4 bakes × 33 dev pairs).
- `run/ckpt/*.best.bin` — the trainer's best-val-on-fit `--out` twins of
  each cell bake (the evaluated `{name}.bin` files are the final-epoch
  `ckpt_epoch{E-1}` dumps, renamed); kept as evidence.
- `run/features.csv` + `.manifest.json` — the w986 extraction pass over
  all 11,888 admitted rows with the producer surface.
- `segments/` — `konfig-{train,eval}-{segment,admission}.json` (the new
  KonFiG originsplit TRAIN/dev2 legs).
- `admit_2b.py` — the segment/admission builder; `specs/` — the fitted
  DVIFM spec copy used.

Immutable evidence: do not delete; rename to `*.bak` if superseded.
