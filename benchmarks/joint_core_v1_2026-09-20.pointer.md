# Pointer: joint-core-v1 evidence (not in git)

Record: `benchmarks/joint_core_v1_2026-09-20.{md,json}`
Plan: `docs/PLAN_JOINT_CORE_SET_2026-09-19.md` (cifix worktree);
guards `docs/FITTED_CONSTANT_GUARDS_2026-09-19.md`; split registration
`docs/DATA_SPLITS.md` addendum 2026-09-20.

All evidence >30 KB lives under:

    /mnt/v/output/zensim/joint-core-v1/

Key artifacts:

- `_MANIFEST.json` — dataset manifest: build commits, per-input sha256
  (32 files), cluster/seed rules, per-leg kernel provenance.
- `pairs/pairs_core.tsv` — 52,963 rows with ref/dist paths, per-cell
  `encode_sha`, kernel, band, class, leg. `pairs/pairs_provenance.tsv`,
  `pairs/pairs_dvifm_sdr.tsv` (50,463 SDR rows), `pairs/errors.tsv`.
- `plan/` — renditions/cells/reused-row plans + selector coverage check.
- `refs/`, `dists/`, `cells/`, `cache/` — rendered references (plain
  Mitchell sharpen=0), encoded distorted bytes, cell outputs.
- `features/` — per-leg ZSTD parquets + `_MANIFEST.json`
  (`ceiling_rev3#b782e349`); `features/perm30/` (30-col row-permuted
  control tables); `features/hdr_pq/` (PQ era, own manifest);
  `fresh_944.csv` / `konfig_944.csv` extraction outputs + manifests.
- `dvifm/` — `core_ycbcr_{y,cb,cr}.bin` f16 capped block caches +
  `.index.jsonl` + `.hist` full-population 256×256 (C̃,m) histograms;
  per-plane extraction feature CSVs + manifests.
- `fits/repro/` — 5-seed core reproduction (specs, train logs, bakes).
- `fits/perm30/` — 5-seed permuted-column control + permuted dev copies.
- `fits/dvifm/` — `native3_fit.json`, `native3_fit.log`,
  `surfaces_native3/grid_*.csv` (per-cell C₀×β surfaces with
  mse_refit/srocc/krocc columns), plus the ≤10-min budget fit variant.
- `coverage_report.json` — bands/classes/legs/codecs/rungs/deciles.
- Equivalence record: `dvifm-equiv` outputs (see .md §DVIFM).

dHash audit + adjudication scratch lived in `~/tmp/devin/dhash_audit/`,
`flagged_pairs.tsv`, `flagged_confirm.tsv` (21 flags, all false-positive).
