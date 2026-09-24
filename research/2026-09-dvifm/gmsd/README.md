# gmsd lane — preserved research code (2026-09-22)

Record: [`benchmarks/gmsd_2026-09-22.md`](../../../benchmarks/gmsd_2026-09-22.md)
(+ `.json`, `.pointer.md`). Verdict: GMSD adopts no feature into zensim; it is a
cost/baseline bar (KADID SELECT SROCC 0.848, KonFiG val 0.759; 0.83 ms at 1024²
single-thread).

- `*.py`, `*.sh` — the lane's extraction, fit, scoring, peer-eval and speed
  scripts, moved here from `tools/gmsd/` unchanged (docstrings still say
  `tools/gmsd/`).
- `zgeom_gms_dev_map_dev.patch` — lane commit `6584c03e`'s source change
  (`ZgeomSpec{gms_dev, map_dev}`, `--zgeom-gmsdev/--zgeom-mapdev`). It applies on
  top of the unmerged transplant/block5 zgeom stack (`59cea473`, `cf517c03`), not
  on `main`; the screen found no adoption case, so it is kept as a patch, not
  source.

The speed arm itself is live in `zensim-bench/benches/ssim2_speed_bar.rs`
behind `--features gmsd-arm`.
