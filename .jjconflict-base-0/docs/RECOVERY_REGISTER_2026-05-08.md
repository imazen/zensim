# zensim recovery register — 2026-05-08

Compiled from a read-only sweep of the v06-rebalance / v07-e1-ablation / v06-moe / v06-film / v04-mlp branches and worktrees, plus on-disk bake artifacts under `/mnt/v/output/zensim/`. See `~/work/zen/RECOVERY_PLAN_2026-05-08.md` for the cross-repo plan.

## Verdict table

| branch | commit (short) | date | item | numbers (dataset:metric) | verdict | files |
|---|---|---|---|---|---|---|
| v04-mlp | (see `4metric_overnight_FINAL_2026-05-01.md`) | 2026-05-01 | **V0_5 SSIM2-proxy MLP** (218k synth, source-disjoint 80/20) | **CID22:0.8934 KADID:0.8505 TID:0.8492** | **kept — current CID22 leader** | `/mnt/v/output/zensim/synthetic-v2/runs/v04_mlp_ssim2_holdout_20260501T045510.bin` (60932 B) |
| v04-mlp | same | 2026-05-01 | V0_4-smooth (RankNet + magnitude-matching) | CID22:0.8910 KADID:0.8400 TID:0.8336 | superseded by V0_5 | `runs/v04_smooth_konjnd_train_20260501T164012.bin` |
| v04-mlp | same | 2026-05-01 | V0_6 dct_hf (zenanalyze tail features) | CID22:0.8935 KADID:0.8496 TID:0.8416 | kept (alternative CID22 leader, +0.0001 over V0_5; equal-class with V0_5) | `runs/v06_dct_hf_20260501T164958.bin` |
| (main, fmt-stripped) | `0218b0ed...` | 2026-04-30 | V0_4 mixed-supervision (synth+KADID_train+TID_train) — currently shipped under `__experimental_versions` | CID22:0.8893 KADID:0.8432 TID:0.8401 | superseded — V0_5 beats on all three | `zensim/weights/v0_4_2026-04-30.bin` (== `runs/v04_mlp_v5znpr2_20260430T044620.bin`) |
| v06-rebalanced-corpus | `95a201e` | 2026-05-06 | V0_6 + FiLM rebalanced (5 per-class bakes + manifest) | val_mean=**0.8457** but on KADID/TID/KonJND only — **CID22 missing from val set** | unverified for CID22; needs re-bench | `/mnt/v/output/zensim/v06-rebalance/runs/v06_film_rebal_20260506T081152.{bin,c[0-4]_*.bin,film_manifest.tsv}` |
| v06-rebalanced-corpus | same | 2026-05-06 | V0_6 baseline rebalanced (no FiLM) | KADID:0.8424 TID:0.8258 KonJND:0.9535 (no CID22 val) | superseded by FiLM | `runs/v06_baseline_rebal_20260506T064045.bin` |
| v06-rebalanced-corpus | same | 2026-05-06 | V0_6 + cclass rebalanced (one-hot content class) | KADID:0.8488 TID:0.8386 KonJND:0.9501 | superseded by FiLM (+0.0071) | `runs/v06_cclass_rebal_20260506T064045.bin` |
| v06-content-class | (parent of rebalance) | 2026-05-05 | precursor to v06-rebalance — procedural content generators (gen-screen, gen-doc, gen-line, gen-chart, gen-mixed) | n/a — synthesis tooling | kept (corpus tooling) | `zensim--v06-rebalance/benchmarks/v06_cclass/{synth_nonphoto.py,encode_synth_via_zenjpeg.sh,build_rebalanced_csv.sh,build_cclass_tsv.py,expand_size_variants.py}` |
| v06-film | `560540e` | 2026-05-05 | V0_6 + FiLM **on photo-only corpus** (pre-rebalance) | KADID +0.0183, TID +0.0199, **CID22 −0.0040** | superseded by v06-rebalanced-corpus | `runs/v06_film_*.bin` (older variants) |
| v06-moe | `cf48ba2` | 2026-05-05 | MoE trainer + bake format + inference (behind cargo feature) | architecture only — no training run | unverified — code intact | `docs/moe_architecture.md`; `runs/v06_moe_*.bin` if any (none on disk per agent) |
| v07-e1-ablation | `ad2e82e` | 2026-05-05 | zenjpeg-420-e1 fill ablation (0/5/10/20/50/100 %) | every fraction regresses on KADID+TID+CID22 vs V0_6 baseline | **abandoned** — JPEG bias 56% → 63% with 100% fill | `benchmarks/v07_e1_subsample_ablation_2026-05-05.md` |
| v07-e1-ablation | (same) | 2026-05-05 | V0_7 dct_hf + low-band oversample 0.5 | CID22 60-75:0.65 KADID ≤0:0.66 — fails to improve | superseded by V0_6 dct_hf | `runs/v07_dct_hf_lowband050_20260505T112826.bin`, `v07_dct_hf_nobias_*.bin` |

## Top cherry-picks for main (anti-bloat: each has documented improvement)

1. **Replace shipped V0_4 with V0_5** (`runs/v04_mlp_ssim2_holdout_20260501T045510.bin` → `zensim/weights/v0_4_2026-04-30.bin` slot under `__experimental_versions`). Improvement: CID22 +0.004, KADID +0.007, TID +0.009. Same byte format. Just swap the file + change profile docstring.
2. **v06-rebalance corpus tooling** (`benchmarks/v06_cclass/synth_nonphoto.py` etc) — recover into a `tools/corpus/` directory in zensim. Documents how to procedurally generate non-photo training content. Not a code-path change; a recipe to keep.
3. **moe_architecture.md** + **scale-invariance.md** + **zenjpeg_e1_fill_plan_2026-05-01.md** + the e1-ablation negative result → preserve in `docs/recovered/`. Zero ship cost; documents tried-and-rejected design space.

## Drop / archive (no measured improvement)

- All sibling worktrees with last commit < 2026-04-25 OR no benchmarks past that date: `diffmap-public-ctors`, `phase-4-zenblend`, `zengrid-analysis`, `zero-weight-elide`. Move to `archive/<branchname>` namespace per user direction.
- The `v_next/` Python trainer in `scripts/v_next/` (this session's work) is to be **removed** once the zentrain port lands. It's a strictly-less-capable version of what already existed in the v06-* branches' Rust trainer.
- `unified_v15r/v15rc/v14/v12.parquet` corpus as a TRAINING input — keep as eval/cross-codec coverage (in zenmetrics-managed sweep), but the canonical SSIM2-target training base remains `training_safe_synthetic_extended.csv` (CID22 source tiles included).

## Validation gap to close before declaring a champion

v06-rebalance FiLM was selected on **KADID + TID + KonJND val_mean**, NOT CID22. The user's gold standard is CID22. Before claiming v06-rebalance FiLM beats V0_5, run `dataset_metric_baseline --cid22 ... --v04-bake .../v06_film_rebal_20260506T081152.bin` (and optionally per-class with the c0..c4 bins).

Note: FiLM produces 5 per-class bins. Plain `dataset_metric_baseline` will load the master `.bin` only; for true per-class evaluation we need the FiLM dispatch (zenanalyze cclass classifier + class-routed predict). That dispatch lives in the v06-rebalance Rust trainer's bake — needs porting to zentrain or Python.

## Notable design docs to preserve

- `zensim--v06-moe/docs/moe_architecture.md` — MoE math + bake format design
- `zensim--v04-mlp/benchmarks/4metric_overnight_FINAL_2026-05-01.md` — 230-line comprehensive 4-metric eval; cross-validation anchors
- `zensim/docs/NEXT_TIER_DATA_PLAN.md` (already on main) — Squintly + condition-aware tier roadmap
- `zensim/docs/v_next_status_2026-05-07.md` (already on main) — V0_4 runtime + v15r vast.ai sweep failure analysis
- `zensim--v06-rebalance/benchmarks/v06_cclass/V0_6_REBALANCED_RESULTS_2026-05-06.md` — V0_6 rebalanced headline numbers
- `zensim--v07-e1-ablation/benchmarks/v07_e1_subsample_ablation_2026-05-05.md` — the explicit "skip e1 fill" verdict

## Artifact inventory (post-recovery)

- Baked models in `/mnt/v/output/zensim/synthetic-v2/runs/`: ~30 bins, 2 MB total
- Eval CSVs: ~15 MB in `synthetic-v2/runs/` per-pair scores
- Training CSVs: 340k row `training_safe_synthetic_extended.csv` + features.bin (260 + 419 MB)
- Metric ledger: 38 MB `metric-ledger.jsonl` append-only
- v06-rebalance run dir + 5 per-class FiLM bakes: ~520 KB

All on /mnt/v (durable). R2 mirror status: synthetic-v2 base mirrored; v06-rebalance run dir not yet mirrored.
