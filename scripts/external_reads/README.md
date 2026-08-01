# external_reads — the committed seven-domain external-validation runner

Closes decision-surface-audit gap 3 (`benchmarks/decision_surface_audit_2026-07-31.md`):
the 2026-07-28/29 external-validation wave left every study's analysis runner
only in its `/mnt/v/output/zensim/<study>/` artifact dir, so nobody but the
original sessions could execute the freeze plan's Phase-4 re-run
(`zenpapers:docs/zensim-final-metric-plan-2026-07-31.md` §7). This directory is
the named, committed owner.

## Layout

- **`run_external_reads.py`** — the canonical runner. `--from-stored` (default):
  rescore the STORED feature tables + labels — ~11 s for the full probe read
  set, no video decode. Scorers: `probe944` (the registered UPIQ-SDR ridge
  probe, gated vs the recorded head before any study look), `s228` (stored
  streamed-score column, zero-fit), **`bake:<path.bin>` — the Phase-4 mode:
  point it at the final V1 bakes** (944 feature contract, forwarded via
  `predict_features_with_bake`). LOO reads (`loo944`, `loo956`) recompute the
  delta tables from the stored per-drop `bake_verdict` JSONs and check them
  against the stored deltas files. All correlations via
  `scripts/lib/zen_stats.panel_batch` (canonical Rust panel; tie-correct
  midrank Spearman) — no scipy in the compute path.
- **`asrun/<study>/`** — provenance copies of every as-run analysis script +
  pre-registration PROTOCOL.md, frozen byte-identical below a provenance
  header (source path, sha256(source), build commit; `PROVENANCE.txt` per dir
  lists everything, including header-free JSON copies). They are archival
  records — some call scipy directly (they predate the stats-rule batch
  migration) and some read session-scratch `~/tmp` inputs that no longer
  exist (the pooled tables in the artifact dirs are the stored equivalents).
  Do not extend them; extend the runner.

Full re-extraction (hours of video decode) is DOCUMENTED, not automated:
`run_external_reads.py --list-extract` prints per-study pointers into each
artifact dir's `COMMANDS.md` (exact commands, input shas, build commits) and
the committed extractor examples.

## Read families

| read | stored data (sha256 first-16) | labels | registered protocol | recorded number (probe944 / s228) |
|---|---|---|---|---|
| `upiq` (hdr-dmean) | `hdr-dmean-2026-07-29/upiq_sdr_956.csv` (`c726c542d270bd87`, 3,779×956), `upiq_hdr_944.csv` (`d3958cada8d8bbf8`, 380×944) | `/mnt/v/datasets/upiq/upiq_subjective_scores.csv` (JOD, in-table) | `benchmarks/hdr_dmean_commensurability_2026-07-29.md` + artifact `PROTOCOL.md` | probe gate = q3_heads.944: **kor 0.9346, nar 0.7688**, pooled 0.7597, cv 0.9363 / s228 q1_readout: 0.7145 / 0.7145 / 0.9456 |
| `sihdr` | `sihdr-transfer-2026-07-29/sihdr_feats_944.csv` (`392684641a4f8a3c`, 2,172×944) | `/mnt/v/datasets/si-hdr/experiment_results/experiment_results.csv` (324 labeled) | `benchmarks/sihdr_transfer_2026-07-29.md` | l1 zero-shot pooled: 0.4208 / 0.3440 |
| `hdrvdc` | `hdrvdc-conditions-2026-07-29/hdrvdc_pooled_944.csv` (`567731161871559e`, 580×944 per (video,config)) | `/mnt/v/datasets/hdr-vdc/HDR_VDC_JOD_Scores.csv` (464) | `benchmarks/hdrvdc_conditions_2026-07-29.md` | q2 pooled legs i/ii/iii: 0.6695 / 0.6517 / 0.7462 (s228: 0.7141 / 0.7162 / 0.8114) |
| `avt` | `avthdr-validation-2026-07-29/avthdr_pooled_944.csv` (`81d392e72dd67fc0`, 195×944) | `/mnt/v/datasets/avt-vqdb-uhd-1-hdr/subjective_scores/mos_ci.csv` | `benchmarks/avthdr_validation_2026-07-29.md` | q1 **pooled 0.7742** (av1 0.7553, hevc 0.8410, vvc 0.7071) / s228 0.7245 |
| `chug` | `avthdr-validation-2026-07-29/chug_feats_frame_level/` (8 CSVs, aggregate `efae53826a250c41`, 2,400 rows) + `chug_sample.tsv` (`64f96bc4b3de324b`, 300 pairs) | `/mnt/v/datasets/chug/chug.csv` (`mos_j`) | `benchmarks/avthdr_validation_2026-07-29.md` (CHUG leg) | pooled: 0.7245 / 0.7525 (imperfect-reference caveat attaches) |
| `rousselot` | `rousselot-chroma-2026-07-29/{hddtb,k4dtb}_feats_944_k179.csv` (`697f9006207b9091` / `6ba268c100072706`, 96×944 each) + `pairs_manifest.json` (`47b750b10f510169`, MOS in-manifest) | in `pairs_manifest.json` (openpyxl-parsed MOS xlsx) | `benchmarks/rousselot_chroma_validation_2026-07-29.md` | s228 comparators: hddtb_all 0.8841, 4kdtb_all 0.8282, hddtb chroma-pure 0.8354 (probe/bake reads are new measurements — the study's L1/L2 are LOSO-CV instrument numbers, owned by the as-run copy) |
| `loo944` (BANDVIS) | `bandvis-loo-2026-07-28/{verdicts944/,loo944_deltas.json}` (`1682f815b7e49fcf`); instrument parquet `/mnt/v/zen/zensim-training/ext944-instrument-2026-07-28/` | — (LOO over canonical eval legs) | `benchmarks/bandvis_loo_944_2026-07-28.md` | **append2 Σ −0.0687 ≤ 0 PASS** (removal hurts — load-bearing) |
| `loo956` (CSFW G6) | `csfw-g6-loo-2026-07-29/{verdicts956/,loo956_deltas.json}` (`c441d55990580ad8`); instrument parquet `/mnt/v/zen/zensim-training/ext956-instrument-2026-07-29/` | — | `benchmarks/csfw_g6_loo_2026-07-29.md` | csfw Σ +0.0608 → G6 FAIL, lanes default-OFF (by design) |

Tower mirrors: `zensim-sihdr-2026-07-29/`, `zensim-hdrvdc-2026-07-29/`,
`zensim-avthdr-2026-07-29/`, `zensim-rousselot-2026-07-29/` under
`/mnt/tower/output/`; `zensim-extreads-2026-07-31/` mirrors the three dirs
that previously had none (hdr-dmean, bandvis-loo, csfw-g6-loo).

## Reproduction record (2026-07-31, from stored tables)

`--scorer probe944` (default; all reads + LOO verify, 10.7 s wall): probe gate
reproduced the recorded hdr-dmean head to ≤5e-7 — measured diffs ≤2.2e-16 —
including **Korshunov 0.9346082397263838** and **Narwaria 0.7688482648531633**;
**AVT pooled probe 0.7741882112891073** and every other recorded probe read
exact; LOO delta tables recomputed from stored verdicts at 0.0 diff (296 + 264
cells). `--scorer s228`: all 15 recorded zero-fit reads exact (≤2.2e-16).
`--scorer bake:` smoked with the lin944 twin
(`bandvis-loo-2026-07-28/bakes/lin944.bin`): upiq 0.7979 pooled / avt 0.5726 /
chug 0.7210 — new measurements, no recorded values, as expected for an
SDR-twin. Logs: `~/tmp/extreads-runner-{probe944,s228,bake}.log`.

## Phase-4 usage (freeze capstone)

```sh
# the external read set against a final bake (V1-SDR / V1-HDR candidates):
python3 scripts/external_reads/run_external_reads.py \
    --scorer bake:zensim/weights/<final_v1>.bin --json phase4_reads.json

# the registered probe + zero-fit baselines (recorded-number regression):
python3 scripts/external_reads/run_external_reads.py
python3 scripts/external_reads/run_external_reads.py --scorer s228
```

UPIQ freeze-bar note: §5's "UPIQ pooled > 0.7536" row is owned by
`scripts/hdr/upiq_panel.py` (372-feature dial-grid path, PU-linear features —
a different pipeline from these 944 stored-table reads; both are canonical,
see the audit's gate-owner table).
