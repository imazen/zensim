# rev4 featbank — worklog (2026-09-23)

Design lane, no fitting, no extraction, **no human label read**. Every measurement below reads feature
columns or schemas only. The `human_score` column of the CID22 TRAIN cache holds SSIMULACRA2 peer scores
(per `baseline_recovery_2026-09-14.md`), and it was not read either.

## Step 1 — workspace

`cd /home/lilith/work/zen/zensim && jj git fetch && jj workspace add ../zensim--rev4-featbank -r main@origin`
→ parent `a14e563e` ("docs: Rev4 program records E1b").

## Step 2 — reading (records cited in the plan)

Read in full or by section: `docs/REV4_EXPERIMENTS_2026-09-23.md`, `FEATURE_SYSTEM_DESIGN_2026-09-05.md`,
`FEATURE_SET_IDS.md` §1-2, `FEATURE_DEFECTS_AUDIT_2026-09-05.md` §1-2 and §5, `PLAN_FEATURE_REV3_2026-09-09.md`,
`REV3_OPTIMIZATION_AND_COLOR_PLAN.md`, `DATA_SPLITS.md` (header, §3, §8, the exposure ledgers),
`WAVE_PLAYBOOK.md` step 4.

Benchmarks:
- `feature_ceiling_2026-09-13.md`, `complete_feature_audit_2026-09-14.md`, `scale_selective_944_2026-09-13.md`,
  `baseline_recovery_2026-09-14.md`, `dvifm_faithful_2026-09-22.md`, `dvifm_verdict_2026-09-20.md`;
- `feature_cost_frontier_2026-08-31.md` lines 150-180 and 286-296;
- `sota944_campaign_2026-08-03.md` Appendix H and H.R;
- sibling workspaces: `zensim--transplant/benchmarks/block5_2026-09-22.md`,
  `zensim--gmsd/benchmarks/gmsd_2026-09-22.md`;
- `~/tmp/devin/LANE_ZGEOM_DONE.md` and `LANE_GEOMETRY_DONE.md`;
- `../zenpapers/docs/zensim-720-feature-gaps-2026-07-26.md` lines 24-28 and 248-272.

Three read-only sonnet sub-agents summarised, with file:line citations:
- (a) `feature_cost_frontier`, `free_features*`, `linear_projections_2026-07-03.md` and
  `e2_optimal_model_720_vs_372_2026-07-23.md`;
- (b) the csf/csfw/bandvis records and the `feature_defs.rs`/`feature_v2.rs` inventory;
- (c) harvest L07/L09 claims, checked against records.

Numbers the plan uses from (a) and (b) were spot-checked by direct `sed -n`: FC lines 286-296 and 170-180;
the e2 LOO table lines 156-168; `feature_v2.rs` lines 742, 765, 4142-4213 and 6618-6622, plus
`C_BLOCK`/`BLOCK_LATTICE` at 215/580.

Harvest checks from (c): L07-05, L07-11 and L09-28 VERIFIED; L07-27 NOT FOUND as a cited path. No zenfleet
per-node extraction throughput exists in any record.

## Step 3 — existing cache schema (no labels)

```
cd /var/tmp/zensim-validation-2026-09-14/baseline-recovery && python3 (pyarrow schema read)
```

Output:
- `cid22-train944.parquet`: 17611 rows, 951 cols {string 4, double 906, int64 40, bool 1}, 1 row group, no
  kv meta.
- `safesyn-train944.parquet`: 196086 rows, 951 cols {string 3, double 907, int64 40, bool 1}.
- The 39 int64 feature columns (the 40th int64 is `row_id`) are f720 721 754-772 805 806 822 823 856 857 873
  874 907 908 927 928 932 933 937 938 942 943.
- Non-feature columns: `ref_basename`, `human_score`, `row_id`, `reference_pixels_sha256`,
  `pixels_identical`, plus `codec`/`knob` (CID22) or `original_oracle`/`source_path` (SafeSyn).
- The manifest keys are `era, feature_set_id, formula_revision, populated_feature_ids, producer_surface,
  sampling`, with no `build_commit`.
- `sha256sum cid22-train944.parquet` = `fb666c4255e9b6df6f4d6778829708f3bcbcd3bcc2127c6fb90d6e9b38f33a7a`,
  matching `CID22_VERIFIED.json`. The SafeSyn sha `6044fdc8…` is taken from `SAFESYN_VERIFIED.json` and was
  not re-hashed.
- The audit JSONL line 1 carries `distorted_pixels_sha256` and `reference_pixels_sha256` (schema
  `canonical-feature-audit-v1`).

## Step 4 — f32 cast and storage probe

Command:
`nice -n19 python3 benchmarks/rev4_featbank_2026-09-23/f32_cast_probe.py /var/tmp/zensim-validation-2026-09-14/baseline-recovery/cid22-train944.parquet`

- script sha256 `08ef4bf165e9e37677344a2e9299a3f461745ab6fba1bf79ca3be332dc5bf227`;
- log `/var/tmp/rev4-featbank/f32_cast_probe_cid22.log`, sha256
  `0bd8fd87e6edfd8e6ee4c58032ff32442d463f482e1e2a92fc8f719d6fa188cd`.

Output lines:
```
basic f0-f155: f32-exact cells 0.1159 max rel 5.960e-08
masked f228-f299: f32-exact cells 0.0045 max rel 5.946e-08
iw f300-f371: f32-exact cells 0.0044 max rel 5.955e-08
v2 f372-f719: f32-exact cells 0.0106 max rel 5.960e-08
overall max rel 5.960017224449779e-08 outside f32 range False f32-subnormal 0
rows 17611 live cols 905 structural-zero ids [720, 721, 754, ... 942, 943]   (39 ids)
zstd3_dict 81622870 B 4635 B/row exact_roundtrip True
zstd3_bss 48164863 B 2735 B/row exact_roundtrip True
source parquet bytes 131368060
```

An earlier ad-hoc run of the same probe with a `row_id` column added (not committed) gave
zstd-3+BSS 48,194,078 B and zstd-9+BSS 47,706,024 B. That is the plan's "zstd-9 only 1% smaller"
statement.

## Step 5 — sources of the decisions in the plan

| plan claim | source |
|---|---|
| BLOCKINESS form contrast-invariant | `zensim/src/feature_v2.rs:742-744` `bounded_excess`, `:215` `C_BLOCK = 1e-3`, `:4194-4209` lattice loop, `:580` `BLOCK_LATTICE = 8` |
| slot 25 inert on glob | block5 record, "Slot-25 interaction" |
| JPEG sign flip | `rev4_e1b_crosscodec_2026-09-23.md` via `REV4_EXPERIMENTS` step 2 text (484/510, 85/94, 162/162) |
| next free ID 986 | sub-agent (b): `feature_defs.rs:2172`, `:2612` |
| zenfleet Feature job kind | `zenmetrics/crates/zenfleet-core/src/job.rs:174-190` (read directly) |
| sample-seed caveat | `docs/WAVE_PLAYBOOK.md` step 4 (Sep 14 correction); memory note seed-subsets-not-equal |
| full944 1024² cost 58.726 / 19.160 ms | `scale_selective_944_2026-09-13.md` runtime table |

Nothing here is a recomputation of a human-label statistic.
