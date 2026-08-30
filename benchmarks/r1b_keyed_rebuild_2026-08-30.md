# R1b — THE KEYED REBUILD (all-944-live regime `folded720append2pools`)

**Lane owner:** R1b (registered in `benchmarks/balance_campaign_2026-08-28.md`,
`## R-LANE EXECUTION + THE GATES CANON` ~L1870). This is a NEW document; the
campaign ledger folds these rows in — nothing here is appended there.

**Status:** OPEN — registration + lineage landed; extraction + gates + bars
follow in this same file as they land. Every table below states its own
measurement status; nothing is claimed before it is measured.

---

## 1. WHY R1b EXISTS (the defect it repairs)

The 954 linear candidates (`head954_kon`, `wlin954b`) can be scored today on
**kon-504 and cid22 only**. The other three round-6 bars — `hfnl`, `nonphoto`,
`imazen26` — live in the D1 validate slices
(`ext944-canonical-2026-08-01/ext_{nonphoto,imazen26,hfnlproxy}.parquet`), and
those slices carry **exactly two non-feature columns: `ref_basename` and
`human_score`**. There is no `encode_sha`, no `encoded_filename`, no `q`, no
`codec` — so a carrier block extracted at another root cannot be attached to
the same pairs. The `(ref, score)` fingerprint join was measured at **0–20 %**
match (score generations differ) and is the documented join-trap: **never
retried.**

The repair is not a better join. It is to **rebuild both sides from the
lineage that still carries keys**, and to re-extract every leg and slice in
**ONE pass at ONE width** so no cross-width fusion is ever needed.

**USER AMENDMENT (2026-08-30), load-bearing:** do **NOT** fuse carriers from
the 720-width root. v1 features diverge across extraction widths (measured:
f237 median rel 7.6e-2, f333 2.05e-1 — the padded-width divergence), so a
fused table mixes numerics. Instead every leg and slice is re-extracted at the
all-live regime **`folded720append2pools`** (`V1PoolsMode::Full`: f0..155
folded v1-basic, **f156..371 v1 pools LIVE**, f372..719 v2-348, f720..943
append+append2 — 944 slots, no structural zeros).

**Regime purity (absolute):** `folded720append2pools` rows are their own
regime. They are NEVER column-mixed with the zero-block 944 rows
(`folded720append2`), with `folded720append2carriers`, or with 720/372/v1
rows. Every output file carries a `regime` column and a manifest naming it.

---

## 2. PRE-REGISTERED BARS (frozen 2026-08-30, BEFORE any fit lands)

This section is written before a single R1b fit exists. The bars are the
**round-6 bars** exactly as registered in the campaign ledger
(`## REGISTRATION — B FULL REMEASURE + W-LIN ROUND 6`), read on the newly
keyed slices, with B under the same ruler.

**Axes read (five, no others promoted after the fact):**

| axis | instrument (rebuilt, keyed) | direction |
|---|---|---|
| `kon-504` | `ext_konjnd_jpeg_val` (504 pairs, JPEG half) | higher better |
| `cid22` | `ext_cid22val` (4,292 pairs, 49-ref GOLD holdout) | higher better |
| `hfnl` | `ext_hfnlproxy` (validate-family D1 slice) | higher better |
| `nonphoto` | `ext_nonphoto` (validate-family D1 slice) | higher better |
| `imazen26` | `ext_imazen26` (validate-family D1 slice) | higher better |

**PASS bar (round-6, unchanged):** `kon ≥ 0.40` AND `hfnl ≥ 0.40` AND
`cid22 ≥ 0.845` AND `nonphoto ≥ 0.865` AND `imazen26 ≥ 0.875`.
**STRETCH:** `kon ≥ 0.45` AND `hfnl ≥ 0.45`.

**Reference row (B / harbor-line), quoted from the ledger, to be RE-MEASURED
on the rebuilt slices under the same ruler:** kon 0.5935 · cid22 0.8764 ·
hfnl 0.503 · nonphoto 0.864 · imazen26 0.831.

**FALSIFIER (frozen):** if no rebuilt-root candidate clears `kon ≥ 0.40` at
`cid22 ≥ 0.845` on the KEYED slices, the "carriers enable the 944-class
linear" reading does not survive the move from the fused 720-width tables to a
single-width all-live extraction, and R1b's registered outcome is that the
954/pools linear lane closes with the pair-of-profiles shape standing.

**Stat ownership:** every number is produced by `bake_verdict` /
`zensim_validate::panel` (the `panel` binary). No statistic in this document is
computed by any other code. Signed SROCC (`srocc_signed`) is what is read and
quoted; `bands[].srocc` is never read (absolute-value defect, CLAUDE.md).

**What is NOT claimed by this registration:** the rebuilt slices are a
DIFFERENT population from the 944-era slices (different extraction regime,
and — for the D1 slices — the same rows only if the key rebuild reproduces
them exactly, which §5 gates). Cross-document comparisons against pre-R1b
published numbers are therefore direction-only unless a gate in §5 says
otherwise.

---

## 3. LINEAGE — where every leg's KEY actually lives

The central finding of the lineage pass, and the reason R1b is cheap for most
legs: **the canonical 944 extraction was driven from pairs TSVs that are all
still on disk**, and those TSVs carry `(ref_path, dist_path, human_score)` in
**row order identical to the stored parquets**. For those legs the pair
identity *is* the key — stronger than an `encode_sha`, because it names the
exact bytes on both sides.

Driver of record: `scripts/canonical_corpus/extract_944_canonical.sh`
(`ZENSIM_AB_MODE=foldapp2` → R1b re-runs it at `foldapp2pools`).

| leg | rows | key source | keyable |
|---|---|---|---|
| `ext_cid22val` | 4,292 | `/mnt/v/dataset/cid22/CID22_validation_set/cid22val_pairs_ab.tsv` | YES (ref,dist) |
| `ext_konjnd_jpeg_val` (kon504) | 504 | `/mnt/v/output/zensim/v2-backfill-2026-07-20/konjnd_jpeg_val_pairs.tsv` | YES |
| `ext_cid22_train201` (cid22t) | 17,611 | `/mnt/v/output/zensim/v2-backfill-2026-07-20/cid22_train201_pairs.tsv` | YES |
| `ext_safesyn_full` | 111,068 | `/mnt/v/output/zensim/v2-ab-2026-07-19/safesyn_jpeg_FULL_pairs_ab.tsv` | YES |
| `ext_kadid` | 10,125 | `/mnt/v/dataset/kadid10k/kadid_pairs_ab.tsv` | YES |
| `ext_tid` | 3,000 | `/mnt/v/dataset/tid2013/tid_pairs_ab.tsv` | YES |
| `ext_aic3` / `ext_aic4` | 600 / 300 | `v2-ab-2026-07-19/aic3_pairs_ab.tsv`, `v2-backfill-2026-07-20/aic4_pairs.tsv` | YES |
| `ext_csiq` / `ext_live` | 866 / 779 | `/mnt/v/dataset/csiq/csiq_pairs.tsv`, `/mnt/v/datasets/LIVE/live_r2_pairs.tsv` | YES |
| `ext_sdr25` | 50 | `v2-backfill-2026-07-20/sdr25_pairs.tsv` | YES |
| `konjnd_bpg` train/val | 8,060 / 2,020 | rebuilt by `scripts/canonical_corpus/build_konjnd_bpg_944.py` from `/mnt/v/datasets/KonJND-1k/konjnd_full_scored.csv` | YES (rebuildable) |

Row counts verified equal to the stored ext944 parquets for all eleven TSV
legs (`wc -l` − 1 vs `ParquetFile.metadata.num_rows`).

**The bigcodec family (D1 slices + tbig + teacher legs)** is the part that was
genuinely keyless, and its lineage is different — recorded in §4.

---

## 4. THE BIGCODEC LINEAGE (the D1 slices, tbig, tbig_hf, teacher legs)

**Owner of the slices:** `scripts/canonical_corpus/build_eval_slices_944.py`
— fully deterministic (global stride per slice over a concatenation of the 4
lossy views in a fixed order), so the exact row set is reproducible and can be
re-cut *with* keys. The current slices are the `--split validate` cut
(2026-08-28 D1 re-slice; the test-family cuts retired to touch-once).

**The keys exist one level up.** The views the slices are cut from —
`ext944-canonical-2026-08-01/bigcodec/<ds>/{train,validate,test}_944.parquet`
— carry `origin_id, ref_filename, encoded_filename, codec, q,
knob_tuple_json, score_ssim2, score_zensim`. `build_eval_slices_944.py` reads
only `ref_filename` + `score_ssim2` + features and **drops the rest**. That
drop is the whole defect. `tbig_944_200k.parquet` already keeps
`encoded_filename` — the tbig leg was keyed all along.

**`encoded_filename` resolves to bytes through the canonical-picker pairs
tables:** `/mnt/v/output/canonical-picker-2026-06-27/<ds>/pairs.<split>.parquet`
carries `ref_path` (an individually-addressable
`s3://codec-corpus/clean-picker-corpus-2026-06-26/*.png`), `dist_member`
(= `encoded_filename`), and `dist_tar`.

**MEASURED OBSTRUCTION (verified on R2, 2026-08-30):** the per-file
`dist_path` objects (`s3://zentrain/canonical/2026-06-27/<ds>/encodes/…`) **do
not exist** — that prefix is empty and its `_regroup/` job carries a `.FAILED`
marker. The distorted bytes live only inside the
`.../variants/box-{0..7}.tar` archives (3–9 GB each, 8 per codec). Byte
recovery for these legs is therefore a tar-member extraction, not an object
GET. Its cost and cheapest mechanism are recorded in §6.

---

## 5. GATES (pre-declared; every one must be reported, none may be relaxed)

| gate | statement |
|---|---|
| **G-A pools identity** | For every leg that also exists at the v1/372 regime, the rebuilt `f156..f371` must equal the v1 values pairwise. Report max abs and max rel. The zensim gate `folded720_v1_pools_match_v1_path` proves BIT-identity at exact widths; the padded-width class diverges by v1's own pad wart, so this gate reports the divergence rather than asserting zero, and states which fixtures are exact. |
| **G-B orientation** | `scripts/canonical_corpus/check_target_orientation.py` on every target column of every rebuilt table. KADID must read **+0.582360**. A corpus with no recoverable raw truth reports SKIPPED = "not checked", never "passed". |
| **G-C holdout** | `check_holdout_overlap --threshold 10` for any newly keyed TRAINING rows, against the CID22-49 references. Screening only at d≤16; d≤10 flags require montage + sign-off before any blocklist action. |
| **G-D regime purity** | Every rebuilt file carries `regime = folded720append2pools`; an assertion refuses any table whose f156..371 block is all-zero (that would be a `folded720append2` row that leaked in) and any row-count/schema mismatch vs its lineage. |
| **G-E row identity** | For each rebuilt leg, row order and `human_score` must match the stored ext944 leg exactly (the pairs TSV is the same input in the same order). Any mismatch aborts that leg. |

**Bans in force:** CID22 human MOS is never a training target. KADID
`human_score` on the ext tables is already correctly oriented (post
2026-08-05) — a post-rebuild value is **never** negated. No ZNPR v2. Bakes are
produced only by the trainer / `zenpredict` / `bake_dial_refit`.

---

## 6. EXTRACTION PLAN + COST

**Extractor (owner, unchanged):** `zensim/examples/v2_ab_extract` at
`ZENSIM_AB_MODE=foldapp2pools`. Verified live at HEAD before any run: a 2-pair
TID probe emits 946 columns with **216 of 216** `f156..f371` slots non-zero.
The mode has been in the extractor since `f19b8469`; the `ced6f52a` activity
lever is in the measured cost.

Local legs run under `run-heavy --mem 20G --jobs 8` (the box is shared by
three other agents; those caps are not exceeded). The bigcodec legs' cost is
§4's tar problem and is priced before anything is launched.

**Outputs:** `/mnt/v/zen/zensim-training/r1b-pools944-2026-08-30/<leg>_pools944.parquet`,
each with a `_MANIFEST.json` carrying `build_commit`, input shas, row counts,
and the regime tag. Mirrored per `~/work/zen/DATA_PROVENANCE.md` and
registered there.

---

## 7. RESULTS

*(empty by construction — filled only by measurements, in the order they land)*
