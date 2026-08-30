# R1b — THE KEYED REBUILD (all-944-live regime `folded720append2pools`)

**Lane owner:** R1b (registered in `benchmarks/balance_campaign_2026-08-28.md`,
`## R-LANE EXECUTION + THE GATES CANON` ~L1870). This is a NEW document; the
campaign ledger folds these rows in — nothing here is appended there.

**Status (2026-08-30, end of lane):** the INSTRUMENT is DELIVERED — the D1
validate slices are keyed with a row-identity proof, every leg and slice is
re-extracted at ONE width in the all-live regime, the gates pass (one caught a
live repro hazard), and the five round-6 bars are readable for a 944-class
candidate and for shipped B **on identical pairs** for the first time. The
pre-registered ARM measurement ran to its registered end and is reported with
its caveat (§8.3): a faithful implementation of the ledger's recipe shows the
pool block to be worth ~nothing, but it does not reproduce the ledger's
baseline, so it does not falsify the carrier finding. §9 lists what is open and
why, priced. Nothing here is claimed before it is measured.

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

## 7. LINEAGE RESULT — the keyed pair tables (LANDED 2026-08-30)

### 7.1 The eleven TSV legs — keyed by construction

Every canonical local leg's pairs TSV is on disk with a row count EQUAL to its
stored parquet, so the `(ref_path, dist_path)` pair *is* the key and the
rebuild is a re-run of the same driver at a different regime. Verified by
`wc -l − 1` vs `ParquetFile.metadata.num_rows` (§3 table) and by a full
existence scan of both sides of all 149,195 pairs.

**One defect found by that scan, recorded not papered over:** `ext_tid`'s
canonical TSV names `/mnt/v/dataset/tid2013/reference_images_png/I25.png`; the
corpus holds **`i25.png`** (lowercase — TID2013's own inconsistency; the source
BMP is `i25.bmp` too). The 2026-08-01 canonical run produced 3,000 rows, so the
uppercase name resolved then and does not now. The extractor SKIPped those 120
rows and `extract_944_canonical.sh`'s row-count guard correctly ABORTED the leg
— the guard did its job. Repaired with a NEW dated pairs TSV
(`r1b-pools944-2026-08-30/pairs/tid_pairs_ab_r1b_i25case.tsv`, the one path
substituted, 3,000 rows) fed through the new `ZM944_PAIRS_<LEG>` override; the
substitution is recorded in the run manifest. No corpus file was renamed or
created. Every other leg scanned clean: **0 missing refs, 0 missing distorted
files across the other 146,195 pairs.**

### 7.2 The three D1 validate slices — keyed, with a row-identity PROOF

`build_eval_slices_944.py` reads only `ref_filename` + `score_ssim2` + features
from the bigcodec views and drops the identity columns; that drop is the whole
"keyless" defect. `--emit-keys` / `--keys-only` now writes the identity sidecar,
and `validate_slice_family_filter.py` applies ONE keep-index to the feature
table and the sidecar together (it refuses if they disagree), so the two can
never drift.

| slice | pre-filter rows | after validate-family filter | G-KEY row identity vs the stored canonical slice |
|---|---|---|---|
| `ext_imazen26` | 10,037 | **6,953** (87 origins) | **OK** — `ref_basename` equal row for row |
| `ext_nonphoto` | 10,042 | **6,142** (61 origins) | **OK** |
| `ext_hfnlproxy` | 10,179 | **7,717** (87 origins) | **OK** |

Each keyed row carries `row_index, ref_basename, human_score, view_row,
origin_id, ref_filename, encoded_filename, codec, q, knob_tuple_json,
score_ssim2, score_zensim, split`. The registered claim "the D1 cuts carry no
key" is now **false by construction** — and the gate is what makes it a proof
rather than an assumption.

Interesting negative: `encoded_filename` is distinct on all 20,812 slice rows,
so for THESE slices the sha-sharing hazard (different sources encoding to
identical bytes — the 2026-08-30 write-back defect) does not bite. The key is
still stated as `(ref_basename, encoded_filename)`, never `encoded_filename`
alone, because the hazard is a property of the corpus, not of this cut.

### 7.3 Where the bigcodec bytes actually are (MEASURED, not assumed)

`resolve_bigcodec_pair_uris.py` joins the key tables to the canonical-picker
`pairs.validate.parquet` on `encoded_filename` through
`join_safety.safe_key_join_arrow` (a pyarrow-native sibling of
`safe_metric_join` added in the same owner — this box has pyarrow and no
pandas; identical refusal semantics: ref-only key, missing key, non-unique
side). **100 % of all 20,812 rows resolve**; the run aborts otherwise.

| dataset | imazen26 | nonphoto | hfnlproxy | how the bytes are reachable |
|---|---|---|---|---|
| `zenjpeg_lossy` | 1,930 | 1,698 | 1,379 | per-file object GET (`…/encodes/<member>`) |
| `zenwebp_lossy` | 1,223 | 1,091 | 774 | per-file object GET |
| `zenavif_lossy` | 1,968 | 1,734 | 4,710 | byte-range into `variants/box-N.tar` |
| `zenjxl_lossy` | 1,832 | 1,619 | 854 | byte-range into `variants/box-N.tar` |

- The reference PNGs are **local** (`/mnt/v/output/clean-picker-corpus-2026-06-26`,
  4,497 files) — no fetch needed on that side.
- The `encodes/` prefix **exists for zenjpeg / zenwebp / zenpng / zenjxl-lossless
  and is EMPTY for zenavif / zenjxl-lossy**. The `_regroup` `.FAILED` marker is
  a red herring (that run died in 3 s on an unset endpoint variable); the real
  run and its recovery pass covered originals→png→webp→jpeg→jxl-lossless and
  **never attempted avif or jxl-lossy** — unexecuted, not failed.
- For those two, member-level `variant_index.tsv` files
  (`member \t offset \t size \t name`, built by zenmetrics
  `scripts/jobsys/index_tar_byterange.py`) already exist under
  `s3://zentrain/jobs/bf-zavif-t{0..7}/` and `bf-zjxlm-t*`, so the bytes come
  out by byte range. **No whole-tar download is needed** — the four lossy runs
  total 151.90 GiB of tar, against ~1 GiB of member bytes for these slices.

`fetch_bigcodec_bytes.py` materialises both modes (s5cmd batch `cp` for
objects, indexed range GETs for tar members, indexed size asserted per member)
and emits the local `(ref_path, dist_path, human_score)` TSV the zensim
extractors consume. Every requested member must land non-empty or the run
aborts.

### 7.4 Decode capability — the one real gap, and how it was closed

`v2_ab_extract` reads its pairs through `zen_io::decode_rgb8`, which handles
**png / jpg / bmp only**. The bigcodec distorted sides are `.avif` / `.jxl` /
`.webp` / `.jpg`, so the slices could not go through the same extractor as the
eleven local legs — and using a *different* extractor for them would reintroduce
exactly the cross-regime hazard R1b exists to remove.

Closed by adding a `--decode-list <tsv> --out-dir <dir>` mode to
`zensim-bench/examples/verify_bitstream_decode` — the file that already owns
the four zencodec decode paths (jpeg / avif / jxl / webp). It reuses those
functions verbatim (no second decode path), writes RGB8 back out as PNG through
`zenpng::encode_rgb8`, and skips anything already written. The decoders are the
same `zencodec` implementations the fleet uses. Known and recorded: the JXL
decode path is `zenjxl-decoder`, which is NOT the `jxl-oxide` the 2026-06
generator used — so R1b's JXL rows are re-decoded through today's decoder, and
that is a property of the rebuild, not a defect to hide.

## 2b. PRE-REGISTERED FIT (frozen 2026-08-30, before any R1b fit exists)

The measurement R1b exists for is the **matched-mix TRUE-linear arm** — the one
the campaign's "LINEAR-QUESTION ANSWERED" section measured at kon 0.1644 →
0.4887 on 720-width-FUSED tables that could only be read on two axes. R1b
re-runs it at ONE width, on KEYED rows, with all five bars readable.

**Recipe (IDENTICAL across arms; nothing but the feature block changes):**
legs `safesyn 1.0 + cid22t 1.5 + kadid 0.5 + tid 0.5`, target `human_score`,
per-corpus min-max frames, shaped space (`scripts/sota944/screen944_monotone.tsv`),
solver **BVLS + sign-mask** (`benchmarks/feature_sign_mask_2026-05-26.tsv`;
f372+ free), owners `bake_dial_refit gram` → `bake_dial_refit fit-lasso
--solver bvls --emit-fit-npz`. No new fit code.

**Three arms, one variable:**

| arm | f156..f371 | what it isolates |
|---|---|---|
| `A0-zero` | structural zeros (stored `ext944-canonical-2026-08-01`) | the no-carrier baseline |
| `A1-carr` | the TEN carrier slots live, other 206 zeroed | the `fused944native` carriers regime, at native slots, on keyed rows and ONE width |
| `A2-pools` | ALL 216 pool slots live (`folded720append2pools`) | the user's amendment — the whole block, live |

`A1-carr` is produced from the `A2-pools` table by zeroing the 206 non-carrier
columns, so A1 and A2 are the SAME pixels, the SAME extraction, the SAME
binary — the only difference is which slots the fit may see. A0 is the stored
zero-block root at the same rows.

**Bars read:** the five of §2, on the KEYED slices, plus **B under the same
ruler**: B is a 372-input bake, so the same 20,812 slice pairs are ALSO
extracted at `ZENSIM_AB_MODE=v1` (372) and B is scored on those — the first
time a 372-class model and a 944-class model are read on literally the same
pairs for the family axes. cid22 and kon-504 already share pairs across roots
(same pairs TSVs), so those two are same-ruler by construction.

**SCOPE, stated before the numbers (not after):** the full-mix `cid` head and
therefore the `wlin954b` BLEND need the tbig / tbig_hf / teacher (`tsafesyn`,
`ttbig`) / kadis legs at this regime. Those are 200k–5.7M-row bigcodec legs
whose rebuild is a fleet job, not a local one. R1b delivers the HEAD arm — the
arm the linear-question result rests on — and prices the blend's remaining
cost. Any blend number would require column-mixing regimes, which is refused.

**What a result means (decision rule, frozen):**
- If `A2-pools`/`A1-carr` clear `kon ≥ 0.40` at `cid22 ≥ 0.845` on the keyed
  slices, the carrier finding survives the move to one width + keyed rows.
- If they do not, the 720-width fusion was carrying the effect and the
  registered R1b outcome is that the 954 linear lane closes.
- The hfnl / nonphoto / imazen26 readings are NEW information either way —
  they have never been measurable for this arm.

### 2c. ROBUSTNESS VARIANT R-1 (declared 2026-08-30 BEFORE running it)

The §2b head, run on both arms, reads **KonJND NEGATIVE on BOTH** (A0 −0.2062,
A2 −0.1911). A head that is anti-correlated on an axis is degenerate on that
axis, so the arm delta there is a difference between two broken reads, not
evidence about carriers. That is a defect in my instantiation of the recipe's
one free parameter, not a result.

**Declared before running:** variant **R-1** = the identical recipe with the
per-corpus min-max target framing REMOVED (`--target human_score`,
`--target-scale 100`, no `--target-minmax01`); everything else — legs,
weights, space, screen, solver, sign mask, anchor — unchanged, and run on BOTH
arms. It is reported whatever it says, alongside the registered §2b numbers,
which are NOT withdrawn. No further variants are declared; if R-1 is also
kon-degenerate, the honest conclusion is that this head recipe does not
reproduce the campaign's kon head and the arm comparison stands only on the
axes where the head is healthy.

### 2d. DISCRIMINATING ARM A3 (declared 2026-08-30 BEFORE running it)

§8.3 leaves two explanations open for the near-null: (i) my head differs from
the ledger's in some unrecorded way, or (ii) the 720-width FUSION carried the
effect. They are separable with one run, so it is declared and run.

**A3 = the IDENTICAL §2b recipe and driver, on the ledger's own
`fused944native-2026-08-30` tables** (the carriers written into their native
slots by 720-width fusion — the exact tables the "+0.3243 carrier effect" and
the `kon 0.4570 / cid22 0.8726` pinned-form reading were produced on). Row
counts match my legs exactly (safesyn 111,068 / cid22t 17,611 / kadid 10,125 /
tid 3,000 / cid22val 4,292 / kon504 504).

**Decision rule, frozen before the run:**
- If A3 reproduces a LARGE KonJND lift over A0 (order +0.2 or more), the effect
  lives in the FUSED tables and not in a single-width all-live extraction ⇒
  explanation (ii), and the user's amendment was the right call.
- If A3 is also near-null, my head is not their head ⇒ explanation (i), and
  R1b says nothing about the carrier finding beyond "not reproduced by this
  recipe".
- Any other outcome is reported as-is. This is ONE run; no variants follow.

**KADID orientation is checked on the fused root before the fit** — that root
was built on 2026-08-29 from the ext lineage, so it may carry either
orientation; the gate decides and the result is recorded.

## 8. RESULTS

### 8.1 The rebuilt roots (what exists now)

| root | regime | contents |
|---|---|---|
| `/mnt/v/zen/zensim-training/r1b-pools944-2026-08-30/` | `folded720append2pools` | the 11 canonical local legs (149,195 rows) + the 3 keyed D1 validate slices (20,812 rows) = **14 corpora / 170,007 rows**, all 216 pool slots live |
| `/mnt/v/zen/zensim-training/r1b-ctrl944-2026-08-30/` | `folded720append2` | 5-leg SAME-BINARY zero-block control (18,521 rows), built only to gate the pools tables |
| `/mnt/v/zen/zensim-training/r1b-samepair{372,944}-2026-08-30/` | v1-372 / pools-944 | the family-axis slices restricted to rows that HAVE a full v1-372 vector, so a 372-input bake and a 944-class candidate read identical pairs |

Every parquet carries a `regime` column AND `zensim_regime` parquet key-value
metadata; the root manifests carry `regime` + a `regime_purity` line.

### 8.2 GATES — all results

| gate | result |
|---|---|
| **G-KEY** (slice row identity) | **PASS 3/3** — the keyed sidecars name exactly the rows the stored slices hold, `ref_basename` row for row |
| **G-P1** (pools vs SAME-BINARY control, non-pool block) | **PASS 5/5 controlled legs** — **728 of 728** non-pool columns BIT-IDENTICAL. The regime flag changes the pool block and nothing else. |
| **G-P2** (pool block state) | control **216/216 zero**, pools **216/216 live**, on every controlled leg |
| **G-E** (row identity + target vs the stored 2026-08-01 root) | **PASS 11/11** — `ref_basename` and `human_score` EQUAL row for row |
| **G-B** (target orientation) | KADID **+0.582360** (after repair, below), TID **+1.000000**, CSIQ/LIVE **SKIPPED = not checked** |
| **G-D** (regime purity) | enforced in the promoter: a `*pools` mode with an all-zero f156-371 block ABORTS, and so does a folded mode with live slots |
| drift vs the 2026-08-01 root (REPORTED, not gated) | **22 of 728** non-pool columns differ, all in the append block, max abs ~1e-8, max rel ~5e-6 — extractor-version drift between `ec3bdd6a` (2026-08-01) and `ced6f52a` (today), identical on every leg |

**G-B caught a live repro hazard, which is the gate earning its keep.**
Re-extracting KADID from its canonical pairs TSV reproduces the
**pre-2026-08-05 INVERTED target** (gate reads **−0.582360**), because the
2026-08-05 orientation correction was applied to the ext *parquets* and not to
the pairs TSV that feeds a re-extraction. Repaired on both new roots with the
owner (`fix_ext_kadid_orientation.py`), cross-checked against the
independently-built canonical at **5.55e-17**. Anyone re-extracting any ext leg
from a pairs TSV inherits this and must re-run the gate — the DATA_SPLITS §1.4
hazard note describes the repro direction, not this one.

### 8.3 THE ARMS — the pre-registered measurement

Head recipe exactly as §2b (BVLS + sign-mask, shaped, 4 legs), run identically
on the two roots; §2c's declared R-1 variant repeats it without the per-corpus
min-max target framing. Read on the KEYED slices. Signed SROCC throughout.

| arm | cid22 | KonJND-504 | nonphoto | imazen26 | hfnl (pooled) | hfnl (per-ref) |
|---|---|---|---|---|---|---|
| **A0-zero** (f156-371 = 0) | +0.8311 | −0.2062 | +0.8207 | +0.8310 | +0.1544 | +0.6133 |
| **A2-pools** (216 slots live) | **+0.8332** | **−0.1911** | **+0.8219** | +0.8311 | **+0.1781** | **+0.6252** |
| Δ (pools − zero) | **+0.0021** | **+0.0151** | **+0.0012** | **+0.0001** | **+0.0237** | **+0.0119** |
| R-1 A0-zero (no min-max) | +0.8646 | −0.3994 | +0.8126 | +0.8304 | +0.1539 | +0.5760 |
| R-1 A2-pools | +0.8652 | −0.4105 | +0.8116 | +0.8298 | +0.1853 | +0.5789 |
| Δ (R-1) | +0.0006 | −0.0111 | −0.0010 | −0.0006 | +0.0314 | +0.0029 |

**Result, stated plainly: making the ENTIRE v1 pool block live moves this head
by ~0.002 on cid22 and by ~0.01–0.02 on every other axis — nothing like the
+0.3243 KonJND effect the campaign measured for the ten carriers on
720-width-FUSED tables.** The direction is consistent (pools ≥ zeros on 5 of 6
axes in the registered variant) but the magnitude is at noise scale, and it
does not depend on the target framing: R-1 reproduces the same near-null.

**The honest caveat, stated before anyone quotes this as a falsification.**
This head is **KonJND-degenerate in every arm and both framings** (signed
−0.19…−0.41; KonJND is conventionally read as |SROCC| because its target is a
PJND parameter, so as a magnitude these are 0.19–0.41 with the sign carrying no
extra information here). The campaign's no-carrier arm read KonJND **+0.1644**
at cid22 0.8249, mine reads −0.2062 at 0.8311 — **so this is not the same head,
and R1b has NOT reproduced their baseline.** Their exact argv is not
recoverable: the 954 heads were fit ad hoc and no driver was committed (the
`wlin-2026-08-29` bakes exist, the commands do not). What R1b measures is
therefore: *a faithful implementation of the recipe as it is described in the
ledger shows no carrier/pool effect at one width on keyed rows.* Two
explanations remain open and R1b does not choose between them — (i) an
unrecorded difference in their head, or (ii) the 720-width FUSION carrying the
effect, which is the possibility the user's amendment was designed to exclude.
The driver here IS committed (`scripts/r1b_linear_arms.sh`), so this arm is
reproducible in a way its predecessor is not.

### 8.3b ARM A3 — the discriminating run, and what it settles

A3 ran the §2b driver, unchanged, on the ledger's OWN
`fused944native-2026-08-30` tables (KADID orientation gate on that root first:
**+0.582360 OK**).

| arm | feature source | cid22 | KonJND-504 (signed) | kadid | tid |
|---|---|---|---|---|---|
| A0-zero | stored 944, f156-371 = 0 | +0.8311 | −0.2062 | +0.8688 | +0.8429 |
| A2-pools | **R1b, ONE width, 216 slots live** | +0.8332 | −0.1911 | +0.8699 | +0.8427 |
| A3-fused | **the ledger's 720-width-fused carriers** | +0.8341 | **−0.1914** | +0.8691 | +0.8427 |

**Two conclusions, both clean, per the rule frozen in §2d:**

1. **The fusion is NOT the cause.** The same recipe on the ledger's own fused
   tables gives the same near-null (kon −0.1914 vs A2's −0.1911, cid22 +0.8341
   vs +0.8332). So R1b lands on explanation **(i)**: my head is not their head,
   and **R1b does NOT falsify the carrier finding** — it reports that a
   faithful reading of the recipe *as written in the ledger* does not reproduce
   it, and that the missing information is the ledger's unrecorded argv.

2. **The rebuild validates itself.** A2 and A3 differ by **0.0003 on KonJND and
   0.0009 on cid22** — a single-width all-live extraction and a 720-width
   fusion are behaviourally equivalent for this head. That is the strongest
   available evidence that the R1b tables are a faithful substrate: they
   reproduce, from clean one-width extraction on keyed rows, what the fused
   tables do, while carrying the keys the fused tables never had.

The practical consequence for the lane: the 954/pools linear question is
*re-openable at any time* on a substrate that is keyed, one-width, gate-passed
and driver-reproducible — which is what R1b was for. What it needs from the
other lane is one thing only: **the argv of the head that read kon +0.1644 /
0.4887.**

### 8.4 B UNDER THE SAME RULER — the axes that were unmeasurable

Same pairs, each model on its native regime: B (372 inputs) on the v1-372 twin,
the arms (944) on the pools twin, over the row set where BOTH exist (rows too
small for v1's 4th scale carry only 279 features and are excluded — MEASURED
453/6,953 imazen26, 422/6,142 nonphoto, 493/7,717 hfnlproxy, ~6.5 % each,
counted and recorded, never silently dropped).

| model | cid22 | \|KonJND-504\| | nonphoto | imazen26 | hfnl | hfnl/ref |
|---|---|---|---|---|---|---|
| **B (shipped, 372)** | **0.8763** | **0.5183** | **0.9093** | **0.9142** | **0.3553** | **0.6279** |
| A0-zero (944) | 0.8311 | 0.2062 | 0.8773 | 0.8806 | 0.2398 | 0.6215 |
| A2-pools (944) | 0.8332 | 0.1911 | 0.8784 | 0.8815 | 0.2474 | 0.6233 |
| R-1 A0-zero | 0.8646 | 0.3994 | 0.8709 | 0.8834 | 0.2342 | 0.5760 |
| R-1 A2-pools | 0.8652 | 0.4105 | 0.8779 | 0.8891 | 0.2521 | 0.5789 |

**This table is the thing R1b was built to make possible** — before it, the
nonphoto / imazen26 / hfnl columns for a 944-class candidate and for B were
read on DIFFERENT pairs and were direction-only. B leads on every axis; no arm
comes close on the family axes.

**Bars (§2): FAIL on all four arms** — every arm misses `kon ≥ 0.40` (as a
magnitude R-1 reaches 0.40/0.41, but at cid22 0.865 it still misses
`nonphoto ≥ 0.865` and `imazen26 ≥ 0.875`), and all four miss `hfnl ≥ 0.40`.
The registered FALSIFIER language applies to this head only, not to the lane:
what is falsified is *this* reconstruction, and §8.3 says why that is weaker
than a falsification of the carrier finding.

### 8.5 Instrument caveats, measured rather than asserted

**(a) The two v1-372 extractors differ per-feature but NOT on the bar.**
`v2_ab_extract ZENSIM_AB_MODE=v1` and the canonical
`zensim-bench extract_features_372col` differ by up to **|d| 0.0927 (f129)** on
the same 504 KonJND pairs. Scoring B on each: KonJND **−0.518703** vs
**−0.518294** — **|ΔSROCC| 0.0004**. The per-feature difference does not move
the axis, so the 372 twin stays on `v2_ab_extract`, which is the only one of
the two that preserves pairs-TSV ROW ORDER (the canonical extractor reorders,
and its `ref_basename` is not row-unique, so its output cannot be aligned back
to a keyed table at all — a real limitation of that tool for keyed work).

**(b) A fresh canonical extraction of cid22 has drifted from the 2026-05-15
stored table by +0.0060 SROCC for B** (0.8822 fresh vs 0.87627 stored; the
stored value is the one B's published 0.8764 lives on). Three and a half months
of extractor evolution. Recorded because it bounds how precisely any
cross-era 372 number can be compared.

**(c) B's KonJND-504 reads 0.5183 here against the ledger's 0.5935.** Same
504 JPEG-half pairs, same PJND target range (22.46–69.98), and the two
extractors agree to 0.0004 — so this is an INSTRUMENT difference in which
504-row table the ledger's number was taken on, not extractor drift. R1b
quotes its own instrument and does not adjudicate; anyone comparing to the
ledger's 0.5935 must first establish which table produced it.

**(d) v1's feature vector length is size-dependent — in BOTH extractors.** A
rendition too small for the 4th scale emits 3 x 93 = 279 features. Measured on
the R1b slices: 453/6,953 imazen26, 422/6,142 nonphoto, 493/7,717 hfnlproxy
(~6.5 % each), identical counts from `v2_ab_extract` and the canonical
extractor. So this is a property of v1, not of a tool. Those rows have no 372
vector and are excluded from every same-ruler read, counted in the manifest.
The 944 side has no such problem (the folded path emits a fixed width), which
is a real robustness advantage of the 944 regime worth recording.


## 9. WHAT IS OPEN, WITH THE MEASURED REASON

| item | status | measured reason |
|---|---|---|
| full-mix `cid` head + the `wlin954b` BLEND at this regime | **NOT DONE** | needs `tbig_200k` (208,169 rows), `tbig_hf`, the teacher legs (`tsafesyn`/`ttbig`) and `kadis50k` at `folded720append2pools`. Their bytes are the bigcodec tarball corpus; R1b proved the fetch path (100 % resolution, byte-range indexes, decode) on 20,812 pairs at 910 MB — the 200k-class leg is the same machinery at ~10x, a fleet job with the `zensim-foldapp2pools` metric (zenmetrics `905ae73d`) once its executor image lands. Producing a blend from heads fit at DIFFERENT regimes is refused. |
| `hdrmix` at 954/pools | **NOT KEYABLE at this regime** | no 720-width or SDR-route extraction of `hdr_v3mix` exists; the leg is HDR-route 944-native. Unchanged from the campaign's own check. |
| `kadis` at pools | keyable, not built | KADIS-700k rows are keyed by `source_id` + a persisted `distorted_url`; the fetch is the same shape as the bigcodec one. Cost, not blocker. |
| arm **A1-carr** (ten carriers only) | **NOT RUN** | A2 (all 216 live) already shows a near-null effect, so the ten-slot subset cannot show more than the whole block; running it would only refine a null. Registered in §2b, honestly not executed. |
| B's ledger KonJND-504 0.5935 vs R1b's 0.5183 | **UNRESOLVED** | §8.5(c): an instrument difference in which 504-row table the ledger's number came from. Not adjudicated here. |
| CID22 holdout audit (G-C) on newly keyed TRAINING rows | **NOT APPLICABLE** | R1b introduced NO new training rows. Every leg is the same pair list as the stored canonical root (G-E: `ref_basename` equal row for row on 11/11), so the 2026-08-01 audit carries over unchanged. The three keyed slices are eval-only. |
| the 22-column append drift vs the 2026-08-01 root | reported, not chased | max abs ~1e-8 / max rel ~5e-6, identical on every leg; harmless within a root, and every R1b comparison is within-root. |

## 10. WHAT R1b CHANGED IN THE TOOLING (all committed, all through owners)

- `build_eval_slices_944.py` — `--emit-keys` / `--keys-only` / `--verify-against`.
- `validate_slice_family_filter.py` — filters the key sidecar under ONE keep-index with the feature table, refuses if they disagree, and gates row identity.
- `join_safety.py` — `safe_key_join_arrow`, a pyarrow-native sibling of `safe_metric_join` with identical refusal semantics (this box has pyarrow, no pandas; the alternative was a bespoke join in a builder, which is what that module exists to prevent).
- `resolve_bigcodec_pair_uris.py` / `fetch_bigcodec_bytes.py` — `encoded_filename` → bytes, object GET or indexed byte range, 100 %-resolution gate, per-member size gate.
- `extract_944_canonical.sh` — `ZM944_MODE`, `ZM944_LEGS`, `ZM944_PAIRS_<LEG>`.
- `promote_ext944_canonical.py` — `EXT944_MODE` (regime column + parquet metadata + manifest), `EXT944_LEGS` with manifest MERGE, `EXT944_EXTRA_LEGS`, `EXT944_N_FEAT`, `EXT944_VERIFY_ROOT` (bit-identity gate), `EXT944_DRIFT_ROOT` (reported), `EXT944_BASENAME_MAP`.
- `verify_bitstream_decode` — `--decode-list` mode reusing its four zencodec decoders, non-RGB8 layouts routed through the canonical `zenpixels_convert::RowConverter` (4,417 of 20,655 canonical-picker AVIF members are `bd10` cells decoding to Rgb16).
- `build_r1b_samepair_roots.py` — the row-restricted 372/944 pair, with the short-v1-row count recorded.
- `bake_verdict` — the wrong-regime guard now ASKS the root (`root_declared_regime`) instead of assuming every `--regime 944` root feeds zeros; verified both directions; regression test added.
- `scripts/r1b_linear_arms.sh` — the committed arm driver (the predecessor 954 heads had none).
