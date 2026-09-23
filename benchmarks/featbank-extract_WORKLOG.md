# featbank-extract WORKLOG (Devin lane, quarantined)

Brief: `/home/lilith/tmp/zensim-paper/rev4/FEATBANK_EXTRACT_brief.md` +
`DEVIN_COMMON.md`. Workspace `zensim--featbank-extract` on `main@origin`
(e6ce1565). Bookmark: `quarantine/devin/featbank-extract`. Never pushed.
Manifest: `/home/lilith/tmp/devin/rev4_featbank-extract_manifest.tsv`.
Bank root: `/var/tmp/rev4-featbank/bank/`. Scratch: `/var/tmp/rev4-featbank/`.
`CARGO_TARGET_DIR=/var/tmp/featbank-extract-target` (not yet used).

## 2026-09-23 — lane setup and input survey (all commands cwd as noted)

- 12:18–12:44Z — reconnaissance (read-only): confirmed inputs exist; cid22/safesyn
  audits carry `canonical-feature-audit-v1` with `reference_pixels_sha256` +
  `distorted_pixels_sha256`; ceiling `human_*.parquet` carry NO pixel-hash
  columns and the ceiling extraction ran without `--audit-jsonl`
  (`executed-feature_screen_ceiling.py` line ~222) — pixel hashes for the three
  ceiling sets must be re-derived. `/mnt/v` and `/home` report ~3 GB free —
  writes confined to `/var/tmp`. `/mnt/tower` not mounted.
- 12:45:29Z — claimed `/home/lilith/work/zen/zensim/.workongoing`
  (`devin-featbank-extract … Part A cache conversion`).
- 12:45–12:46Z — `jj workspace add ../zensim--featbank-extract -r main@origin`
  (parent `e6ce1565`), cwd `/home/lilith/work/zen/zensim`. Exit 0.
- 12:47Z — pinned input sha256s (`sha256sum`, cwd
  `/var/tmp/zensim-validation-2026-09-14/baseline-recovery` and
  `~/work/zensim-validation-2026-09-13/ceiling/final`); full table in
  `benchmarks/featbank-extract_prereg_2026-09-23.md`. Verified
  `cid22-train944.parquet` sha256 `fb666c42…` and `safesyn-train944.parquet`
  `6044fdc8…` match the design-plan values; `CID22_ADMISSION.json`'s
  `extractor_sha256` `7c7ffbbf…` equals the on-disk binary's sha256.

## Pixel-hash re-derivation probe (25 rows, preflight before any full run)

- ~12:55Z (approx; exact minute not captured) — built
  `/var/tmp/rev4-featbank/probe_pairs.tsv` (25 rows: ref/dist
  paths + row_id from `audits/human-full944-h128-full-s5101.jsonl`, human_score
  written as 0 placeholder) and `/var/tmp/rev4-featbank/probe_expected.json`.
- ~12:56Z (approx) — cwd `/var/tmp/rev4-featbank`:
  `ZENSIM_FORMULA_REV=3 ZENSIM_ROOT_FORM=sqrt RAYON_NUM_THREADS=4
   /home/lilith/work/zensim-validation-2026-09-14/native-integrity-admission/extract-native-admission
   --corpus pairs-tsv --path probe_pairs.tsv --out probe_feats.csv --full-944
   --audit-jsonl probe_audit.jsonl`
  exit 0; output tail `scored 25/25 pairs in 0.2s (0 failed)`,
  `Wrote 25 rows × 944 features to /var/tmp/rev4-featbank/probe_feats.csv`.
- Verification (python3, same cwd): `pixel-hash mismatches: 0` over all 25 rows
  vs the recorded probe audit; `feature parity: checked 25 mismatch 0` —
  re-extracted f64 f0..f943 bit-identical to the stored ceiling parquet columns.
  Conclusion: the pinned binary reproduces the ceiling era decode+extract
  exactly; its audit JSONL is a valid pixel-hash source for kadid/tid/select.

## 2026-09-23T13:50Z — era findings (decisive)

- ext944-canonical-2026-08-01 parquets are NOT Rev3: probe of 10 tid pairs via
  extract-native-admission (rev3/sqrt, --full-944) vs ext_tid.parquet = 0/120
  bit-exact rows (same-pair nearest row maxdiff ~1e-5..0.84; f12 0.68 vs 2.14).
  ext944 = v2_ab_extract foldapp2 era (build ec3bdd6, 2026-08-01).
  CONCLUSION: ext944 caches cannot serve as Rev3 bank rows. Convert ONLY
  baseline-recovery + ceiling (rev3-verified); FRESH-EXTRACT all other sets.
- Decode/pixel-hash parity PROVEN across eras: konfig_pairs.tsv recorded
  dist_px_sha256 == extract-native audit distorted_pixels_sha256, 5/5 rows
  (e.g. 9c418c165fb2b9b0 SRC01_colordiffusion_0.png).
- Ceiling human extraction COMPLETED (was queued on heavy.lock):
  ceiling_human_audit.jsonl 11125 rows sha256 eaf2893a85223b10a0f009f64a9647b069798ccb48621304042d91373c1ec8ba
  ceiling_human_feats.csv 11125 rows sha256 f17c5a294149d3f79f5b8dd9577ffc70321346f20ff1e99972291519ce9fdb4e
  ceiling_human_pairs.tsv sha256 caf94cfc8b9bfad6ff2bb527dc07e49733a296c27c5a1e0d9d543ed513cce430
- Ceiling human corpora = {kadid 8125, tid 3000} ONLY (INPUTS.json); no
  konjnd_half in human rows; kadid TERMINAL refs not in ceiling (8125=train+select).
- Baseline-recovery audits complete and positionally aligned:
  cid22-train-audit.jsonl 17611 rows; safesyn-train-audit.jsonl 196086 rows.
- Pair TSV sources for fresh extraction (all verified present):
  kadid_pairs_ab.tsv (10125) tid_pairs_ab.tsv (3000) csiq_pairs.tsv (866)
  cid22val_pairs_ab.tsv konfig_pairs.tsv (1090, sha 44bde2d8 == manifest)
  konjnd_bpg_{train,val}_pairs.tsv aic3_pairs_ab.tsv aic4_pairs.tsv
  konjnd_jpeg_val_pairs.tsv (504 = 1 pair/ref, JPEG half).
- CID22 A/B ref lists: docs/DATASET_HISTORY.md 2026-09-19 entry (A=25, B=24).
- KonJND jpeg split: ref last-digit {7,9}=terminal (100), rest=select (404).
- KonFiG origin split (manifest): train {SRC06,28,50} val {SRC01,03,31,45}
  test {SRC07,09,17}.

## 2026-09-23T14:0xZ — assembler + full-corpus parity proof

- Coordinator scope expansion (received earlier this session): fold sets =
  KADID all views incl SELECT, TID, KonFiG train+originsplit val, KonJND-BPG,
  CID22-A(25), AIC-3 CTC; pixels-only confirmation = CID22-B, AIC-4 sample,
  KonJND JPEG, CSIQ, MCL-JCI. Disk rule: /home >=20 GB, all outputs /var/tmp.
- `convert_cache.py` rewritten as unified assembler: parquet-convert path +
  fresh-extract path sharing `emit_set()` (keys.parquet + sidecar + labels +
  _MANIFEST.json, dedup by content-addressed pair_key).
- DECISIVE verification: fresh ceiling extraction keyed by row_id is
  BIT-EXACT vs ceiling parquets — 13,872,080 feature cells, 0 mismatches
  across human_fit/dev/test/half. extract-native-admission rev3 == ceiling era.
- Extractor quirk found in source (`extract_features_372col.rs:340`): feats
  CSV is `rows.sort_by(ref_name)` — NOT pairs order. Audit jsonl IS pairs
  order and echoes extra TSV cols into extra_targets; CSV echoes them too.
  Fix: all fresh pairs files rebuilt with dense `row_id` column; assembler
  binds audit/csv/pairs keyed on row_id (never positional).
- Pixel-identical stimuli: KADID level-1 no-op distortions decode to
  identical pixels -> identical pair_key. emit_set dedups keys+features
  (first stimulus wins), labels keep ALL stimuli w/ source_row_id +
  n_stimuli. kadid_train 5000->4880 keys (120 collapsed),
  kadid_select 3125->3050 (75 collapsed).
- Structural-zero ids corrected to the authoritative 39 (verified identical
  across cid22-train944, safesyn-train944, human_fit, human_test):
  720,721,754-772,805,806,822,823,856,857,873,874,907,908,927,928,932,933,
  937,938,942,943.
- Ref basename normalization: parquet stores `kadid:I02`, audit stores
  `I02.png` — normalize via prefix-strip + stem + lowercase.
- Parquet-convert DONE (4 sets, f32 mismatch 0 everywhere):
  tid2013      3000/3000 keys  2971 B/row
  kadid_train  5000->4880      2931 B/row
  kadid_select 3125->3050      2961 B/row
  cid22_train  17611/17611     2770 B/row (plan's 2735 + ~35 B/row pair_key)
  safesyn queued under heavy.lock (196086 rows).
- Label roles per ruling (2026-09-23): confirmation sets emit NO labels
  file (pixels-only): cid22_b, aic4, konjnd_jpeg_{select,terminal}, csiq,
  mcljci, kadid_terminal. Potential/fold/train sets emit usable labels
  (copy-through): kadid_{train,select}, tid2013, konfig_{train,val},
  konjnd_bpg_{train,val}, cid22_a25, aic3, cid22_train, safesyn.
- CID22 val pairs 4292 rows = A 2192 (25 refs) + B 2100 (24 refs).
- Fresh pairs rebuilt w/ row_id: kadid_terminal 2000, konfig_all 1090,
  konjnd_bpg_{train 8060, val 2020}, cid22val 4292, aic3 600, aic4 300,
  konjnd_jpeg_val 504, csiq 866, mcljci 5000. Extraction requeued on
  heavy.lock (extract_all.sh + extract_mcljci.sh).

## 2026-09-23T14:xxZ — extraction + assembly complete; lane DONE

- safesyn convert under heavy: 196086/196086 keys, 2775 B/row, f32_mismatch=0.
- All 10 fresh extractions completed, 0 failures (extract_all.sh 127s total;
  mcljci 237s). Watcher `assemble_when_ready.sh` assembled each set on
  row-count-verified raw files.
- cid22_a25 initially emitted 2111/2192 — A-ref `ularapi_Semarang_City_Logo`
  is mixed-case; fixed by lowercased CID22_A/B_REFS_L membership (select uses
  basename_norm). Re-ran: 2192/2192.
- Extract-path end-to-end proof: keyed ceiling triple -> tid2013 sidecar
  bit-identical (3000 rows x 905 populated cols, 0 mismatch).
- bank_report.py: 18 sets, 0 problems, 248983 unique keys / 249227 stimuli.
- Stale `bank/_sealed/kadid_select` from draft config removed (select labels
  are unsealed potential-exposed).
- Confirmation collapse counts: kadid_terminal 48, csiq 1.
- FEATBANK_EXTRACT_DONE.md written to /home/lilith/tmp/zensim-paper/rev4/.
- /home 68G free (>=20 rule); all outputs under /var/tmp.

## 2026-09-23T14:4xZ — Opus review corrections (PROMOTE WITH CORRECTIONS)

Review `/home/lilith/tmp/zensim-paper/rev4/REVIEW_FEATBANK_EXTRACT.md` verdict
PROMOTE WITH CORRECTIONS. Applied corrections 1,2,4,5,6,7,8,9; correction 3
drafted as proposed DATA_SPLITS text in the DONE file; correction 10 confirmed
by coordinator message (cited in DONE).

- Corr 4 (option a): moved every pairs/raw file carrying a held-out
  human_score under /var/tmp/rev4-featbank/_sealed/ —
  pairs/{cid22val,aic4,csiq,konjnd_jpeg_val,kadid_terminal,mcljci}.tsv and
  raw/<same six>.{feats.csv,audit.jsonl,feats.csv.manifest.json,
  feats.csv.producer.bin}. sha256s unchanged by move; README written at
  _sealed/README.md. Disclosure corrected in DONE.
- Corr 2/5/6: cid22_a25 label = "CID22 human MCOS/100"; label_scale field
  added to every manifest (ssim2 raw / ssim2/100 / MCOS/100 / source-unit
  descriptions); build_commit fixed to "unknown(binary 7c7ffbbf…)" for the
  ceiling-converted sets.
- Corr 7: DONE arithmetic fixed (stimuli 5000+3125+2000=10125); collapse note
  now says distinct distortions can decode to identical pixels.
- Corr 8: convert_cache.py split (19.5KB) + featbank_sets.py (18.1KB), both
  under the 30KB committed-file limit; worklog timestamps corrected;
  mcljci.tsv provenance recorded (built from
  /var/tmp/datasets/mcl-jci/pairs_mcljci_src.tsv + appended row_id column,
  sha256 b7f4667d52e5b58ee5502d025d3c2c7c5867ef202ae77f7fbb4c43dc652c91b1 —
  predates build_pairs.py's row_id change, absent from pairs/_sha256.tsv);
  manifest TSV expanded to a full file list; commit messages rewritten in
  What/Commands/Outputs(sha256)/Numbers form.
- Re-emitted all 18 sets against sealed paths: all 47 bank parquet files
  BYTE-IDENTICAL to the promoted versions (only manifests changed);
  bank_report.py re-run: 18 sets, 0 problems, 248983 keys / 249227 stimuli.
- Corr 9: prereg amendments A (dedup counts 4880/3050) and B (kadid_select
  unsealed per D1) recorded in the prereg file and DONE.

## Verbatim command record (correction 8)

Extraction (under `~/tmp/devin/heavy` = flock /home/lilith/tmp/devin/heavy.lock
+ run-heavy):
```
BIN=/home/lilith/work/zensim-validation-2026-09-14/native-integrity-admission/extract-native-admission  # sha256 7c7ffbbf…
env ZENSIM_FORMULA_REV=3 ZENSIM_ROOT_FORM=sqrt RAYON_NUM_THREADS=8 \
  $BIN --corpus pairs-tsv --path <pairs.tsv> --out <raw>/<n>.feats.csv \
       --full-944 --audit-jsonl <raw>/<n>.audit.jsonl
```
All 10 runs: exit 0, "scored N/N pairs … (0 failed)". Output sha256s are
recorded per set in each bank `_MANIFEST.json` `source_sha256` (pairs/audit/
feats). Ceiling re-derivation identical form on ceiling_human_pairs.tsv ->
ceiling_human_audit.jsonl sha eaf2893a…, feats.csv sha f17c5a29….

Assembly (no lock; pyarrow joins only):
```
cd /home/lilith/work/zen/zensim--featbank-extract && \
  python3 scripts/rev4_featbank/convert_cache.py --set <name>
```
All 18 sets: exit 0, `f32_mismatch=0`; safesyn under `~/tmp/devin/heavy --mem
24G` (peak-RSS 8.13GiB, rc=0). Bank validation:
`python3 scripts/rev4_featbank/bank_report.py` -> `{"n_sets": 18,
"problems": [], "total_unique_keys": 248983, "total_stimuli": 249227}`.

Parity checks (python3 one-liners, cwd anywhere): ceiling feats.csv vs ceiling
parquets keyed by row_id -> `TOTAL 13872080 bad 0`; keyed ceiling triple ->
tid2013 sidecar -> `tid joined rows: 3000 mismatched cells: 0 missing keys: 0`.

## 2026-09-23 ~23:5xZ — MISSING (c)(d)(e) closure (coordinator instruction)

- Wrote `benchmarks/rev4_featbank_extract_2026-09-23.{md,json,pointer.md}`
  (4.4/24.4/13.5 KB): per-set counts, roles, label_scale, build_commit,
  binary sha256, sealed locations, promote verdict; pointer = sha256 of all
  66 bank files + all 31 _sealed files.
- Registry: appended consumer set
  `basic+peaks+masked+iw+v2+append+append2@w944/ceiling_rev3#e3db6aab`
  (populated-905 slots; hash computed via the
  `feature_set_id::slots_hash8` FNV-1a algorithm, validated by reproducing
  `b782e349` on slots 0-943). Registry JSON re-validated.
- DATASET_HISTORY: appended dated `## 2026-09-23` bank entry.
- DATA_PROVENANCE (other repo): proposed row text placed in DONE file only.
- Tower: `ssh tower df -h <array>` → 20T avail.
  `nice -n19 rsync -a bank _sealed` →
  `…:/mnt/user/coefficient/output/zensim/rev4-featbank-2026-09-23/` (935M).
  First rsync flattened both dirs into the dest root (trailing-slash bug);
  dest dir removed and re-synced correctly. sha256 verified local==remote
  on 3 random files: bank/_MANIFEST.json 813cd920…,
  bank/csiq/_MANIFEST.json d9ce0601…,
  _sealed/raw/csiq.feats.csv.producer.bin 915cc4c9….
- DONE file updated: MISSING (c)(d)(e) marked CLOSED; (a) Part B, (b)
  measure-first run, (f) recompute lines remain open.

## 2026-09-24 ~00:0xZ — MISSING (f) recompute lines (coordinator instruction)

- New `scripts/rev4_featbank/verify_bank_f32.py`: standalone re-verifier —
  rebuilds every source stimulus's (pair_key, f64 row) from original inputs
  (parquets+audits for converted sets; pairs+audit+feats for extracted) and
  bit-compares np.float32(src) vs sidecar row at its pair_key. ~21 s.
  Result: `TOTAL stimuli=249227 cells=225550435 bad=0`.
- Ceiling parity re-run verbatim: `TOTAL 13872080 bad 0 missing 0`.
- bank_report re-run rewrote bank/_MANIFEST.json created_utc (sha
  813cd920->b208bc08); remote copy verified identical modulo timestamp and
  re-synced; mirror spot-check re-verified 3/3 (b208bc08/d9ce0601/915cc4c9).
- DONE file: MISSING (f) CLOSED; "Recompute" section holds one-line
  commands + verbatim outputs for every headline number.
