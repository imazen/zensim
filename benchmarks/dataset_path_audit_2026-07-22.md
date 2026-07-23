# `/mnt/v/dataset` (singular) vs `/mnt/v/datasets` (plural) — reference audit (2026-07-22)

Ripgrep audit after noticing corpora referenced under BOTH path forms.

## Counts (zensim repo)
- **256** refs to `/mnt/v/dataset/` (singular) — the DOMINANT convention.
- **52** refs to `/mnt/v/datasets/` (plural).

## What lives where (disk truth)
- **Singular `/mnt/v/dataset/`** = the older FR-IQA corpora, EXTRACTED + working:
  `kadid10k` (116 refs), `tid2013` (50), `cid22` (42), `aic3_ctc_epfl` (24),
  `aic4_sample` (16), `pipal`, `konfig-iqa`, `csiq` (source images).
- **Plural `/mnt/v/datasets/`** = NEWER datasets, working: `LIVE`, `KonJND-1k`,
  `jpeg-ai-sdr25`, `kadis700k`, `fill4-6codec-2026-07-01`, `aic3-btc-ptc`,
  `aic-hdr2025`, `upiq` — PLUS raw-archive (.zip/.rar) copies of some singular corpora.

## Findings (severity: LOW — no live functional breakage)

1. **6 corpora exist under BOTH paths, DIVERGENT** (not symlinks): `csiq`,
   `kadid10k`, `tid2013`, `pipal`, `konfig-iqa`, `pieapp`. In 4 of them the plural
   copy is just the raw archive (`kadid10k.zip`, `tid2013.rar`, `pipal/*.zip`) —
   benign redundancy; live refs correctly use the singular (extracted) path.
2. **`csiq` is genuinely SPLIT** — source images under `/mnt/v/dataset/csiq`
   (singular), but `dst_imgs/` + `csiq_pairs.tsv` + `csiq.DMOS.xlsx` under
   `/mnt/v/datasets/csiq` (plural). `build_fr_corpus_pairs.py::build_csiq` already
   spans both (SRC=singular, DST/OUT=plural) — WORKS but fragile. The one case
   worth consolidating.
3. **`pieapp`**: singular dir is EMPTY; only plural has data. (No singular pieapp
   refs exist, so nothing broken — but a future singular ref would fail.)
4. **1 broken ref**: `/mnt/v/dataset/konjnd-1k` (3×) — path does NOT exist (real
   corpus is `/mnt/v/datasets/KonJND-1k`, correctly used 13× elsewhere). All 3
   broken refs are in **historical docs only** (`docs/HISTORY-2026-05-v0x-era.md`,
   `site/COMPARE_PLAN_2026-05-12.md`, `benchmarks/cycle_7_..._2026-05-12.md`) —
   **zero live scripts**. Dead-doc cosmetic issue.

## Convention going forward
There is no single canonical prefix — it's two storage eras. **Rule of thumb:**
older FR-IQA corpora → `/mnt/v/dataset/<name>` (singular, extracted); newer
downloads → `/mnt/v/datasets/<name>` (plural). Check disk before hardcoding a new
ref.

## CONSOLIDATION DONE (2026-07-22 → 07-23, user directive "consolidate, move duplicate archives to tower")

1. **csiq → fully consolidated under SINGULAR `/mnt/v/dataset/csiq/`.** The
   distorted images already existed there (per-distortion dirs `awgn/blur/...`,
   150 each — identical to the old plural `dst_imgs/`); only the DMOS xlsx + pairs
   TSV were plural-only. Moved DMOS to singular, regenerated
   `/mnt/v/dataset/csiq/csiq_pairs.tsv` (866 pairs, 0 missing, 0 plural refs),
   updated `build_fr_corpus_pairs.py::build_csiq` (SRC/DST/OUT/xlsx all singular).
   The old plural `/mnt/v/datasets/csiq/` (dup dst_imgs + moved DMOS/pairs) →
   backed up to tower (`plural-extracted/`) + removed local.
2. **Duplicate raw archives → tower** `/mnt/tower/v-datasets-archives-2026-07-22/`
   (size-verified before local rm; recoverable): kadid10k (`*_images.zip` +
   plural `.zip`), tid2013 (`.rar`), csiq (`dst_imgs.zip`/`src_imgs.zip`), cid22
   (`CID22.zip` 7.3 GB full-library + `CID22_validation_set*.zip`), konfig-iqa
   (`.zip`). ~17 GB reclaimed locally. Extracted working data KEPT in place.
3. **Empty plural archive-only dirs removed** (tid2013).
4. **ALL raw archives relocated** (2026-07-23): 16 archives total — the 11
   duplicate/split ones above PLUS 5 single-path raw archives (upiq_dataset.zip,
   aic4_sample, LIVE databaserelease2.zip, jpeg-ai-sdr25 ×2) whose extracted forms
   were verified present. **0 archives remain local**; **21 GB on tower**, every
   one size-verified before local rm. Recover any via
   `/mnt/tower/v-datasets-archives-2026-07-22/<corpus>/`.

**Orchestration note (for future archive moves):** `rsync -a` to the tower NFS
export FAILS on `chgrp` ("Operation not permitted") — data transfers but rsync
returns code 23. Use `cp` (or `rsync --no-o --no-g --no-p`). And never `set -e`
with a glob that may not match (`*.rar` where none exist exits the whole script).

**Result — each corpus now has ONE canonical path:** the FR-IQA set under singular
`/mnt/v/dataset/` (kadid10k, tid2013, cid22, aic3_ctc_epfl, aic4_sample, csiq,
pipal, konfig-iqa); the newer datasets under plural `/mnt/v/datasets/` (LIVE,
KonJND-1k, jpeg-ai-sdr25, kadis700k, fill4, aic-hdr2025, upiq). No corpus is split
across both anymore. Dead-doc `konjnd-1k` refs (historical) left as-is.
