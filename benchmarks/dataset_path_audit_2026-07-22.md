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
ref. The only genuine consolidation candidate is **csiq** (unify its source +
dst/pairs under one path); the rest is benign redundancy + dead-doc refs.
