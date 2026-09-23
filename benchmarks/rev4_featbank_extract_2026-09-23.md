# Rev4 feature bank — existing-family extract (2026-09-23)
**Verdict:** PROMOTE WITH CORRECTIONS (`REVIEW_FEATBANK_EXTRACT.md`); corrections 1–9 applied. `bank/` is the promoted existing-family bank. No feature here is shown to help.
**Bank:** `/var/tmp/rev4-featbank/bank/` — 18 sets, 249,227 stimuli → 248,983 unique `pair_key`s (244 pixel-identical stimuli collapsed; verified bit-identical features). 696 MB, 47 parquet data files.
**Machine-readable:** `rev4_featbank_extract_2026-09-23.json` · **File hashes:** `rev4_featbank_extract_2026-09-23.pointer.md`
## Producer

| | |
|---|---|
| feature_set_id | basic+peaks+masked+iw+v2+append+append2@w944/ceiling_rev3#b782e349 |
| era / formula / root_form | ceiling_rev3 / rev3 / sqrt |
| input_contract | legacy-rgb8 |
| binary | extract-native-admission |
| binary sha256 | `7c7ffbbfa033e8ca1a8f103d472b61ccde061c2394b03d519af2852ee8eeda87` |
| build_commit | `unknown(binary 7c7ffbbfa033e8ca1a8f103d472b61ccde061c2394b03d519af2852ee8eeda87)` |
| env | ZENSIM_FORMULA_REV=3 ZENSIM_ROOT_FORM=sqrt RAYON_NUM_THREADS=8 |
| pair_key | sha256(utf8(ref_px_sha256_hex)‖utf8(dist_px_sha256_hex)‖utf8('legacy-rgb8')) |
| sidecar | 905 populated ids as f32; 39 structural-zero ids omitted (list per-set _MANIFEST.json) |
| bank consumer set id | `basic+peaks+masked+iw+v2+append+append2@w944/ceiling_rev3#e3db6aab` (slots 0-719,722-753,773-804,807-821,824-855,858-872,875-906,909-926,929-931,934-936,939-941) |

## Per-set counts and label handling

| set | role | stimuli | keys | collapsed | label_scale |
|---|---|---|---|---|---|
| aic3 | fold+potential | 600 | 600 | 0 | human (AIC-3 CTC, source units) |
| aic4 | confirmation | 300 | 300 | 0 | none (pixels only) |
| cid22_a25 | fold+potential | 2,192 | 2,192 | 0 | CID22 human MCOS/100 |
| cid22_b | confirmation | 2,100 | 2,100 | 0 | none (pixels only) |
| cid22_train | train | 17,611 | 17,611 | 0 | ssim2 raw (peer-SSIM2 oracle; observed 3.01–94.11) |
| csiq | confirmation | 866 | 865 | 1 | none (pixels only) |
| kadid_select | fold+potential | 3,125 | 3,050 | 75 | human (KADID dmos-derived, source units) |
| kadid_terminal | confirmation | 2,000 | 1,952 | 48 | none (pixels only) |
| kadid_train | train | 5,000 | 4,880 | 120 | human (KADID dmos-derived, source units) |
| konfig_train | train | 327 | 327 | 0 | human (KonFiG q_jnd-derived, source units) |
| konfig_val | fold+potential | 436 | 436 | 0 | human (KonFiG q_jnd-derived, source units) |
| konjnd_bpg_train | train | 8,060 | 8,060 | 0 | ssim2/100 (gpu_ssimulacra2/100; observed −0.649…0.962) |
| konjnd_bpg_val | fold | 2,020 | 2,020 | 0 | ssim2/100 (gpu_ssimulacra2/100; observed −0.649…0.962) |
| konjnd_jpeg_select | confirmation | 404 | 404 | 0 | none (pixels only) |
| konjnd_jpeg_terminal | confirmation | 100 | 100 | 0 | none (pixels only) |
| mcljci | confirmation | 5,000 | 5,000 | 0 | none (pixels only) |
| safesyn | train | 196,086 | 196,086 | 0 | ssim2 raw (peer-SSIM2 oracle; observed −743.9…100.0) |
| tid2013 | train | 3,000 | 3,000 | 0 | human (TID2013 mos, source units) |

## Key and label handling

- **Keys** are content-addressed: pixel-identical stimuli share one `pair_key` row in `keys.parquet`/`features*.parquet`; every source stimulus keeps a row in `labels*.parquet` with `source_row_id`; `keys.n_stimuli` records multiplicity. Join keys↔labels via `source_row_id`/label row order — never assume one stimulus per key.
- **Fold / train / potential sets** emit `labels__human.parquet` (source-unit human labels) or `labels__ssim2_oracle.parquet` (ssim2 raw or /100 — see `label_scale`); copy-through only, nothing fitted or selected.
- **Confirmation sets** (cid22_b, aic4, konjnd_jpeg_select, konjnd_jpeg_terminal, csiq, mcljci, kadid_terminal) emit **no labels file** — pixels only, per ruling D4.
## Sealed source replicas

Held-out-score-bearing extraction sources live sealed at `/var/tmp/rev4-featbank/_sealed/` (see its README; review correction 4a — moved, not deleted or regenerated):

- `_sealed/pairs/`: aic4.tsv, cid22val.tsv, csiq.tsv, kadid_terminal.tsv, konjnd_jpeg_val.tsv, mcljci.tsv
- `_sealed/raw/`: the matching `.audit.jsonl`, `.feats.csv`, `.feats.csv.manifest.json`, `.feats.csv.producer.bin` per set

Bank-side label-bearing sets keep their `pairs/`/`raw/` sources unsealed (labels are in-role there). The promoted bank itself carries no confirmation labels.
