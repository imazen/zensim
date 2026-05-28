# Pointer: corruption-corpus multi-metric scoring outputs

## What

Multi-metric scoring of the 672-entry × {corruption, q20, q10} corruption
corpus (codec-corpus#7, `gb82_dog__reference.png`) with 5 metrics:
ssim2-gpu, butteraugli-max-gpu, butteraugli-pnorm3-gpu, cvvdp,
dssim-gpu. Generated 2026-05-28 via
`scripts/v_next/corruption_corpus_multimetric_chunked.py`.

## Files (block storage — too large for the repo)

| File | Size | Description |
|---|--:|---|
| `corruption_multimetric_2026-05-28.tsv` | 225 KB | Per-pair scores: name, family, region, sev, kind (corruption/q20/q10), 5 metric columns × 2017 rows |
| `corruption_multimetric_analysis.md` | 4.7 KB | Aggregated tables: gate pass rate per metric, per-region, per-family, discriminative gap |
| `cell5_corruption_gate.log` | 4.4 KB | V0_2-methodology-372 Cell 5 bake gate-pass log |
| `v47_strict_qat_native_2026-05-27__corruption_gate.log` | 4.4 KB | Current Profile::A v47 gate-pass log |
| `v02_bvls_NO_shaping_2026-05-28__corruption_gate.log` | 4.4 KB | v02-bvls NO shaping gate-pass log |
| `v02_bvls_shaped_2026-05-28__corruption_gate.log` | 4.4 KB | v02-bvls WITH shaping gate-pass log |

## Location

- Local: `/mnt/v/output/zensim/corruption_gate_results/`
- Not mirrored to R2 yet — small enough that block-storage suffices
  for now. Mirror on next R2 sync if user requests.
- Not mirrored to Tower yet — same.

## Headline finding

**butteraugli-max-gpu wins the corruption-gate use case by 2-4×**:

| metric | pass@q20 | pass@q10 |
|---|--:|--:|
| **butter-max-gpu** | **72.2%** | **60.9%** |
| Cell 5 (zensim 4.8 KB linear) | 38.4% | 20.4% |
| v47 (Profile::A, 27 KB MLP) | 19.6% | 10.7% |
| v47 + TILEMIN tile-min pooling | 36.9% | 18.3% |

Full analysis: `benchmarks/corruption_corpus_multimetric_2026-05-28.md`.

## Reproduction

```bash
python3 scripts/v_next/corruption_corpus_multimetric_chunked.py
# → ~9 min on a 7950X with GPU, writes /tmp/corruption_multimetric_2026-05-28.tsv
python3 scripts/v_next/corruption_corpus_analyze.py /tmp/corruption_multimetric_2026-05-28.tsv
# → prints the aggregated tables to stdout
```

## sha256

```
98f63607ba4f6a1495ac4d96132fa7b4869ed3078f3bd24f7353e00e8cf64413  corruption_multimetric_2026-05-28.tsv
```
