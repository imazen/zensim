# CHROMAQ per-cell rows (moved out of git)

The per-cell tables (1,760 rows: family, level, bytes, every judge and every bake score) live in block storage, not in the repo:

| file | sha256 | location |
|---|---|---|
| `rows_full.tsv.gz` | `51ccaffdd66a66ce98be65049934137ac821672f6c51571d075f4d7c44b1a96a` | tower `output/zensim/chromaq-2026-10-04/rows_full.tsv.gz`; local `~/tmp/chromaq/out/rows_full.tsv.gz` |
| `rows_noaic3.tsv.gz` | `db0df6f51a5d3ce63a26911335773ac07ba19bccb1c08b75131f32fd235b5974` | tower `output/zensim/chromaq-2026-10-04/rows_noaic3.tsv.gz`; local `~/tmp/chromaq/out/rows_noaic3.tsv.gz` |

Produced by `analyze.py` from the sweep TSVs and the `serve_custom_bake --pairs` scores (see `../chromaq_2026-10-04.md`).
They were committed in 9b3f1af9 by mistake (over the 30 KB limit) and removed in the next commit.
