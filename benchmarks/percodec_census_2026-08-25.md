# Per-encoder k2/k3 census (zensim-target loop, secant on, tol ±2, 9 refs × t{70,80,88} = 27 cells)

| codec | k | census (±2 / 27) | median iters |
|---|--:|--:|--:|
| zenavif | 2 | 13/27 | 2 |
| zenavif | 3 | 23/27 | 3 |
| zenjpeg | 2 | 9/27 | 2 |
| zenjpeg | 3 | 19/27 | 3 |
| zenwebp | 2 | 7/27 | 2 |
| zenwebp | 3 | 18/27 | 3 |

Instrument: `zensim-target` CLI (`target_search`, `ZENSIM_TARGET_SECANT=1`) over the 27-cell
instrument (9 refs = city/dog/girl + 3 CID + 3 gb82-sc crops × targets 70/80/88), tolerance ±2,
max-iterations = k. Drives each codec's REAL encoder; census = fraction converged within ±2 at
budget k. NOTE (per-codec ownership, feedback_per_codec_loop_ownership): this uses the SHARED
zensim-target loop; the directive wants each codec to own its loop — jxl + zenavif already do (their
OWN censuses: jxl benchmarks/zensim_secant_2026-08-25.md; zenavif runs its own encode_rgb8_with_target).
This table is the cross-codec baseline; zenjpeg/zenwebp/svt/aom/gainmap own-loop censuses follow their
own-loop implementations. Copy: /mnt/v/output/zensim/percodec-census-2026-08-25/.
