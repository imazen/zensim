Devin-executed; Opus-reviewed 2026-09-23 (REVIEW_PAPER_MEASURE.md): PROMOTE WITH CORRECTIONS

# Peak memory, N-process throughput and cold vs warm latency — zensim and peers (2026-09-23)

Measurement lane for the zensim companion paper ("memory + throughput", user-approved
2026-09-22). Every number below comes from a run made for this record, or is an
arithmetic difference of two such runs (and says so). Raw outputs:
[`paper_memory_2026-09-23.pointer.md`](paper_memory_2026-09-23.pointer.md).
Compact machine-readable headline JSON: [`paper_memory_2026-09-23.json`](paper_memory_2026-09-23.json). The complete original tables are SHA-pinned in the pointer.

Reviewer independently confirmed 16 MP/8T heap share: zensim D 134,647,456 B (max RSS 224,976 KiB), butteraugli 3,625,201,630 B, CVVDP-4k 3,557,352,199 B. The 1 MP/N=16 throughput values are CONTENDED: GMSD 13,756.885 pairs/s and zensim B 252.2457 pairs/s.

## Results

Every table below is produced verbatim by
`zenmetrics--paper-memory/benchmarks/heaptrack/paper_memory/summarize.py`
(commit `a6355cde` on `quarantine/devin/paper-memory`, zenmetrics workspace)
from `/var/tmp/paper-memory/{memory.tsv, tput.tsv, cold/cold_warm_{1t,8t}.json}`;
sha256s in the pointer file and WORKLOG. Memory cells are allocation measures
only; throughput and cold/warm cells are flagged `CONTENDED` where foreign
load crossed the gate — every throughput cell and every cold round in this
run is so flagged (a shared box; the numbers are reported, not used for
uncontended claims).

### Scalar score path

| metric | 1 MP 1T | 1 MP 8T | 16 MP 1T | 16 MP 8T | 40 MP 1T | 40 MP 8T |
|---|---:|---:|---:|---:|---:|---:|
| zensim PreviewV0_2 (published 0.2.x profile) | 27.2 MB (26.0 B/px); RSS 37 MiB | 40.9 MB (39.0 B/px); RSS 49 MiB | 410.9 MB (24.5 B/px); RSS 492 MiB | 461.6 MB (27.5 B/px); RSS 539 MiB | 975.0 MB (24.3 B/px); RSS 1,164 MiB | 1,056.4 MB (26.3 B/px); RSS 1,242 MiB |
| zensim B (unreleased default) | 27.4 MB (26.1 B/px); RSS 38 MiB | 42.0 MB (40.0 B/px); RSS 50 MiB | 411.5 MB (24.5 B/px); RSS 493 MiB | 464.7 MB (27.7 B/px); RSS 543 MiB | 975.6 MB (24.3 B/px); RSS 1,165 MiB | 1,059.5 MB (26.4 B/px); RSS 1,244 MiB |
| zensim D | 16.7 MB (15.9 B/px); RSS 26 MiB | 33.1 MB (31.6 B/px); RSS 40 MiB | 67.6 MB (4.0 B/px); RSS 161 MiB | 134.6 MB (8.0 B/px); RSS 220 MiB | 113.4 MB (2.8 B/px); RSS 339 MiB | 214.2 MB (5.3 B/px); RSS 430 MiB |
| zensim Rev3 fast (Y60 ×5 ensemble) | 15.5 MB (14.7 B/px); RSS 23 MiB | 27.8 MB (26.5 B/px); RSS 33 MiB | 62.9 MB (3.7 B/px); RSS 148 MiB | 114.0 MB (6.8 B/px); RSS 194 MiB | 105.3 MB (2.6 B/px); RSS 318 MiB | 179.9 MB (4.5 B/px); RSS 381 MiB |
| zensim Rev3 rich (basic228 ×5 ensemble) | 17.4 MB (16.6 B/px); RSS 27 MiB | 34.1 MB (32.5 B/px); RSS 41 MiB | 68.3 MB (4.1 B/px); RSS 161 MiB | 135.4 MB (8.1 B/px); RSS 220 MiB | 114.1 MB (2.8 B/px); RSS 340 MiB | 215.0 MB (5.4 B/px); RSS 429 MiB |
| GMSD (zenmetrics port of libgmsd) | 0.110 MB (0.1 B/px); RSS 10 MiB | 0.209 MB (0.2 B/px); RSS 10 MiB | 0.177 MB (0.0 B/px); RSS 100 MiB | 0.576 MB (0.0 B/px); RSS 100 MiB | 0.231 MB (0.0 B/px); RSS 233 MiB | 0.971 MB (0.0 B/px); RSS 234 MiB |
| SSIMULACRA2 (fast-ssim2) | 159.5 MB (152.1 B/px); RSS 158 MiB | 159.5 MB (152.1 B/px); RSS 158 MiB | 2,550.3 MB (152.0 B/px); RSS 2,468 MiB | 2,550.3 MB (152.0 B/px); RSS 2,468 MiB | 6,743.1 MB (168.2 B/px); RSS 5,893 MiB | 6,743.1 MB (168.2 B/px); RSS 5,893 MiB |
| butteraugli (max + 3-norm, one call) | 152.2 MB (145.2 B/px); RSS 163 MiB | 223.7 MB (213.3 B/px); RSS 196 MiB | 2,433.2 MB (145.0 B/px); RSS 2,741 MiB | 3,625.2 MB (216.1 B/px); RSS 3,091 MiB | 5,819.8 MB (145.1 B/px); RSS 5,784 MiB | 8,631.0 MB (215.3 B/px); RSS 8,374 MiB |
| DSSIM (dssim-core) | 117.5 MB (112.1 B/px); RSS 137 MiB | 132.5 MB (126.4 B/px); RSS 159 MiB | 1,878.7 MB (112.0 B/px); RSS 1,904 MiB | 2,131.5 MB (127.0 B/px); RSS 2,159 MiB | 4,489.9 MB (112.0 B/px); RSS 4,531 MiB | 5,101.4 MB (127.2 B/px); RSS 4,737 MiB |
| IW-SSIM (zenmetrics port) | 147.5 MB (140.7 B/px); RSS 161 MiB | 147.5 MB (140.7 B/px); RSS 161 MiB | 2,367.8 MB (141.1 B/px); RSS 2,446 MiB | 2,367.8 MB (141.1 B/px); RSS 2,446 MiB | 5,661.1 MB (141.2 B/px); RSS 5,681 MiB | 5,661.1 MB (141.2 B/px); RSS 5,680 MiB |
| CVVDP CPU port, standard_4k display | 222.4 MB (212.1 B/px); RSS 223 MiB | 222.6 MB (212.3 B/px); RSS 222 MiB | 3,557.0 MB (212.0 B/px); RSS 3,493 MiB | 3,557.4 MB (212.0 B/px); RSS 3,493 MiB | 8,500.7 MB (212.0 B/px); RSS 8,341 MiB | 8,500.8 MB (212.0 B/px); RSS 8,342 MiB |
| CVVDP CPU port, standard_fhd display (JPEG AIC) | 222.4 MB (212.1 B/px); RSS 223 MiB | 222.5 MB (212.2 B/px); RSS 222 MiB | 3,557.0 MB (212.0 B/px); RSS 3,493 MiB | 3,557.4 MB (212.0 B/px); RSS 3,493 MiB | 8,500.7 MB (212.0 B/px); RSS 8,341 MiB | 8,500.7 MB (212.0 B/px); RSS 8,342 MiB |
| DVIFM-ish talk-faithful-luma, float | 46.2 MB (44.1 B/px); RSS 58 MiB | 46.2 MB (44.1 B/px); RSS 58 MiB | 738.3 MB (44.0 B/px); RSS 748 MiB | 738.3 MB (44.0 B/px); RSS 748 MiB | 1,764.3 MB (44.0 B/px); RSS 1,774 MiB | 1,764.3 MB (44.0 B/px); RSS 1,774 MiB |
| DVIFM-ish talk-faithful-luma, integer | 13.9 MB (13.2 B/px); RSS 26 MiB | 13.9 MB (13.2 B/px); RSS 26 MiB | 220.9 MB (13.2 B/px); RSS 338 MiB | 220.9 MB (13.2 B/px); RSS 338 MiB | 527.9 MB (13.2 B/px); RSS 743 MiB | 527.9 MB (13.2 B/px); RSS 743 MiB |
| DVIFM-ish serving-gate-ycbcr3, float | 46.2 MB (44.1 B/px); RSS 64 MiB | 46.2 MB (44.1 B/px); RSS 64 MiB | 738.3 MB (44.0 B/px); RSS 812 MiB | 738.3 MB (44.0 B/px); RSS 812 MiB | 1,764.3 MB (44.0 B/px); RSS 1,945 MiB | 1,764.3 MB (44.0 B/px); RSS 1,945 MiB |
| DVIFM-ish serving-gate-ycbcr3, integer | 16.0 MB (15.2 B/px); RSS 28 MiB | 16.0 MB (15.2 B/px); RSS 28 MiB | 254.5 MB (15.2 B/px); RSS 366 MiB | 254.5 MB (15.2 B/px); RSS 365 MiB | 608.1 MB (15.2 B/px); RSS 825 MiB | 608.1 MB (15.2 B/px); RSS 825 MiB |

### Map path

| metric | 1 MP 1T | 1 MP 8T | 16 MP 1T | 16 MP 8T | 40 MP 1T | 40 MP 8T |
|---|---:|---:|---:|---:|---:|---:|
| zensim PreviewV0_2 (published 0.2.x profile) | 39.6 MB (37.7 B/px); RSS 49 MiB | 46.9 MB (44.7 B/px); RSS 61 MiB | 628.5 MB (37.5 B/px); RSS 684 MiB | 643.5 MB (38.4 B/px); RSS 692 MiB | 1,496.7 MB (37.3 B/px); RSS 1,622 MiB | 1,508.9 MB (37.6 B/px); RSS 1,600 MiB |
| zensim B (unreleased default) | 39.7 MB (37.8 B/px); RSS 51 MiB | 50.3 MB (48.0 B/px); RSS 63 MiB | 628.0 MB (37.4 B/px); RSS 685 MiB | 641.5 MB (38.2 B/px); RSS 697 MiB | 1,496.1 MB (37.3 B/px); RSS 1,623 MiB | 1,505.9 MB (37.6 B/px); RSS 1,609 MiB |
| zensim D | 39.7 MB (37.8 B/px); RSS 51 MiB | 50.3 MB (48.0 B/px); RSS 63 MiB | 628.0 MB (37.4 B/px); RSS 684 MiB | 641.5 MB (38.2 B/px); RSS 698 MiB | 1,496.1 MB (37.3 B/px); RSS 1,623 MiB | 1,505.9 MB (37.6 B/px); RSS 1,612 MiB |
| zensim Rev3 fast (Y60 ×5 ensemble) | 90.5 MB (86.3 B/px); RSS 74 MiB | 93.3 MB (89.0 B/px); RSS 86 MiB | 1,440.1 MB (85.8 B/px); RSS 1,098 MiB | 1,449.8 MB (86.4 B/px); RSS 1,141 MiB | 3,437.8 MB (85.7 B/px); RSS 2,612 MiB | 3,445.2 MB (85.9 B/px); RSS 2,697 MiB |
| zensim Rev3 rich (basic228 ×5 ensemble) | 135.1 MB (128.9 B/px); RSS 141 MiB | 138.0 MB (131.6 B/px); RSS 154 MiB | 2,145.3 MB (127.9 B/px); RSS 2,148 MiB | 2,155.5 MB (128.5 B/px); RSS 2,192 MiB | 5,122.5 MB (127.8 B/px); RSS 5,122 MiB | 5,129.8 MB (127.9 B/px); RSS 5,207 MiB |
| GMSD (zenmetrics port of libgmsd) | 9.5 MB (9.1 B/px); RSS 19 MiB | 9.6 MB (9.2 B/px); RSS 19 MiB | 151.2 MB (9.0 B/px); RSS 244 MiB | 151.5 MB (9.0 B/px); RSS 243 MiB | 361.1 MB (9.0 B/px); RSS 578 MiB | 361.7 MB (9.0 B/px); RSS 578 MiB |
| butteraugli (max + 3-norm, one call) | 152.2 MB (145.2 B/px); RSS 163 MiB | 223.6 MB (213.2 B/px); RSS 199 MiB | 2,433.2 MB (145.0 B/px); RSS 2,741 MiB | 3,625.2 MB (216.1 B/px); RSS 3,605 MiB | 5,819.8 MB (145.1 B/px); RSS 5,784 MiB | 8,671.2 MB (216.3 B/px); RSS 8,004 MiB |
| DSSIM (dssim-core) | 117.5 MB (112.1 B/px); RSS 137 MiB | 126.2 MB (120.4 B/px); RSS 156 MiB | 1,878.7 MB (112.0 B/px); RSS 1,904 MiB | 2,134.7 MB (127.2 B/px); RSS 2,096 MiB | 4,489.9 MB (112.0 B/px); RSS 4,531 MiB | 5,100.8 MB (127.2 B/px); RSS 4,808 MiB |
| CVVDP CPU port, standard_4k display | 311.9 MB (297.5 B/px); RSS 308 MiB | 312.0 MB (297.5 B/px); RSS 312 MiB | 4,988.6 MB (297.3 B/px); RSS 4,858 MiB | 4,988.7 MB (297.3 B/px); RSS 4,907 MiB | 11,922.1 MB (297.3 B/px); RSS 11,604 MiB | 11,922.2 MB (297.3 B/px); RSS 11,633 MiB |
| CVVDP CPU port, standard_fhd display (JPEG AIC) | 299.3 MB (285.4 B/px); RSS 296 MiB | 299.4 MB (285.5 B/px); RSS 300 MiB | 4,787.2 MB (285.3 B/px); RSS 4,667 MiB | 4,787.3 MB (285.3 B/px); RSS 4,714 MiB | 11,440.8 MB (285.3 B/px); RSS 11,145 MiB | 11,440.9 MB (285.3 B/px); RSS 11,174 MiB |
| DVIFM-ish talk-faithful-luma, float | 46.2 MB (44.1 B/px); RSS 58 MiB | 46.2 MB (44.1 B/px); RSS 58 MiB | 738.3 MB (44.0 B/px); RSS 748 MiB | 738.3 MB (44.0 B/px); RSS 748 MiB | 1,764.3 MB (44.0 B/px); RSS 1,774 MiB | 1,764.3 MB (44.0 B/px); RSS 1,774 MiB |
| DVIFM-ish serving-gate-ycbcr3, float | 47.1 MB (44.9 B/px); RSS 65 MiB | 47.1 MB (44.9 B/px); RSS 65 MiB | 752.6 MB (44.9 B/px); RSS 859 MiB | 752.6 MB (44.9 B/px); RSS 859 MiB | 1,798.5 MB (44.9 B/px); RSS 1,977 MiB | 1,798.5 MB (44.9 B/px); RSS 1,977 MiB |

### Bounded-memory (strip / stream) path

| metric | 1 MP 1T | 1 MP 8T | 16 MP 1T | 16 MP 8T | 40 MP 1T | 40 MP 8T |
|---|---:|---:|---:|---:|---:|---:|
| zensim B (unreleased default) | 25.2 MB (24.0 B/px); RSS 35 MiB | 47.3 MB (45.2 B/px); RSS 56 MiB | 314.5 MB (18.7 B/px); RSS 389 MiB | 532.8 MB (31.8 B/px); RSS 605 MiB | 716.9 MB (17.9 B/px); RSS 898 MiB | 1,073.6 MB (26.8 B/px); RSS 1,274 MiB |
| SSIMULACRA2 (fast-ssim2) | 119.9 MB (114.3 B/px); RSS 122 MiB | 119.9 MB (114.3 B/px); RSS 122 MiB | 841.1 MB (50.1 B/px); RSS 891 MiB | 841.1 MB (50.1 B/px); RSS 891 MiB | 2,359.9 MB (58.9 B/px); RSS 1,859 MiB | 2,359.9 MB (58.9 B/px); RSS 1,858 MiB |
| butteraugli (max + 3-norm, one call) | 96.7 MB (92.2 B/px); RSS 107 MiB | 133.9 MB (127.7 B/px); RSS 154 MiB | 720.1 MB (42.9 B/px); RSS 828 MiB | 903.9 MB (53.9 B/px); RSS 1,107 MiB | 1,505.3 MB (37.5 B/px); RSS 1,722 MiB | 1,819.7 MB (45.4 B/px); RSS 2,124 MiB |
| IW-SSIM (zenmetrics port) | 104.3 MB (99.5 B/px); RSS 115 MiB | 104.3 MB (99.5 B/px); RSS 114 MiB | 716.2 MB (42.7 B/px); RSS 850 MiB | 716.2 MB (42.7 B/px); RSS 849 MiB | 1,406.8 MB (35.1 B/px); RSS 1,689 MiB | 1,406.8 MB (35.1 B/px); RSS 1,689 MiB |
| CVVDP CPU port, standard_4k display | 176.0 MB (167.8 B/px); RSS 115 MiB | 176.0 MB (167.8 B/px); RSS 115 MiB | 1,478.7 MB (88.1 B/px); RSS 1,097 MiB | 1,478.7 MB (88.1 B/px); RSS 1,096 MiB | 2,946.4 MB (73.5 B/px); RSS 2,336 MiB | 2,946.4 MB (73.5 B/px); RSS 2,336 MiB |

| metric | 1 MP 1T | 1 MP 8T | 16 MP 1T | 16 MP 8T | 40 MP 1T | 40 MP 8T |
|---|---:|---:|---:|---:|---:|---:|
| DVIFM-ish talk-faithful-luma, integer | 0.708 MB (0.7 B/px); RSS 10 MiB | 0.708 MB (0.7 B/px); RSS 11 MiB | 2.5 MB (0.1 B/px); RSS 102 MiB | 2.5 MB (0.1 B/px); RSS 102 MiB | 3.5 MB (0.1 B/px); RSS 237 MiB | 3.5 MB (0.1 B/px); RSS 236 MiB |
| DVIFM-ish serving-gate-ycbcr3, integer | 2.0 MB (1.9 B/px); RSS 12 MiB | 2.0 MB (1.9 B/px); RSS 12 MiB | 7.6 MB (0.5 B/px); RSS 107 MiB | 7.6 MB (0.5 B/px); RSS 107 MiB | 11.1 MB (0.3 B/px); RSS 244 MiB | 11.1 MB (0.3 B/px); RSS 244 MiB |

### Cached reference and per-worker increment (heap)

| metric | size | cached reference (MB) | one warm compare adds (MB) | per extra concurrent worker, K=4 / K=8 (MB) |
|---|---|---:|---:|---:|
| zensim PreviewV0_2 (published 0.2.x profile) | 1 MP | 17.0 | 14.4 | API: no shared cached ref / API: no shared cached ref |
| zensim PreviewV0_2 (published 0.2.x profile) | 16 MP | 268.3 | 208.7 | API: no shared cached ref / API: no shared cached ref |
| zensim B (unreleased default) | 1 MP | 17.0 | 14.5 | 14.7 / 14.7 |
| zensim B (unreleased default) | 16 MP | 268.3 | 209.3 | 210.1 / 210.1 |
| zensim D | 1 MP | 17.0 | 16.4 | 21.7 / 21.6 |
| zensim D | 16 MP | 268.3 | 66.7 | 89.6 / 89.6 |
| SSIMULACRA2 (fast-ssim2) | 1 MP | 84.0 | 62.9 | 96.5 / 96.5 |
| SSIMULACRA2 (fast-ssim2) | 16 MP | 1,342.4 | 1,006.4 | 1,543.6 / 1,543.6 |
| butteraugli (max + 3-norm, one call) | 1 MP | 102.8 | 59.9 | 82.6 / 79.0 |
| butteraugli (max + 3-norm, one call) | 16 MP | 1,644.3 | 956.7 | 1,286.3 / 1,299.3 |
| DSSIM (dssim-core) | 1 MP | 64.1 | 53.5 | 62.6 / 41.5 |
| DSSIM (dssim-core) | 16 MP | 1,023.5 | 855.2 | 781.7 / 671.5 |
| IW-SSIM (zenmetrics port) | 1 MP | 37.1 | 110.4 | API: no shared cached ref / API: no shared cached ref |
| IW-SSIM (zenmetrics port) | 16 MP | 592.8 | 1,775.0 | API: no shared cached ref / API: no shared cached ref |
| CVVDP CPU port, standard_4k display | 1 MP | 191.0 | 0.000 | API: no shared cached ref / API: no shared cached ref |
| CVVDP CPU port, standard_4k display | 16 MP | 3,053.7 | 0.000 | API: no shared cached ref / API: no shared cached ref |

### Synthetic-pair control (1 MP, 8 threads)

| metric | 1 MP 1T | 1 MP 8T | 16 MP 1T | 16 MP 8T | 40 MP 1T | 40 MP 8T |
|---|---:|---:|---:|---:|---:|---:|
| zensim PreviewV0_2 (published 0.2.x profile) | — | 39.4 MB (37.6 B/px); RSS 49 MiB | — | — | — | — |
| zensim B (unreleased default) | — | 39.9 MB (38.1 B/px); RSS 51 MiB | — | — | — | — |
| zensim D | — | 32.1 MB (30.6 B/px); RSS 41 MiB | — | — | — | — |
| zensim Rev3 fast (Y60 ×5 ensemble) | — | 27.4 MB (26.1 B/px); RSS 33 MiB | — | — | — | — |
| zensim Rev3 rich (basic228 ×5 ensemble) | — | 33.7 MB (32.1 B/px); RSS 41 MiB | — | — | — | — |
| GMSD (zenmetrics port of libgmsd) | — | 0.208 MB (0.2 B/px); RSS 9 MiB | — | — | — | — |
| SSIMULACRA2 (fast-ssim2) | — | 159.5 MB (152.1 B/px); RSS 158 MiB | — | — | — | — |
| butteraugli (max + 3-norm, one call) | — | 223.6 MB (213.2 B/px); RSS 229 MiB | — | — | — | — |
| DSSIM (dssim-core) | — | 131.4 MB (125.3 B/px); RSS 160 MiB | — | — | — | — |
| IW-SSIM (zenmetrics port) | — | 147.5 MB (140.7 B/px); RSS 160 MiB | — | — | — | — |
| CVVDP CPU port, standard_4k display | — | 222.5 MB (212.2 B/px); RSS 222 MiB | — | — | — | — |
| CVVDP CPU port, standard_fhd display (JPEG AIC) | — | 222.5 MB (212.2 B/px); RSS 222 MiB | — | — | — | — |
| DVIFM-ish talk-faithful-luma, float | — | 46.2 MB (44.1 B/px); RSS 58 MiB | — | — | — | — |
| DVIFM-ish talk-faithful-luma, integer | — | 13.9 MB (13.2 B/px); RSS 26 MiB | — | — | — | — |
| DVIFM-ish serving-gate-ycbcr3, float | — | 46.2 MB (44.1 B/px); RSS 64 MiB | — | — | — | — |
| DVIFM-ish serving-gate-ycbcr3, integer | — | 16.0 MB (15.2 B/px); RSS 27 MiB | — | — | — | — |

### N independent single-threaded processes

| metric | size | N=1 pairs/s | N=4 | N=8 | N=16 | N=16 ÷ N=1 | Σ max RSS at N=1/4/8/16 (GiB) | status |
|---|---|---:|---:|---:|---:|---:|---|---|
| zensim B (unreleased default) | 16mp | 1.42 | 5.46 | 9.94 | 11.89 | 8.39× | 0.48 / 1.94 / 3.88 / 7.75 | CONTENDED |
| zensim D | 16mp | 2.72 | 9.41 | 13.64 | 13.32 | 4.90× | 0.16 / 0.63 / 1.25 / 2.51 | CONTENDED |
| zensim Rev3 fast (Y60 ×5 ensemble) | 16mp | 5.91 | 19.03 | 20.70 | 19.05 | 3.22× | 0.14 / 0.58 / 1.16 / 2.31 | CONTENDED |
| GMSD (zenmetrics port of libgmsd) | 16mp | 64.79 | 244.64 | 463.28 | 661.87 | 10.22× | 0.10 / 0.39 / 0.78 / 1.56 | CONTENDED |
| SSIMULACRA2 (fast-ssim2) | 16mp | 0.69 | 2.39 | 2.91 | NOT-RUN projected 38.8 GiB (N x single-p | — | 2.42 / 9.69 / 19.38 / — | CONTENDED,NOT-RUN |
| butteraugli (max + 3-norm, one call) | 16mp | 0.49 | 1.41 | 1.64 | NOT-RUN projected 47.1 GiB (N x single-p | — | 2.94 / 11.77 / 23.54 / — | CONTENDED,NOT-RUN |
| zensim B (unreleased default) | 1mp | 24.70 | 94.00 | 169.89 | 252.25 | 10.21× | 0.05 / 0.21 / 0.42 / 0.83 | CONTENDED |
| zensim D | 1mp | 35.07 | 138.18 | 249.68 | 369.44 | 10.53× | 0.03 / 0.10 / 0.20 / 0.41 | CONTENDED |
| zensim Rev3 fast (Y60 ×5 ensemble) | 1mp | 80.01 | 311.70 | 568.57 | 729.47 | 9.12× | 0.03 / 0.10 / 0.20 / 0.41 | CONTENDED |
| GMSD (zenmetrics port of libgmsd) | 1mp | 1,047.13 | 4,106.56 | 8,029.50 | 13,756.89 | 13.14× | 0.01 / 0.04 / 0.08 / 0.16 | CONTENDED |
| SSIMULACRA2 (fast-ssim2) | 1mp | 14.37 | 48.31 | 52.93 | 61.98 | 4.31× | 0.15 / 0.62 / 1.24 / 2.47 | CONTENDED |
| butteraugli (max + 3-norm, one call) | 1mp | 11.06 | 34.89 | 32.56 | 35.09 | 3.17× | 0.16 / 0.64 / 1.28 / 2.55 | CONTENDED |

### Cold vs warm (cold_warm_1t.json)

| metric | size | cold launch median (p10–p90) ms | decode | construct | first score | warm per-call median ms | cold ÷ warm |
|---|---|---:|---:|---:|---:|---:|---:|
| process floor (no work) | 256² | 1.09 | | | | | |
| zensim B (unreleased default) | 256² | 4.91 (4.76–5.61) | 1.56 | 0.00 | 2.27 | 2.003 | 2.45 |
| zensim D | 256² | 4.35 (4.18–4.96) | 1.56 | 0.00 | 1.72 | 1.260 | 3.45 |
| zensim Rev3 fast (Y60 ×5 ensemble) | 256² | 4.09 (3.89–4.72) | 1.56 | 0.25 | 1.17 | 0.803 | 5.09 |
| GMSD (zenmetrics port of libgmsd) | 256² | 2.75 (2.61–3.34) | 1.56 | — | — | 0.065 | 42.21 |
| SSIMULACRA2 (fast-ssim2) | 256² | 6.34 (6.17–7.01) | 1.56 | — | — | 3.518 | 1.80 |
| butteraugli (max + 3-norm, one call) | 256² | 8.38 (8.05–9.32) | 1.55 | — | — | 4.990 | 1.68 |
| contended rounds / warm launches | 256² | 20 / 20 | | | | 18 | |
| process floor (no work) | 1024² | 1.09 | | | | | |
| zensim B (unreleased default) | 1024² | 70.98 (69.34–72.44) | 25.26 | 0.00 | 43.86 | 41.276 | 1.72 |
| zensim D | 1024² | 55.17 (54.48–56.54) | 25.17 | 0.00 | 28.59 | 28.574 | 1.93 |
| zensim Rev3 fast (Y60 ×5 ensemble) | 1024² | 40.12 (39.52–41.02) | 25.20 | 0.26 | 13.48 | 12.953 | 3.10 |
| GMSD (zenmetrics port of libgmsd) | 1024² | 27.49 (27.18–29.02) | 25.13 | — | — | 0.955 | 28.78 |
| SSIMULACRA2 (fast-ssim2) | 1024² | 105.21 (101.67–110.64) | 25.03 | — | — | 72.294 | 1.46 |
| butteraugli (max + 3-norm, one call) | 1024² | 134.20 (128.79–140.40) | 25.09 | — | — | 93.401 | 1.44 |
| contended rounds / warm launches | 1024² | 20 / 20 | | | | 18 | |
| process floor (no work) | 4096² | 0.96 | | | | | |
| zensim B (unreleased default) | 4096² | 1066.24 (1062.26–1075.45) | 346.71 | 0.00 | 702.04 | 705.920 | 1.51 |
| zensim D | 4096² | 722.60 (720.06–739.03) | 347.12 | 0.00 | 368.71 | 369.438 | 1.96 |
| zensim Rev3 fast (Y60 ×5 ensemble) | 4096² | 521.53 (520.95–528.72) | 347.25 | 0.28 | 168.33 | 170.181 | 3.06 |
| GMSD (zenmetrics port of libgmsd) | 4096² | 368.01 (366.67–370.53) | 346.92 | — | — | 16.209 | 22.70 |
| SSIMULACRA2 (fast-ssim2) | 4096² | 1760.57 (1734.50–1780.45) | 346.78 | — | — | 1415.229 | 1.24 |
| butteraugli (max + 3-norm, one call) | 4096² | 2501.46 (2486.80–2563.33) | 347.27 | — | — | 2175.201 | 1.15 |
| contended rounds / warm launches | 4096² | 10 / 10 | | | | 18 | |

### Cold vs warm (cold_warm_8t.json)

| metric | size | cold launch median (p10–p90) ms | decode | construct | first score | warm per-call median ms | cold ÷ warm |
|---|---|---:|---:|---:|---:|---:|---:|
| process floor (no work) | 256² | 0.99 | | | | | |
| zensim B (unreleased default) | 256² | 4.11 (3.46–5.81) | 1.63 | 0.00 | 1.13 | 0.605 | 6.80 |
| zensim D | 256² | 3.92 (3.24–5.84) | 1.64 | 0.00 | 1.07 | 0.361 | 10.86 |
| zensim Rev3 fast (Y60 ×5 ensemble) | 256² | 4.08 (3.80–5.78) | 1.62 | 0.25 | 1.04 | 0.413 | 9.87 |
| GMSD (zenmetrics port of libgmsd) | 256² | 2.89 (2.56–4.21) | 1.63 | — | — | 0.035 | 82.23 |
| SSIMULACRA2 (fast-ssim2) | 256² | 7.20 (6.61–8.96) | 1.64 | — | — | 3.538 | 2.03 |
| butteraugli (max + 3-norm, one call) | 256² | 7.48 (5.68–9.37) | 1.63 | — | — | 2.453 | 3.05 |
| contended rounds / warm launches | 256² | 20 / 20 | | | | 18 | |
| process floor (no work) | 1024² | 1.06 | | | | | |
| zensim B (unreleased default) | 1024² | 34.20 (33.79–34.85) | 25.25 | 0.00 | 7.17 | 6.696 | 5.11 |
| zensim D | 1024² | 33.08 (32.81–33.71) | 25.27 | 0.00 | 6.72 | 5.682 | 5.82 |
| zensim Rev3 fast (Y60 ×5 ensemble) | 1024² | 31.56 (31.27–32.18) | 25.39 | 0.26 | 4.85 | 4.112 | 7.67 |
| GMSD (zenmetrics port of libgmsd) | 1024² | 26.88 (26.51–27.50) | 25.27 | — | — | 0.129 | 208.85 |
| SSIMULACRA2 (fast-ssim2) | 1024² | 103.03 (99.61–105.80) | 25.24 | — | — | 70.161 | 1.47 |
| butteraugli (max + 3-norm, one call) | 1024² | 77.95 (73.29–82.07) | 25.08 | — | — | 34.796 | 2.24 |
| contended rounds / warm launches | 1024² | 20 / 20 | | | | 18 | |
| process floor (no work) | 4096² | 1.18 | | | | | |
| zensim B (unreleased default) | 4096² | 478.98 (472.43–488.18) | 338.94 | 0.00 | 121.77 | 137.424 | 3.49 |
| zensim D | 4096² | 433.03 (429.79–446.96) | 339.20 | 0.00 | 87.23 | 90.527 | 4.78 |
| zensim Rev3 fast (Y60 ×5 ensemble) | 4096² | 398.73 (395.70–409.33) | 341.07 | 0.27 | 48.66 | 51.937 | 7.68 |
| GMSD (zenmetrics port of libgmsd) | 4096² | 344.91 (343.85–353.97) | 338.53 | — | — | 2.111 | 163.35 |
| SSIMULACRA2 (fast-ssim2) | 4096² | 1730.89 (1713.44–1752.30) | 340.67 | — | — | 1408.774 | 1.23 |
| butteraugli (max + 3-norm, one call) | 4096² | 1306.12 (1299.21–1402.46) | 340.40 | — | — | 991.140 | 1.32 |
| contended rounds / warm launches | 4096² | 10 / 10 | | | | 18 | |



## Method

### One binary, every arm

All three measurements run through zenmetrics' single-call heaptrack driver
`benchmarks/heaptrack/drivers/cpu_profile` (`cpu-profile <metric> <mode> W H`, one
metric call per process — the driver the GMSD lane's heaptrack matrix used). This lane
extended it (zenmetrics workspace `paper-memory`, `src/paper.rs`, private to the
driver, not pushed) with: the zensim profiles PreviewV0_2 / B / D, the two frozen Rev3
ensembles, the CVVDP display presets, DVIFM-ish, a real-pair input, and the `map`,
`ref_only`, `warm_ref`, `workers<K>`, `stream`, `rep_n<K>` and `rep_t<S>` modes. Every
arm is timed or allocated from the same interleaved sRGB8 pair buffers, so no arm pays
a conversion another is spared, and the cold path decodes the same PNGs for every arm.

| arm | implementation and configuration |
|---|---|
| zensim PreviewV0_2 / B / D | `Zensim::new(profile)` at zensim main@origin `ffc14647`; process revision unset = Rev1, the served revision of the named profiles |
| zensim Rev3 fast / rich | `BakeScorer::ensemble` over the five frozen R915 members each (`R915_y60_h32_s*`, `R915_basic228_h128_s*`, weights 0.2), sha256 verified against `FROZEN.json` before the runs (identities in `speed_matrix_2026-09-18.notes.md`); map path = `prepare_steering(bin 8)` + one comparison |
| SSIMULACRA2 | fast-ssim2 (resolved version recorded below), features `imgref` as zenmetrics' consumers build it (no `rayon`) |
| butteraugli | butteraugli (local sibling, default features `rayon`+`avx512`); ONE call returns both the max-norm and the libjxl 3-norm, so both norms share every memory and time number |
| DSSIM | dssim-core (resolved version recorded below), `threads`; map = `set_save_ssim_maps(1)` |
| IW-SSIM | zenmetrics `iwssim` port, with its `parallel` feature (zenmetrics' consumers build it `std`-only; the driver used to as well) |
| CVVDP | zenmetrics CPU `cvvdp` port with its `parallel` feature, at BOTH `standard_4k` (the board's display) and `standard_fhd` (the JPEG AIC display) — photometry and geometry from the vendored `display_models.json` |
| GMSD | zenmetrics `crates/gmsd` at `212603c6` (the GMSD-opt lane's integer sRGB8 luma + fused decimation, bit-exact to libgmsd per that lane's parity gate), feature `parallel` |
| DVIFM-ish | our reimplementation of the DVIFM configuration described in a public talk (not the author's code), dvifmish commit `49aaf667` — the commit of that lane's final scoring binary `dvifmish-49aaf667` (sha256 `69cd2309…`), read through git so the lane's uncommitted working copy is not compiled. Presets `talk-faithful-luma` (the CLI default) and `serving-gate-ycbcr3` (three planes), float and integer arithmetic |

### Inputs

One real photographic pair (the dvifmish speed fixtures `ref_6048x4032.png` /
`dist_6048x4032.png`), cut to size by the driver's `pair-fixtures`: 256×256,
1024×1024 (1 MP) and 4096×4096 (16 MP) are identical centre crops of both images;
7000×5728 (40 MP) exceeds the 24 MP source, so the centre window is extended by
reflect-101 mirroring at the edges (the dvifmish `speed_fixtures` construction),
applied to reference and distorted alike so the distortion is carried along — no
resampling at any size. Sizes by measurement: memory at 1, 16 and 40 MP; throughput at 1 and
16 MP; cold/warm latency at 256×256, 1 MP and 16 MP. Each size is written both as
PNG (for the cold path) and as raw interleaved `.rgb` (for every other path, so no
decode is inside a measured loop). File hashes are in the pointer file's `SHA256SUMS`.

### Peak memory (`memory_matrix.py`)

One metric call per process, one process per cell (the GMSD lane's `gmsd_matrix.sh`
shape). Two instruments per cell, in separate processes:

* **heaptrack 1.5** — peak heap, recorded as heaptrack_print's rounded figure and as the
  exact byte count (sum of the `--flamegraph-cost-type peak` stacks); the two are
  cross-checked per cell. The driver allocates the two sRGB8 inputs itself
  (2·W·H·3 bytes, one exact-size allocation each), recorded as `inputs_bytes`; the
  metric's own share is `heap_peak_bytes − inputs_bytes`.
* **`/usr/bin/time -v`**, un-instrumented — whole-process maximum resident set size,
  wall, user and system time.

Modes: `full` (one score), `map` (score plus the metric's per-pixel map, where it has
one), `ref_only` / `warm_ref` (reference-side precompute, and one comparison against
it), `strip` / `stream` (the peers' own bounded-memory walkers, zensim B's strip path,
and DVIFM-ish's integer streaming form), and `workers<K>` (K = 1, 4, 8). `full`, `map` and
`strip` cells run at 1 thread (pinned to cpu 2) and 8 threads (cpus 0–7). A content
control repeats every `full` arm at 1 MP / 8 threads on the GMSD lane's synthetic pair.
1-min loadavg is recorded
before and after every cell. Memory is measured at every listed size; no size is
derived from another.

### N-process throughput (`nproc_throughput.py`)

N independent single-threaded processes (`RAYON_NUM_THREADS=1`) started together, one
per core, each `cpu-profile <metric> rep_t<S> W H` on the same raw pair: construct the
scorer once, one untimed warm-up call, then score until S seconds have passed. Pinning:
N=1 on cpu 2; N≤8 on cpus 0..N−1 (CCD0, the 96 MiB L3); N=16 on cpus 0–15 (both CCDs,
no SMT siblings). Reported per cell: `pairs_per_s` (sum of each process's calls over
its own loop seconds), `overlap` (fraction of the window every loop ran concurrently)
and `rss_sum_kib` (sum of max RSS, an upper bound on the concurrent total). A cell
whose projected memory (N × the N=1 process's max RSS) exceeds the 36 GiB budget is
not run and is listed as such.

### Cold vs warm latency (`cold_warm.py`)

The GMSD lane's cold protocol (arms round-robin in a shuffled order per round, two
discarded warm-up launches per arm, a `floor` arm = the same binary with no arguments)
driven through `cpu-profile`, so every arm pays the same PNG decode.
**COLD** = wall time of one launch on the PNG pair: process start, loading, two PNG
decodes, conversion to RGB8, scorer construction and one score; the driver splits it
into `t_synth_ms` (read + decode), `t_setup_ms` (construction) and `t_first_ms` (the
score). **WARM** = `rep_n<K>` on the raw pair: per-call median and p95 over K calls
after the first, from launches interleaved across arms; the median of the launch
medians is reported with its spread. Run at 1 thread (cpu 2) and 8 threads (cpus 0–7).

### Quiet-box rule

The box is shared. Timing and throughput cells are not skipped under load; they are
flagged. Per cell or round: /proc/stat busy core-seconds minus the measured processes'
own user+sys = `foreign_core_s`. A cell is **CONTENDED** when that averages more than
0.5 of a core over the cell, or when the 1-s pre-gate still saw more than 1.0 busy core
after waiting up to 60 s. CONTENDED numbers are reported with the flag and never mixed
unflagged into a comparison. Memory cells record loadavg only: peak heap and max RSS
are allocation measures, not timings, and the wall/user/sys times `time -v` prints in a
memory cell are never used as speed numbers.
