# Cross-path audit, 2026-10-28

Raw artifact root: `/mnt/v/output/zensim/crosspath_2026-10-28/`.

These exploratory measurements do not establish universal performance or memory
parity. See [scope and limits](../docs/CROSS_PATH_AUDIT.md).

- `combined/comparisons.tsv`: candidate/control ratios,
  paired process counts and cellwise 95% bootstrap intervals. A 2% non-regression
  margin was used. All original samples remain in follow-up comparisons.
- `combined/memory.tsv`: three fresh-process
  `/usr/bin/time -v` maximum-RSS measurements per cell and binary.

Candidate production change: `4b4ad4736334`, with `zensim/src/diffmap.rs` SHA256
`202d35360da81d1e3e5953538d573d5df546f90b68b2ff9754dfd004537d4033`.
Baseline production: `d15a54862106a3df8e14d2617b2fc7642d85b955`, plus the identical
audit harness. The research source snapshot manifest aggregate SHA256 is
`df35a9214e41a3b117848b04bdf3534249440f330a01fe1ad37ab812ad890b46`.

| Binary | SHA256 |
|---|---|
| candidate | `86b90443858bf08cbea0837a4f0717fad075849ed9740d567623921b03ad1c3c` |
| baseline | `f3c69776d80a446ba5a27157b53b54d22e65793ee5368d0f3e6f7d97777bc127` |
| research | `95d8071e3c8c402f83da3f9c078e3d4e0066348af14dde0583254ddea3f41319` |

Hardware: Intel Core Ultra 7 265K, Linux, enforced performance-core cpuset.
Serial work used one performance core; parallel work used eight. Compiler:
Rust 1.99.0 (`b940084d7`), LLVM 23.1.1. Release builds used thin LTO and runtime
SIMD dispatch, without `target-cpu=native`. Timing engine: zenbench 0.1.11.

Reproduce using frozen executables and the same model bakes, under the resource
wrapper, exclusive heavy-work lock and enforced uniform CPU cpuset:

```sh
just path-audit-paired --arm candidate=BIN/candidate --arm baseline=BIN/baseline \
  --arm research=BIN/research --sizes 65x63,1024x768 --rounds 8 \
  --memory-rounds 3 --output matrix1
# The retained followup-cells.txt identifies the 38 unresolved first-pass cells.
just path-audit-paired --arm candidate=BIN/candidate --arm baseline=BIN/baseline \
  --arm research=BIN/research --sizes 65x63,1024x768 --rounds 24 \
  --memory-rounds 0 --cells "$(cat followup-cells.txt)" --output matrix2
just path-audit-paired --arm candidate=BIN/candidate --arm baseline=BIN/baseline \
  --arm research=BIN/research --sizes 2048x1536 --rounds 0 \
  --memory-rounds 3 --output large-memory
just path-audit-report matrix1 matrix2 large-memory --output combined
```

The research adapter requires `PATH_AUDIT_GENERAL_BAKE` and
`PATH_AUDIT_CODEC_BAKE`. Their SHA256 values are respectively
`e460a214c1e0c4f48bc7e3bb4705df548f9f172807cc63f39ba17cb1d94e05a0` and
`f7ecd60299482210fe33586e8d4674fe9ba163f17f2dbaf54c6e99d5ef44af63`.
Raw packets, source manifests, binaries and invalid probe evidence are retained
outside git; their durable location and confirmation results will be added when
the fixed confirmation run finishes.
