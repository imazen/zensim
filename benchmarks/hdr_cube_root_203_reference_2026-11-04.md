# Anchored HDR cube-root reference — 2026-11-04

The public HDR fixtures are in-memory strided f32-nits / f32-code / u16-code buffers.
The research `hdr_continuity` example accepts only 8-bit RGB/RGBA PNGs, so it cannot
consume these buffers directly. A scratch program instead used the research
`BakeScorer::with_hdr_front_end(CubeRoot { unit_nits: 203.0 })` on exactly the
public fixture bytes. No file was added to or changed in the research checkout;
Cargo's target and lockfile lived under scratch.

Reference: imazen/zensimx `48a4bd3f152c059c9c55ef535add102bc9c65d4e`, verified clean before and after the run.
The front end comes from `94095e060047cbae3c257188f4f9e1b2a785ea54` and its bake
scorer integration from `41ea3ecfc0da147e2b4525b3225f52ea3dc44c6a`.

- A bake SHA-256: `d5e2a56db35c32620d9f9d9c19ba3bc78bd8fea2b8cd933cde8447428d85e62a`.
- Public fixture generator SHA-256: `992ba71f79cd3f58369a4176317b5dedea92afbf9ad9a955a11c19b944ff8f44`.
- Reference: [scratch program source](hdr_cube_root_203_reference_2026-11-04.rs).
- Output: [166 golden rows](../zensim/tests/fixtures/hdr_a_cube_root_203.tsv).

To reproduce, copy the reference program as `src/main.rs` in a standalone scratch
Cargo package. Set `ZENSIM_HDR_FIXTURES` to the absolute path of
`zensim/examples/split_parity/hdr.rs` in the public checkout. Its manifest uses edition 2024 and these dependencies:

```toml
[workspace]
[dependencies]
zensim = { path = "<read-only-zensimx>/zensim", default-features = false, features = ["avx512", "threads", "feature-regime-v2"] }
zenpredict = { git = "https://github.com/imazen/zenanalyze", rev = "f0903489300aea240ffae2e3f1359f8f2b9a0ba4" }
```

Set `TMPDIR` and `CARGO_TARGET_DIR` to disk-backed scratch paths, set
`ZENSIM_FORMULA_REV=7`, and run this under the exclusive heavy-work lock and
resource wrapper (16 GiB hard cap):

```sh
cargo run --release --manifest-path <scratch>/Cargo.toml -- <public-crate>/zensim/weights/rev7_mix_h128_2026-11-03.bin
```

All 166 resulting score-bit values match the public `compute_hdr` test exactly.
This covers the full existing fixture matrix, including identity cases.

The public continuity test uses the research example's sRGB EOTF and 203 cd/m²
normalization on 14 existing SDR synthetic pairs, each serial and parallel.
Maximum absolute score gap: `0.00055716912091697`; each comparison must be at most
`1e-3`. The separate [research PNG panel](https://github.com/imazen/zensimx/blob/41ea3ecfc0da147e2b4525b3225f52ea3dc44c6a/benchmarks/hc1_continuity_2026-11-04.tsv)
reported a rounded maximum `0.0003` for F seed 0 over 24 pairs; that observation
is specific to its fixture panel.

Resource-wrapper diagnostics (correctness runs, not performance comparisons):

```text
research oracle: run-heavy: done rc=0 34s | peak-RSS 1.14GiB | min-avail 37914MiB | peak-load 4.91
public HDR tests: run-heavy: done rc=0 10s | peak-RSS 1.08GiB | min-avail 37820MiB | peak-load 8.37
```
