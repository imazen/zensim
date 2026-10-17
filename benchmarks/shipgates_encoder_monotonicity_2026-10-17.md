# SVT encoder amendment and G-ADDR

The owner excludes avif-rav1e from required rows and judges AVIF on the SVT port. Original results remain archived.

The amended ladder has7,097 rows (9,593 minus2,496 rav1e rows). The amended standard grid has4,494 rows:3,024 unchanged non-AVIF rows plus1,470 existing SVT ladder rows for its exact35 AVIF source families. All substituted reference RGB hashes match. SVT keeps its own native quality axis; the standard AVIF sample grid changes from40 to42 rungs per source.

The standard grid records `zenavif` with empty backend knobs. Its contemporary zenavif default is Zenravif (zenrav1e), at historical source `bde705fc0057a008e123f4f2e83f9f39cfde06ad`, `src/encoder.rs`. This identifies the default route; the exact as-run encoder build/backend receipt was not preserved. These legacy AVIF rows remain diagnostics; required AVIF rows now explicitly identify `avif-svt`.

Strict A7r comes from the frozen Rust evaluator, SHA256 `3be3ef6fa3b123b50ccdbbe65692608767d3b2a0c1a876cec76afccfa4efd4b7`. It selects four rungs from the SSIMULACRA2 mentor using a0.5-point materiality margin and requires strict increases plus the registered off-clamp condition. Its bar is the mentor’s own strict fraction on the same derived grid. A1–A6 and A8r retain their report-only role; this recompute does not qualify the contract tier.

Direction agreement is diagnostic only. It uses the same selected edges and clamp condition, accepts a candidate change with the mentor’s sign, and treats mentor changes smaller than0.5 as ties. No selected mentor edge is a tie here. Both ladder-level agreement and individual material-edge agreement are shown. It changes no gate or threshold.

Every reconstructed strict decision/count matches the native owner. Peer scores retain the original rendered-pixel binding; CVVDP-FHD retains its frozen Q=10×JOD mapping. No labels, new encodes or builds were used.

| Grid | W-C | diff+frag-820 | diff-410 | CVVDP-FHD | SSIMULACRA2 |
|---|---|---|---|---|---|
| standard | FAIL | FAIL | FAIL | PASS | PASS |
| ladder | PASS | PASS | FAIL | PASS | PASS |

## standard

| Metric | Codec | Strict represented | Mentor strict bar | Strict verdict | Direction represented (diagnostic) | Agreeing material edges |
|---|---|---:|---:|---|---:|---:|
| W-C | avif-svt | 35/35 | 35/35 | PASS | 35/35 | 105/105 |
| W-C | jpeg | 20/22 | 21/22 | FAIL | 21/22 | 65/66 |
| W-C | jxl | 33/33 | 33/33 | PASS | 33/33 | 99/99 |
| W-C | webp | 16/16 | 16/16 | PASS | 16/16 | 48/48 |
| diff+frag-820 | avif-svt | 35/35 | 35/35 | PASS | 35/35 | 105/105 |
| diff+frag-820 | jpeg | 20/22 | 21/22 | FAIL | 21/22 | 65/66 |
| diff+frag-820 | jxl | 33/33 | 33/33 | PASS | 33/33 | 99/99 |
| diff+frag-820 | webp | 16/16 | 16/16 | PASS | 16/16 | 48/48 |
| diff-410 | avif-svt | 34/35 | 35/35 | FAIL | 34/35 | 104/105 |
| diff-410 | jpeg | 20/22 | 21/22 | FAIL | 21/22 | 65/66 |
| diff-410 | jxl | 33/33 | 33/33 | PASS | 33/33 | 99/99 |
| diff-410 | webp | 16/16 | 16/16 | PASS | 16/16 | 48/48 |
| CVVDP-FHD | avif-svt | 35/35 | 35/35 | PASS | 35/35 | 105/105 |
| CVVDP-FHD | jpeg | 21/22 | 21/22 | PASS | 22/22 | 66/66 |
| CVVDP-FHD | jxl | 33/33 | 33/33 | PASS | 33/33 | 99/99 |
| CVVDP-FHD | webp | 16/16 | 16/16 | PASS | 16/16 | 48/48 |
| SSIMULACRA2 | avif-svt | 35/35 | 35/35 | PASS | 35/35 | 105/105 |
| SSIMULACRA2 | jpeg | 21/22 | 21/22 | PASS | 22/22 | 66/66 |
| SSIMULACRA2 | jxl | 33/33 | 33/33 | PASS | 33/33 | 99/99 |
| SSIMULACRA2 | webp | 16/16 | 16/16 | PASS | 16/16 | 48/48 |

## ladder

| Metric | Codec | Strict represented | Mentor strict bar | Strict verdict | Direction represented (diagnostic) | Agreeing material edges |
|---|---|---:|---:|---|---:|---:|
| W-C | avif-svt | 39/39 | 39/39 | PASS | 39/39 | 117/117 |
| W-C | jpeg | 28/39 | 26/39 | PASS | 34/39 | 112/117 |
| W-C | jxl | 25/26 | 25/26 | PASS | 24/26 | 76/78 |
| W-C | webp | 39/39 | 39/39 | PASS | 39/39 | 117/117 |
| diff+frag-820 | avif-svt | 39/39 | 39/39 | PASS | 39/39 | 117/117 |
| diff+frag-820 | jpeg | 28/39 | 26/39 | PASS | 34/39 | 112/117 |
| diff+frag-820 | jxl | 25/26 | 25/26 | PASS | 24/26 | 76/78 |
| diff+frag-820 | webp | 39/39 | 39/39 | PASS | 39/39 | 117/117 |
| diff-410 | avif-svt | 38/39 | 39/39 | FAIL | 38/39 | 116/117 |
| diff-410 | jpeg | 25/39 | 26/39 | FAIL | 36/39 | 114/117 |
| diff-410 | jxl | 25/26 | 25/26 | PASS | 24/26 | 76/78 |
| diff-410 | webp | 39/39 | 39/39 | PASS | 39/39 | 117/117 |
| CVVDP-FHD | avif-svt | 39/39 | 39/39 | PASS | 39/39 | 117/117 |
| CVVDP-FHD | jpeg | 32/39 | 26/39 | PASS | 29/39 | 105/117 |
| CVVDP-FHD | jxl | 26/26 | 25/26 | PASS | 25/26 | 77/78 |
| CVVDP-FHD | webp | 39/39 | 39/39 | PASS | 39/39 | 117/117 |
| SSIMULACRA2 | avif-svt | 39/39 | 39/39 | PASS | 39/39 | 117/117 |
| SSIMULACRA2 | jpeg | 26/39 | 26/39 | PASS | 39/39 | 117/117 |
| SSIMULACRA2 | jxl | 25/26 | 25/26 | PASS | 26/26 | 78/78 |
| SSIMULACRA2 | webp | 39/39 | 39/39 | PASS | 39/39 | 117/117 |

All original and amended rows, including excluded rav1e and legacy standard AVIF diagnostics, are in [the complete row table](shipgates_encoder_monotonicity_2026-10-17.tsv). Direction agreement can decrease as well as increase: on the ladder’s JXL row the candidates change from25/26 strict to24/26 direction agreement; the two criteria answer different questions.

## Remaining AVIF work

G-RD and G-TARGET require zenav1-svt (`svt-rs`) only. Completed Zenravif smoke cells remain diagnostics. The current native CQ/map driver explicitly selects Zenravif and4:4:4; the SVT seam requires4:2:0 and has no map hookup in that driver. The dispatcher refuses further AVIF work through the legacy driver. SVT map steering and the targeting adapter still need implementation and end-to-end validation; no SVT RD or targeting pass is claimed. JXL remains paused for priority generation.

## Evidence

Source: `c954661253fe`; frozen evaluator and model pins, derived-grid hashes and exact selected edges are bound in [the campaign pointer](shipgates_2026-10-16.pointer.md). Canonical RESULT SHA256 `2f468bf22f10561995945e4cfe28738e9e8edb06b698f064c26e742c9a9c59ff`.

```text
run-heavy: done rc=0 10s | peak-RSS 0.40GiB | min-avail 22828MiB | peak-load 34.27
```

This was a one-thread,4GiB light recompute without the paper lock. The resource line describes this contended run, not a quiet speed qualification.
