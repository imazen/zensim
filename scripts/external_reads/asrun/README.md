# `asrun/` — frozen as-run provenance copies

Byte-identical archival copies of every external-read study's as-run analysis
scripts + pre-registration `PROTOCOL.md`, below a provenance header. See
[`../README.md`](../README.md) for the read families, stored tables, and the
canonical runner. **Do not extend these files — extend the runner**
(`../run_external_reads.py`).

## BLESSED: the `avthdr/` + `hdrvdc/` ffmpeg drivers (user, 2026-09-02)

The workspace rule is **imazen-only imaging/codec software**
(`~/work/zen/CLAUDE.md`). These two directories are a **sanctioned exception**,
blessed as-is by the user on 2026-09-02. The stored numbers stay valid and
citable; this is a provenance caveat, not an invalidation. Registry entry:
`benchmarks/eval_annotations.json` →
**`extreads-ffmpeg-blessed-avthdr-hdrvdc-2026-09-02`** (documentation-only
scope — it applies to no fulleval cell).

**Scope: EVAL-ONLY.** Both domains are external-validation reads. Neither has
ever been training input.

**Why ffmpeg was the route.** Both datasets distribute only coded video, and we
have no imazen video demux/decode path:

| domain | distributed as | decoder(s) used |
|---|---|---|
| `hdrvdc` (HDR-VDC) | **AV1** — SVT-AV1 v1.5.0 preset 4, 10-bit yuv420p, limited range (`hdrvdc/PROTOCOL.md:63`) | ffmpeg **libdav1d** (`:96`) |
| `avthdr` (AVT-VQDB-UHD-1-HDR) | **three** coded codecs — 65 av1 `.mkv` + 65 hevc `.mp4` + 65 vvc `.266` = 195 bitstreams, plus FFVHUFF lossless `.mkv` sources as references (`avthdr/PROTOCOL.md:68`) | **two pinned builds**: system ffmpeg 4.4.2 (libdav1d / native hevc / native ffvhuff), and a BtbN static **n7.1.5** for the native `vvc` decoder that 4.4.2 lacks — vvc runs two-step, raw decode piped into the *same* system-ffmpeg filter chain so csc/scale stays common-mode across codecs (`:131-137`) |

> Note the codec facts above against the shorthand: HDR-VDC is **AV1, not
> HEVC**, and AVT is **three codecs, not one**. Corrected 2026-09-02 by reading
> the drivers.

**ffmpeg's role is wider than demux+decode.** swscale also performs the
YCbCr→R'G'B' conversion (`in_color_matrix=bt2020`, `in_range=tv` →
`out_range=full`, `accurate_rnd+full_chroma_int`, to `rgb48le`) and a **Lanczos
a=3 resample** to the 4K display frame, plus a second 1920×1080 far-viewing leg
for `hdrvdc`. So a foreign colour converter and resampler are in the chain, not
only a decoder. Both are registered common-mode approximations — identical
across every leg, so leg deltas are unaffected.

**Everything downstream of the pixels is imazen-owned:** extraction is zensim's
own `target/release/examples/hdrvdc_features_extract` running
`compute_folded720_append2_features_hdr(..., csfw_block:false)` at mode 944,
profile `codec_target()`; all correlations go through the canonical Rust panel.

## The routine path does not run ffmpeg at all

Extraction was **one-time**. Every rescoring reads the stored pooled
944-feature tables — `hdrvdc_pooled_944.csv` (580×944, sha256 `567731161871559e…`)
and `avthdr_pooled_944.csv` (195×944, sha256 `81d392e72dd67fc0…`):

```sh
python3 scripts/external_reads/run_external_reads.py --from-stored   # default
```

~11 s for the full seven-domain read set, zero video decode.

**A from-scratch re-extraction needs fresh user sign-off** (or an imazen
video-decode capability: AV1 + HEVC + VVC demux/decode, plus zenpixels csc and
zenresize Lanczos). Practical hazard: the vvc leg staged its n7.1.5 binary under
`~/tmp/tools/`, which is volatile scratch and is likely gone.
