# HDRVID — Rev5 re-decode of the external HDR video panels (2026-10-10)

Owner, verbatim 2026-10-10: "do HDR-VDC with rav1d-safe now, and non-av1
videos with ffmpeg 8.1". This replaces the deleted July frames
(`../asrun/{hdrvdc,avthdr}`) with retained, hash-pinned display frames and
Rev5 by_v2fy-420 features for E31's registered external reports. Both
panels stay **external-eval, report-only**; nothing here is training data.

## Pipeline

| step | owner | notes |
|---|---|---|
| inventory | `decode.py plan` | file names + README grammar, sha256, ffprobe 8.1.3 metadata and packet counts; no label file |
| AV1 decode (all HDR-VDC, AVT av1) | `tools/hdrvid_decode` → rav1d-safe `f3132ee6` | ffmpeg 8.1.3 only stream-copies to IVF (`-c copy -f ivf`) |
| HEVC / VVC / FFVHUFF decode | ffmpeg 8.1.3, native decoders | `-f rawvideo -pix_fmt yuv420p10le`, built without swscale, so no conversion is possible |
| YCbCr → R'G'B' | zenavif `85dd0d2b` `yuv_convert` | BT.2020 NCL, limited → full, {9,3,3,1}/16 chroma, fixed point to 10-bit codes |
| display frame | zenresize `e3975fb9` `Filter::Lanczos` (a=3) | on PQ code values; 3840×2160; display-size frames pass through |
| far leg (HDR-VDC) | zenresize Lanczos-3 | 1920×1080 from the stored 16-bit 4K frame |
| storage | zenpng `37c942ed` | 16-bit RGB PNG, `round(clamp(v,0,1)·65535)` |
| features | `zensim-validate` `hdrvid_extract` | `research::extract_hdr`, `HdrEncoding::Pq{peak}`, Rev5, by_v2fy 420 IDs, 1825-slot f64 transport |
| tables | `extract.py tables` | per (video, config) mean of the eight frame vectors |
| report | `run_e31video.py` → `v40_panels.py --mode e31video` | frozen V40 packet runtime, binaries, cells and control pins |

Frames: `N` = the reference's packet count; kept frames
`floor((j+0.5)N/8)`, j = 0..7; every test must decode exactly `N` frames.
HDR-VDC configs (July registration): A 4K Pq{1000}; B 4K Pq{700};
C 4K Pq{700} dimmed; D 1080p Pq{700}; E 1080p Pq{700} dimmed. Dimming is
the experiment shader (PQ decode at 10 000 cd/m², ÷8, re-encode) on both
images, via linear-srgb's ST 2084 functions. AVT uses A only.

## FFmpeg 8.1.3 build

Source `https://ffmpeg.org/releases/ffmpeg-8.1.3.tar.xz`, SHA-256
`7138d28c96d9d3e3af4ee3d8cad72741f8ffb40da90c1112235dea3ecd3178a3`,
signature verified against the FFmpeg release key
`FCF9 86EA 15E6 E293 A564 4F10 B432 2F04 D676 58D8`. Configure:

```
--disable-everything --disable-autodetect --disable-doc --disable-network
--disable-swscale --disable-swresample --enable-static --disable-shared
--enable-decoder=hevc,vvc,ffvhuff --enable-encoder=rawvideo
--enable-demuxer=matroska,mov,vvc,ivf --enable-muxer=rawvideo,ivf,null,obu
--enable-parser=hevc,vvc,av1
--enable-bsf=av1_metadata,av1_frame_merge,av1_frame_split,hevc_mp4toannexb,vvc_mp4toannexb
--enable-protocol=file,pipe --enable-filter=null,format,select,setpts,trim
```

## Differences from the July chain (registered approximations)

July used libdav1d / ffmpeg 4.4.2 and n7.1.5 decoders and swscale
(`accurate_rnd+full_chroma_int` rgb48, `flags=lanczos`) for conversion and
resampling, and f64 exact ST 2084 for dimming. HDRVID uses rav1d-safe,
ffmpeg 8.1.3 decode-only, the zenavif recipe (10-bit RGB codes), zenresize
Lanczos-3 in f32, and linear-srgb's f32 ST 2084. The cross-checks quantify
these differences; see the results pointer.
