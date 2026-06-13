# HDR model build-out — session progress + plan (2026-06-13)

Goal (user): build an HDR model for zensim **and** make a better
`zensim-a`. The through-line: one luminance-aware representation
(PU front-end + luminance-CSF) powering both `Profile::A` (SDR) and a
new `PreviewV0_5Hdr` (HDR) — the CVVDP unification the
`docs/HDR_PLAN.md` calls the standing "P0".

## Live baseline (reproduced this session)

`upiq_pu_score` on the 380-pair UPIQ HDR subset (`scripts/upiq_eval.py`):

| profile | HDR-band SROCC |
|---|--:|
| zensim Profile::A | **0.6933** (matches recorded 0.694) |
| PreviewV0_2 | 0.6871 |
| PreviewV0_1 | 0.6525 |

Bar = **PU-SSIM 0.740**; decisive target HDR-VDP-2 0.812; stretch
PU-PieAPP 0.875. Gap to bar ≈ 0.046. Diagnosis on file (and confirmed):
*"it's the representation, not the weights"* — linear + MLP both cluster
0.63–0.69. Two unblocks needed: an HDR **training** corpus, and chunk 3
(luminance-dependent CSF).

## Data unblock (the key finding)

`HDR_PLAN.md` thought only UPIQ's 380 *validation* pairs existed. But
`hdr-corpus-convert` already produced **76 PQ-PNG HDR references** from
the local HEIC/UltraHDR sources:
`/mnt/v/output/imazen-26-png-v3/{1400-nature(47),1200-interiors(20),1000-photos(6),1600-food(3)}/**/*.hdr.png`
(16-bit, cICP transfer=16 PQ, 1.0 = 203-nit ref white). Source corpus
`~/work/codec-corpus/imazen-26/` has 90 HEIC + 33 gain-map JPGs (~62%
reconstruct to HDR; Apple HEICs w/o gain-map params degrade to SDR).

UPIQ stays **validation-only** (same discipline as CID22 for SDR).

## Pipeline validated end-to-end this session

zenmetrics has a wired HDR sweep + faithful scorers. Built with
`--features png,jxl,cpu-metrics,gpu-cvvdp,gpu-butteraugli,gpu-cuda,hdr,sweep`
(binary at `~/work/zen/zenmetrics/target/release/zen-metrics`, 106 MB).
zenmetrics patches `zensim = { path = "../zensim/zensim" }` → builds
against LOCAL zensim (so `compute_pu_linear` is available).

- **Smoke (2 refs):** cvvdp monotone (q70→9.88, q30→9.48 JOD), encoded
  bytes persisted content-addressed, no GPU OOM at 12 MP, ~4.3 s/cell.
- **Spread probe:** `q` is useless — JXL-HDR **floors at ~q16** (q1..16
  byte-identical). The `distance` knob is the lever:
  cvvdp 9.97→8.17, butteraugli_max 2.08→36.1 across distance 1→25.
- **UPIQ val distribution:** HDR images are **1920×1080 (2 MP)**; HDR
  JOD spans only **−2.5→0** (narrow, high-quality band → discriminate
  *within* high quality; rank is the gate).

## REVERSED DECISION: native res, NOT downscaled (downscaling kills the signal)

First plan was downscale to 1920 (val-match + dodge the encoder memory
cap). **Falsified by measurement.** Same nature-glacier ref:

| ref size | cvvdp span (distance 1→25) |
|---|--:|
| native 12 MP | **1.8 JOD** (9.97→8.17) — good signal |
| 2560 (4.9 MP) | 0.08 JOD — flat |
| 1920 (2 MP) | 0.08 JOD — flat |

The distortion signal lives in the **high-frequency camera detail**
(sensor noise / fine texture) that JXL actually struggles with. Any
quality downscale (Lanczos) low-passes it away → JXL then encodes the
smooth result near-losslessly at every quality → cvvdp ≈ 9.9 flat. So
downscaling destroys the training target. (butteraugli_max keeps some
spread ~2→5 but butter is a weaker HDR predictor: 0.628 UPIQ vs cvvdp
0.758 — training toward it caps below today's 0.69.) → **native res.**

The `hdr_pq_downscale` tool (zensim-validate, image+PQ+cICP, tests pass)
is KEPT — it's correct and useful for a future multi-size axis — just
not used for v1.

**Encoder memory cap fix (user-provided lever):** native 12–24 MP HDR
encodes exceed jxl-encoder's default ~2 GiB budget. zenjxl's
`encode_with_metadata` never threaded `ResourceLimits.max_memory_bytes`
into the `EncodeRequest`. Fixed: `codec.rs` now maps
`ResourceLimits.max_memory_bytes → jxl_encoder::Limits::with_max_memory_bytes`
on the request; the HDR sweep (`sweep/hdr.rs encode_jxl_hdr`) sets a
16 GiB ceiling via `.with_limits(ResourceLimits::default().with_max_memory(16<<30))`.
(zenjxl + zenmetrics commits.)

**Open corpus-quality concern:** even native res gave only 1.8 JOD span
on the HF-rich glacier; smoother refs may be flatter. JXL is a *very*
good codec — UPIQ used JPEG/JPEG-XT (worse, more visible artifacts). For
a stronger corpus the real levers are MORE codecs (wire zenavif HDR
encode — currently decode-only in the sweep) and/or harder distortion,
plus more/harder refs. v1 = JXL-native-cvvdp to get a first signal;
multi-codec is the v2 corpus upgrade.

## Feature extraction path (confirmed)

`Zensim::new(ZensimProfile::A).compute_pu_linear(ref_nits, dist_nits,
w,h, w*3, w*3).features()` → **372-dim PU-XYB vector**. MLP profiles
force `compute_all_features` (metric.rs:2242), and `.features()` is
pub/un-gated (metric.rs:749). `NitsImage{pub rgb: Vec<f32>, ...}` (zenmetrics
hdr.rs) is interleaved f32 cd/m² — the exact `compute_pu_linear` input.
`hdr::decode_to_nits` (ref) + `sweep::hdr::decode_encoded_to_nits` (dist
from persisted JXL) are both pub → a small zenmetrics bin/example can
emit features keyed to the sweep's encoded filenames. No sweep-loop
surgery (run.rs HDR path is complex — don't touch it).

## Plan / tasks (TaskList #1–6)

1. ✅ Reproduce UPIQ baseline (0.6933).
2. **HDR corpus**: downscale 76 refs → sweep `--codec zenjxl --hdr
   --metric cvvdp --metric butteraugli-gpu --knob-grid
   '{"distance":[0.5,1,1.5,2,3,4,6,8,11,14,18,25]}' --q-grid 50
   --encoded-out-dir` → scores TSV + encoded artifacts (~912 pairs).
   Staging: `/mnt/v/output/zensim-hdr-train/2026-06-13/`.
3. **Features**: zenmetrics bin → 372 PU features per pair, join to scores.
4. **Train** `PreviewV0_5Hdr` (fine-tune from A or fresh) on features →
   cvvdp/butter target; validate held-out UPIQ. Expect re-fit caps ~0.74.
5. **Chunk 3 CSF** (luminance-dependent per-channel) if capped — the real
   lever; also lifts SDR.
6. **Better Profile::A**: apply shared CSF to SDR path, retrain, validate
   SDR panels + dial. Additive `PreviewV0_5Hdr` is API-safe; any
   `Profile::A` rotation needs methodology doc + user sign-off.

## Repro commands

```
# baseline
cargo build --release -p zensim-validate --bin upiq_pu_score
./target/release/upiq_pu_score --out /tmp/s.csv
python3 scripts/upiq_eval.py --scores /tmp/s.csv --score-col zensim_a

# zenmetrics HDR build
cd ~/work/zen/zenmetrics && cargo build --release -p zen-metrics-cli \
  --features png,jxl,cpu-metrics,gpu-cvvdp,gpu-butteraugli,gpu-cuda,hdr,sweep
```
