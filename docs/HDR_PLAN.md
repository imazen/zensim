# zensim HDR support — process, plan, and test methodology

_Synthesized 2026-06-01 from the `zenpapers` corpus (4-agent fan-out over
`docs/iqa-methods/`, the converted-paper markdown under `/mnt/v/input/papers/`,
and the on-disk datasets). Every constant below is either a public ITU/SMPTE
spec value or cited to a corpus file. Where a value is not in the corpus, the
canonical source is named — do not fabricate._

## 0. TL;DR

To score HDR (PQ/HLG) content correctly, an IQA metric must stop working in
relative `[0,1]` and work in **absolute display luminance (cd/m²)**, then either
(1) **PU-encode** that luminance and run the existing SDR metric, or (2) feed
absolute luminance into a **luminance-dependent CSF** (CVVDP/HDR-VDP). zensim
already has the opponent-color bones (XYB ≈ CVVDP's DKL) and a cube-root
nonlinearity that is structurally the slot PU occupies. The pragmatic,
field-validated path is **PU-encode the existing XYB metric**; the
luminance-dependent CSF is a follow-on that also improves SDR.

The binding constraint is **data**: only ~380 HDR codec pairs exist on disk
(UPIQ), enough to *validate/calibrate* but not to *train*. A trained
`PreviewV0_5Hdr` waits on a generated HDR codec sweep (and/or the unreleased
AIC-HDR2025).

## 1. The process — how CVVDP & friends added HDR support

### The universal enabling step: absolute luminance via a display model
A pixel code value becomes emitted display light through a transfer function +
display geometry (peak/black/ambient):

- **PQ / SMPTE ST 2084** — EOTF, peak 10000 cd/m². `code 0.5 → 92.25`, `1.0 → 10000`.
- **HLG / ITU-R BT.2100** — inverse-OETF (per channel) + OOTF system gamma
  (depends on the RGB triple's luminance; `γ = 1.2` at 1000-nit peak).
- **SDR sRGB / BT.709** — `L = (Y_peak − Y_black)·EOTF_sRGB(V) + Y_black + L_refl`.
  zensim's existing `standard_4k`: 200 / 0.2 / 0.3979 → `code 128 → 43.73 cd/m²`.

### Two families
1. **PU-encode an SDR metric** (PU-SSIM, PU-PSNR, PU-PieAPP, HDR-VDP front-end,
   UPIQ). Apply a **Perceptually-Uniform encoding** `PU21(L)` — the integral of
   1/(luminance detection threshold), *constrained so SDR sRGB at 0.1–80 cd/m²
   maps back to ~0–255*. The SDR metric then runs unchanged; **PU replaces the
   gamma/decode nonlinearity** (it is the only nonlinearity between absolute
   luminance and the metric, not an extra stage). PU is luminance-only; color is
   handled by the opponent transform. PU08 = Aydın 2008 (t.v.i.-derived);
   **PU21 = Mantiuk & Azimi 2021** (CSF-derived).
2. **Native luminance-dependent CSF** (CVVDP, HDR-VDP). The pipeline is
   *identical* SDR↔HDR; the HDR-specific part is that the **CSF reads local
   absolute adaptation luminance** — peak sensitivity and peak spatial frequency
   `ρ_m(Y)` both shift with luminance. Everything else (DKL transform, masking,
   Minkowski pooling β=2/2/4/4, JOD output) is luminance-agnostic.

### Key field result (do not skip)
A good SDR metric + PU-encoding is a **very strong baseline**; native-HDR
metrics do not reliably beat it on professional HDR-WCG content (Sugito 2022;
ZJUHDR 2025). CVVDP's edge comes mostly from **color + masking**, not HDR per
se. → Start with PU-encoding; treat the native luminance-CSF as the
quality enhancement, not the entry ticket.

### Corpus citations
- UPIQ (the SDR+HDR unification): Mikhailiuk et al. 2021, arXiv 2012.10758 →
  `/mnt/v/input/papers/68/6845a362…md`.
- PU21: Mantiuk & Azimi 2021 → `/mnt/v/input/papers/8c/8c72e282…md`.
- CVVDP: Mantiuk et al. 2024 → `/mnt/v/input/papers/84/847a1669…md`.
- HDR-VDP-3 → `/mnt/v/input/papers/35/3507a04c…md`; stelaCSF `13/135eb3e4…`;
  castleCSF `b9/b978a0a6…`.
- Synthesis (the one doc that ties it to zensim):
  `zenpapers/docs/iqa-methods/vdp-csf-perceptual-math.md`; UPIQ realignment in
  `…/subjective-scaling-jod.md`.

## 2. Where zensim sits — the exact pipeline change

zensim today (`zensim/src/color.rs`): sRGB→linear→**XYB** (opsin matrix +
cube-root), on relative `[0,1]` input. Its XYB opponent space ≈ CVVDP's DKL
(`Ach↔Y, RG↔X, YV↔B`), and the cube-root is the slot PU occupies.

**Minimal correct insertion (PU path):**
```
linear RGB (relative [0,1])
  → [NEW: display model]  scale to absolute cd/m²   (transfer.rs — DONE, chunk 1)
  → opsin matrix (unchanged — linear LMS mix, luminance-scale-agnostic)
  → [PU21 replaces cube-root]  per channel on absolute luminance
  → XYB-like PU planes → existing feature bank
```
PU does **not** precede the opsin matrix (it expects linear input). The minimal
variant applies PU to the Y channel only (color handled separately, per PU21);
the fuller variant replaces cube-root on all three channels.

**Three gaps (cost):**

| Gap | Fix | Cost | Status |
|---|---|---|---|
| No absolute-luminance input | display model relative→cd/m² (`transfer.rs`) | cheap | **DONE (chunk 1)** |
| Cube-root is luminance-agnostic | `PU21(L)` swap on the absolute path | moderate | chunk 2 |
| Constant 4-tap CSF `[0.5,1.0,0.8,0.4]` (`cvvdp_features.rs:102`) | luminance-dependent, **per-channel** CSF, seed `s_ch=[1.0(Y),1.7(X),0.237(B)]` | moderate — **also lifts SDR** (standing P0) | chunk 3 |

**Reuse:** zenmetrics ships a pure-Rust CVVDP (`crates/cvvdp/src/params.rs:203-257`)
with the verified PQ/HLG/sRGB display-model constants and a conformance harness
(`cvvdp-conformance`, vs pycvvdp goldens @ 1e-3 JOD). The display-model math is
plain ITU constants (reusable); the **castleCSF LUT is AGPL** (`cvvdp-gpu`) — do
NOT copy it; keep zensim's own CSF.

**PR #39 (`feat/hdr-iqa-source`)** adds the `ColorTransferFunction` signal
(Srgb/Linear/Bt709/Pq/Hlg) + the HDR-refusal guard. That guard is the
prerequisite to *lift* on the PU path; `ColorTransferFunction` is how a source
selects its display model. The HDR work is the reason #39 exists — merge it.
(The `display::DisplayProfile` module — `peak_nits`/`ambient_lux`/`ppd` — flagged
"speculative/unused" in the 2026-06-01 release audit, is the display-model home.)

## 3. Verified constants (spec-exact; implemented in `transfer.rs`)

- **PQ ST 2084 EOTF**: `m1=2610/16384=0.15930175`, `m2=2523/4096·128=78.84375`,
  `c1=3424/4096=0.8359375`, `c2=2413/4096·32=18.851562`, `c3=2392/4096·32=18.6875`,
  `L_max=10000`.
- **HLG BT.2100 inverse-OETF**: `a=0.17883277`, `b=1−4a=0.28466892`,
  `c=0.5−a·ln(4a)=0.55991073`; lower segment `v²/3` for `v≤0.5`.
- **HLG system gamma**: `1.2` for peak ≤1000; else `1.2 + 0.42·log10(peak/1000) − 0.07623·log10(amb/5)`.
- **sRGB display model (`standard_4k`)**: `Y_peak=200, Y_black=0.2, Y_refl=0.39788736`
  (= 250 lux · 0.005 / π).
- **PU21 coefficients**: NOT in corpus. Canonical source: `gfxdisp/pu21`
  (`pu21_encoder.{m,py}`, per-variant 7-param fits `banding`/`banding_glare`).
  **Fetch before chunk 2.**
- **CVVDP per-channel color weights** `s_ch = [1.0 (Ach), 1.7 (RG), 0.237 (YV)]`
  (CVVDP paper, verified) — seeds the chunk-3 per-channel CSF.

## 4. Test methodology

### Layer 1 — reference-parity unit tests (no training data; all platforms incl. i686/wasm)
Display-model + transfer invariants vs published ITU/SMPTE values. **DONE** in
`transfer.rs::tests`: `pq_eotf(0.5)=92.25`, `pq_eotf(1.0)=10000`,
`srgb→43.73 nits @128`, `hlg(0.5)=1/12`, `hlg_system_gamma(1000)=1.2`, PQ
monotonicity, HLG segment continuity. Next: PU21 round-trip + monotonicity
(self-consistent invariant — no external golden needed); ULP-parity of the
front-end against zenmetrics' cvvdp color path (which tests at 1e-3–1e-4).

### Layer 2 — metric parity vs pycvvdp (runnable locally)
`pycvvdp 0.5.4` is installed (`/home/lilith/.local/bin/cvvdp`, verified → 8.78 JOD).
Use it as the golden generator. **Compare rank only (SRCC/KRCC)** — never raw
JOD vs zensim's 0–100 (different scales). Golden-JSON pattern:
`zenmetrics/crates/cvvdp-conformance/`. CLI:
`cvvdp -t dist -r ref -d standard_hdr_pq --device cpu -m cvvdp --result out.csv`.

### Layer 3 — subjective validation on UPIQ
Full Mohammadi 2025 panel (SROCC+PLCC+KROCC+OR+PWRC+Z-RMSE) after the VQEG
5-param logistic → JOD (UPIQ ships `fit_logistic.m`). **Stratify HDR vs SDR**
(`is_hdr` column) **and by luminance band** (the HDR-specific failure mode is the
highlight band). Baselines to beat (precomputed in `upiq_objective_scores.csv`):
**PU-SSIM 0.70 / HDR-VDP-2 0.81 / PU-FSIM 0.84 → PU-PieAPP 0.945** (the bar).
Krasula AUC ladder to match: PU-PieAPP 0.92 > PU-FSIM 0.83 > HDR-VDP-2 0.71.

### CI gating (no graceful skips)
- **Cheap/deterministic → run everywhere:** all transfer/display invariants, PU
  round-trip, front-end ULP-parity, the stats panel on a fixed table.
- **Gated (caller-controlled via env var, CI workflow → justfile → test):**
  golden regeneration from the live `cvvdp` CLI (needs pycvvdp+torch); full UPIQ
  EXR validation (2.4 GB decode) is a benchmark job, results committed to
  `benchmarks/`.

## 5. Data reality

| Dataset | HDR codec pairs on disk | Distortion | Subjective | Role |
|---|---|---|---|---|
| **UPIQ** (Narwaria+Korshunov) | **380 / 30 ref** — EXR inside `/mnt/v/datasets/upiq_dataset.zip` (2.4 GB, unextracted); scores in `/mnt/v/datasets/upiq/` | JPEG-MSE/SSIM, JPEG-XT A/B/C | JOD | **held-out HDR validation + JOD calibration** |
| UPIQ SDR (TID2013+LIVE) | 3779 / 54 | synthetic + codec | JOD | SDR train / scale-anchor |
| **AIC-HDR2025** (Jenadeleh/Sneyers 2025, arXiv 2506.12505) | **0 — not released** (README-only at `/mnt/v/datasets/aic-hdr2025/`) | JXL/AVIF/JPEG-AI/JPEG-XT, **PQ** | JND | ideal compression-HDR train+val once released — re-check `github.com/jpeg-aic/AIC-HDR2025` |
| TMIQD | unopened zip | tone-mapping (not codec) | MOS | low priority |

380 HDR codec pairs is **validation, not training**. To *train* `PreviewV0_5Hdr`:
generate an HDR codec sweep (encode PQ/HLG sources through JXL/AVIF across a
q-grid, score with pycvvdp / PU-SSIM2) mirroring the synthetic-v2 SDR pipeline,
in PU/PQ space.

## 6. Phased plan + status

- **Chunk 1 — display-model foundation** (`transfer.rs`: PQ/HLG/sRGB → cd/m² +
  reference-parity tests). **✅ DONE 2026-06-01** — 6 tests green, decision-
  independent, zero SDR regression, internal (`pub(crate)`).
- **Chunk 0 — UPIQ validation harness** (`scripts/upiq_eval.py`: joins the UPIQ
  JOD truth + objective scores, runs the canonical `panel` binary stratified
  ALL/HDR/SDR; no stat math reimplemented). **✅ DONE 2026-06-01** — reproduces
  the published baselines to <0.001 (`benchmarks/upiq_baselines_2026-06-01.md`:
  PU-PieAPP 0.945, PU-FSIM 0.841, HDR-VDP-2 0.815, PU-SSIM 0.696). It also
  reproduces the **PU-encoding payoff** on our own data: FSIM's HDR-band SROCC
  0.457 → PU-FSIM 0.719 (+0.26 from PU-encoding the same metric) — the empirical
  case for chunk 2. Needs only the two CSVs (no 2.4 GB EXR extraction). The
  `--scores` flag is the plug-in point for zensim-HDR scores once chunk 2 lands;
  EXR extraction + pycvvdp rank-parity is the remaining (image-side) half.
- **Chunk 2 — PU21 front-end** (stacked PR on #39). **2a ✅ DONE 2026-06-01**:
  `pu21.rs` — `pu21_encode`/`decode` with the verified `gfxdisp/pu21`
  `banding_glare` coefficients (100 cd/m² → ~256), 6 reference-parity tests.
  **2b/2c ✅ WIRED + VALIDATED 2026-06-01 (partial pass).** The PU-XYB path is
  are always compiled (the `hdr` feature gate was removed): `Zensim::compute_pu_linear_planar`
  (absolute-luminance planes → opsin → PU21 → XYB → existing pyramid/features),
  `color::linear_to_pu_xyb_planar_into`, `streaming::compute_multiscale_stats_pu_linear_planar`.
  SDR path byte-identical (separate functions; default API unchanged). Validated
  end-to-end on the 380-pair UPIQ HDR subset
  (`zensim-validate/src/bin/upiq_pu_score.rs` → `scripts/upiq_eval.py`):
  **HDR-band SROCC 0.694 (best config)** vs PU-SSIM 0.740 / PU-FSIM 0.719 — far
  above the no-PU SDR baselines (FSIM 0.457) but ~0.05 short of the bar. It's the
  *representation* not the weights (linear + MLP all cluster 0.63–0.69); chroma
  de-emphasis (X 14→4) helped +0.01. Further formulation tuning is **not** done
  on UPIQ (held-out — would overfit); clearing the bar needs an HDR *training*
  corpus (chunk 4), which is data-blocked per §5. Full numbers:
  `benchmarks/upiq_pu_validation_2026-06-01.md`. Original design spec retained
  below for reference.

  ### Chunk 2b/2c — PU-XYB wiring spec (do this next)

  **2b — the PU-XYB conversion** (`zensim/src/color.rs`). Add a scalar
  `linear_abs_to_pu_xyb_planar_into(pixels_cdm2: &[[f32;3]], display:
  DisplayModel, variant: Pu21Variant, x_out, y_out, b_out)` mirroring
  `linear_to_positive_xyb_planar_inner` (color.rs:1454) with three changes:
  1. **No `[0,1]` clamp** (HDR exceeds 1) — clamp instead to `[PU21_L_MIN,
     display.y_peak]` after scaling relative→absolute via
     `DisplayModel::sdr_linear_to_luminance` (or `pq_to_luminance` for PQ).
  2. **Replace `cbrt_midp()` (color.rs:1517-1519) with `pu21_encode(·, variant)`**
     applied to each opsin-mixed channel.
  3. **DESIGN DECISIONS to make explicit (then validate on the harness, not by
     eye):** (a) the `K_B0` absorbance bias + `-cbrt(K_B0)` term (color.rs:1464)
     — PU already encodes absolute light, so the absorbance offset likely drops;
     (b) the opponent centering biases `0.42/0.01/0.55` (color.rs:1481-1483) are
     tuned to cube-root XYB ranges and will need re-centering for PU's `[0,~600]`
     range. Start scalar-only (HDR is rare; SIMD later); keep the SDR path
     byte-identical (separate function, separate dispatch).

  **2c — dispatch + guard lift** (`source.rs`, `metric.rs`, `streaming.rs`).
  When `source.color_transfer_function().is_hdr()` (PR #39's signal), route to a
  PQ/HLG composite (decode via `transfer.rs` → absolute luminance) → 2b's
  PU-XYB conversion, and **lift `reject_hdr_input` for that path only** (PR #39's
  guard stays for the un-wired transfers). Add the public HDR API
  (`ColorTransferFunction` → `DisplayModel` selection; promote `transfer`/`pu21`
  surface as needed).

  **GATE (mandatory before merge):** the wired PU path MUST be scored on the
  UPIQ HDR subset via `scripts/upiq_eval.py --scores` and **clear PU-SSIM
  (HDR-band SROCC 0.74) — ideally approach PU-FSIM (0.72) / PU-PieAPP**. A
  PU-XYB that compiles + "doesn't error" but doesn't beat PU-SSIM on the HDR
  band is wrong color math, not a feature. Do not merge on a compile.
- **Chunk 3 — luminance-dependent per-channel CSF** (also lifts SDR — the
  standing P0 from `vdp-csf-perceptual-math.md`). Seed `s_ch=[1.0,1.7,0.237]`;
  CSF *shape* from castleCSF (exact coeffs are OCR-garbled → `gfxdisp/castleCSF`
  MATLAB, MIT) or learn the LUT.
- **Chunk 4 — generate HDR codec training sweep → train `PreviewV0_5Hdr` →
  validate on UPIQ (+ AIC-HDR when it releases).**

## 7. Reference tools (local)

- `pycvvdp 0.5.4` — `/home/lilith/.local/bin/cvvdp` (runs; JOD + `pu-psnr-y`;
  display models in `pycvvdp/vvdp_data/display_models.json`).
- `zenmetrics/crates/{cvvdp,cvvdp-conformance}` — pure-Rust CVVDP + the golden
  conformance harness pattern to mirror (PQ/HLG constants at `cvvdp/src/params.rs:203-257`).
- UPIQ data: `/mnt/v/datasets/upiq/` (CSVs + `fit_logistic.m`) + `…/upiq_dataset.zip`.
- `gfxdisp/pu21`, `gfxdisp/castleCSF` — NOT cloned locally; fetch coefficients
  when chunks 2/3 need them.
