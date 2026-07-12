# Profile B (SDR) + B-HDR — full methodology & byte-exact reproduction

**Date:** 2026-07-12
**Status:** shipped. `ZensimProfile::B` (SDR) and `ZensimProfile::BHdr` (HDR
sibling) are the deterministic linear profiles. This doc is the single
consolidated "how B was built and how to recreate it byte-for-byte" record,
per the shipping-policy methodology-doc requirement in `CLAUDE.md`. It ties
together the detailed sub-docs rather than duplicating them:

- Fit / ensemble catalog + Pareto: [`linear_projections_2026-07-03.md`](linear_projections_2026-07-03.md)
- Pinned input shas + best-results table: [`provenance_best_results_2026-07-04.md`](provenance_best_results_2026-07-04.md)
- Winsor + dial-top-extend + near-lossless resolution: [`jxl_nearlossless_dial_2026-07-05.md`](jxl_nearlossless_dial_2026-07-05.md) §7–§11
- Knob validation with real encoders (mechanics / MOS-consistency / span / reach / ceiling): [`b_knob_validation_real_encoders_2026-07-11.md`](b_knob_validation_real_encoders_2026-07-11.md)
- One-command byte-repro: [`../scripts/reproduce_b.sh`](../scripts/reproduce_b.sh)

---

## 1. What B is (identity)

| | Profile B (SDR) | Profile B-HDR |
|---|---|---|
| Enum variant | `ZensimProfile::B` | `ZensimProfile::BHdr` |
| `name()` | `zensim-b` | `zensim-b-hdr` |
| Shipped bake | `weights/b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin` | `weights/bhdr_linear_shaped_anchored2_2026-07-04.bin` |
| Bytes | 7,325 | 11,684 |
| sha256 (16) | `b6fe5233ee9c752d` | `373eac56e7a07d6d` |
| `include_bytes!` site | `profile.rs:905` (`linear_bake_b_cid80`) | `profile.rs:915` (`linear_bake_bhdr_shaped`) |
| Wire format | ZNPR v3 | ZNPR v3 |

**Architecture (both):** a **linear** model, not an MLP —

```
372 features ──[372 winsor_p99 feature_transforms]──> clamp per-feature
             ──[standard scaler: (x−mean)/scale]──> z
             ──[single linear layer 372→1, IDENTITY activation, f16 weights]──> raw
             ──[output_calibration_spline: 30-knot monotone PCHIP]──> dial ∈ [0,100]
```

No hidden layer. That is the whole point: the entire model is one dot product
plus a monotone 1-D spline, which is why it is ~7 KB, has **no collapse mode**,
and reproduces **byte-for-byte** (Gram-exact least-squares solves — no SGD, no
seed, no nondeterminism). The spline is rank-preserving, so SROCC is invariant
under it; it exists only to map the raw score onto the [0,100] dial.

**Routing note:** `params_pu_linear()` sends `ZensimProfile::B` to the B-HDR
weights when the caller feeds absolute nits (HDR path); the SDR bake above is
used on the standard 0–1 / sRGB path. See `profile.rs`.

---

## 2. Lineage (how the raw weights were fit)

B (SDR) = **`ens-Pline-cid80`**, a raw-space convex blend of two linear heads:

```
ens-Pline-cid80  =  0.8 · cid_head  +  0.2 · kon_head        (blended in RAW output space)
```

| Head | Bake name | Fit method | Training corpus | Target column | Non-zero wts |
|---|---|---|---|---|---|
| **cid** (80%) | `hdrmix-lasso0.002-raw` | Lasso, λ=0.002, τ=0 | `hdr_v3mix` (7,410 rows) | cvvdp-mix = `0.5·clip01(ssim2) + 0.5·clip01((JOD−6)/4)` | 35 |
| **kon** (20%) | `canonhdr15-bvls-raw` | BVLS (bounded-variable LSQ), τ=0.005 | safesyn + cid22_train + kadid + tid + hdr_v3mix | per-corpus anchor (ssim2-derived) | (dense) |

Why this blend: the cid head is the **best linear CID22 rank** we found
(0.874 standalone) but weak on KonJND (0.37); the kon head sets the **KonJND
record** (0.6696 standalone) but trails on CID22. The 0.8/0.2 raw-space blend
buys +KonJND at a sub-noise CID22 cost — the CID22↔KonJND trade is **non-linear
in our favor** in raw space (documented in `linear_projections_2026-07-03.md`).
The blend is done on **raw** outputs (pre-spline) and then a single shared
spline is fit on the blended raw — mixing post-spline would scramble rank.

**Every head is CID22-49-MOS-clean.** `hdr_v3mix` and `canonhdr15` carry no
CID22-49 human MOS. `cid22_train` in the kon head is the **ssim2-anchored**
train split (NOT MOS — verified, `DATA_SPLITS.md`), which is training-legal.
The blend never sees a held-out human label.

---

## 3. Guards applied to the raw bake (calibration)

The raw ensemble bake (`b_sdr_linear_cid80_anchored_2026-07-04.bin`, 823 B,
sha `7b326ac5`) is finished with two deterministic post-hoc steps, both via the
`bake_dial_refit` Rust binary (never hand-edited bytes):

1. **`add-winsor`** — attaches 372 `winsor_p99` feature-transform bounds fit at
   `[p0.1, p99.9]` per feature on the **inclusive winsor corpus**
   (`/mnt/v/output/zensim-jxl-nearlossless/inclusive_winsor_corpus.parquet` =
   hdr_v3mix **+** the zenjxl near-lossless SDR sweep). This clamps OOD feature
   excursions that would otherwise expose the linear model's unbounded tail
   (f155 in particular).
   - **Why the *inclusive* corpus (2026-07-07 fix):** the predecessor fit its
     winsor bounds on hdr_v3mix **alone**; those bounds sat *above* the SDR
     near-lossless feature range, clamping 245/372 features constant there and
     pinning B's near-lossless dial at ~91.5 while ssim2/A reached ~96. Adding
     the near-lossless SDR sweep to the *fit corpus* frees those lower bounds:
     near-lossless dial climbs 91.5→96.1 and per-image near-lossless rank-vs-ssim2
     0.657→0.771, at **zero human-MOS cost** (CID22 0.8763→0.8764, KonJND
     0.5474→0.5466). This flaw was user-caught. Detail: `jxl_nearlossless_dial_2026-07-05.md` §7–§8.

2. **`extend-top`** — extends **only the dial TOP** by the training-fitted
   concave saturation (lstsq on the multiband anchor
   `multiband_anchor_dial100.parquet`, `target_score` col) so near-lossless
   codec-knob configs resolve toward 100. The bottom and in-distribution spline
   are kept **verbatim**, so rank is identical to the winsor bake.

Result → `b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin` (SHIPPED B).

**B-HDR** is finished differently (its data ceiling is ~92.8, so it uses
`bake_dial_refit bottom-extend` / `shared-anchor` rather than top-extend; a
research densify lives in `scripts/v_next/hdr_anchor_dense_refit.py`).
Its `bottom-extend` variant was **withdrawn** — negatives are valid scores.

---

## 3b. B-HDR — creation recipe & exact relationship to B (rechecked 2026-07-12)

> **Status note (2026-07-12, later same day):** this section describes the
> `anchored2` bake (`373eac56`). A cvvdp-mix sibling (`7d7f2123`) was promoted into
> the `BHdr` slot the same day, then a same-day audit found the promotion's UPIQ
> significance claim invalid post-selection — see
> [`bhdr_improvement_split_lineage_2026-07-12.md`](bhdr_improvement_split_lineage_2026-07-12.md)
> §5–§7 for the promotion, the audit, and the disposition.

B-HDR is **not** "B refit on HDR data." It is an **independent linear fit** with a
different target, different feature preprocessing, a different scaler, and a
different dial. The only things it shares with B are the ZNPR-v3 linear
architecture and the runtime routing that lets `ZensimProfile::B` *reuse* B-HDR's
weights on the absolute-nits path. This section is the corrective record — a prior
note claimed the two "share an anchor scale, seam ≤0.92pt," which is true only in
the mid-range overlap; **their calibrated dial ranges differ by construction.**

### How B-HDR was created

| Step | B-HDR (`bhdr_linear_shaped_anchored2_2026-07-04.bin`, sha `373eac56`) |
|---|---|
| Fit | **`hdr-lasso0.001-shaped`** — Lasso λ=0.001, τ=0 (zerobias) |
| Training corpus | **`hdr_v3`** = `hdr_zenjxl_v3_traindigits_2026-07-03.parquet` — 7,410 HDR-JXL renditions |
| Target column | **`human_score` = ssim2-based** (NOT the cvvdp-mix `hdr_v3mix` that B's cid head uses) |
| Feature regime | **PU-linear** (absolute nits, `compute_pu_linear_extended_features`) — not the SDR sRGB→XYB shell |
| Feature transforms | **"shaped": per-feature MIXED** — `winsor_p99` / `quantile_bins` / `clip_then_log1p` / `log1p` / `signed_cbrt` / `yeo_johnson` (chosen per feature by fit lift) |
| Scaler | **real standardization** (`(x−mean)/scale`, non-trivial mean/scale per feature) |
| Dial | **`anchored2`** = shared-anchor refit v2 → **25-knot** PCHIP, knots raw `[0.297, 1.023]` → dial **`[25.88, 92.75]`** |

The **shaping** (mixed transforms + real scaler) is what lifted UPIQ from ~0.65
(raw features) to **0.7313** — the shaping, not the choice of ssim2 target, drove
the gain.

### Relationship to B — side by side

| aspect | **B (SDR)** | **B-HDR** |
|---|---|---|
| Fit | `ens-Pline-cid80` blend (0.8·cid + 0.2·kon) | single `hdr-lasso0.001-shaped` |
| Target | cvvdp-mix (cid head) + ssim2-anchor (kon head) | **pure ssim2** |
| Feature regime | SDR shell (u8/sRGB→XYB) | **PU-linear (nits)** |
| Transforms | **uniform** `winsor_p99` ×372 | **shaped** (mixed per feature) |
| Scaler | **IDENTITY** (mean 0 / scale 1) | **real standardization** |
| Spline | 30-knot, raw `[−1.974, 4.076]` | 25-knot, raw `[0.297, 1.023]` |
| Calibrated dial | **`[0.00, 100.00]`** (explicit floor + extend-top knots) | **`[25.88, 92.75]`** (its HDR data range) |

> The architecture block in §1 (line 36) writes "standard scaler" for both; that is
> exact for B-HDR and *degenerate* for B — B's shipped scaler is the **identity**
> (all means 0, all scales 1), so B's preprocessing is winsor-clamp → dot →
> spline with no re-centering.

The two are coupled by exactly **two** mechanisms, both verified:

1. **Runtime routing.** `params_pu_linear()` returns `&PROFILE_B_HDR` for
   `Self::B`, so when a caller feeds absolute nits, `ZensimProfile::B` extracts
   PU-linear features and forwards B-HDR's shaped weights. **`B.compute(nits) ≡
   BHdr` is verified byte-for-byte** by
   `profile::descriptor_hdr_routing_tests::compute_routes_descriptor_flagged_hdr_to_pu_linear`
   (`via_compute == via_pu`, `b == bh` within 1e-12). So "B on HDR" and "B-HDR"
   are the *same* function — B-HDR is simply the name for B's PU-linear branch.
2. **Partially-shared dial anchor.** Both dial splines were fit against the same
   multiband anchor family, so **in the mid-range** the two dials read within
   ~1 point of each other. They **diverge at the endpoints** (below) — the
   "shared anchor" is a mid-range overlap, not a shared range.

### Material findings (these update the docs + memory)

1. **B and B-HDR do NOT share a calibrated dial range.** B is calibrated with
   explicit knots across the full **`[0,100]`** (deliberate floor + `extend-top`).
   B-HDR's knots cover only **`[25.9, 92.8]`** — its honest HDR-JXL data floor and
   ceiling — and it **extrapolates** outside that band (on the real UPIQ HDR
   stratum the anchored2 B-HDR mapped to `[−37.3, 86.1]` median 7.0, re-measured
   2026-07-12 via the pred-dump path; an earlier note said "[0.00, 86.11]" which
   clipped the extrapolated lower tail — the runtime clamps at −100, not 0).
   So the earlier "shared anchor, seam ≤0.92pt"
   claim is correct only *inside* `[25.9, 92.8]`; a target dial of, say, 15 or 96
   lands in B-HDR's **extrapolated** (uncalibrated) region, not a calibrated one.
   This is the precise statement of the "92.8 data ceiling" limitation for
   near-lossless HDR — and its symmetric `~25.9` floor.
2. **B-HDR's target was never A/B'd against the cvvdp-mix under shaping.** B's cid
   head trains on `hdr_v3mix` (cvvdp-mix); B-HDR trains on `hdr_v3` (pure ssim2).
   The cvvdp-mix target was only ever tried on **raw** features (`hdrmix-lasso0.001-raw`,
   UPIQ 0.6488) — a **shaped-cvvdp-mix** fit does not exist. So "ssim2 is B-HDR's
   best HDR target" is **unproven**; it is the best among what was fit, and B-HDR
   likely inherits ssim2's HDR ceiling, which plausibly explains its ~0.73 plateau
   below the heavy HDR specialists.
3. **B-HDR is not the single max-UPIQ linear candidate.** `canon-ridge1e-05-raw-tau0.02`
   reached UPIQ **0.7425**, but it is an SDR-corpus fit with poor KADID (0.4063);
   B-HDR was chosen as the **HDR-trained + shaped + balanced** pick, which is the
   defensible choice for an HDR-only profile but is *not* the raw UPIQ argmax.

**Net:** B-HDR is a legitimate, self-contained lightweight HDR metric that B
*reuses* on the nits path — not a derivative of the SDR bake. The open threads
are (a) B-HDR's dial is calibrated only over `[25.9, 92.8]` (near-lossless and
heavy-HDR are extrapolated), and (b) the shaped-cvvdp-mix HDR target is untried.

---

## 4. Byte-exact reproduction

```bash
# From the zensim repo root. Requires the /mnt/v fit inputs (hdr_v3mix,
# the zenjxl near-lossless sweep, the multiband anchor).
bash scripts/reproduce_b.sh            # → asserts sha b6fe5233, cmp byte-identical to shipped
```

The script rebuilds the shipped B from the committed 823 B raw bake:

```
b_sdr_linear_cid80_anchored_2026-07-04.bin (823 B, sha 7b326ac5)   [committed]
  │  bake_dial_refit add-winsor  --fit-corpus inclusive_winsor_corpus.parquet
  │                              --lo-pct 0.1 --hi-pct 99.9
  ▼
  b_winsor.bin
  │  bake_dial_refit extend-top  --anchor multiband_anchor_dial100.parquet
  │                              --target-col target_score
  ▼
  b_sdr_linear_cid80_inclwinsor_dense_dial.bin  ==  SHIPPED (sha b6fe5233) ✓
```

To reproduce the **raw** 823 B bake from parquet corpora (the fit stage, one
level deeper than `reproduce_b.sh`):

```bash
# scripts/v_next/linear_projections_2026-07-03.py  (subcommands: gram | fit | ensemble | finalize)
#   gram     : build Gram matrices per corpus (deterministic full-data)
#   fit      : cid head = hdrmix-lasso0.002 ; kon head = canonhdr15-bvls
#   ensemble : ens-Pline-cid80 = 0.8·cid + 0.2·kon  (raw space)
#   finalize : fit shared spline on blended raw → f16 → zenpredict bake (v3)
```

All stages are deterministic (Gram-exact solves; fixed percentiles for winsor;
lstsq + fixed percentiles for extend-top). Determinism proof: 44/44 refits
byte-identical across the full pipeline (`linear-probe/determinism_check.py`).

---

## 5. Data-lineage table

| Artifact | Path | Rows | sha256 (16) | CID22-MOS contam |
|---|---|---:|---|---|
| `hdr_v3mix` (cid head + winsor fit) | `/mnt/v/output/zensim-multicodec-probe/hdr_zenjxl_v3mix_traindigits_2026-07-03.parquet` | 7,410 | `31e08c70…` | none (HDR-JXL, no CID22-49) |
| canonical train (kon head) | `canonical-2026-05-21/train/{safesyn,cid22_train,kadid,tid}.parquet` | 196,086 + … | see set `_MANIFEST.json` | cid22_train = **ssim2-anchored**, train-legal |
| inclusive winsor corpus | `/mnt/v/output/zensim-jxl-nearlossless/inclusive_winsor_corpus.parquet` | 10,810 × 372 | (rebuilt by `build_inclusive_winsor_corpus.py`) | none |
| multiband anchor (extend-top) | `canonical-2026-05-21/train/multiband_anchor_dial100.parquet` | — | — | none (synthetic dial anchor) |
| raw ensemble bake | `weights/b_sdr_linear_cid80_anchored_2026-07-04.bin` | — | `7b326ac5…` | derived, clean |
| **SHIPPED B** | `weights/b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin` | — | `b6fe5233…` | derived, clean |
| **SHIPPED B-HDR** | `weights/bhdr_linear_shaped_anchored2_2026-07-04.bin` | — | `373eac56…` | derived, clean |

---

## 6. Held-out panel (rank quality) — shipped B (b6fe5233)

Mohammadi 2025 panel on the held-out corpora (CID22-49, AIC-3, AIC-4, KonJND,
UPIQ are validation-only; KADID/TID are train==val memorization, integrity
guards only). Numbers from `bake_verdict` on the quarantined dial grid +
canonical val parquets:

| Corpus | SROCC | Role |
|---|---:|---|
| CID22-49 | **0.8764** | gold-standard held-out human MOS |
| AIC-3 | 0.7774 | held-out JND (compression) |
| AIC-4 | 0.8906 | held-out reconstructed-JND |
| KonJND | 0.5466 | PJND anchor (`|SROCC|`; signed-negative is structural) |
| UPIQ | 0.6846 | HDR/SDR unified perceptual |

The 07-05 predecessor (b78adb15) posted CID22 0.8763 / KonJND 0.5474; the
07-07 winsor+dial change moves only the near-lossless region, so all other
columns track within noise.

**Dial gates** (the second mandatory panel): inversions 0.026, dead-zone
0.0005, monotonicity 0.974, G-RANGE 0 rows extrapolating — **all PASS**.

---

## 7. Knob quality (B as a codec quality dial)

Full validation with real encoders in `b_knob_validation_real_encoders_2026-07-11.md`.
Summary — B is on-par-or-better than ssim2 and Profile A as a dial, and
**decisively better on independent-reference consistency at scale**:

| Test | B | A (v47) | ssim2 |
|---|---:|---:|---:|
| Mechanics: \|ρ\| vs q | 0.953 | — | — |
| Mechanics: strict-mono rate | 79.9% | — | — |
| MOS consistency: resid-SD (score units) | ±6.33 | ±6.60 | ±6.04 |
| Normalized span (reachable dial fraction) | **0.68** | 0.64 | 0.40 |
| At-scale η²(butteraugli \| metric-decile) | **0.582** | 0.344 | — |

Per-codec reachable zone (dial units, ≥50% of ladders): avif 10–91, jpeg 27–82,
webp 27–82, jxl 39–81. JXL distance→B target table (near-lossless):
d0.04 ≈ **B 96.0** (±0.2), d0.5 ≈ 92.7, d1.0 ≈ 90.2 — see
`jxl_nearlossless_dial_2026-07-05.md` Part 11.

---

## 8. Honest gaps (what B does worse / where it is limited)

- **B ↔ A is a TRADEOFF, not dominance.** B wins the human-MOS holdouts
  (CID22, AIC-3, AIC-4); A wins ssim2-agreement on the ~1M-cell codec sweeps.
  A **still ships** as the profile `codec_target()` returns today. Deprecating
  A does not mean A is worse everywhere — see the deprecation rationale below.
- **Top-shoulder reach:** on jxl/webp, B's raw output ceilings such that the
  85–92 dial band is compressed (a raw-output ceiling, recalibratable but only
  by trading away calibration; `extend-top` cannot open it because that band is
  mid-spline, not the saturation tail). Documented in the knob-validation doc.
- **KonJND** (0.5466) is below the G5 0.70 floor — a characterized Pareto limit
  shared with every profile; the kon head alone reaches 0.6696 but at a CID22
  cost the blend declines to pay.
- **avif at-scale join** has a coverage gap in the fill4 sidecar (no avif codec
  tags) — genuine, noted, not a naming bug.

---

## 9. What ships / publishing note

`ZensimProfile::{A, B, BHdr}` are all default-compiled today. The crate
`include` list in `zensim/Cargo.toml` currently ships **only** A's bake
(`v47_strict_qat_native_2026-05-27.bin`) — B's and B-HDR's `include_bytes!`
sources are NOT in `include`, so a packaged crate would fail to build. Fixing
that (adding both linear bakes to `include`) is a prerequisite for publishing
B. Tracked as part of the publish-prep in this session's CHANGELOG entry.
