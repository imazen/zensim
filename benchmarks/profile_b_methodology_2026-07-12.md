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
