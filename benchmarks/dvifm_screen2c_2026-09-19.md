# DVIFM block-visibility — PHASE 2c: substitution + codec-distortion screen (2026-09-19)

**Q1 verdict: NOT-A-SUBSTITUTE.** Adding the 30 DVIFM columns to the fast
`y60` tier *reduces* the within-reference primary metric on all 10 paired
seeds (mean paired Δ = −0.0045). The columns do not close any of the gap to
`basic228` — they widen it (gap closed = −2.46). DVIFM does carry real
standalone signal (`dvifm30` alone = 0.857 per-ref mean vs 0.005 for its
permuted twin, +10/−0), but the signal does not survive composition with a
real base, exactly the 2b story.

**Q2 verdict: NEGATIVE** on the TRAIN-role codec-proxy panel. The 2b
ordering does NOT hold: `basic228+dvifm30` no longer beats its permuted
twin (Δ(B−C) = −0.0021, |Δ| ≤ 2·SD/√10 = 0.0085, +5/−5) — the within-image
signal visible on human-labelled rows does not exist against SSIMULACRA2
proxy labels. The Q1 rule applied on codec rows is also NOT-A-SUBSTITUTE
(Δ(y60dvifm − y60) = −0.0132, +0/−10). On JPEG specifically — where the
8-lattice blockiness hypothesis should show first — the y60 pair delta is
the most negative of any codec (within-ladder −0.1105 ± 0.0146).

`dvifm_block` stays registered and default-OFF. No model is qualified.

Preregistration: `benchmarks/dvifm_screen2c_prereg_2026-09-19.md` (committed
`4d6946db` before any fit; driver extensions `8ea2e7ef`; base `main@origin`
`8bff0ff7`). Companion data: `benchmarks/dvifm_screen2c_2026-09-19.json`;
evidence root `/mnt/v/output/zensim/dvifm-screen2c-2026-09-19/` (pointer
file below).

## MISSING first

1. **Codec labels are PROXIES, not human judgments.** Every Q2 number is
   measured against registered `score_ssim2` (SSIMULACRA2) full-reference
   proxy labels on the Sept-5 ladder anchor panel. A codec-panel result —
   in either direction — is not human-label evidence. The human-label
   answer is Q1's, and it is negative.
2. **Q2 codec panel is small**: 373 fit + 121 eval rows over 18 + 6
   source origins (the Sept-13 frozen admission's fit/dev families). The
   six families it labelled `test` (origins 1634, 1220, 8134, 7050, 7004,
   7058) were excluded entirely — never opened, extracted, or retagged.
   Per-codec cells have 5–6 origins each; per-codec deltas are wide-SE.
3. **Within-ladder saturation on codec rows.** On WebP, JXL and AVIF-SVT,
   per-(origin,codec) ladder SROCC saturates at 1.0000 for both y60 arms —
   those per-codec within-ladder deltas are exactly 0 and uninformative.
   Only JPEG and pooled (cross-ladder) rows carry discriminating signal.
4. **`dvifm30`-alone had not converged at E=100** (Q1): its dev per-ref
   mean was still rising E50→E100 (+0.0019 vs SD/√10 = 0.0006). The
   standalone-carry number (0.857) is a lower bound. All *decision-pair*
   arms (y60 / y60dvifm / y60perm30 / basic228) declined E50→E100 — the
   2b overfit signature — so the Q1 rule stands at E=100; the
   preregistered E=150 escalation applies to Q2, which needed none.
5. **One fixed permutation** (train seed 6619; dev 6620; dev2 6621 — the
   identical draws 2b used), not a permutation distribution.
6. **dev2 is small** (436 KonFiG originsplit-val rows over 4 references).
7. **No E=150, no frozen EVAL/test/terminal data, no CID22/AIC/KonJND-val/
   KADID{7,9}/secret-holdout reads, no subsampling decision exercised**
   (494-row panel « 40k cap), **no production qualification.**

## Design (all preregistered)

Identical trainer/head/loss/LR cycle/pairs-per-epoch/final-epoch-checkpoint
as 2b; the same ten paired seeds 9201,…,9219; E ∈ {50,100}. Q1 arms on the
2b human rows (8,327 fit / 3,125 dev / 436 dev2): `y60`, `y60dvifm`,
`y60perm30`, `dvifm30`, `perm30`, `basic228`. Q2 arms on the codec panel:
`basic228`, `basic228dvifm`, `basic228perm30`, `y60`, `y60dvifm`.
Permuted arms jointly row-permute f956..f985.

**2b-cell reuse (Q1):** 40 cells (basic228 + y60, E∈{50,100} × 10 seeds)
were byte-identical — verified trainer sha256 `9e77cb78…`, identical
`human_train.parquet` / `human.features.bin` / `human_dev2.features.bin`
against the 2b manifest — copied, not refit; each reused bake sha256 is
cited in `q1-human/RESULT.json` (`reuse.cells`) and its eval legs were
re-scored on the 2c tables. New fits: Q1 80, Q2 100.

**Driver correction delivered:** final-epoch checkpoint bakes are stamped
with `zentrain.formula_revision` + `zentrain.feature_set_id` inside the fit
cell (via `bake_dial_refit append-meta`, values from the run's extraction
manifest) before the bake sha256 is recorded — no post-hoc step.

**Q2 admission:** `admit_2c.py` reproduces the Sept-13 scale study's panel
selection verbatim over the registered ladder anchor
(`ladder-2026-09-05/anchor`), re-roled under `train-eval-only-v1`: 18
fit-families → TRAIN (373 rows), 6 dev-families → EVAL (121 rows), 6
test-families dropped. Sidecars `segments/codec-{train,eval}-admission.json`
name the authority chain (Sept-8 canonical family map sha `9d07a0f6…`,
ladder + anchor manifests); `validate_source_admission` + `validate_rows`
enforce it. Extraction wall time for the full 494-row panel (under the
first-500-row gate): **70.8 s**; no subsampling.

## Q1 — substitution screen (human-labelled rows)

Primary = within-reference panel mean (`zenstats::per_group_srocc`,
band=origin, 25 refs). Secondary = pooled signed SROCC. E=100, 10 seeds.

| pair | Δ per-ref mean | SD | 2·SD/√10 | signs | Δ pooled | signs |
|---|---:|---:|---:|---:|---:|---:|
| y60dvifm − y60 | **−0.00454** | 0.00239 | ±0.00151 | +0/−10 | −0.00560 | +0/−10 |
| y60dvifm − y60perm30 | **+0.00795** | 0.00183 | ±0.00116 | +10/−0 | +0.00707 | +10/−0 |
| basic228 − y60 | +0.00185 | 0.00146 | — | +9/−1 | +0.00441 | +9/−1 |
| dvifm30 − perm30 | +0.85224 | 0.00535 | — | +10/−0 | +0.83863 | +10/−0 |
| y60dvifm − basic228 | −0.00639 | 0.00234 | — | +0/−10 | −0.01001 | +0/−10 |
| dvifm30 − basic228 | −0.08434 | 0.00294 | — | +0/−10 | −0.09300 | +0/−10 |

**GAP CLOSED** = Δ(y60dvifm − y60) / Δ(basic228 − y60): **−2.46** on the
primary metric, **−1.27** pooled. The denominator is small (+0.00185 —
stated, not smoothed); the numerator is *negative*, so the ratio is
meaningful only as a sign: DVIFM moves the fast tier away from basic228.

**Rule:** SUBSTITUTE-CANDIDATE requires Δ(y60dvifm − y60) > +2·SD/√10 AND
Δ(y60dvifm − y60perm30) > +2·SD/√10. The first conjunct is −0.0045 —
**NOT-A-SUBSTITUTE.** (The permuted-control delta is positive and
significant — real signal — but signal without base-composition gain is
the 2b INFO-NOT-USEFUL pattern, not a substitute.)

Paired bootstrap over the 25 dev references (10,000 resamples, seed 8819):

| pair | median Δ | [q025, q975] | frac positive |
|---|---:|---:|---:|
| y60dvifm − y60 | −0.0045 | [−0.0073, −0.0019] | 0.34% |
| y60dvifm − y60perm30 | +0.0080 | [+0.0063, +0.0097] | 99.99% |
| basic228 − y60 | +0.0019 | [−0.0006, +0.0043] | 90.5% |
| dvifm30 − perm30 | +0.8587 | [+0.8523, +0.8650] | 100% |

### Q1 per-seed table (dev per-ref mean, E=100)

| seed | basic228 | y60 | y60dvifm | y60perm30 | dvifm30 | perm30 |
|---:|---:|---:|---:|---:|---:|---:|
| 9201 | 0.9416 | 0.9388 | 0.9334 | 0.9259 | 0.8545 | +0.0030 |
| 9203 | 0.9405 | 0.9396 | 0.9364 | 0.9275 | 0.8596 | +0.0065 |
| 9205 | 0.9411 | 0.9398 | 0.9338 | 0.9262 | 0.8584 | −0.0032 |
| 9207 | 0.9424 | 0.9414 | 0.9349 | 0.9263 | 0.8611 | +0.0065 |
| 9209 | 0.9422 | 0.9389 | 0.9375 | 0.9274 | 0.8570 | −0.0002 |
| 9211 | 0.9433 | 0.9390 | 0.9352 | 0.9286 | 0.8568 | +0.0061 |
| 9213 | 0.9417 | 0.9388 | 0.9352 | 0.9270 | 0.8568 | +0.0118 |
| 9215 | 0.9386 | 0.9392 | 0.9349 | 0.9279 | 0.8576 | +0.0114 |
| 9217 | 0.9410 | 0.9394 | 0.9376 | 0.9268 | 0.8512 | +0.0046 |
| 9219 | 0.9410 | 0.9401 | 0.9307 | 0.9264 | 0.8572 | +0.0012 |

`perm30` per-ref means carry the `Orientation::Auto` pooled-sign
convention (its pooled ρ is near zero and flips sign across seeds/legs);
the raw signed per-reference values are in `cells/*.json`
(`per_ref_srocc`). Its dev2 per-ref mean reads +0.087 for the same
reason (pooled dev2 ρ = −0.086).

dev2 leg (436 KonFiG rows, 4 refs; E=100 per-ref means): basic228 0.8715,
y60 0.8643, y60dvifm 0.8480, y60perm30 0.8354, dvifm30 0.8016, perm30
0.0866 — the same ordering as dev.

Fit-side overfit signature (E=100, dev): y60dvifm fit SROCC 0.9699 (vs
y60 0.9537) while dev per-ref drops 0.9395→0.9350 — the same signature 2b
recorded for basic228+dvifm30.

## Q2 — codec-distortion screen (proxy-labelled rows)

TRAIN-role codec panel, PROXY labels (`score_ssim2` SSIMULACRA2), 373 fit /
121 eval rows, source-family-disjoint by origin; dev leg, E=100.

| pair | Δ per-ref mean | SD | 2·SD/√10 | signs | Δ pooled | signs |
|---|---:|---:|---:|---:|---:|---:|
| B−A (basic228dvifm − basic228) | +0.00470 | 0.0245 | ±0.01551 | +6/−4 | −0.00120 | +5/−5 |
| B−C (basic228dvifm − basic228perm30) | **−0.00213** | 0.0135 | ±0.00851 | +5/−5 | +0.00362 | +7/−3 |
| y60dvifm − y60 | **−0.01319** | 0.0067 | ±0.00423 | +0/−10 | −0.00891 | +1/−9 |
| y60 − basic228 | +0.01907 | 0.0177 | — | +8/−2 | +0.01570 | +9/−1 |

**Verdict: NEGATIVE** — |Δ(B−C)| = 0.0021 ≤ 2·SD/√10 = 0.0085. On
proxy-labelled codec rows the DVIFM block does not even separate from its
row-permuted twin; the within-image ordering signal measured on
human-labelled rows does not transfer to the proxy codec panel. The Q1
substitution rule on codec rows: Δ(y60dvifm − y60) = −0.0132, +0/−10 →
**NOT-A-SUBSTITUTE**.

Ancillary observation (recorded, not a screen verdict): `y60` *beats*
`basic228` on the proxy codec panel (+0.0191 per-ref, +8/−2; +0.0157
pooled, +9/−1) — the fast-tier features track SSIMULACRA2 ordering better
than the full basic228 block. Proxy-label caveat applies.

Paired bootstrap over the 6 dev origins (10,000 resamples, seed 8819 —
only 6 origins, so the interval is coarse):

| pair | median Δ | frac positive |
|---|---:|---:|
| basic228dvifm − basic228 | +0.0044 | 63.8% |
| basic228dvifm − basic228perm30 | −0.0019 | 44.1% |
| y60dvifm − y60 | −0.0130 | 33.4% |

(frac-positive < 50% for y60dvifm − y60 despite 10/10 seed signs: with 6
origins the resample distribution is dominated by the largest per-origin
deficits.)

### Q2 per-codec table (dev, E=100, mean over 10 seeds)

pooled = per-family pooled signed SROCC; ladder = mean of
per-(origin,codec) within-ladder SROCC; Δ columns are paired seed deltas
of the ladder mean.

| codec | arm | pooled | ladder | Δ y60dvifm−y60 (ladder) |
|---|---|---:|---:|---:|
| jpeg | basic228 | 0.9228 | 1.0000 | −0.1105 ± 0.0146 |
| jpeg | basic228dvifm | 0.8877 | 0.9860 | |
| jpeg | basic228perm30 | 0.8951 | 0.9713 | |
| jpeg | y60 | 0.9601 | 0.9684 | |
| jpeg | y60dvifm | 0.8990 | 0.8579 | |
| webp | basic228 | 0.9679 | 0.9483 | +0.0000 ± 0.0000 (saturated) |
| webp | basic228dvifm | 0.9692 | 0.9500 | |
| webp | basic228perm30 | 0.9111 | 0.9583 | |
| webp | y60 | 0.9760 | 1.0000 | |
| webp | y60dvifm | 0.9858 | 1.0000 | |
| avif_svt | basic228 | 0.9933 | 0.9983 | +0.0183 ± 0.0104 |
| avif_svt | basic228dvifm | 0.9914 | 1.0000 | |
| avif_svt | basic228perm30 | 0.9894 | 1.0000 | |
| avif_svt | y60 | 0.9837 | 0.9817 | |
| avif_svt | y60dvifm | 0.9907 | 1.0000 | |
| jxl | basic228 | 0.9026 | 0.9060 | +0.0000 ± 0.0000 (saturated) |
| jxl | basic228dvifm | 0.9378 | 0.9580 | |
| jxl | basic228perm30 | 0.9528 | 0.9740 | |
| jxl | y60 | 0.9828 | 1.0000 | |
| jxl | y60dvifm | 0.9785 | 1.0000 | |

JPEG is where the block-visibility hypothesis should show first; instead
it shows the *largest* within-ladder loss for the DVIFM-augmented fast
tier. The `identity` family (constant target 100) reports SROCC 0.0 —
degenerate by construction, recorded not interpreted.

### Q2 per-seed table (codec dev per-ref mean, E=100)

| seed | basic228 | basic228dvifm | basic228perm30 | y60 | y60dvifm |
|---:|---:|---:|---:|---:|---:|
| 9201 | 0.9465 | 0.9797 | 0.9729 | 0.9876 | 0.9685 |
| 9203 | 0.9730 | 0.9611 | 0.9716 | 0.9895 | 0.9686 |
| 9205 | 0.9541 | 0.9785 | 0.9700 | 0.9768 | 0.9679 |
| 9207 | 0.9342 | 0.9608 | 0.9730 | 0.9794 | 0.9690 |
| 9209 | 0.9766 | 0.9728 | 0.9760 | 0.9928 | 0.9716 |
| 9211 | 0.9537 | 0.9836 | 0.9755 | 0.9919 | 0.9707 |
| 9213 | 0.9836 | 0.9507 | 0.9716 | 0.9775 | 0.9683 |
| 9215 | 0.9816 | 0.9500 | 0.9717 | 0.9805 | 0.9700 |
| 9217 | 0.9803 | 0.9853 | 0.9698 | 0.9928 | 0.9866 |
| 9219 | 0.9680 | 0.9763 | 0.9680 | 0.9735 | 0.9693 |

## Execution

Exact commands (each under `~/work/zen/scripts/run-heavy --mem 16G --jobs 8`):

```
python3 scripts/lib/feature_screen.py recipe_q1.json q1-human --ceiling-stage prepare   # extract 39.4 s
python3 scripts/lib/feature_screen.py recipe_q1.json q1-human --ceiling-stage fit       # 40 reused + 80 new fits (~5.0k s train)
python3 scripts/lib/feature_screen.py recipe_q1.json q1-human --ceiling-stage audit     # 6 bakes × 33 pairs, max consumed-feature |Δ| ≤ 2.8e-17
python3 scripts/lib/feature_screen.py recipe_q1.json q1-human --ceiling-stage report
python3 admit_2c.py                                                                    # codec segments/admissions
python3 scripts/lib/feature_screen.py recipe_q2.json q2-codec --ceiling-stage prepare   # extract 70.8 s / 494 rows
python3 scripts/lib/feature_screen.py recipe_q2.json q2-codec --ceiling-stage fit       # 100 fits (~4.8k s train)
python3 scripts/lib/feature_screen.py recipe_q2.json q2-codec --ceiling-stage audit     # 5 bakes × 5 pairs
python3 scripts/lib/feature_screen.py recipe_q2.json q2-codec --ceiling-stage report
```

Commits: prereg `4d6946db`; driver `8ea2e7ef`; this record follows.
Recipe sha256: q1 `448ac697…`, q2 `320b732b…` (full hashes in the run
`RESULT.json` identities). `/mnt/v` free space stayed ≥ 80 GB throughout
(~102 GB); phase output ≈ 4 GB « 40 GB cap.

Toolchain (byte-identical to 2b): extractor `91e10440…`, trainer
`9e77cb78…`, predict `c9fcfdd6…`, panel `f11857c2…`, refit `31d016be…`.

One driver fix over 2b: the `per_group_srocc` consistency check now
encodes `Orientation::Auto`'s pooled-polarity rule plus the band-spread
filter — a garbage-control arm (pooled ρ < 0, `perm30`) reported the
sign-flipped per-ref mean by design and falsely tripped the 2b check.

## What was NOT done

No E=150 escalation (no Q2 arm was still rising; the one Q1 flag is a
standalone-carry diagnostic, not a decision arm). No frozen EVAL, test or
terminal data — including the codec panel's six historical test families.
No CID22 human scores / 49-ref gold set, AIC-3/4/AIC2026, KonJND
validation, KonFiG test, KADID/TID terminal refs {7,9}, or any secret
holdout. No DVIFM constant optimisation, no X/B channels, no i16 kernel
work, no production qualification, no public API change, no push.
`dvifm_block` remains default-OFF.
