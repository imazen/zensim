# BANDVIS dst-activity plane — opt-in toggle + adjudication protocol (2026-08-02)

Mission: SOTA-944 campaign **P1.5** (`docs/PLAN_SOTA944_CAMPAIGN_2026-08-01.md`)
— implement the RECORDED-DEFERRED fix from
`benchmarks/append2_bandvis_gates_2026-07-27.md` REMAINDERS #3 (the V3(b)/(c)
dst-side cross-fire: dither/blocking texture in the DISTORTED image fires
BANDVIS_GAIN because the flatness mask was REF-side by design) as an OPT-IN,
default-OFF toggle, and run its recorded characterization/acceptance so the
campaign can adjudicate which math the expensive bigcodec extraction uses.
The `bandvis_dither_retest_2026-07-28.md` DEFER verdict is superseded by the
P1.5 sequencing decision (2026-08-02): the plane is built NOW so the
adjudication can compare, not because the RESHAPE was wrong.

Workspace `zensim--bandvis-dst` on main@origin (`5246d978`). Host: 7950X/WSL2;
**the box is NOT quiet** — the P1 kadis-944 backfill agent runs heavy
extraction concurrently (its `.workongoing` line is live). All perf numbers
below are interleaved A/B pairs so load bias mostly cancels; absolute ms/pair
is NOT comparable to the quiet-box gates-doc numbers and is flagged as such.

## What shipped

`V2NewFeatureToggles::append2_dst_activity` (default **false**; requires
`append2_block`, asserted). When ON:

- **Phase A** additionally computes a **dst-activity plane** for the Y
  channel only: `activity_dst = box_blur(|dst − mu2|)` — the exact
  distorted-side twin of the existing ref chain
  `activity = box_blur(|src − mu1|)`; `mu2` already exists from the fused
  blur pass, so the marginal cost is one `abs_diff` + one 2-pass box blur
  per Y strip (the recorded ≈+5%-class chain). Buffer is lazily grown so
  the OFF-mode heap profile is unchanged.
- **The BANDVIS band terms become per-side self-masked**:
  `b_src = band(|∇²src|; δ) · (1 − sat(act_src, C_ACTIVITY))` (unchanged) and
  `b_dst = band(|∇²dst|; δ) · (1 − sat(act_dst, C_ACTIVITY))` (was
  `act_src`). Third const-split kernel instantiation
  (`BANDVIS=true, BV_DSTACT=true`); the OFF instantiations emit today's
  exact operation sequences.
- Driver: `ZENSIM_AB_MODE=foldapp2* / foldcsfw*` + `ZENSIM_APPEND2_DSTACT=1`
  (SDR and both HDR routes; same CSV shape).

### Design decision recorded (the one degree of freedom the remainder left open)

The gates doc specified "a dst-activity plane" but not the mask algebra.
Three candidate forms were considered BEFORE measurement:

| form | b_src | b_dst | verdict |
|---|---|---|---|
| **A: per-side self-mask (SHIPPED)** | `band·flat_src` | `band·flat_dst` | literal "dst texture self-masks"; visibility is masked by the viewed image's own activity; identity-exact-0 preserved bitwise (identical planes ⇒ bitwise-identical activity twins ⇒ FR pair exactly 0); texture-as-deband earns the directionally-defensible LOSS credit (retest doc §2: "dithered quantization genuinely looks less banded") |
| B: joint mask | `band·flat_src·flat_dst` | `band·flat_src·flat_dst` | identity-safe, strictly-stronger suppression, but erases the deband-credit direction (dither-as-deband reads neutral, not LOSS) and couples the src term to dst content |
| C: dst term ref∧dst, src term ref | `band·flat_src` | `band·flat_src·flat_dst` | REJECTED: violates identity-exact-0 (`b_dst < b_src` on any non-flat identity pair ⇒ spurious LOSS) |

Form A also fixes a latent miss the ref-side mask had: heavy posterize of a
textured source (texture fully collapsed inside one quantization step) is
REAL new visible banding, which the ref-side mask suppressed; the self-mask
lets it fire.

## Pre-registered fixture matrix (gates written BEFORE the ON-arm measurements)

Fixtures verbatim from the gates doc V3 / `append2_bandvis_behavior`
(256² diagonal sRGB ramp 32..224; posterize/dither/lattice builders
unchanged). "OFF (recorded)" = the 2026-07-27 gates-doc numbers, which the
toggle-off path must keep bit-for-bit.

| id | fixture | OFF (recorded) | ON gate (pre-registered) | prediction |
|---|---|---|---|---|
| F1 ladder | posterize 7/6/5/4/3-bit GAIN, per scale | every rung fires >0.05; unimodal; 5b>4b>3b cap rolloff | SAME three invariants hold with the toggle ON (real banding must keep firing; resonance structure preserved) | amplitudes mildly lower (band contours contribute a little self-activity) |
| F2 noise-dither | 4-bit + hash-noise dither vs undithered 4-bit, GAIN ratio @s3 | ratio 1.72 — FIRES (pinned) | **ratio < 1.0 — MASKS** (the fix's primary claim) | ~0.2–0.5 (dither act ≈ 2–3× C_ACTIVITY ⇒ flat_dst ≈ 0.25–0.35 vs ≈0.9) |
| F2b ordered-Bayer | 4-bit Bayer 4×4, per-scale GAIN | [0.169, 0.207, 0.361, 0.643] | every scale ≤ the OFF value (suppression, no polarity flip) | s0–s2 strongly reduced; s3 partially (coarse-scale aliased pattern survives downscale into act_dst too) |
| F3 lattice | ±6-code 8-px DC lattice, max-scale GAIN | 0.433 (pinned >0.1 cross-fire) | max GAIN < 0.5× the OFF value; BLOCKINESS unchanged (>0.1) | ~2× suppression (lattice act ≈ C_ACTIVITY ⇒ flat_dst ≈ 0.5) |
| F4 src-texture | ±16-code textured src, 4-bit posterize, GAIN ratio vs clean @s3 | 0.099 (PASS < 0.5) | still < 0.5 (dst keeps quantized texture ⇒ self-mask carries the suppression the ref-side mask used to) | ratio may move but stays well under 0.5 |
| F5 deband credit | 4-bit banded src → smooth ramp dst | LOSS 0.414 > GAIN 0.324 | LOSS > GAIN preserved AND LOSS not below the OFF value (b_src math unchanged; b_dst can only shrink on a smooth dst) | LOSS ≈ unchanged, GAIN slightly down |
| F6 identity | src == dst | gain/loss exactly 0.0 | exactly 0.0 with the toggle ON (bitwise: activity twins identical ⇒ FR pair exact-0) | exact |
| F7 lanes-only | real textured/quantized pair, OFF vs ON full 944 | — | ONLY `f924 + s*5 + {BANDVIS_GAIN, BANDVIS_LOSS}` may differ; all other 942 slots BIT-identical; and the BANDVIS lanes DO differ on a dither fixture (toggle is live) | — |
| F8 serial ≡ parallel | toggle ON, 944 | — | bitwise equal | — |
| F9 HDR route | PU posterized log-ramp (gates-doc V4 fixture) + HL/SDR-zero fixtures, toggle ON | BANDVIS fired [0.171, 0.455, 0.477, 0.254]; HL bins fire / SDR-zero | HL bins + conditioner + first-924 unchanged (F7 on the HDR route); PU BANDVIS still fires on the banded HDR ramp (some scale > 0.05) | route-independent mechanism |
| F10 byte-stability OFF | `v2_ab_extract` CSVs: aic3-100 × {fold 720, foldapp 924, foldapp2 944, foldapphdr100 924} + kadis-hdr sample × foldapp2hdrpq; my binary (toggle absent/off) vs main-tip binary | — | **byte-identical (`cmp`) on every CSV — HARD GATE**; full test suite green with zero relaxations | — |
| F11 perf | foldapp2 vs foldapp2+DSTACT, aic3-100, 1-thread compute-only µs, interleaved A/B rounds | foldapp2 = +1.79% over foldapp (quiet box) | measure the DSTACT marginal; recorded expectation ≈ +5%; report honestly if higher (loaded-box caveat above) | +2–6% (abs+2-blur on 1 of 3 channels; SIMD lane adds 1 load + 3 ops per 8 px) |

### Acceptance protocol status (the recorded protocol, adapted to what exists)

1. **Characterization fixtures (PRIMARY evidence here)**: F1–F9 above.
2. **LIVE-YT-Banding ON-vs-OFF read**: videos on Tower
   (`/mnt/tower/input/datasets/live-yt-banding/videos`, sha-verified
   2026-07-28), labels local, the July frame-sampling pipeline script and
   its OFF-arm master CSV (960 pairs, current math — toggle-off is
   byte-stable) both survive in `~/tmp`. Plan: re-extract the same 8
   frames/video, score BOTH arms from the SAME fresh frames
   (`ZENSIM_APPEND2_DSTACT` 0/1), drift-check fresh-OFF vs the July master,
   then run the committed `benchmarks/bandvis_lyb_eval_2026-07-28.py` per
   arm. Pre-registered read: GAIN's pooled/fold SROCC (the weak polarity,
   −0.163/−0.154 recorded) should IMPROVE (more negative) if dst-masking
   removes dither/texture false fires; LOSS s3 (the workhorse, −0.447)
   must NOT degrade materially (bar: stays within 0.03 of the OFF arm |per
   this run's protocol|, both arms same frames). This is a
   frame-sampled FR probe, NOT CAMBI's temporal protocol — same caveat as
   the July run.
3. **LOO-on-944-bake**: NOT run here — it belongs to P3's model wave (the
   E2 LOO-positive criterion + the texture_dissim gate-pair read from the
   retest doc §4). Stated explicitly: no training-side claim is made in
   this doc.

## Results — ARM 1 (multiplicative self-mask INSIDE the FR pair): FALSIFIED on its primary claim

Fixture matrix run (`append2_dst_activity_behavior_matrix`, commit `840cea27`
math; full table in the test output / `~/tmp/bandvis-dst/append2-tests-run2.log`;
fixture-extraction motion proven by `append2_bandvis_behavior` reproducing
every gates-doc number bit-for-bit: ladder 0.647/0.414/0.121, ratio 1.715,
b2 0.099, bayer [0.169, 0.207, 0.361, 0.643], lattice 0.433, deband
0.324/0.414).

| gate | registered | measured | verdict |
|---|---|---|---|
| F2 noise-dither ratio @s3 | < 1.0 (MASKS) | OFF 1.715 → ON **1.959** | **FAIL — the fix's primary claim falsified.** Undithered (real banding) GAIN fell 0.414→0.357 while dither fell only 0.710→0.699: the mask suppresses REAL banding MORE than dither |
| F2b bayer per scale | every scale ≤ OFF | [0.169, 0.188, 0.336, 0.641] vs OFF [0.169, 0.207, 0.361, 0.643] | pass in direction, but reductions are 0–9% — cosmetic |
| F3 lattice max GAIN | < 0.5 × OFF (0.217) | 0.321 (s3 0.433→0.247, −43%) | FAIL the bar (real −26% overall reduction, under-bar) |
| F1 ladder | rungs >0.05; unimodal-interior; rolloff | rungs ✓ (3b 0.068); rolloff ✓ (0.623>0.357>0.068); interior peak **missed by 0.6%** (peak relocated to 7b 0.6271 vs 5b 0.6231 — coarse rungs self-mask, 2-code steps ≈ the δ optimum don't) | partial |
| F4 src-texture | < 0.5 | ratio 0.103 (OFF 0.099) | PASS |
| F5 deband | LOSS>GAIN and LOSS ≥ OFF | 0.357 > 0.337 ✓ but LOSS fell 0.414→0.357 and the margin collapsed 0.089→0.020 (b_dst mask on the SMOOTH dst is ≈1 ≥ the banded src's flat ⇒ gain UP 0.324→0.337) | direction pass, margin gate FAIL |
| F6/F7/F8 structural | exact-0 / lanes-only / serial≡parallel | all pass (`append2_dst_activity_lanes_only_identity_and_parallel`) | PASS |
| F9 HDR | lanes-only + still fires | PU GAIN ON [0.159, 0.408, 0.413, 0.257] (OFF [0.171, 0.455, 0.477, 0.254]) | PASS |

**Mechanism (the load-bearing finding):** `bounded_excess(a, b, c) =
(a−b)⁺/(a+b+c)` is SCALE-INVARIANT in its arguments wherever one side
dominates (`t·(a−b)/(t·(a+b)+c) ≈ (a−b)/(a+b)` for `t·(a+b) ≫ c = 1e-4`).
A multiplicative flatness mask inside the pair therefore CANNOT suppress a
dominant dst band term — dense dither keeps `b_dst ≫ b_src` and its
per-pixel excess stays ≈ saturated regardless of the mask. Suppression
materializes only where the mask flips the a-vs-b BALANCE, i.e. where the
src carries its own in-band term — which on u8 fixtures is the smooth
ramp's micro-staircase floor under REAL banding rungs. Net effect: the
self-mask attacks the true-positive side harder than the false-positive
side. This also explains the SHIPPED math retroactively: the ref-side
flat inside the pair is equally ratio-cancelled (V3(b2)'s 10× "source
masking" comes from band_s FR-cancellation, not from the flat factor),
which is WHY V3(b) measured dither firing in the first place.

Additional arm-1 finding: dst dither now fires **LOSS** strongly at fine
scales (s0 0.519→0.649, s1 0.198→0.356 on ramp→dither) — the u8 ramp's own
staircase reads as "structure the dither removed". Directionally defensible
(dither DOES hide the staircase) but it converts a GAIN cross-fire into a
LOSS cross-fire at damaging magnitude — LOSS is the LYB workhorse
(−0.447), so this is a real hazard, not a freebie.

## ARM 2 pre-registration (BEFORE its measurement): visibility-weighted POOLING

The mechanism dictates the fix shape: the flatness must enter OUTSIDE the
ratio, as a per-pixel POOLING weight, where normalization cannot cancel
it. Registered arm-2 math (same dst-activity plane, same toggle, same
kernel instantiation — only the BV_DSTACT=true combine changes):

```
(g0, l0) = bounded_excess_pair(band(curv_dst), band(curv_src), C_BV)   # PURE band terms
gain_px  = g0 · flat_dst          # new banding counts where the DST is locally flat (visible)
loss_px  = l0 · flat_src          # removed banding counts where the SRC was locally flat
```

Properties (derived, registered): identity stays EXACT-0 (band twins
bitwise-equal ⇒ excess 0 ⇒ 0·w = 0); F4's src-texture suppression is
carried by the band_s FR-cancellation (which the arm-1 analysis proved is
the real mechanism) PLUS the dst weight; the deband LOSS keeps its
direction with a cleaner margin (gain cannot rise — it is weighted by
flat_dst ≤ 1 and g0 ≈ the OFF gain on clean fixtures).

Pre-registered arm-2 gates (fixtures unchanged):

| id | gate |
|---|---|
| A2-F2 | noise-dither ratio @s3 < the OFF 1.715 (target < 1.0; HONEST caveat registered up front: at s3 dither activity after 3 downscales approaches banding-contour activity, so the s3 fixture ratio may not cross 1.0 — the REAL cross-fire lives at s0–s2 per the retest doc's real-content table, so A2-F2b is the primary suppression read) |
| A2-F2b | bayer GAIN at s0–s2 each < 0.6 × OFF (dense dither activity ⇒ flat_dst ≈ 0.25–0.4 at fine scales) |
| A2-F3 | lattice max GAIN < 0.5 × OFF max (0.217) |
| A2-F1 | rungs all > 0.05; cap rolloff monotone (amplitudes will drop ×flat_dst ≈ 0.5–0.7 at contours — registered as expected, not a regression) |
| A2-F4 | ratio < 0.5 |
| A2-F5 | LOSS > GAIN with margin ≥ the OFF margin ratio (LOSS/GAIN ≥ 1.28 = OFF's 0.414/0.324) |
| A2-F6/7/8/9 | unchanged structural gates |
| A2-LOSS-hazard | ramp→dither LOSS at s0/s1 must NOT exceed the OFF values (the arm-1 hazard must not reproduce; l0·flat_src with flat_src ≈ 1 on the smooth ramp ⇒ registered risk: l0 itself fires on the staircase — measure and record) |

Ship rule (registered): the toggle ships the arm winning A2-F2b/A2-F3
subject to A2-F1/F4/F6-F9 holding; if BOTH arms fail their suppression
gates, the toggle ships arm-2 math as the better-mechanism candidate but
the ADJUDICATION VERDICT is "extract bigcodec with the toggle OFF" and
the plane remains a research surface for P3's LOO round.

## Results — ARM 2

_(filled after the arm-2 block above was committed)_
