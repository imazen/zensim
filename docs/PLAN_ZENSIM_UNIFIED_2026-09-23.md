# zensim unified: one model that fixes the measured weaknesses — proposal, September 23, 2026

**User direction (2026-09-23):** "get to a new version of zensim that addresses all the weaknesses discovered
and offers a unified and simplified model for the paper to cover."

This plan does not replace the [controlling production plan](PRODUCTION_PRIORITIES_2026-09-15.md). It adds
two things to it:
1. **One target model.** P0–P6 qualify a *single* composition. Qualifying a fast/rich pair, or anything
   carried forward from the profile zoo, is out of scope.
2. **The weaknesses P0–P6 do not name.** These were found by the September 22–23 evidence audit behind the
   zensim-2026 companion paper (`zenpapers/papers/zensim-2026/`; reconciliation and gap list in its
   harvest notes).

The rules, split protections, owners and gates of the production plan all still bind this work.

## What "unified and simplified" means

| Today | Unified target |
|---|---|
| Nine named models that a reader must keep apart: PreviewV0_2 (crates.io), A, B (default), BHdr, C, CHdr, D, Rev3 fast, Rev3 rich; feature regimes 156/228/372/720/944 across three arithmetic revisions | **One model**: one feature contract (Rev3 arithmetic, one registered feature set), one trained head, one calibration spline, one public score. The older profiles become deprecated aliases for one release, then are removed. |
| SDR and HDR as separate bakes (B vs BHdr, C vs CHdr) | One composition over declared input, with SDR and HDR in one TRAIN mix (P4's first hypothesis). An internal metadata-selected head is allowed only if TRAIN evidence forces it. Users still set one number. |
| Scalar and map computed by different routes, which can disagree (the identity bug of 2026-09-18) | The map is the scalar's own attribution. One extraction; identity handled by a single owner at every entry point, including `*_with_ref*` (store a source hash in the prepared reference). |
| Integrity folded into the score in some bakes | A separate, thresholded integrity flag (P5 contract); never folded into the quality score. |
| Arithmetic revision chosen through a process-wide environment variable | Declared by the bake only (partly done: narrow plans already serve the declared revision). |

Speed does not force a fast/rich split. At 1024², one thread, both Rev3 ensembles are well inside the budget
against fast-ssim2 (records in `benchmarks/speed_matrix_2026-09-18.md`). Choose the one feature set that
passes P2/P4. If a fast mode survives at all, it must be a strict prefix of the same model (coarse scales
only, same head family), never a separate model.

## Weakness ledger → where it gets fixed

Evidence for each row lives in the paper's claim registry and harvest notes. P# = production-plan section.

| # | Measured weakness | Fix owner |
|---|---|---|
| W1 | Local ordering: the shipped default trails SSIMULACRA2 within-image on CID22, on AIC-3 and on JPEG-AIC forced choices; B8–B9 band ordering is weak; floor ordering fails cells the mentor passes | **P2** (within-image ranking supervision on TRAIN) |
| W2 | Cross-codec ordering. The forced-choice exam separates metrics only on cross-codec questions. | **New, part of P2.** Add cross-codec TRAIN pairs at matched quality (same source, different codec) and test them as a stratum. No human holdout is used for fitting. |
| W3 | Near-threshold (KonJND) failure. Nine mechanisms falsified; the quantity is absent from the training signal. | **Squintly** (paid human data), per the production plan. Until then, state the limitation; no metric-derived proxy. |
| W4 | Label circularity. Most targets are SSIMULACRA2-derived, so "ties SSIMULACRA2" is partly inheritance. | **New.** A TRAIN label mix that is not dominated by one teacher: human legs where roles allow, plus agreement-filtered multi-teacher targets (two-reference agreement already works for inversions). Report teacher share per bake. |
| W5 | Dial contract. The default fails G-ADDR rows that SSIMULACRA2 passes; there are near-lossless inversions and codec-floor ordering failures. | **P1/P2**, with G-ADDR as a hard gate (existing rule) |
| W6 | Targeting tails too wide at 3 shots; the richer model has the worst tails | **P1** |
| W7 | Spatial steering shows no independent-judge byte saving; maps fail on text and screens | **P3** |
| W8 | HDR: SDR-trained weights regress on UPIQ; BHdr trails the HDR specialists | **P4** |
| W9 | No qualified integrity head; progressive-AC truncation miss | **P5** |
| W10 | Small images: served B is slower than fast-ssim2 at 64²; fixed per-call overhead dominates thumbnails | **New, part of P6.** Measure α (the per-call fixed cost) for the unified model; target ≤ fast-ssim2 at 64². |
| W11 | Determinism is scoped, not global: FMA vs unfused (scalar/wasm) scores differ up to 2.8e-2; threading and libc effects existed historically | **New, part of P6.** One cross-tier determinism gate for the unified model, with the tolerance stated per tier; aim for bit-identical scores across thread counts. |
| W12 | Memory contract never measured: per-worker incremental memory, and peers only historical | **P6** (the paper-memory lane is measuring the current state now) |
| W13 | Identity cannot be certified on `*_with_ref*` entries | **New, small.** Keep the source hash in the prepared reference. Needs a public-API change, so user approval first. |
| W14 | Profile/era sprawl confuses users and evaluations (mixed-era tables, a stale README) | The **unification** itself, plus one README/API correction batch at release |
| W15 | CVVDP and other peers configured suboptimally in older board rows | The **paper lanes' fixes** (display-matched CVVDP, butteraugli 3-norm) become the board defaults |

## Order of work (compute-aware)

The machine is shared with the DVIFM-ish wave, and `/home` has only a few GB free. Heavy steps use the lab
lock. Execution goes to Devin swe-2 lanes; Opus is used only for rulings and final judgment.

1. **U0 — pick the single feature contract** (TRAIN only, uses existing caches). Decide between basic228 and
   Y60 as the one set, or register a smaller set that removes real work, using P2's first screen as the
   decider. Output: one registered `feature_set_id` for the unified model.
2. **U1 — P2 + W2 + W4 in one bounded TRAIN screen.** Within-image and cross-codec ranking supervision, and a
   less teacher-dominated label mix, on the chosen feature set. Five seeds only for a surviving arm.
3. **U2 — P4 mixed SDR/HDR training** of the U1 winner, one composition.
4. **U3 — P1 controller tails** against the frozen U2 composition.
5. **U4 — P5 integrity head** as a separate flag on U2's features.
6. **U5 — P3 spatial value.** The map is U2's attribution; JXL first.
7. **U6 — P6 runtime.** Includes W10 (small-image overhead), W11 (determinism gate) and W12 (memory), with
   full qualification.
8. **Release shape** (user decision): profile aliases, the deprecation window and the public API delta
   (W13). Then the paper covers this one model.

Each step gets a preregistration, a fresh result directory, and a DONE report with the MISSING list first.

## Decisions for the user

- **D1.** Retire the older named profiles to deprecated aliases in the next release? Proposed: yes; crates.io
  0.2.7 users keep PreviewV0_2 behaviour through the alias for one release.
- **D2.** Allow the public-API change for identity certification on prepared references (W13)?
- **D3.** W4 label mix: which human legs may enter TRAIN beyond today's roles? The current rules keep CID22-49,
  AIC and KonJND out.
- **D4.** Paper timing: publish the methodology companion now (it stands on the recorded evidence) and a
  short model paper when the unified model qualifies, or hold the whole paper for the unified model?
- **D5.** AGPL headers on five files in the MIT/Apache crate (open since September 18).
