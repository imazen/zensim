# Cross-scale features (A6) against the streaming-only walk — design (2026-07-28)

> **OUTCOME (2026-07-28, same day): KILLED AT F1 — honest-stop, nothing
> merged.** The wave was implemented in full (forms i+ii, all §6 gates
> green in-workspace, bitwise two-pass reference parity) and the §8 F1
> kill-test ran pre-merge on 600 aic3 pairs: **median R² 0.99988** (min
> 0.99888, 9/9 lanes ≥ 0.99) of every XSW lane explained by the
> same-scale pools — the v1-IW death signature one scale up, decisively
> past the pre-registered 0.99 bar. §9's "single riskiest claim" was
> false. No 964 regime exists; the CSF chunk-3 block keeps f944. Record,
> mechanism analysis, escapee addendum, and the preserved unmerged
> implementation head (`3ca485b5`): see
> `benchmarks/append3_f1_killtest_2026-07-28.md`. One implementation
> correction is folded into that doc: the §3.2 occupancy proof has an
> odd-height tail-preemption hole (the last coarse strip can emit one
> row before the fine tail strip; fixed by a pending-activity
> mechanism). This design stays as the reference for the analysis that
> survives it: the timing math, the fold analysis, and the now-measured
> conclusion that ref-only activity-vocabulary weighting is in-span at
> ANY scale pairing.

Design only, no implementation. Decides which of gap-audit A6's three forms
(`zenpapers:docs/zensim-720-feature-gaps-2026-07-26.md` §2 fact 5, §5 A6) are
buildable in the streaming-only foldapp walk, by what mechanism, at what cost,
and in what order — against the walk as actually built
(`src/feature_v2_stream.rs` + `feature_v2.rs::foldapp_streaming_walk`, read at
`45f5f117`; architecture per `docs/STREAMING_FOLDAPP_C0_DESIGN_2026-07-26.md`).
Landing shape follows the append2 precedent
(`benchmarks/append2_bandvis_gates_2026-07-27.md`, commit `3fe73189`).

Primary sources (corpus, absolute paths):
- IW-SSIM (Wang & Li, TIP 2011):
  `/mnt/v/input/papers/4f/4ff403becbdcc10b9b6dbc03a83044df776c22aba705b8029fa99c6dd6f4a55e.md`
- HaarPSI (Reisenhofer et al., 2017):
  `/mnt/v/input/papers/50/502ca14e1004ea713a240d5934c48c96819cb2b89e4dc7b93471c53dfe1c7a18.md`

Provenance caveat: the IW-SSIM corpus md's equation bodies are
extraction-garbled (pdf-oxide dropped the inline math). Everything below is
recovered from the surviving prose + equation-number structure and
cross-checked against the surrounding derivation text; formulas are cited by
equation number, and the two load-bearing structural facts (neighborhood
composition, where the parent enters) are stated verbatim-adjacent from the
readable text. Do not quote numeric equation bodies from that md without
re-extracting the PDF.

## 1. What the primaries actually establish (and one lineage correction)

**IW-SSIM.** Five-scale Laplacian pyramid; per-scale SSIM maps; pooling
weights from a Gaussian-scale-mixture model. The neighborhood vector at each
coefficient is **"3×3 spatial neighborhood coefficients together with one
parent coefficient"** (N=10). GSM: `x = √z·u` (Eq 4), `C_u` estimated once
per subband from all windows (Eq 29), `z` per-location ML (Eq 30); distortion
channel `y = g·x + v` with `g, σ_v²` by per-window least squares (Eqs 31–33);
perceptual channel adds white Gaussian noise to both branches (Eqs 7–8). The
weight is total perceptual information minus shared:
`w = I(x;x'|z) + I(y;y'|z) − I(x;y|z)` (Eq 12), closed form in the
eigenvalues of `C_u` (Eqs 26–28), **monotonically increasing in the
distortion residual σ_v²** ("more weights are given to the regions with
larger distortions"). Weights apply at scales 1..M−1 (Eqs 45–46); MS-SSIM
scale exponents kept; no training, no new parameters. Results: information
weighting alone turns PSNR competitive (IW-PSNR); IW-SSIM best overall on all
six databases; on JPEG AIC-3 it is #2/27, SROCC 0.944, behind only CVVDP
(gaps doc §1, §3 — the near-threshold regime zensim's open gap lives in).

**The lineage correction that reorders A6's forms.** In IW-SSIM the parent
coefficient enters ONLY the pooling weight — the per-scale similarity maps
are pure same-scale SSIM. The cross-scale mechanism that earned #2/27 is
**coarser-scale-informed pooling of same-scale maps**, i.e. form (i)'s class,
not a parent term inside a normalizer. The parent-in-normalizer lineage is
VIF's vector-GSM (11/27 mid-pack on AIC-3, 459 ms) — and NLPD, the divisive-
normalization champion at #5, uses **same-band neighbors only, no parent**.
So form (iii) has weaker direct evidence than the A6 one-liner suggests; the
evidence mass sits on forms (i) and, secondarily, on decay structure.

**HaarPSI.** Six 2D Haar filters (scales j∈{1,2,3} × horizontal/vertical),
à-trous style — full resolution, no decimation, hence no cross-resolution
alignment problem (zensim's decimated pyramid has exactly that problem; §3 is
about solving it). Similarity `HS^(k)` = logistic of the mean over scales 1–2
of `S(|g_j^(k)∗f₁|, |g_j^(k)∗f₂|, C)` (Eq 10); weight
`W^(k) = |g₃^(k)∗f|` — the scale-3 response, low-frequency relative to the
similarity band (Eq 11) — combined `max(W_{f₁}, W_{f₂})` over BOTH images
(Eq 13); one weighted average over space and orientation (Eq 12); C=30,
α=4.2. Mean SROCC 0.9279 (color) at 24 ms vs FSIM 0.9076 at 142 ms — the
frequency-separated weighting beats FSIM's phase congruency at 1/6 cost.
Note for adaptation honesty: HaarPSI's weight is max-both-sides and
H/V-separated; the foldable zensim form below is reference-only (the fold
constraint) and orientation-free (orientation is A10's problem, deliberately
last per the gaps doc).

**House priors that bound expectations.**
- The cheap same-scale IW proxy is dead: v1's `1 + k·blur(|s−μ|)` weight
  regressed on basic+peaks at median R² 0.9980
  (`benchmarks/iw_pool_underuse_investigation_2026-05-25.md`). What A6-i
  claims is that the PARENT-scale weight field is not in that span — that
  claim is falsifiable and §8 pins the measurement.
- The one existing cross-scale feature, `EDGE_WIDTH_CHANGE`
  (`feature_v2.rs:5329-5350`: finalize-time
  `decay = mean_grad(s)/(mean_grad(s−1) + C_GRAD_DECAY)` per side,
  `1 − bounded_sim(decay_src, decay_dst, C_EDGEWIDTH)`) is classified
  K2 — "~neutral, non-foldable" (`zenpapers:docs/zensim-bake-subset-plans-
  2026-07-26.md` §0). That is the closest relative of form (ii), and it is a
  weak prior for the whole decay-statistics family.

## 2. The three forms, restated against the as-built walk

The walk (`foldapp_streaming_walk`): one pass, per-scale-cursor producer.
`produce()` converts ADVANCE_ROWS=128 scale-0 rows (both sides, 3 channels)
and cascades `downscale_2x_into` so scale s+1's `hi = ⌊hi_s/2⌋`;
`find_ready()` emits the shallowest scale whose next 128-row kernel strip has
rows `[y0, y0+strip_h+HALO_P)` present (HALO_P=10; the upper halo — the lower
halo is retention, floor `next_ks·128 − 10`); `retire()` drops rows below
`min(kernel floor, downscale floor = child_hi·2)`. Per strip: Phase A (blur
chains → mu1/mu2/ssq/s12/activity/bs2 in strip scratch), Phase B (dense +
gradient + append + fold-v1 kernels → per-(scale,ch) f64 accumulators).
Finalize replays the per-scale epilogue on merged accums. Nothing but
accumulators and the rolling raw planes survives a strip.

- **Form (i) — coarser-scale weight maps for finer-scale pooling.** New
  weighted-mean pools `Σ w_parent·v / Σ w_parent` of existing per-pixel maps
  at scale k, with the weight field from scale k+1's reference side.
- **Form (ii) — scale-decay statistics.** Finalize-only scalars relating
  per-scale accumulator means across scales.
- **Form (iii) — parent coefficient in the MSCN normalizer.** Extend
  `n_i = r_i/√(var_i + C_MSCN_VAR)` (`idx_append::MSCN_DIFF_*`,
  `feature_v2.rs:3066-3070`) with a parent-residual term in the denominator.

## 3. Feasibility in the streaming walk — the ordering analysis

### 3.1 The timing facts (derived from the producer, not assumed)

Work at one scale pair (k, k+1); all row numbers at the named scale.

- Fine strip `ks` (scale k) is emitted when `hi_k ≥ 128·ks + 128 + 10`
  (last strip: `plane_h`).
- Coarse strip `m` (scale k+1) is emitted when `hi_{k+1} ≥ 128·m + 138`,
  i.e. `hi_k ≥ 256·m + 276`.
- Therefore coarse strip m runs AFTER fine strips 2m (ready at 256m+138) and
  2m+1 (ready at 256m+266), and BEFORE fine strip 2m+2 (ready at 256m+394):
  between hi_k = 276 and 394 (mod 256m) the only ready strip at these two
  scales is coarse m, and `find_ready`'s shallowest-first scan finds scale
  k's next strip not-ready. **The coarse Phase-A planes for a fine region
  arrive one-to-two fine strips AFTER that fine region was processed — and
  exactly one coarse strip's worth of fine block data is ever in flight.**
- The coarse RAW rows for fine strip ks's span, `[64·ks, 64·ks+64)`, DO
  already exist when strip ks runs (`hi_{k+1} = ⌊hi_k/2⌋ ≥ 64ks+69` at
  emission) — raw coarse data is available early; only the canonical
  BLURRED coarse planes (mu1/activity) lag.

Two structural prohibitions that kill the "just reorder" family of ideas:

- **P-A: no re-granularizing coarse Phase A.** Running scale-(k+1) blur in
  finer sub-strips to make its activity available earlier changes
  `box_blur_v_from_copy`'s f32 running-sum history ⇒ changes every existing
  scale-(k+1) plane value ⇒ breaks the append2-precedent gate that the first
  N features stay bit-stable even when the new toggle is ON
  (`append2_layout_identity_and_first924_bit_stable` class). Forbidden.
- **P-B: no second pass over fine maps.** Any "compute fine maps, revisit
  them once coarse data exists" scheme at full resolution requires retaining
  per-pixel fine maps ≈ whole-plane materialization — precisely the
  machinery the streaming-only cutover deleted (1.03 GB → 221 MB). A second
  DECODE pass is equally dead (the walk owns single-pass semantics; inputs
  stream). This variant is architecturally poisonous; it is descoped, not
  deferred.

### 3.2 Form (i): FEASIBLE — strip-lagged deferred pooling via 2×2 block-sum carry

The mechanism rides on an exact identity. Let `v` be a per-pixel map at scale
k, `w` a weight field at scale k+1, and upsample `w` by nearest neighbor
(block-constant over each 2×2 fine block). Then

```
Σ_fine w↑(p)·v(p) = Σ_coarse w(q)·B(q),   B(q) = Σ_{p ∈ 2×2 block of q} v(p)
Σ_fine w↑(p)      = 4·Σ_coarse w(q)
```

so the weighted pool needs **no upsampled weight plane and no retained fine
map — only the 2×2 block sums of v**, which are half-width rows. Because
dims floor-halve, every coarse pixel has a complete 2×2 fine block; the
trailing odd fine row/col has no parent and is excluded from this pool
(documented boundary semantic, same family as `downscale_2x`'s drop).

Walk integration (Y-only first, scales k = 0..2 weighted by k+1 = 1..3):

1. **Fine strip ks, Phase B (new side step):** compute `v` per pixel from the
   strip scratch planes (mu1/mu2/ssq/s12 + raw rows are all present) and
   accumulate 2×2 block sums into a rolling half-width buffer owned by the
   Y accumulator set. Row pairing = `downscale_2x`'s (rows 2j, 2j+1).
2. **Coarse strip m, Phase B (new side step):** its Phase A just produced the
   canonical scale-(k+1) activity rows `[128m, 128m+138)∩plane`. For each
   coarse pixel q in the strip: `w(q) = sat(act_ref(q), C_ACTIVITY) +
   IW_WEIGHT_FLOOR` (the exact iw-pool weight vocabulary of
   `feature_v2.rs:1812`, evaluated one scale up), MAC `w·B(q)` and `w` into
   the per-(fine-scale k) `WeightedSum` in f64. Then retire the consumed
   block-sum rows.
3. **Finalize:** `WeightedSum::finish()` per (map, k) — bounded [0,1] since
   `B ≤ 4` × a [0,1]-saturated map and den = 4Σw with the iw floor keeping
   den > 0 (identity-pair safe: v ≡ 0 ⇒ pool = 0 exactly).

Properties, checked against §3.1:

- **Buffer bound: 128 coarse rows per (map, scale pair).** Fine strips 2m and
  2m+1 fill block-sum rows [128m, 128m+128); coarse strip m drains them
  before fine strip 2m+2 can add more (the 276-vs-394 ordering above). Last
  strips truncate; sub-64 inputs are pre-padded to 64 (unchanged), so the
  minimum coarse plane is 8 rows.
- **Determinism/parity:** strip emission order is deterministic; both new
  steps run in Y's Phase B, which exclusively owns the Y accumulator state in
  serial AND channel-parallel modes — `parallel_matches_serial_exactly`
  class holds by construction. Accumulation stays per-(scale,ch), f64, in
  emission order (C0 fact 5).
- **No tuned-kernel widening.** The fine-map values are recomputed in a small
  side kernel over the scratch planes rather than exported from the dense
  kernel — the §A.14/§A.16 register-pressure rule stays untouched (the
  BANDVIS lesson: new work lands as separate/const-gated code, never by
  widening the hot kernel; here it does not even need a const split).
- **Which maps:** `ssim_i` (IW-SSIM's own carrier) and `mse_i` (the IW-PSNR
  result: information weighting turns pointwise error competitive). `art/det`
  variants are follow-ups behind the §8 falsifier, not in the first landing.
- **Which weight:** reference-side coarse activity in the iw vocabulary
  (`sat(act,C_ACTIVITY)+FLOOR`, "busy parent = important"). This is the
  HaarPSI weight direction (band magnitude ↑ importance) with the canonical
  activity plane standing in for `|g₃∗f|`. The masked-vocabulary complement
  (`1 − sat(act)`: "fine error under a FLAT parent") is one extra
  WeightedSum on the same block sums — cheap, and it is the banding/blocking
  direction; include it for `mse_i` only (see §7 layout).
- **Adjacent-parent (k ← k+1) only.** HaarPSI technically weights scales 1–2
  by scale 3; a fixed-coarse variant (k ← 2 or 3 for all k) needs a 4–8×
  block carry with cross-TWO-scale lag (buffer ×~3, longer retention). The
  trained head can approximate the fixed-coarse combination from the
  adjacent-parent set; not worth the machinery in v1 of the family.

**Rejected mechanism (recorded):** on-demand recomputation of the coarse
weight at fine-strip time from the raw coarse rows (they ARE present, §3.1).
Costs a duplicate mu1+activity chain at quarter area per (scale,Y) AND
produces weight values that are NOT the canonical activity plane (different
wide-buffer f32 V-state init) — CPU for semantic mud, versus a 128-row f32
carry buffer. Deferred pooling wins on both axes. On-demand remains the
fallback if the carry buffer ever conflicts with a future retention change.

### 3.3 Form (ii): FEASIBLE and free — but mostly trainer-derivable; ship only the two escapees

Finalize-only arithmetic on per-scale accumulator means; zero walk changes;
the `EDGE_WIDTH_CHANGE` epilogue (`feature_v2.rs:5329`) is the exact
precedent. The E3 audit (`bake-subset-plans` §0: features that are exact
functions of already-emitted features "add nothing an MLP can't synthesize…
only add value to LINEAR bakes or to v1-free subsets") splits the candidates:

- **E3-trapped — do NOT ship as extraction features:** cross-scale ratios of
  emitted means (`MSCN_DIFF_MEAN[s+1]/MSCN_DIFF_MEAN[s]`, GMS decay, art/det
  decay). Every such ratio is an exact smooth function of two emitted
  features. For MLP heads it is synthesizable; for linear/BVLS bakes it is a
  legitimate precomputed nonlinearity — which a TRAINER-SIDE derived column
  provides at zero extraction cost and zero regime churn. Verdict: derived
  columns in zentrain for the linear track; nothing in the walk. This is the
  honest answer to "what marginal value remains and for which model class."
- **Escapee 1 — reference spectral-decay conditioners (`XS_REF_DECAY`).**
  `gvar₁[s] = Σs²/n − (Σs/n)²` per scale from `AppendAccum::{sum_s,sum_s2}`
  exists but is NOT emitted (only `bounded_excess` pairs of gvar₁ vs gvar₂
  are — `GLOBAL_CGAIN/CLOSS`), so
  `sat(gvar₁[s+1]/(gvar₁[s]+C), ·)` is not in any model's span. It is a
  reference-only NSS conditioner (1/f spectral-slope proxy; the BRISQUE
  two-scale rationale), correct-0 steering class like `PJND_FRAGILITY` /
  `LUMA_MEAN_REF`. Not R-class: every image has a spectral slope
  (conditioner, not rare-fire error signal).
- **Escapee 2 — src-vs-dst decay similarity (`XS_DECAY_SIM`).**
  `1 − bounded_sim(gvar₂-decay, gvar₁-decay, C)` per scale pair — "the
  distorted image's energy decays across scales differently" (blur steepens
  decay, grain/sharpening flattens it). Not an exact function of emitted
  features (gvar values are collapsed by bounded_excess before emission).
  Honest prior: this is `EDGE_WIDTH_CHANGE`'s concept on window variance
  instead of gradient means, and edge_width measured K2-neutral — expect
  little, it costs nothing, and §8's falsifier kills it cheaply.

### 3.4 Form (iii): FEASIBLE-BUT-DEFERRED — ref-parent variant only, and not first

The faithful "each side normalized by its own parent" MSCN extension needs
the upsampled parent residual for BOTH sides inside the fine append kernel —
per §3.1 the canonical parent mu1 lags, so it forces the on-demand mechanism:
two extra quarter-area blur chains (src+dst) per (Y, scale pair) + a bilinear
(or nearest) upsample + a const-instantiation widening the append kernel
(which already spills registers — `feature_v2.rs` kernel doc). Estimated
+2–4% CPU for a mechanism whose pedigree §1 just demoted (VIF-lineage, not
IW-SSIM's; NLPD wins WITHOUT a parent term).

If it is ever built, build the **ref-parent variant**: one shared parent term
`σ′_i² = var_i + C_MSCN_VAR + c_p·P₁²` with `P₁` = upsampled REFERENCE parent
residual in BOTH denominators. Identity-safe (n₁ = n₂ still exact), only ONE
on-demand chain (ref side), and the added denominator term is reference-only
— the fold gradient class is unchanged from shipped MSCN_DIFF (§4). The
dst-parent variant adds cross-scale dst terms to the map's gradient and a
second chain; it is the last variant to consider, not the first.

Verdict: architecturally feasible (single-pass strip locality survives via
on-demand computation from raw coarse rows), but deferred behind form (i) by
evidence and cost. Build only if the §8 duel says the cross-scale signal is
real AND concentrated at pixel level rather than pooling level.

## 4. Foldability (the diffmap-fold constraint)

Constraint (gaps doc §5): foldable = mean of a per-pixel map, reference-only
weights allowed; the shipped fold spatializes weighted pools as `n·w·v/Σw`
with Σw from the strip pass (`top5_v2_full_scoreboard_2026-07-23.md:69-84`,
masked `w=1−sat(act)`, iw `w=sat(act)+FLOOR`; M3 ≈0 → +0.364).

- **Form (i): foldable, exactly.** `n·w↑·v/Σw↑` with `w↑` the
  block-constant reference-only parent-activity field — the identical
  two-pass shape, one scale up. ∂score/∂v(p) = w↑(p)/Σw↑ is constant given
  the reference: the folded map IS the model gradient, same as masked/iw.
  Wiring into `compute_v2_diffmap_channel_scale` is mechanical (the diffmap
  path materializes per-scale planes; no streaming constraint there).
  The HaarPSI-faithful `max(w_ref, w_dst)` weight is NOT foldable under the
  house constraint (dst-dependent weight ⇒ the fold stops being the
  gradient) and would also need a dst-side activity plane that does not
  exist (~+5% CPU, the BANDVIS-dst-mask class). If ever wanted, it is a
  scalar-regime sibling, not a fold citizen.
- **Form (ii): not foldable, by construction.** Ratios of pooled means are
  not means of any per-pixel map (K2/edge_width precedent; D4/D8 lesson
  class). The two escapees don't need to fold: `XS_REF_DECAY` is
  reference-only (correct-0 in any steering fold, `PJND_FRAGILITY` class);
  `XS_DECAY_SIM` is scalar-regime material for the rank models
  (504/Ebothg regime), like the dev2 family.
- **Form (iii): folds as a plain mean, like MSCN_DIFF.** The map is
  per-pixel, mean-pooled; internal dst-dependence (var₂) is already the
  shipped MSCN_DIFF situation. Ref-parent variant: the new denominator term
  is reference-only ⇒ ∂map/∂dst keeps MSCN_DIFF's structure (no new
  cross-scale dst coupling). Dst-parent variant: ∂map/∂dst acquires
  cross-scale terms — coarse dst error masks fine dst error inside the map —
  legal as a plain mean but the folded map's local-attribution quality
  degrades (the fold ignores ∂w-like internal terms); one more reason the
  ref-parent variant is the buildable one.

## 5. Cost budget (against the +1.79% append2 precedent; baseline foldapp2 59.6 ms/pair aic3-100 1T, 221.04 MB @12 MP)

All CPU figures are PROJECTIONS to be measured against the gate — the
append2 wave projected +1–2% and measured +1.79%; same discipline here.

| form | CPU (projected) | memory | new passes |
|---|---|---|---|
| (i) XSW pools: 2 map buffers + 3 WeightedSums, Y, scales 0..2 | +1–2% (side kernel ~20 ops/px over Y s0..2 ≈ 33% of pyramid pixels, reading planes already in scratch; block-sum ~2 adds/px; deferred MAC ~0.25 ops/fine-px-equiv) | +~3.6 MB @12 MP: 2 f32 carry buffers (ssim; mse shared by both mse pools), 128 coarse rows × Σ_{k}(w/2^{k+1}), k=0..2 → 2×128×4×0.875·w B = 896·w B ≈ **1.5 strip-plane units** (SP₀ = 148·w·4 B = 2.37 MB @ w=4000); + O(scales) f64 accums | none (both steps ride existing strips) |
| (ii) 2 escapee families, Y, 3 scale pairs | ~0 (finalize f64 arithmetic on existing sums) | 0 | none |
| (iii) ref-parent MSCN_P, Y, scales 0..2 | +2–4% (1 quarter-area blur chain per scale pair + upsample rows + append-kernel const-instantiation; register-pressure risk on a kernel already at ~19 row-lane accums) | +1 coarse mu1 row-window ≈ 0.2 SP₀ | on-demand mini-blur per fine strip |

Gate proposal for the (i)+(ii) landing wave: **CPU ≤ +2% vs foldapp2**
(drop `XSW_MSE_FLAT` first, then the whole mse buffer, if over); **RAM
≤ +6 MB @12 MP heaptrack** (unlike append2 this wave CANNOT be
RAM-identical — the carry buffers are real; the gate names the number
instead of pretending). Form (iii) is not in the wave and gets its own
budget if ever built.

## 6. Landing shape — append3 at f944+ (follows append2 exactly)

- `V2NewFeatureToggles::append3_block`, default **false**, `assert!`-requires
  `append2_block` (f944+ sits after append2's 20 — same index-density rule
  that makes append2 require append).
- Layout: `f944 + scale·APPEND3_PER_SCALE + local`, **Y-only** (documented
  layout deviation, the append2 precedent), `APPEND3_PER_SCALE = 5`:

  | local | name | form | scale semantics |
  |---|---|---|---|
  | 0 | `XSW_SSIM` | (i), iw-vocab parent weight | pair (s, s+1) stored at s; s=3 emits 0.0 |
  | 1 | `XSW_MSE` | (i), iw-vocab parent weight | same |
  | 2 | `XSW_MSE_FLAT` | (i), masked-vocab parent weight (fine error under flat parent — the banding/blocking direction) | same |
  | 3 | `XS_REF_DECAY` | (ii), ref-only conditioner | pair (s, s+1) at s; s=3 emits 0.0 |
  | 4 | `XS_DECAY_SIM` | (ii), scalar-regime | same |

  4 scales × 5 = 20 slots, full vector **964**;
  `FeatureRegime::Folded720Append3`; `append3_features()` accessor. All
  scale-3 locals are structural zeros (index-stable, deprecate-by-absence —
  cleaner than edge_width's copy-last-slot compat hack; append3 defines
  fresh semantics).
- Storage note: pair-(s,s+1) features live at the FINE index s (the map
  being pooled is scale s's) — mirrors `EDGE_WIDTH_CHANGE`'s convention.
- Entries: `compute_folded720_append3_features[_hdr]` + toggle on the
  `_streaming` batch forms; driver modes `foldapp3` / `foldapp3hdr100` /
  `foldapp3hdrpq`.
- HDR route: same machinery over PU planes, no per-route constants needed
  (the weights are relative fields; `C_ACTIVITY`'s PU-domain anchoring
  carries the SAME approximately-transfers caveat as every append family —
  re-anchoring is the chunk-3/regime wave's job, not append3's).
- Gates, mirroring append2's V1–V5 verbatim where applicable:
  - V1 byte-stability OFF: aic3-100 CSVs `cmp`-identical for fold/foldapp/
    foldapp2 + HDR modes; full suite green, zero relaxations.
  - V2 sanity: 964 layout + regime + accessor; **first-944 bit-stable with
    append3 ON**; serial ≡ parallel at 964; all 20 slots ∈ [0,1]; identity
    pair ⇒ XSW_* and XS_DECAY_SIM exactly 0, XS_REF_DECAY ∈ (0,1];
    odd-dims + h<128 + tall-thin + sub-64 fixtures (the carry buffer's
    boundary semantics are where the bugs would live).
  - V3 behavior: synthetic divergence fixtures — same fine error placed
    under flat-parent vs busy-parent regions must separate `XSW_MSE` vs
    `XSW_MSE_FLAT` (the mechanism's existence proof); blur ladder moves
    `XS_DECAY_SIM` monotonically.
  - V5 perf/RAM: the §5 gates, measured, quiet box, 4 interleaved rounds.
- Regime note (verbatim policy from append2): 964 rows are additive-only and
  OPT-IN; they join the NEXT extraction wave — never mix into draining
  924/944 tables. The in-flight 924 backfill
  (`zenmetrics/PLAN_FLEET_924_BACKFILL_2026-07-27.md`) is untouched; with
  the toggle off nothing changes anywhere.

## 7. Trap-class implications for the trainers (S/E/R bookkeeping)

Written for `bake-subset-plans` §0's inventory; add on landing:

- **NEW S-row (S7): `app3[XSW_*]` vs `v2[iw_*/masked_*]` (same scale).**
  Near-collinear on typical content — both are activity-vocabulary weighted
  pools of the same maps; they diverge exactly where fine error's SAME-scale
  context and PARENT-scale context disagree (blocking inside locally-smooth
  regions of a globally-busy image; fine grain under coarse flatness). This
  is the v1-IW death's second chance and must be treated as such: BVLS not
  Adam, swap-don't-add vs the same-scale iw pools if LOO says subsume, and
  the enriched mass that separates the twins is blocking/banding-in-smooth +
  near-threshold KADIS (the p1_fulldata lesson that mass, not architecture,
  moved KonJND).
- **NEW near-collinear cluster note: three weighted-mse pools.**
  `LUM_DARK_ERR`/`LUM_BRIGHT_ERR` (luminance bins), `XSW_MSE`/`XSW_MSE_FLAT`
  (parent activity), and plain `v2[MSE]` all pool `mse_i` under different
  ref-only weights — an E2-partition-caveat-class warning (non-orthogonal by
  design), not an exact dependency.
- **E3 rows:** cross-scale ratios of emitted per-scale means (MSCN/GMS/art/
  det decay) are E3 — trainer-side derived columns for the linear track
  only, never extraction slots (§3.3). `XS_REF_DECAY` and `XS_DECAY_SIM`
  escape E3 (gvar₁/gvar₂ are not emitted, only their bounded_excess
  collapse); `XSW_*` escape E3 (new Σw·v sums, not functions of emitted
  scalars).
- **R-risk: none expected.** XSW pools fire on every distorted pair;
  XS_REF_DECAY is a content conditioner. XS_DECAY_SIM could be near-constant
  on mild-distortion corpora — watch for R-class behavior in the P12 run
  before giving it rank-model weight.
- **Sign policy:** XSW_* and XS_DECAY_SIM are distortion-increasing
  ([0,1], 0 at identity) — sign-safe candidates pending the standard
  direction sweep; XS_REF_DECAY is a conditioner (free-sign, conditioner
  handling like LUMA_MEAN_REF).

## 8. Evidence-ranked build order, with falsifiers

1. **A6-i XSW pools + A6-ii escapees, one wave (the append3 landing).**
   Evidence: IW-SSIM #2/27 AIC-3 (parent-informed WEIGHTING is the
   mechanism, §1); HaarPSI 0.9279 @ 24 ms (coarse-band weighting beats
   phase congruency at 1/6 cost); the M3 fold result (+0.364) says weighted
   pools are where coherent-regime capacity lives; (ii) rides free.
   Falsifiers, in measurement order, each sufficient to kill its target:
   - **F1 (mechanism-dead check, cheapest):** regress each XSW feature on
     the same-scale masked/iw pools + basic over the canonical corpus. If
     R² ≥ 0.99 across content classes — the v1-IW death signature
     (`iw_pool_underuse_investigation`'s 0.998) — the parent-scale field is
     in the same-scale span; kill XSW, keep only (ii)'s escapees.
   - **F2 (P12-residual):** rerun the residual-boost instrument on a 964
     extraction. XSW family below TEXTURE_DISSIM's 0.114 CV-R² floor ⇒ no
     marginal value ⇒ kill.
   - **F3 (P13 TWIN-DUEL + LOO):** 964 bake; LOO must show the XSW family
     LOO-positive (family hurts when removed) somewhere in the coherent
     slice (CID22-coherent, KonJND) — the E2-criterion shape BANDVIS owes.
     Neutral-everywhere ⇒ S7 says swap-don't-add failed ⇒ drop from
     default bakes, keep extraction slots (deprecate-by-mask, house rule).
   - **F4 (decay escapees):** if `XS_REF_DECAY` adds nothing over
     `GRAD_SRC_MEAN`/`LUMA_MEAN_REF` in the conditioner role and
     `XS_DECAY_SIM` repeats edge_width's K2 neutrality ⇒ zero-fill both at
     the next layout revision. Expected-cheap to lose.
2. **A6-iii ref-parent MSCN_P — conditional, after F1–F3 read out.** Build
   only if XSW survives F1–F3 AND the duel's divergence rows concentrate at
   pixel level (flat-parent fine error mispooled rather than mispredicted)
   — that is the one signature separating "cross-scale masking belongs in
   the normalizer" from "in the weight." Its own falsifier at build: LOO
   delta vs plain MSCN_DIFF on the same bake; identical ⇒ the parent term
   is redundant with the pooling-level mechanism, kill permanently.
3. **Never in this family's scope:** faithful GSM/eigenvalue information
   weights (needs both-sides + covariance per pixel — cost class of the
   §6-banned VIF), dst-inclusive max-weights (needs the nonexistent dst
   activity plane; scalar-regime sibling at best), fixed-coarse HaarPSI
   pairing (cross-two-scale carry; head can approximate), any order
   statistics in the new pools (D4).

## 9. Honest verdict summary

| form | verdict | mechanism | foldable | cost | order |
|---|---|---|---|---|---|
| (i) coarse-weighted pooling | **FEASIBLE — build first** | strip-lagged deferred pooling: 2×2 block-sum carry (≤128 coarse rows/pair), MAC at coarse strip's Phase B against canonical activity | yes — exact masked/iw two-pass shape, ref-only weight | +1–2% CPU, +~3.6 MB | 1 |
| (ii) scale-decay stats | **FEASIBLE — mostly don't ship** | finalize-only (edge_width precedent); E3-trapped members become trainer-side derived columns; only the 2 escapees land | no (scalar-regime; conditioner is correct-0) | ~0 | 1 (rides along) |
| (iii) parent-in-normalizer | **FEASIBLE-BUT-DEFERRED** | on-demand quarter-area ref-mu1 chain + upsample; ref-parent variant only | as plain mean (MSCN_DIFF class); ref-parent keeps gradient class | +2–4% CPU, kernel-widening risk | 2, conditional |
| any second-pass / re-granularization variant | **POISONOUS — descoped** | violates P-A (bit-stability-when-ON) or P-B (single-pass, no materialized planes) | — | — | never |

The doc's single riskiest claim, stated as such: that the parent-scale
activity field escapes the span that killed v1's same-scale IW proxy. F1 is
deliberately first and deliberately cheap because if that claim is false,
everything else in §8.1 is noise.

## Open questions (for the coordinator / user)

1. Gate arithmetic: append2 accepted +1.79% against ≤+2%. Stacking append3's
   projected +1–2% puts the foldapp3 route at ~+3–4% over foldapp — is the
   per-wave gate (each append wave ≤+2% over its predecessor) the standing
   policy, or is there a cumulative ceiling for the 924→964 route that
   should be pinned now?
2. `XSW_MSE_FLAT` (masked-vocabulary parent weight) is the marginal slot —
   it exists because the flat-parent direction is the banding/blocking
   story, but it is also the most collinear with BANDVIS's ref-flatness
   mechanism at adjacent scales. Keep in the first landing or hold at 4
   locals (drop to `APPEND3_PER_SCALE = 4`, vector 960)?
3. The E3-trapped decay ratios need a zentrain derived-column registry so
   the linear track actually gets them — is there an existing home for
   derived columns, or does that land as a small zentrain feature alongside
   F2's P12 rerun?
4. Layout arithmetic pins 964 (or 960 per Q2) as the next regime tag —
   should the HDR backfill wave (which will re-extract everything under the
   chunk-2 front-end) wait for append3 so the fleet runs once, or run at
   944 with append3 joining the wave after (two extraction regimes in
   flight)?
