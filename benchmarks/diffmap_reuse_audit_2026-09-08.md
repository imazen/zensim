# Diffmap reuse and Rust serving audit — September 8, 2026

The first implementation packet is **integration of existing attribution
machinery with the complete candidate scorer**. Do not build another map,
controller, or native codec experiment harness. The previous roadmap omitted
substantial completed work, including a JPEG research workspace and the later
AVIF attribution experiment.

This is a source/history audit plus existing attribution tests. It is not a new
native-encoder measurement or model qualification. The September 8
[reachable-target protocol](../docs/TARGET_STEERING_PROTOCOL_2026-09-08.md)
governs new experiments; historical results below retain their original limits.

## Search and recovery

Inspected the local checkout and reachable development machines, including the
requested workstations, macOS, ARM machines and the storage server. Search
covered Git heads/remotes/stashes and registered worktrees, `jj` workspaces and
matching changes, selected copied source trees, and relevant Claude memories.
Git-only discovery is insufficient: several research directories have `.jj`
without `.git`.

- Across eight core repositories, inventoried **450 distinct Git tips**.
- Found **467 distinct matching `jj` changes** across the six scorer/encoder
  repositories; their commit objects were already in the local repositories.
- Fingerprinted **527 matching Rust file instances in 31 selected source
  roots**. Every blob was already in the corresponding local Git object store.
  This includes working files in copied/dirty trees, not just their HEADs.
- The JPEG experiment `46a6ff3083d4568ca1cd10a83681eab730381dc2` is in
  `zenjpeg--zensim-diffmap-rd`, **not an ancestor of current JPEG main**.
  Its `encode/zq.rs` is byte-identical across the two inspected workspace
  copies: SHA-256 `a9f9083c8ddcbba5e38dc003e5393b812a4d32633968d2652f0c19c120333773`.
  An unapplied recovery patch and manifest were saved in the operator's work
  share. No checkout, worktree or dirty file was reset, switched or removed.
- Remote memory comparison found three older versions of local zensim notes
  and one additional JXL memory, chiefly about allocation and encoder gates.
  The older notes do not supersede the later local corrections.

Coverage is bounded: four configured SSH endpoints rejected their changed host
keys and were not bypassed. Some copied worktrees have broken repository
pointers; their source can be inspected without treating their Git state as
valid. The storage server was searched with native filesystem/Git tools because
it has no Python. Depth limits, exclusions, inaccessible paths and partial
initial scans are recorded privately. This is not a claim to have searched
every archived object or an unreachable machine.

Private evidence: `~/tmp/zensim-fleet-diffmap-audit-2026-09-08/` contains host
inventories, ref unions, `jj` output, source hashes, memory differences, access
failures and test logs. Fleet addresses and machine identities stay there.

## What to reuse

| Component | Existing implementation | Decision |
|---|---|---|
| Spatial steering signal | `zensim/src/attribution.rs`: signed attribution density, rectangle sums, basic and folded-944 paths, retained planes, stale sessions, direct binned accumulation | This is the map owner. The older `DiffmapWeighting::ModelSensitivity` signal fold is a separate approximation; do not make it the new product path by accident. |
| Complete candidate scalar | `zensim/src/metric/bake.rs`: `BakeScorer`, declared IDs, heads, spline, affine, ensemble and corruption composition | Extend this owner to provide the sensitivities used by attribution. Preserve its exact score semantics. |
| JXL | `jxl-encoder/src/vardct/zensim_loop.rs` and `examples/zensim_diffmap_rd.rs`: H3 magnitude steering, fused/binned attribution, stale maps, emit-best, global and tile secants, split-role maps, traces | Reuse the current corrected encoder and these arms. H3 and stale/fused integration already landed; they are not new milestones. |
| JPEG research | `zenjpeg--zensim-diffmap-rd`, commit `46a6ff30`: custom bake mount, signed/absolute gradient-map arms, global q correction and `zq_rd_probe` | Recover the useful caller/instrument contract into the current `target-zq` owner selectively. Do not merge the old workspace wholesale. Its map is the older signal fold, and its model/gradient cache assumes one image/target/driver per process. |
| JPEG serving | `zenjpeg/src/encode/zq.rs`, `target_quality.rs`, `zq_seed.rs` | Keep the current block-AQ controller, float-native target search and seed owner. Upgrade the actual scorer/map attachment here. |
| WebP | `src/encoder/zensim_target.rs`: global q correction and conditional per-segment diffmap feedback | Retain the controller and segment geometry. Measure engagement and final output; enabling the feature does not prove that spatial feedback ran. |
| AVIF attribution | `examples/zensim_cq_rd.rs`, `scripts/zensim-loop/run_avif_loop.sh`: folded-944 attribution, H3 per-SB hints, hint-engagement probe, separate score/map bakes, scalar and outer controls | Already merged, including `e50a9834`. This is the later native attribution experiment to extend after JXL. |
| AVIF serving/research | `src/two_pass_zensim.rs`, `sb_pool.rs`, `target_quality.rs`; `cooptloop` history | Distinguish the classic diffmap loop, CQ attribution experiment and rav1e inner hints. They have different scoring, budget and configuration contracts. |

The codec owns its production controller. `zensim-target::demo_matrix` remains
the shared scalar comparison instrument; it does not replace the native codec
owners above.

## Chronology changes the interpretation

1. **July 18:** the JPEG workspace mounted custom bakes and model-sensitive
   maps. Its companion zensim fix made `DiffmapResult::score()` use the actual
   profile; earlier loop scores had followed the legacy weighted score.
2. **July 29–August 1:** attribution replaced the signal fold as the intended
   steering mechanism. JXL studies found that coherent maps alone did not
   guarantee useful control; H3 magnitude steering helped the tested MLP,
   while the tested linear B did not benefit. Emit-best and stale single-pass
   paths subsequently landed. See
   [the evolving #69/#70 record](../docs/PLAN_LOOP_STEERING_69.md) and
   [the C1–C3a record](attribution_map_c1_2026-07-29.md).
3. **August 4–7:** append2 coverage, folded-944 retention/fused paths and
   direct binned accumulation followed. `d0f624eb` is already in zensim main.
   The July 31 zenpapers plan's claim that folded-944 retention was unbuilt
   is superseded by these implementations. Current module introductions also
   contain older basic-only descriptions; inspect the later entry points.
4. **August 7, extended through August 29:** the AVIF CQ attribution study
   progressed from harness registration to actual H3 and split-role runs.
   The filename is **not** its last evidence date. The later record reports
   worse targeting for both tested own-map models. At its self-judged
   matched-quality pilot, own-map/pair cost +2.16%/+3.06% bytes; apparent
   aggregate savings had been undershoot. A zero-sum steering fix was also
   negative. Seeds improved the tight-budget pilot, but were fitted from that
   experiment's traces, not the newly required independent training-family
   calibration. See [the complete AVIF record](https://github.com/imazen/zenavif/blob/9ae5199b/benchmarks/zensim_avif_loop_2026-08-07.md).
5. **September 7–8:** JXL reconstruction/quantizer/filter corrections and
   zensim's complete scalar serving/feature-plan cleanup change the substrate.
   Old map gains, tolerances and fixed targets cannot qualify the current
   product without rerunning through the new bounds/split protocol.

The AVIF record's final statement that a target was unreachable after k3 is
also a historical hypothesis, not proof of a codec limit. Its K counts updates
after the seed, so K=3 means four full encodes. New work establishes witnessed
bounds first and reports actual encodes and internal comparisons separately.

## The real Rust serving gap

The missing piece is not “a Rust diffmap implementation.” It is a **single
candidate-bound score/attribution contract**:

- `BakeScorer` has complete scalar composition but no candidate gradient or
  attribution method. The existing profile finite-difference helper has a
  different caller surface and no codec-hint argument. The new candidate path
  must differentiate the same full `score_features` operation used for scoring,
  including codec calibration and all companion models.
- Attribution accepts a caller-supplied gradient. That permits an intentional
  separate map model, but it does not prove that the map belongs to the score
  being returned. Store full scorer/map identities and the feature/formula
  contract together; distinguish same-model and split-role experiments.
- The folded-944 entry returns a v1 `ZensimResult` **whose score is not the
  944 candidate score**, plus a separate feature result which callers forward.
  Its retained extraction uses its established toggles; the candidate surface
  uses `Plan::for_bake`. Joining them requires consumed-feature and scalar
  parity, not replacing the returned scalar after an unchecked extraction.
- Coverage is explicit, not universal. Basic f0–155 and supported later
  integrands exist; f156–371, reference-only slots, SDR-inactive HDR slots and
  other unsupported terms must remain distinguishable. A missing integrand
  cannot silently become “zero effect.” Raw sensitivity magnitudes across
  differently scaled features do not establish missing score contribution.
- A tree corruption gate is discontinuous. The served score may floor to zero;
  a local gradient cannot describe crossing the gate or promise finite-block
  improvements. Keep full scalar execution, explicit map limitations and
  finite intervention checks. Do not silently substitute the perceptual head.
- Existing stale sessions and binned maps are reusable performance work.
  Their saved-work claims differ: a stale map/gradient is not a fresh one, and
  folded-944's fused entry still includes a v1 walk. Measure real work before
  describing the composed path as one pass.

## Next bounded implementation packet

1. Extend `BakeScorer` with the full-surface feature sensitivity operation;
   preregister its exact public signature and native JXL caller before editing.
   Reuse the existing forward and attribution owners. Test the complete head,
   spline, codec-affine and companion compositions, including negative scores,
   dense IDs, discontinuities and nonfinite refusal.
2. Make the existing retained extraction obey the candidate's declared feature
   contract. Establish ordinary-score versus score-plus-map parity before
   optimizing reference/scratch reuse. Return explicit coverage/limitations;
   no unconditional “all models have complete maps” claim.
3. Wire that surface into the existing JXL experiment on the reconstruction-
   corrected encoder. Start with one candidate, its scalar control and active
   versus neutral attribution. Prove quant-field engagement, independently
   decode the emitted bytes and count all inner/outer work.
4. Use train-family-calibrated seeds and per-image witnessed bounds for actual
   1/2/3-shot comparisons. Preserve the original score scale. Expand geometry
   and source coverage before interpreting tails or production performance.
5. Only after the shared contract and JXL result are sound, port the candidate
   attachment into the existing JPEG/WebP owners and revisit AVIF against its
   already-negative attribution controls. No new gain sweep or feature ablation
   merely because an old plan called it unfinished.

## Verification in this audit

At source `394239df` (with only empty local descendants), ran:

```sh
../scripts/run-heavy --mem 16G --jobs 8 \
  env ZENSIM_FORMULA_REV=1 RAYON_NUM_THREADS=8 \
  cargo test -p zensim --lib \
  --features custom-profiles,feature-regime-v2,corruption-head \
  attribution::tests -- --test-threads=8
```

**26 passed, 2 ignored** (performance measurements), 0 failed. This exercises
the existing slot coverage, sum preservation, fused/standalone, folded-944,
binned and stale/recycled-buffer contracts. It does not test an unimplemented
composed-candidate map surface or establish a codec RD/performance improvement.

## Later September 8: JPEG candidate binding and actual AQ repairs

The recovered JPEG experiment now runs through complete candidate scoring and
current attribution in the existing Zq loop. [Codec result](https://github.com/imazen/zenjpeg/blob/main/benchmarks/zensim_candidate_binding_2026-09-08.md),
commit `3d4ad0d77ecb`: no process-global profile/gradient cache, explicit candidate
seed, scalar/neutral/active arms and independently verified returned bytes.
The first pass trace found old neutral clamping and missing final-strip callback
bugs. After repair all 832 blocks of the training control reach the callback;
neutral is byte-exact, active changes maps/bytes/pixels reproducibly. The 69-byte
saving also lowers D, SSIMULACRA2 and Butteraugli quality, so this is engagement
and controller correctness evidence, not improved RD or model qualification.
Actual train-calibrated bounds/budgets and WebP binding are still required.
