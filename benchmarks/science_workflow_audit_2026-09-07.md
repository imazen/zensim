# One useful quality dial — science and workflow audit, 2026-09-07

**Finding:** there is a useful fast baseline in D, but the evidence does not yet
establish a new unified model with the strongest perceptual results, usable
codec floors and excellent encoder-loop cost. B remains the public default;
D's September 5 “ship flip” changed D's calibration, not that alias.

This review checked checkout `0205c45c`, the September 6 measurement records,
current source and local board artifacts, and the original Claude conversation
through September 7. It did not run a new model experiment. The short navigation
page is [SESSION-RESUME](../SESSION-RESUME.md); current integration and exact bake
identities are in [CODEC_TARGET_METRIC](../docs/CODEC_TARGET_METRIC.md).

## What the product must deliver

The user asks for **one target score**, for example 80. Encoders and pickers
choose their own codec settings and reach that perceived quality with few
passes, small output, low latency and low memory. Native codec q/distance,
feature layouts and research profile names are implementation choices.

The scale needs useful resolution near lossless and at each codec's actual
floor; negative scores are valid. Identity should be 100, and distortions
should not beat a perfect copy. Neither a learned score near 100 nor the API's
identity shortcut certifies exact losslessness: compare decoded pixels for that
claim. A monotone spline cannot repair an inverted raw ranking.

A stronger rank correlation is useful evidence, but does not establish a better
targeting product. Measure signed target error, undershoots, hit rate, bytes,
passes, total encode/decode/score time and peak memory together. Show difficult
content and tails as well as medians. Judge claimed byte savings at equal
independently measured quality; saving bytes by missing the quality target is
not a win. The [existing scorecard](../docs/MODEL_SELECTION_SCORECARD.md) already
names the rank, dial, steering, RD and target exams.

## Conversation recovered and chronology checked

Primary transcript for this exact folder:

```text
/home/lilith/.claude/projects/-home-lilith-work-zen-zensim/
  9d242656-d636-45a6-9468-565163baed2d.jsonl
```

It spans **August 25 20:18 UTC–September 7 10:25 UTC**, about 107 MB. The earlier
substantive transcript beside it is
`8dcd6d39-57f0-4a84-97cd-6b9b08084fdb.jsonl` (July 29–August 27). The other two
top-level files are bridge/mode records. The exact-folder index in
`~/.claude/history.jsonl` contains 1,987 entries across 70 session IDs, reaching
back to March. Original user messages, rather than memory paraphrases alone,
were used to recover the decisions. Many earlier full session files are absent.

Prior subordinate transcript copies also exist under
`~/tmp/session-transcripts-2026-09-06/`, and a prior audit under
`~/tmp/transcript_audit/`. These private records remain local.

| Original instruction, UTC | Consequence for this review |
|---|---|
| April 26: adhere tightly to the target with low latency/memory | Measure the whole targeting loop. |
| July 16–17: one obvious, traceable scientific workflow and a consistent target across codecs | Extend existing owners and maintain source-to-output evidence. |
| August 3: a single number all parties understand | A profile name alone cannot establish a shared scale. |
| August 29: prioritize a unified ship candidate; two-/three-shot steering is the main use case | Separate ranker/dial suggestions were not adopted product decisions. |
| September 4–5: unreachable ranges cannot ship; use SSIMULACRA2 as mentor; represent the lowest configurable settings per codec | No arbitrary −50 depth requirement; name the floor instrument and rule. |
| September 5–6: comprehensive research features, efficient exact production subsets, IDs/revisions end to end | Width and observed zero columns are not feature semantics. |
| September 7 10:25:39: “Do we have something that is really good or not?” | Lead with the product verdict and one useful next step. |

The September 6 [six-decision memo](../docs/OPEN_DECISIONS_2026-09-06.md) received
no subsequent original-user ruling in the recovered conversation. Its proposed
model installs, default revisions and C/CHdr toggle change remain proposals.

Chronology was checked using original message timestamps, Git history, dated
findings and their later corrections. Filename date, mtime, an opening DONE
label, or even a copied final paragraph is insufficient. Examples:

- May's apparent same-distance compression win was overridden by the full
  study after independent quality worsened; the decision became opt-in only.
- September 5's width-based serving blocker was largely closed September 6.
  C/CHdr's remaining issue is an actual feature-toggle mismatch, not their
  storage width. See [dense serving](dense_serving_ungate_2026-09-06.md).
- The ladder-board record followed the earlier selector fix on September 6.
  It is the later operative ruler, with 450/508 inputs graded and 58 explicitly
  unmeasured. Those counts describe inputs, not rendered board rows.
- Even that newer record mislabeled C2. Source and JSON show **C2 is the tied
  fraction; C6 is above-identity count**. The population change explains the
  example C2 improvement; it is not improved model identity behavior. Dated
  corrections were added to the [board record](board_ladder_ruler_2026-09-06.md),
  [gate history](dial_addressability_gate_2026-09-04.md) and
  [dataset history](../docs/DATASET_HISTORY.md).

## What is established now

| Evidence | Product interpretation |
|---|---|
| D is served using 28 declared IDs in a 1,420-byte bake. Its lineage meets all five codec-floor bars in the September 6 comparison under `resolvable`, margin 0.5. | Useful fast baseline. This does not imply universal perceptual superiority or a pass under the older `distinct` rule. [D change](d_ship_flip_2026-09-05.md), [ladder comparison](board_ladder_ruler_2026-09-06.md). |
| The constrained MLP improves several human-rank axes at roughly D's measured speed, but fails codec floors and regresses on KonJND. | A promising challenger, not a qualified replacement. [Best-of-all](best_of_all_2026-09-06.md), [D1](../docs/OPEN_DECISIONS_2026-09-06.md). |
| `freeze_check --select` still selects a recipe with incomplete contract coverage and failed floor clauses. | Research selection differs from release qualification. `Verdict::shippable()` requires both addressability tiers to pass; other product gates remain required. [Selection result](board_ladder_ruler_2026-09-06.md). |
| A/B/BHdr/D now have explicit-ID bakes and serving plans; C/CHdr can serve but retain a train/serve activity-toggle difference. | Do not repeat the old “wide models cannot serve” diagnosis or treat their remaining conversion as score-neutral. [Feature system](../docs/FEATURE_SYSTEM_DESIGN_2026-09-05.md). |
| Sparse-D revision-2 refits lose rank/floors; a later scoped guard plus calibration is promising but has no wall-time result. | Do not infer a safe global revision flip or zero runtime cost. [Refit](rev2_refit_2026-09-06.md), [later guard](rev2_d_arms_2026-09-06.md). |
| The targeting library defaults to B; its CLI defaults to experimental `tuner-v4`. | Current examples must explicitly select `--profile codec-target`. Aligning the default is a behavior change that needs compatibility review. [Checked CLI guide](../zensim-target/README.md). |

The constrained network's identity argument also needs its stated assumptions:
28 positive-lower-bound winsor transforms move zero features away from zero.
The [actual best-of-all record](best_of_all_2026-09-06.md) measures raw identity
99.6138 for one bake. Bounding raw output by the pin is not a proof that actual
identity is the argmax. C6's pass on 9,593 cells is measured evidence, not a
global theorem for unseen distortions.

## Cleanup priorities

1. **One current product verdict in the existing board.** Put “qualified”,
   “fails: …” or “missing: …” before the exploratory composite, using the
   existing verdict owners. Show one baseline and one unified challenger first;
   retain the full research table and drilldowns. Missing identity/negative
   probes must not look like a pass. This presentation is a proposed follow-up;
   this change only fixes the requested chart axes.
2. **One reproducible identity for every comparison.** Carry the actual bake and
   executable hashes, feature IDs/revision, extraction/color/decoder semantics,
   image/encoded identities, split, initialization/order seeds, packing and
   calibration chain, plus instrument/rule identity. Extend the existing
   manifest/loader/verdict boundaries; do not create another scorer or registry.
   Reuse stored features for matched rescoring and existing harvest automation.
3. **Treat uncertainty and independence explicitly.** Compare replicated recipes,
   not the best lucky seed. A confidence interval including zero is not proof of
   noninferiority: declare a tolerable loss margin. The existing wave-6 bootstrap
   resamples individual pairs; use paired reference-cluster resampling when
   claiming uncertainty over unseen content. Preserve historical analyses with
   their registered scope. CID22 has been repeatedly used for selection; AIC4
   and SDR25 share stimuli; an SSIM2-derived “human_score” is a teacher proxy.
   See [split policy and exceptions](../docs/DATA_SPLITS.md) and
   [replication results](replication_wave_2026-09-05.md).
4. **Measure the score inside actual consumers.** Keep extraction, forward and
   complete-loop timings distinct; include warm reference reuse, representative
   image sizes, architectures, threads, memory and failure tails. A good scalar
   correlation or analytic gradient does not establish useful spatial steering.
   The [integration guide](../docs/CODEC_TARGET_METRIC.md) identifies the JPEG,
   WebP, AVIF and JXL loop owners and their different success policies.

## Smallest next scientific test

The most informative next model experiment is a **matched codec-floor training
coverage test**, including rav1e's missing anchor ladders. The September 6
best-of-all data comparison improved four codec families when true floor data
was included; it did not clear the bars. Missing rav1e supervision is a concrete
coverage gap, not proof that transfer without such data is impossible.

Use one frozen strong recipe and one changed data arm, content-disjoint training
anchors, all five true codec floors, at least three recorded initialization/order
replicates, and the same packing/calibration procedure. Pin feature revisions,
transforms, decoder and exact codec outputs; grade on a separate registered
instrument population. Require both addressability tiers and predeclared human
quality noninferiority margins, especially near threshold. Follow survivors
with independently judged fixed-target codec runs and a complete cost budget.
If floor gains do not survive held-out content, or they buy unacceptable
perceptual/loop regressions, stop and record that bounded result. No new broad
architecture sweep or default model change is implied. **Not run by this audit.**

## Changes and verification

- The summer-gauntlet codec-q charts now share a score domain derived from
  visible medians, retaining 0 and 100 as context and expanding below/above
  them. Canonical `dial.curves` remains separate from `dial_ladder` qualification.
  Both local HTML boards were regenerated through the existing generator;
  previous HTML files were backed up beside them.
- The fair board contains 850 negative median points, down to −77.13934, now
  visible on an axis reaching −80. The full historical board also exposes a
  legacy value above 100. Chromium checked shared domains, selection changes,
  positive-only reset, actual ECharts extents, light/dark themes and no errors.
  Both boards passed `gauntlet_gates.sh`.
- Fair-board embedded data is identical. Full-board science fields are
  identical; one B training-date metadata fallback was populated by the current
  generator. The fair HTML remains below the 12 MiB target; the full historical
  board remains above it. This was not a payload-size cleanup.
- Current navigation, integration, CLI and evaluation docs were corrected;
  historical tables remain dated. Unsafe old push/cleanup examples were aligned
  with current workspace instructions. `README.crates.md` was regenerated with
  the owning zenutils script recovered from commit `1921bc2`.
- Python/chart checks, standalone CLI help/Cargo metadata, documentation links
  and `git diff --check` were checked. Rust changes are comments only; no model
  bake, scoring arithmetic, CLI default or dataset was changed.

## Reading coverage and limits

All **299 related active/archive Claude memory files** were fully read (99 in
this project, 174 related sibling/global files, 26 archives; 1,678,810 bytes).
An additional July 19 backup contained 101 relevant copies: 77 identical, 22
older versions reviewed by complete diffs, and two full reads (the backup index
and a retired 214 KB audit). Transcript coverage was original user turns and pivotal assistant evidence,
not every tool result in the 107 MB conversation.

The repository inventory contained **660 Markdown files**. Current contracts,
integration documents and selected findings received detailed/full reads;
older campaign logs were reviewed through headings, dated outcomes and later
corrections. **This is not a claim that every paragraph of all 660 files was
read.** The benchmark/script/crate partition records 24 full and 552 partial
reviews; nine additional current root-owned documents were read fully, along
with the root review and seven target/site/regression documents.

Local per-file coverage, hashes, original-message extraction and detailed audit
notes are preserved under `~/tmp/zensim-science-audit-2026-09-07/`. Private
memory/transcript bodies were not copied into the repository.

### Trainer admission and reproduction continuation

The candidate-surface increment is pushed as `7cbc2458f420`. The next increment
moves existing trainer capability guards before data loading and preserves the
same guard owner at library dispatch. Unsupported CPU auxiliary losses now
refuse instead of warning and disappearing. The GPU adapter refuses options it
cannot execute and records its actual final-epoch checkpoint policy and minimum
batch size. CPU defaults that do not apply to GPU are identified explicitly.

Parquet admission now rejects duplicate IDs, mixed aliases and null features;
gapped dense tables continue to refuse until an identity-preserving reader is
available. Training resolves every group root through the existing feature-set
registry before reading rows. Unknown/mixed eras require a reason-bearing
`--historical-replay`, embedded in the bake, and remain unqualified provenance.
Formula revisions are recorded in that same registry, not inferred from width.
The Rust transform-screen loader gained stable top-N and ID-bound options;
the redundant V_20 converter is retired at `7cbc2458` with its recipe preserved.

**Reproduction split audit, before training:** the frozen September 6 control's
CID22 training origins have zero overlap with the gold validation origins;
bigcodec also has zero overlap with the registered held-out origin sets.
KADID is an old train==eval guard. The canonical KonJND dense leg includes the
JPEG validation origins, so **its KonJND result is a memorization/anchor read,
not held-out generalization**. TID's old evaluator surfaces are superseded by
the August 29 train-only ruling. The unchanged recipe is replayed only to test
training preservation. These findings cannot be waived into a new-model pass.

The three seeds remain 4004/4005/4006, 372 identity columns with 228 selected,
H128, 120 epochs, 50,000 pairs per epoch. Packing uses the exact frozen
negative-rich plus identity anchor and the existing Rust pack owner. Evaluation
uses the Rust surface on the pinned post-C root and the named ladder/probes.
New artifacts will live in `cleanup-validation-2026-09-07`; original bakes and
verdicts remain untouched. The wave owner now refuses existence-only reuse,
retains exact argv, and returns failure when any cell fails.
