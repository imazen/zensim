# Task: build `joint-core-v1` — a compact, photo-dominant training core that a full cycle can chew in an hour

Repo `/home/lilith/work/zen/zensim` (jj-colocated; the box is now QUIET — the previous constants run was stopped).
Read FIRST, in full: `docs/PLAN_JOINT_CORE_SET_2026-09-19.md` (this task IS that plan),
`docs/FITTED_CONSTANT_GUARDS_2026-09-19.md`, `docs/PREREG_SCALES_PLANES_2026-09-19.md`, `docs/DATA_SPLITS.md`
(+ its override), `../DATA_PROVENANCE.md` (especially the 2026-08-22 canonical imazen-26 entry and the
VARIANTS-SPEC discipline), `CLAUDE.md`. `/home/lilith/.claude/CLAUDE.md` + `~/work/zen/CLAUDE.md` bind you.
Outputs: `/mnt/v/output/zensim/joint-core-v1/` (hard cap 25 GB; `/mnt/v` must keep ≥80 GB free).

## What to build
A named, versioned TRAIN-role view — `joint-core-v1` — of ~50,000 pairs that can train BOTH zensim heads and a
DVIFM-style constants fit, with these binding properties from the plan:

1. **≥75% camera photography by pairs.** Today's imazen26 leg is only 19% photography from 40 origins; the
   canonical corpus has 111 origins in the photo classes (1000-photos-general, 1200-interiors, 1400-nature,
   1600-food, 2000-people, 3000/3300-museum). Use them. Remaining ≤25%: ≥5% each screen content, document/text,
   line-art/synthetic as GUARD classes; AI-generated imagery capped at 5% and labelled.
2. **Scale ladder: capped at ~1 MP (1024 px long edge), and not tiny-dominated.** ≥55% of pairs at 384–1024 px,
   ~25% at 192–256, ~20% at 64–128. Never above 1024.
3. **Mitchell–Netravali for every resample**, via the imazen `zenresize` crate (read its local README/source
   first; no foreign resamplers), identical kernel both sides of a pair, kernel recorded per variant.
   **First: determine what kernel produced `/mnt/v/output/clean-picker-corpus-2026-06-26` (its `_provenance.tsv`
   does not record one).** If it is not Mitchell, regenerate the renditions you need from the canonical imazen-26
   PNGs rather than reusing them, and say so. Canonical corpus membership/splits come from the codec-corpus
   manifests per DATA_PROVENANCE 2026-08-22 — verify by sha256 against a manifest before building on any
   `/mnt/v` cache.
4. **Never clip a signed teacher target.** Keep raw signed ssim2. If a consumer cannot take negatives, that
   consumer's view drops those rows and records the count — never clamp. Also fix the other end: do not let
   near-lossless rows stack on an exact ceiling (`codec_fit` today is 15.4% at exactly 100).
5. **Clustered, not random selection**: k-means over the existing `feat_*` embedding per leg, centroid-nearest
   members, singletons kept; record cluster sizes and seeds.
6. Legs and shares per the plan, adjusted so (1) holds; every leg keeps its own target column and orientation, and
   legs are never column-mixed. Reference-level disjoint from every eval corpus; dHash audit against CID22-49,
   AIC-3, AIC-4, AIC2026, SDR25, KonJND val, KonFiG test with flags adjudicated, not auto-quarantined.

## The one-hour gate — design for it from the start
- Constants fit ≤10 min for all (plane, level) of a 3-plane variant. Build **2-D histograms of (C̃, m) per
  (plane, level)** in the extraction pass — each grid cell's loss is a sum of per-block terms, so the histogram
  reproduces the per-block grid exactly. Verify that equivalence on one small domain and report the max deviation.
  Per-block records only on a capped, quantised subsample (≤40 KB per row per plane) for the gradient steps.
- Five-seed `basic228/h128` fit ≤40 min, ≤2 concurrent under run-heavy. Whole repeat cycle ≤1 h. Report each
  measured time. If a stage misses its budget, shrink ROWS first — never seeds, never budgets.
- Encoding: imazen codecs only (zenjpeg/zenwebp/zenavif/zenjxl), q grid denser below q60, persist encoded bytes
  content-addressed per the workspace rule, and record per-cell `encode_sha`.

## Admission gates (all four, reported with numbers)
1. **Leader reproduction:** refit `basic228/h128` on the core with the leaders' seeds (17101/03/07/11/13) and
   compare to the frozen R915 result on the leaders' own tables (`/var/tmp/zensim-validation-2026-09-15/recovery/`,
   READ-ONLY). Within seed noise, or state the measured gap.
2. **Convergence:** the development metric must be flat or rising between the 50- and 100-epoch checkpoints. The
   8,327-row estate failed this.
3. **Permuted-column control:** 30 permuted inputs must cost less than seed noise. If they still cost ~0.003
   SROCC, the core is too small — grow it and say so.
4. **Coverage report:** pairs and origins per content class, per codec, per scale step, per quality decile, plus
   the photography share — the numbers that prove (1) and (2) above.

## Records
`benchmarks/joint_core_v1_2026-09-20.{md,json}` + `.pointer.md` (nothing >30 KB in git), MISSING list first,
`_MANIFEST.json` with `build_commit`, per-input sha256, cluster/seed rules and the kernel; an addendum entry in
`docs/DATA_SPLITS.md` registering `joint-core-v1` as a TRAIN-role view (no new role for any corpus); append-only
2-space-indent `board_discussion_sets.json` entry. State plainly which numbers are measured.

## Rules
`jj` only, small commits, **DO NOT PUSH**; heavy work only through `~/work/zen/scripts/run-heavy --mem 16G
--jobs 8 -- … 2>&1 | tee ~/tmp/devin/<n>.log`, ONE at a time; scratch only `~/tmp/devin/`, never `/tmp`; never
delete caches or generated data; `cargo fmt --all -- --check`, `just clippy`, `just lint-scripts` clean at the end;
no public API additions; never relax or `#[ignore]` a test (the 5 pre-existing `zensim-validate/tests/
bake_surface.rs` failures stay); refresh `/home/lilith/work/zen/zensim/.workongoing` (`<UTC ts> devin-core
<activity>`) every ≤2 min (another session may overwrite it — ignore that); touch no other repo; no GitHub writes;
no household names/MACs/LAN IPs in any output (the canonical class names contain one — strip it); no `pgrep -f`.

## Reporting
Progress lines to `~/tmp/devin/core_progress.log` at every step. Terminal file `~/tmp/devin/CORE_DONE.md`:
commits, the coverage report, the four gate results with numbers, every measured stage time against the one-hour
gate, the kernel finding for the existing renditions, and what was NOT done. Or `~/tmp/devin/CORE_BLOCKED.md`.
A measured "the core needs to be bigger" is a good outcome; an unearned pass is the worst one.
