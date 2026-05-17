# SESSION-RESUME — read this first after every compact

This is the **canonical entry point** for a session restart or
post-compaction resume. Read in this order:

1. **This doc** — current state in ~2 minutes.
2. **`CLAUDE.md`** — methodology + workflow + gotchas (load-bearing rules).
3. **`CONTEXT-HANDOFF.md`** — yesterday's state if today's hasn't been
   written yet.
4. **`RESEARCH.md`** — corpus map + workflow recipes + sibling-repo map.
5. **`benchmarks/INDEX.md`** — find any prior experiment by theme.
6. **`scripts/v_next/README.md`** — find any helper script.

Then: `TaskList` to see open tasks. Work on the lowest available ID
that isn't blocked.

## Where we are (2026-05-16 evening)

### What's live in production

- **Shipped bake**: `zensim/weights/v0_18_zerobiased_lz4_2026-05-13.bin`
  (V_18 3-way concat, CID22 SROCC 0.8933 — but **note the methodology
  caveat below**).
- **Multi-bake runtime**: `PreviewV0_4` = V_18 ship + V_20 IS
  calibrated at α=0.4 raw space. CID22 B3 +0.080 lift at agg −0.008.
- **Crate**: zensim 0.3.0, never published. Swap bake bytes in place.

### What landed today (2026-05-16)

5 commits to zensim main + 5 commits to zenanalyze main:

**zensim** — pit-of-success docs + tightening:
- `ec27122e` RESEARCH.md (corpus map, workflow recipes, bakes
  inventory, sibling-repo map)
- `49f8ed1b` scripts/v_next/README.md (39-script index)
- `3d14b2bb` benchmarks/INDEX.md (TOC for 76 methodology docs)
- `ba67ff8c` CONTEXT-HANDOFF.md + CHANGELOG.md refresh
- `0fedd8ac` 8 rustdoc fixes → zero zensim-side rustdoc warnings

**zenanalyze** — same audit pattern applied to sibling repos:
- `c6458d6` preserve pre-existing whitespace in feature_transform.rs
- `0b5c1bf` zenpredict cfg(advanced) gating (5 dead-code warnings)
- `e852388` zenpredict clippy + rustdoc (5 warnings)
- `1cea505` zenpredict-bake clippy + rustdoc (16 warnings)
- `4574c9b` zenpicker API update for zenpredict 0.2.0+

Total: 26 warnings cleared across zenpredict + zenpredict-bake +
zenpicker now builds at HEAD again.

### What landed on 2026-05-15 (the methodology shift)

Bigger picture changes — read CLAUDE.md sections by name:

1. **"SROCC-only verdicts BANNED + ssim2-target training bias"** —
   every ship call requires the full Mohammadi 2025 panel
   (SROCC + PLCC + KROCC + OR + PWRC + Z-RMSE). Prior "falsified
   on SROCC" labels in `benchmarks/v0_20*` are PROVISIONAL.
2. **"CID22 is VALIDATION-ONLY"** — never use CID22 human MOS as
   a training target.
3. **"ZNPR v2 PROHIBITED"** — new bakes must be v3
   (header byte 4 = `0x03`).
4. **"Bash readonly variable gotcha"** — `$GROUPS` / `$EUID` etc.
   silently fail to assign in bash scripts.

### Two parallel critical-path tracks

The session goal is **train a V_22 bake that escapes the
ssim2-target training bias** documented in (1) above. Two parallel
target candidates:

#### Track A — IW-SSIM (Wang & Li 2011)

The IW-SSIM target corpus **completed today**:
`/mnt/v/output/zensim/synthetic-v2/iwssim_targets_safesyn_2026-05-16.parquet`
(196,086 rows, 0 errors, 3.7 hr GPU compute via `piq.information_weighted_ssim`).

Critical-path tasks (in order):
- **T1.1** (#50): Trainer `--target-column NAME` flag
- **T1.2** (#51): IW-SSIM merge script (parquet → features CSV)
- **T1.3** (#52): Train V_22-IW seed=1 against IW-SSIM target

#### Track B — CVVDP (Mantiuk et al.) — 60–77 % done via vast.ai

zenmetrics CVVDP infrastructure: 8 of 11 PINNED TASK items done
(per `~/work/zen/zenmetrics/CLAUDE.md`):

- ✓ Inventory + zen-metrics-cli wiring + versioned column name +
  schema doc + pycvvdp worker + score-pairs CLI + encoder driver
  + dual-impl chunk runner
- **PARTIAL**: vast.ai fan-out + verification GATE (blocked on a
  real-GPU smoke; local WSL2 can't satisfy the dual_impl_chunk.sh
  parity gate)
- **TODO**: production sweep over 2.37M-row safesyn store +
  parquet write-back to `/mnt/v/zen/zensim-training/<date>/unified/`

Critical-path tasks (in order):
- **T2.1** (#53): Push CVVDP dual-image to GHCR after real-GPU smoke
- **T2.2** (#54): CVVDP production sweep across all zensim training corpora
- **T2.3** (#55): CVVDP write-back to unified store
- **T2.4** (#56): Train V_22-CVVDP seed=1 against CVVDP target

CVVDP is psychophysically the strongest target — display
calibration model + HDR-aware + trained on authentic distortions.
Expected to outperform IW-SSIM as the V_22 training target if both
land cleanly.

## How to keep working (loop discipline)

This session's user directive: *"set goal and loop to continue
work, make sure messages are reread after every compact"*.

Concrete shape:

- **`/loop` is active** (dynamic mode, user-arm cadence). Each tick
  re-enters with full context. The loop's prompt is verbatim
  preserved.
- **On each tick**: run `TaskList`, work on the lowest pending +
  unblocked task, mark in_progress when starting, completed when
  done (per the `TaskUpdate` rules in the tool docs).
- **Before any non-trivial work**: re-read this doc, CLAUDE.md, and
  the relevant section of RESEARCH.md / benchmarks/INDEX.md /
  scripts/v_next/README.md.
- **After a compact**: ALL the above docs survive (committed +
  pushed to main). Re-read them in the order at the top of this
  doc, then resume.

## Background tasks running outside this session

- **Vast.ai IW-SSIM agent** (another session): scripts at
  `scripts/v_next/vastai_iwssim/` ready to launch. Deployment
  plan: `benchmarks/iwssim_vastai_deployment_plan_2026-05-15.md`.
  Local IW-SSIM run already completed (output above) so vast.ai
  is now optional / parallel-iteration-only.

## What NOT to do

- Don't ship a bake without a methodology doc per CLAUDE.md
  "Shipping policy".
- Don't use CID22 human MOS as a training target. Period.
- Don't produce ZNPR v2 bakes. Use `bake()` from `zenpredict-bake`,
  not `bake_v2`.
- Don't use SROCC alone as a verdict gate. Full Mohammadi panel.
- Don't relax test expectations to silence failures.
- Don't touch sibling repos under `~/work/zen/` without explicit
  user permission (today's session was explicitly permitted to
  touch zenpredict / zenpredict-bake / zenpicker; that grant
  carries forward).

## Pointers (canonical paths)

| What | Where |
|---|---|
| Pit-of-success entry | `RESEARCH.md` |
| AI-agent operational guide | `CLAUDE.md` |
| Yesterday's state | `CONTEXT-HANDOFF.md` |
| Methodology docs index | `benchmarks/INDEX.md` |
| Python helper script index | `scripts/v_next/README.md` |
| zenmetrics CVVDP state | `~/work/zen/zenmetrics/CLAUDE.md` (PINNED TASK) |
| Shipped bake | `zensim/weights/v0_18_zerobiased_lz4_2026-05-13.bin` |
| IW-SSIM corpus | `/mnt/v/output/zensim/synthetic-v2/iwssim_targets_safesyn_2026-05-16.parquet` |
| Safesyn training corpus | `/mnt/v/zen/zensim-training/2026-05-14-clean/safe_synth_v19_clean_features.csv` |
| Held-out corpora | `/mnt/v/dataset/{kadid10k,tid2013,cid22,konjnd-1k,aic3_ctc_epfl,aic4_sample}/` |
| Full 372-feat CSVs | `/mnt/v/zen/zensim-training/2026-05-15-full-features/*.csv` |
| Unified V_X parquet store | `/mnt/v/zen/zensim-training/2026-05-07/unified/` (7 parquets, 2.37M rows × 351 cols) |
