# ITERATION_PROTOCOL.md — fast, replicable experiment cycles (2026-07-02)

How to run zensim training experiments quickly WITHOUT giving up the
reproducibility/holdout discipline in `docs/DATA_SPLITS.md`. Written after a
day that exposed every slow path: 80-min cells (full per-epoch eval), whole
cells lost to fragile manifest string-surgery, collapse basins invisible to
selection, local OOM/crash losses, and serial local seeds.

## The measured cost model (2026-07-02)

A 200-epoch α-head cell = load (~1-2 min) + 200 × (train step + per-epoch
group eval). Training compute is `pairs_per_epoch`-bound (50k sampled pairs)
— corpus size costs load time + RAM, almost nothing per epoch. **Corrected
2026-07-02: a full-eval v51 cell is ~10-11 min (~3 s/epoch)** — the group-eval
forwards are f32-SIMD-fast (3.4M rows ≈ 0.5 s). Earlier "80-min cell" and
"eval-dominates" claims were clock misreads under 3-way box contention; the
idle-box A/B puts group_eval_cap at ~4%. **The real iteration costs are (a)
serial local execution, (b) wave-level failures from fragile tooling, (c)
collapse cells wasted for lack of honest selection** — levers 2-5 below.

## The five levers (all landed)

1. **`group_eval_cap` (trainer, `[training].group_eval_cap`)** — per-epoch
   diagnostics/selection forward a deterministic stride sample (pre-gathered
   once; no RNG; training byte-stream untouched). **CORRECTED measurement
   (idle-box A/B, 12 epochs): 209s vs 201s wall — ~4%, NOT the earlier-claimed
   4.7×** (that number came from a contended smoke; the full-eval forwards are
   f32-SIMD-fast and a v51 cell was ~10-11 min all along — the '80-min cell'
   was a clock misread during 3-way box contention). Keep the cap as cheap
   insurance that scales when groups grow another 10×, but the REAL speed
   levers are #2 (box parallelism) and #3 (screen-length recipes). Default 0;
   honest final numbers always from bake_verdict.
2. **Hetzner-first (DATA_SPLITS §6)** — all minutes-scale work on the train
   box (ccx63: 48 dedicated cores), 6-8 cells concurrently via
   `scripts/hetzner/runcells.sh`. With lever 1: a 20-cell grid ≈ 1 hour wall.
   Workstation = orchestration + seconds-scale evals only.
3. **`scripts/v_next/make_manifest.py`** — structured manifest generation
   (toml round-trip validated, sha256+rows computed from the files,
   `--stamp-trainer-commit`). String-surgery manifest editing is BANNED — it
   produced two broken waves in one day (duplicate key; empty `rows =`).
4. **Held-out val groups in selection (DATA_SPLITS §5)** — mandatory for new
   recipes; makes `early_stop_patience` and checkpoint selection actually
   reject collapse basins (train==val selection scored a collapsed run 0.909).
5. **Reproducibility gates** (`trainer_commit` + input sha256) — cost nothing
   per run and make every result a replicable claim.

## The screen → confirm pipeline (standard shape for any new idea)

**Stage S (screen, box):** `group_eval_cap=50000`, epochs 100, no QAT,
3 seeds, PAR on the box. Read seed-mean deltas on honest holdouts (CID22-49,
AIC-3/4, held-out KADIS safety, HQ-zone panel) + collapse count.
Kill ideas here — a 3-seed screen is ~20 min wall on the box.

**Stage C (confirm, box):** finalists only. Full 200 epochs + QAT-native +
5 seeds. Full Mohammadi panel + both dials + HQ instrument + per-band tables
+ paired-bootstrap deltas vs the ship. Pre-register the gates BEFORE the run
(CID22 within noise of A; KonJND no-regression; safety in the 0.96–0.98 band
vs the 0.980 oracle; HQ zones ≥ A) — no post-hoc goalposts.

**Verdicts:** single-seed deltas < ~0.02 CID22 SROCC are noise (measured seed
sd 0.013–0.02). Nothing ships without Stage C + a methodology doc.

## Big-box lifecycle (MANDATORY, user rule 2026-07-02)

x86 big boxes (ccx63-class) are EPHEMERAL: `hz.sh retire <name>` (snapshot →
delete) as soon as a grid's results are pulled — never leave one idling.
Restore later with `hz.sh restore <name>` from the newest snapshot (current
base: `zen-train-1-1782989687`). Every grid chain ends with pull → retire;
any session that finds an idle x86 big box (no cells running, status ALLDONE
or stale) retires it on sight.

## Data-contract validation (MANDATORY)

`scripts/v_next/validate_parquet.py` runs on every derived/pulled parquet
before it trains anything (wired into the box bootstrap; run it manually for
locally-built parquets and via `--manifest` for a recipe's full input set).
Fleet + versioning errors it already catches: footerless partial writes (OOM),
schema drift (fN vs feat_N), null/NaN/Inf, out-of-range targets (caught real
negative cvvdp targets on its FIRST run), all-constant feature columns (the
picker join bug), row-count/sha drift vs manifests, split-rule violations.

## Standing anti-patterns (each cost us a cell or a wave)

- Editing manifests with sed/string replace → use make_manifest.py.
- Uncapped heavy jobs next to training cgroups → OOM (run-heavy or the box).
- Long-lived local nohup chains → die with harness crashes; put them on the
  box (its jobs survive) or make every step independently resumable.
- Trusting train==val selection → collapse-blind; held-out val required.
- Comparing single seeds, or scoreboarding vs ssim2 on KADID/TID (in-sample
  for it) — see DATA_SPLITS §3.

## Seed fan-out + box discipline (added 2026-07-02 late)

- **Fan out seeds to fill the box**: a ccx63 (48c) runs PAR=8 cells at
  RAYON_NUM_THREADS=6. Queue 5-seed × N-recipe waves (10+ cells), not serial
  3-cell trickles. Cells are ~7-14 min → a 10-cell wave ≈ 25 min wall.
- **systemd-run for anything long on a box** (`systemd-run --unit=NAME
  --collect bash script...`) — nohup-over-ssh dies when the session drops.
- **Box sizing**: ccx53/43 for ≤5-cell grids; ccx63 only for ≥6-cell waves
  (user cost directive 2026-07-02; the idle-grid incident cost ~$1.90).
- **bake files**: manifests' [bake].file lands via the path symlink in
  /data/derived — pull bins separately from hz.sh pull (which grabs /data/out).

## Per-result reports (MANDATORY, added 2026-07-02)

**Every result gets a visual report; reports collect at
`/mnt/v/output/zensim/reports/` with a regenerated `index.html` viewer —
browsable at <http://172.23.240.1:3300/zensim/reports/>.** This is built into
the standard flow, not an extra step:

- `hz.sh pull` auto-generates a report for every pulled `.bin` that doesn't
  have one (bins land in the probe dir; reports render on the workstation,
  which owns /mnt/v + matplotlib).
- Ad-hoc: `python3 scripts/v_next/bake_report.py --bake X.bin [--label L]`.
- Each report = Cloudinary-style scatter grid (human vs prediction, 4PL
  display fit) over the 6 canonical corpora + the FULL Mohammadi panel per
  corpus (stats via `scripts/lib/zen_stats` — never hand-rolled) + per-corpus
  verdict markdowns + `meta.json` (bake sha256, trainer commit, command).
- Validity labels are baked into every page: CID22 = the 49-ref HOLDOUT
  (ssim2's own validation split — stricter than the Cloudinary all-CID22
  SVG); KADID/TID panels carry an IN-SAMPLE banner (trained here at w=0.5 AND
  ssim2 tuned on them) — integrity guards, never rankings.

## Auto-retirement (added 2026-07-03 — boxes bill by the hour)

**`hz.sh autoretire <ip> <name> [idle_min=25]` MUST be armed right after every
`hz.sh run`.** A detached workstation-side watcher (hcloud token never leaves
this machine) pulls results + bins and snapshot-retires the box when the cell
queue reaches ALLDONE with no trainers/units active — or after `idle_min`
minutes with no trainer running (crashed-run guard). **Billing-aware
(2026-07-03): Hetzner bills per started hour, so after work completes the box
STAYS AVAILABLE through the already-paid hour and retires at minute ≥45 of
the billing hour** (uptime mod 60) — queue follow-on waves into that free
window instead of retire-restore cycling. Veto with
`touch $PROBE/.box_hold` while queuing a follow-on wave; remove the hold to
re-enable. Log: `$PROBE/autoretire_<name>.log`. This turns the lifecycle rule
("retire after pull") into machinery instead of memory.

## Rigor × efficiency — co-equal constraints (USER DIRECTIVE 2026-07-19)

User, verbatim: *"do rigorous science, but also make experiments efficient."* Neither
subordinates the other: an unrigorous fast result is noise, and a rigorous result obtained
10× slower than necessary starves the next ten experiments. Both are DESIGN constraints on
every experiment, checked at design time (the 4-line hypothesis note should say what the
CHEAPEST DISCRIMINATING measurement is, not just the falsification bar).

**The efficiency ladder — always climb from the bottom:**
1. **Pre-registered gate + kill criterion FIRST** (rigor's cheapest tool is also efficiency's:
   a numeric bar decided before measuring kills a dead direction in one round, not five).
2. **Cheapest discriminating measurement first**: asm/spill inspection before wall-clock;
   64-256² fixtures before 1024²+; seed-1 before seed sweeps; a 9-pair bounds smoke before a
   13k-pair screen before a 2.3M-row corpus scan; the stored-feature rescore path
   (`bake_verdict` over parquets, seconds) before ANY re-extraction (hours). Escalate scale
   ONLY when the cheap level cannot discriminate the hypothesis.
3. **Reuse instruments, never rebuild**: the committed zenbench benches, the probe harness,
   the sidecar emitters. Building a new harness for a question an owner already answers is
   both a rigor bug (duplication drift) and an efficiency bug.
4. **Measure on a quiet box or don't measure**: check `uptime` load before benching; a
   contended measurement is WORSE than none (phase-3 2026-07-19 burned a large fraction of
   its budget chasing noise from 10+ concurrent sessions + a stray background process; the
   2026-07-02 "80-min cell" myth came from the same failure). Load-gate + record load next
   to every number.
5. **One lever per phase**: narrow scope beats broad sweeps for engineering experiments —
   phase briefs that changed one thing produced attributable results; the phase that landed
   a kernel rewrite + 7 features together lost its before/after spill diff.
6. **Time-box, then report-and-move-on**: an honest "not attempted, here is the residual
   attribution" (phase-3 SIMD deferral) preserves both rigor and the schedule; silent
   scope-creep destroys both.

**What rigor still never yields** (unchanged by efficiency pressure): pre-registration,
full-panel stats over SROCC-only, held-out discipline, honest failure reporting, golden
byte-stability tests before touching frozen code, and full-scale validation before any
constant lands in source (a sampled scan MISSED the 5.8M D1 explosion that a full scan
caught — sampling is an efficiency tool for ITERATION, never for the final claim).
