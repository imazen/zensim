# ITERATION_PROTOCOL.md — fast, replicable experiment cycles (2026-07-02)

How to run zensim training experiments quickly WITHOUT giving up the
reproducibility/holdout discipline in `docs/DATA_SPLITS.md`. Written after a
day that exposed every slow path: 80-min cells (full per-epoch eval), whole
cells lost to fragile manifest string-surgery, collapse basins invisible to
selection, local OOM/crash losses, and serial local seeds.

## The measured cost model (2026-07-02)

A 200-epoch α-head cell = load (~1-2 min) + 200 × (train step + PER-EPOCH
GROUP EVAL). Training compute is `pairs_per_epoch`-bound (50k sampled pairs)
— **corpus size costs almost nothing in training; the per-epoch eval of every
group row was the bottleneck** (v51: 3.4M forwards/epoch ≈ 3× training
compute → ~25 s/epoch → ~80 min/cell).

## The five levers (all landed)

1. **`group_eval_cap` (trainer, `[training].group_eval_cap`)** — per-epoch
   diagnostics/selection forward a deterministic stride sample (pre-gathered
   once; no RNG; training byte-stream untouched). Measured: 25 → 5.3 s/epoch
   (~4.7×); 200-epoch cell ≈ 18 min. Selection SROCC on 50k rows ≈ full ±0.005.
   Default 0 (= exact historical behavior; old manifests stay
   byte-reproducible). **Standard for new recipes: `group_eval_cap = 50000`.**
   Honest final numbers always come from bake_verdict, never the sampled
   diagnostics.
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

## Standing anti-patterns (each cost us a cell or a wave)

- Editing manifests with sed/string replace → use make_manifest.py.
- Uncapped heavy jobs next to training cgroups → OOM (run-heavy or the box).
- Long-lived local nohup chains → die with harness crashes; put them on the
  box (its jobs survive) or make every step independently resumable.
- Trusting train==val selection → collapse-blind; held-out val required.
- Comparing single seeds, or scoreboarding vs ssim2 on KADID/TID (in-sample
  for it) — see DATA_SPLITS §3.
