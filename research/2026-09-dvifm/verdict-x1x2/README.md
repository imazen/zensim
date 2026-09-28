# verdict-x1x2 — X2 constant-form selection, then X1 standalone verdict on real labels

## What was asked

Brief: `../briefs/lane_verdict_prompt.md`. X2: pick which constant form
(gate vs curve vs prior) DVIFM should ship with, by composite score with a
2-sigma seed-noise margin. X1: score the frozen X2 winner standalone against
peer metrics (fast-ssim2, zensim bakes B/D, R915 ensembles) on every
real-label leg available, and report honestly whether DVIFM is competitive.

## Verdict (from `../reports/LANE_VERDICT_DONE.md` — copied verbatim, corrected once)

- **X2 winner: `gate`** (≤15 knees + head, ≈35 constants). `curve`'s +0.0031
  composite edge is inside 2·seed-noise (gate σ=0.0045); `prior` trails.
  Composite: gate 0.8697, curve 0.8728, prior 0.8362.
- **X1: standalone DVIFM is NOT competitive on real held-out labels** — below
  fast-ssim2 on 4/5 real-label legs (kadid_dev_full and CID22-B
  significantly), and below every zensim bake on every real-label leg with
  one exception (above `bake_prof_d` on konfig_val, +0.035). On CID22-B it is
  significantly below ALL peers (paired Δ −0.08..−0.13, P(Δ≤0)≈1.0).
- For the record: `R915_basic228` beats both fast-ssim2 and DVIFM on
  human_dev (0.931 vs 0.817/0.783), kadid_dev_full (0.945 vs 0.944/0.901),
  konfig_val (0.839 vs 0.735/0.759).

**CID22-B correction (same single read, not a second exposure):** the first
CID22-B pass showed all five bakes collapsing to 0.29-0.32 — this was
diagnosed as a feature-table era mismatch (the lane's re-extracted peer
tables were w986 research-path, but the bakes consume w944/ceiling_rev3) and
corrected by scoring the historical `rev3-public-human-eval-2026-09-14`
w944/ceiling_rev3 parquets instead (pixel≡table corr went from 0.49 to 1.0).
The numbers above are the corrected ones.

## Code in this directory

| file | role |
|---|---|
| `build_verdict.py` | X1 orchestrator: assembles the peer comparison table across every leg |
| `score_dvifm.py`, `score_peers.sh` | score the frozen DVIFM gate spec / peer metrics over each leg |
| `score_x2.py`, `x2_decide.py` | X2: score the gate/curve/prior arms and apply the 2-sigma decision rule |
| `fit_winner.py` | fits the chosen (gate) constant form after X2 selects it |
| `extract_verdict.sh` | feature/label extraction driver for all legs |
| `unseal_cid22b_verdict.py`, `cid22b_read.sh` | the single registered CID22-B read (post-freeze) — and its era-correction rescoring |

## Rust / repo state

Committed directly to the **main checkout's own history**:
- `mxoqvxrz` / `248a4deea8e6` — "verdict lane: DVIFM standalone verdict — gate
  form wins X2; not competitive with fast-ssim2 on real held-out labels"
  (adds `tools/joint_core/fit_forms.py` + the initial benchmark record).
- `zsoyusqz` / `d012f8e66de0` — "verdict lane: correct CID22-B bake scores —
  w986-era feature tables replaced by w944/ceiling_rev3" (the era-correction
  above; updates the benchmark record only).

Both are unpushed but fully committed local history — not at risk.

## How to re-run

1. `extract_verdict.sh` pulls features/labels for every leg (human_dev,
   kadid_dev_full, konfig_val, cid22_dev, safesyn_dev, codec_dev, and the
   single CID22-B read).
2. `score_x2.py` scores the gate/curve/prior constant-form arms;
   `x2_decide.py` applies the preregistered 2-sigma decision rule to pick
   the winner.
3. `fit_winner.py` fits the winning (gate) form's constants.
4. `score_dvifm.py` + `score_peers.sh` score DVIFM and the peer metrics
   (fast-ssim2, B, D, R915_basic228_ens5, R915_y60_ens5) over every leg.
5. `unseal_cid22b_verdict.py` + `cid22b_read.sh` perform (and, on rerun,
   would re-perform) the single registered CID22-B read — **do not open
   this leg more than once**; if re-running for methodology verification
   only, treat CID22-B as spent and use a fresh held-out leg instead.
6. `build_verdict.py` assembles the final comparison table.

**Important caveat carried in the report:** `safesyn_dev`/`cid22_dev`/
`codec_dev` targets are signed-ssim2/100 pseudo-labels, so `fastssim2` scores
1.0 there by construction (circular) — only `human_dev`, `kadid*`,
`konfig_val`, and `cid22b` are real-label legs; only those support the
"NOT competitive" verdict above.

## Artifacts (reference by path)

- `/mnt/v/output/zensim/dvifm-verdict-2026-09-20/` — `cache/`, `eval_meta/`,
  `features/`, `fits/`, `logs/`, `pairs/`, `missing.json`
- Struck w986-era CSVs preserved as `scores/*.w986era` per the report
- Committed benchmark record (already permanent in the repo, not copied here):
  `benchmarks/dvifm_verdict_2026-09-20.{md,json}`
