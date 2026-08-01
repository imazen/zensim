# Decision-surface audit vs the freeze plan's gate table (2026-07-31)

User directive 2026-07-31: "make sure the final dashboard rust has all the
stuff we need to decide on things." This audits every gate in
`zenpapers:docs/zensim-final-metric-plan-2026-07-31.md` §5 against the tools
that exist TODAY, names the measuring owner for each, and lists the gaps.
Verified by grep/read of `bake_verdict.rs`, `eval_report.rs`,
`scripts/run_full_eval.sh`, `scripts/v_next/gauntlet.py` at `03b001df`.

## Gate → owner map

| §5 gate | bar | measured today by | status |
|---|---|---|---|
| CID22 (selected seed) | ≥ 0.89 | `bake_verdict` rank panel (`cid22`); seed selection via the `sdr25` corpus + embedded `best_val` | **NATIVE** |
| KonJND | ≥ 0.40 | `bake_verdict` (`konjnd`, polarity-corrected \|SROCC\|) | **NATIVE** |
| Corruption ordering (via head) | ≥ 0.214 | see "Corruption: two stats" below | **PARTIAL — joint dial+head report missing** |
| CSIQ/LIVE | ≥ best 924-arm | `bake_verdict` corpora `csiq`, `live`; cross-arm comparison via `bake_compare` (MRR + bootstrap CI) or gauntlet sort | **NATIVE** |
| UPIQ pooled (V1-HDR) | > 0.7536 | `scripts/hdr/upiq_panel.py` (+ `upiq_crossdomain_instrument.py`) — Python, self-documented in bake_verdict's "Related specialized evals" table | **PYTHON OWNER** (acknowledged in-tool) |
| Korshunov hold (V1-HDR) | ≥ 0.93 | no committed runner found in this repo — lives in the seven-domain external-read tooling | **NO REPO OWNER — must be named/committed before Phase 4** |
| M3a coherence | ≥ 0.85 EM2-class @16–128px | `diffmap_block_coherence` example → `scripts/run_full_eval.sh` (`m3a_coherence`/`m3a_n` in fulleval JSON) → gauntlet `M3a-attr` column | **FULLEVAL+GAUNTLET** (not bake_verdict-inline; fine) |
| Dial monotonicity + `product_composite` | per bake_verdict gates | `bake_verdict` native (dial panel G1/G3/G4; `product_composite()` in `--full-json`; gauntlet reads, never re-derives) | **NATIVE** |
| Byte-repro | embedded `zentrain.repro`, exit-4 | trainer embed (fatal on fail) + bake_verdict repro badge | **NATIVE** |
| Perf (SDR) ≤ +2% | bench | zenbench extractor/compare benches — a bench artifact, not a verdict stat | **BENCH-OWNED** (correct home) |
| Perf (HDR) ≤ +5% | bench | same; the V5 lever list | **BENCH-OWNED** |
| LOO (append2 ≤ 0) | instrument | extractor-side LOO instrument (gaps-doc §0 program; other work line) | **OUT OF SCOPE here — plan must name the runner** |

## Corruption: two DIFFERENT stats — don't conflate them

1. **Detection-rate gate** (bake_verdict native, `eval_report.rs::corruption_gate`):
   `pass_q20` / `pass_q10` = fraction of corruption entries scoring BELOW an
   honestly-lossy q20/q10 anchor. This is the "butteraugli wins 2-4×" stat.
2. **Severity ordering** (the §5 bar's 0.214@720 / 0.03–0.17@924 numbers): the
   ordering stat from the fulleval corruption instruments, surfaced as the
   gauntlet's corruption column.

The §5 row is stat (2), evaluated on the companion HEAD (the dial's own
ordering is broken at 924 by design — distributional, occlusion≠ablation).
**Both stats are computable for a head bake today** by pointing the existing
instruments at the head's `.bin` — nothing structural is missing. What IS
missing is the JOINT report: one verdict that evaluates the dial bake AND its
companion corruption head and prints the §5 row from the head's numbers while
keeping the dial's own (broken-by-design) numbers for honesty.

## Gaps + queued actions (status updated same-day)

1. **`bake_verdict --corruption-head <head.bin>` — CLOSED (`1134d1cb`).**
   Scores the head on the same corruption grid, section titled as the
   shipping owner, dial-alone numbers kept for honesty, `corruption_head`
   block in `--full-json`. Smoke: self-as-head == dial exactly (pass_q20
   0.1875 both); dial-only path unchanged.
2. **Freeze-bar summary — CLOSED (this commit): `freeze_check` bin.**
   `freeze_check --fulleval <bake.fulleval.json> [--bar csiq=X --bar live=X]`
   prints the §5 table with measured values + PASS/FAIL, externally-owned
   rows as explicit ATTACH rows (never silently omitted), exit 1 on any
   FAIL. Computes NO stats — compares numbers the owning tools produced.
   Smoke on shipped B (cid22-only fulleval): CID22 0.8764 FAIL vs the 0.89
   freeze bar (expected — the bar targets the EM4-class 944 bake), dial
   97.9%/0.0% PASS, repro PASS, 10 ATTACH.
3. **Korshunov runner** — OPEN. The seven-domain external-read set needs a
   committed, named runner before the Phase-4 capstone can be executed by
   anyone but the session that built it. (Owner: the external-validation
   work line; flagged in the plan.)
4. **upiq_panel.py stats provenance — VERIFIED NON-CANONICAL (violation).**
   It calls `scipy.stats.spearmanr` directly (lines 83-118), including
   inside bootstrap loops — exactly the banned pattern, in a freeze-gate
   owner. `zen_stats.srocc` can't be dropped in as-is: it shells the Rust
   `panel` binary PER CALL (fine for aggregates, prohibitive for bootstrap
   iterations). Correct fix per the no-duplication rule: **extend the
   owner** — add a batch mode to the `panel` bin (N pairs in, N panels out,
   one process), expose `zen_stats.panel_batch`, migrate upiq_panel's call
   sites, verify parity with the current scipy numbers (≤1e-9, the proven
   equivalence bound). Queued with the #8 torch/stat tail; must land before
   the UPIQ row gates a real freeze candidate.

## What already exists that the plan can lean on (verified today; +this commit's additions)

- **`freeze_check`** (this commit) — the §5 PASS/FAIL decision surface over a
  fulleval JSON; see gap 2 above for semantics.
- **`bake_verdict --corruption-head`** (`1134d1cb`) — joint dial+head
  corruption report; see gap 1.

- `bake_compare` (A-vs-B decisive, MRR + bootstrap CI) — the right tool for
  every "≥ best other arm" bar.
- `bake_dial_refit gate` — G-RANGE spline-domain tail gate.
- Gauntlet reject-gate greying (CID22 < 0.84 or nonphoto < 0.80) + the
  `M3a-attr` / `M3-coh` / `M3 drop%` columns (d539625e) + `product_composite`
  read from Rust JSON.
- bake_verdict's "Related specialized evals (run separately)" table — the
  report already points at every non-inline eval with exact commands instead
  of silently omitting them.
