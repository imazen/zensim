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
| UPIQ pooled (V1-HDR) | > 0.7536 | `scripts/hdr/upiq_panel.py` (+ `upiq_crossdomain_instrument.py`) — Python, self-documented in bake_verdict's "Related specialized evals" table; stats canonical since `a5bd3e6f` (panel --batch, gap 4) | **PYTHON OWNER** (canonical stats) |
| Korshunov hold (V1-HDR) | ≥ 0.93 | `scripts/external_reads/run_external_reads.py` (gap 3, `ee4a1972`) — probe gate + `--scorer bake:` for the final bakes | **COMMITTED OWNER** |
| M3a coherence | ≥ 0.85 EM2-class @16–128px | `diffmap_block_coherence` example → `scripts/run_full_eval.sh` (`m3a_coherence`/`m3a_n` in fulleval JSON) → gauntlet `M3a-attr` column | **FULLEVAL+GAUNTLET** (not bake_verdict-inline; fine) |
| Dial monotonicity + `product_composite` | per bake_verdict gates | `bake_verdict` native (dial panel G1/G3/G4; `product_composite()` in `--full-json`; gauntlet reads, never re-derives) | **NATIVE** |
| Byte-repro | embedded `zentrain.repro`, exit-4 | trainer embed (fatal on fail) + bake_verdict repro badge | **NATIVE** |
| Perf (SDR) ≤ +2% | bench | zenbench extractor/compare benches — a bench artifact, not a verdict stat | **BENCH-OWNED** (correct home) |
| Perf (HDR) ≤ +5% | bench | same; the V5 lever list | **BENCH-OWNED** |
| LOO (append2 ≤ 0) | instrument | harness committed at `scripts/external_reads/asrun/{bandvis_loo_944,csfw_g6_loo_956}/` (+ artifact COMMANDS.md); stored-verdict verify via `run_external_reads.py --reads loo944,loo956` | **COMMITTED (asrun harness + verify read)** |

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
3. **Korshunov runner — CLOSED (`ee4a1972`).**
   `scripts/external_reads/run_external_reads.py` is the committed, named
   owner of the whole seven-domain external-read set. `--from-stored`
   rescores the stored study tables in ~11 s (no decode) under `probe944` /
   `s228` / `bake:<final.bin>` scorers, gates the registered probe against
   the recorded hdr-dmean head (5e-7) before any study look, and checks
   every recorded number: Korshunov 0.9346 + Narwaria 0.7688 + AVT pooled
   0.7742 + 12 more probe reads + 15 s228 reads all reproduced at ≤2.2e-16;
   BANDVIS/CSFW LOO delta tables recomputed from stored verdicts at 0.0
   diff (296+264 cells). As-run scripts + PROTOCOL.md pre-registrations
   frozen under `scripts/external_reads/asrun/` (35 files, provenance
   headers + sha256s); full re-extraction documented via `--list-extract`,
   not automated. Phase 4 runs `--scorer bake:<final_v1>.bin`.
4. **upiq_panel.py stats provenance — CLOSED (batch mode in `ba94f35b` +
   `1486b2d0`*; migration `a5bd3e6f`).** As specced: `panel --batch` (N
   pairs in, N stat rows out, one process; explicit rows or `#def` bases +
   index-set resamples so the caller keeps the bootstrap RNG; `--stats
   srocc` fast path + `srocc_signed`/`plcc_raw` columns), exposed as
   `zen_stats.panel_batch` / `panel_batch_indexed`, gated by
   `scripts/verify_panel_batch_parity.py` (≤1e-12 vs scipy midrank incl.
   tie-heavy — measured ≤3.3e-16 — plus indexed≡explicit + byte-
   determinism) and `tests/panel_parity.rs --ignored`. upiq_panel.py
   migrated: whole 10k bootstrap = ONE process (2.5 s vs 8.5 s), stdout
   byte-identical on the recorded invocations (0.7081/0.7173/0.8992
   default-feats; 0.7536/0.7834/0.9175 + p 0.3950/0.0799 pulinear);
   scipy only behind the optional `--verify-scipy` flag.
   *Attribution note: the batch-mode content sits inside `ba94f35b`
   (docs(#70)) + `1486b2d0` (style: fmt) — a concurrent session's pushes
   swept this work line's WIP into its commits; the CHANGELOG entry is
   the accurate record.

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
