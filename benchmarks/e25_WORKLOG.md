# E25 — Rev4-trained by_v2fy weights on Rev5 held-out features

2026-10-04 MDT / 2026-10-05 UTC. Registered in `benchmarks/rev5_spec_2026-10-04.md` Addendum D, commit 399acb59, before E25 scoring.

MISSING first: no new confirmatory holdout was read; this is the registered exploratory instrument assessment. External sets and the Rev5-retrained comparison are reported only. No production qualification or default-model change is implied.

**E25 passes E21's registered as-good rule against the Rev4 E24 control.** Fifty frozen control cells (seeds 0–9 × KADID/TID2013/KonFiG/CID22-A25/AIC-3), selected final-epoch weights, were densified with `v2_common.dense_bake` and freshly predicted on their Rev5 held-out tables. No retraining, fitting or selection occurred.

## Exact gate before any E25 number

The successful run first fed all 50 cells' dense bakes their own Rev4 tables. `v2_lodo_mlp.predict` invokes the same `bake_dial_refit predict --score-units` owner as the trainer; its `panel_batch(..., stats="full")` invokes the same Rust metric owner as `result.json`. Every prediction vector and every full held-out metric dictionary equals its stored Rev4 value **exactly**, without tolerance. The global gate is written only after all 50 pass. No Rev5 input, E25 prediction or E25 metric is opened by this command before that gate.

`rev4_gate.tsv` pins all 50 results, source/dense models, feature lists, tables and predictions. The full gate receipt is archived. The log has 50 exact gate lines before its gate PASS line, and every E25 prediction line follows that PASS. Rev4/Rev5 keys and targets have identical order; full input tables, sidecars and keys pass their receipt hashes. All 50 cells use the same 420 by_v2fy IDs.

The local predictor/panel binaries differ from the binaries recorded by some historical cells. The exact gate above proves reproduction of the complete stored held-out results with the current owner binaries; those executed binaries and historical pins are recorded. Executed predictor SHA-256 `81ec2207f593b5b2c6d6314e9f942f817bba63ff15c65f3908704dd1ceff5a1d`; panel SHA-256 `c9c610b8bcddd0cb66c411c10de07ecd283e2030dd26a48845878f5eb403ef16`.

## Primary: E25 minus Rev4 control

E21 rule unchanged: signed mean Δ ≥ −0.002, every source Δ ≥ −0.005, and W2 Δ > −2 SE. `e24_rev5.py` shares one `as_good` helper between E24 and E25; it exactly reproduces both existing E24 decisions. Per-source/W1/W2 and seed-paired uncertainty come from the existing `e13_teacher.score_arms`, with no second scorer. The inherited E13 `adopt` field describes its own superiority rule and is not E25's verdict; `decision.json.primary.as_good` applies the registered E21 rule.

| Quantity | Measured Δ | SE | Gate |
| --- | --- | --- | --- |
| Signed mean | -2.33766692566e-07 | 2.75228711147e-07 | PASS ≥ −0.002 |
| Worst source (AIC-3) | -1.84367236391e-06 | 1.18277019479e-06 | PASS ≥ −0.005 |
| W2 worst three types | 6.56967757474e-06 | 9.80081758841e-06 | PASS > −2 SE |

Per-source signed deltas (10 seed pairs each):

| Source | E25 minus Rev4 | SE |
| --- | --- | --- |
| kadid | -4.41722807776e-08 | 1.11734423135e-07 |
| tid2013 | -9.01457426705e-08 | 3.15502343879e-07 |
| konfig | -1.55782650491e-07 | 4.95867321593e-07 |
| cid22_a25 | 9.64939575021e-07 | 3.70020325622e-07 |
| aic3 | -1.84367236391e-06 | 1.18277019479e-06 |

Coverage: 50/50 cells, 144170 held-out predictions, zero missing cells. All 50 prediction vectors change when fed Rev5 tables; maximum absolute score prediction change is 0.01190185546875. The tiny rank deltas are measured, not a reused Rev4 prediction cache. Rev4 training bakes and their dense transformations have identical weights and bit-identical densifier gate predictions. The offline predictor is requested here; no serving revision stamp or pixel-forward qualification is substituted for that path.

## Reported only: E25 minus Rev5-retrained by_v2fy

| Quantity | Δ | SE |
| --- | --- | --- |
| Signed mean | -0.00109865941805 | 0.00096157218147 |
| Worst source (KonFiG) | -0.00717895135532 | 0.00306843461773 |
| W2 | 0.00657985927569 | 0.00619408779669 |
| W1 reference p10 | -0.00182353822014 | 0.00183435221951 |

| Source | E25 minus retrained | SE |
| --- | --- | --- |
| kadid | -0.000288683226031 | 0.00279909373397 |
| tid2013 | 0.000562064683995 | 0.00197321826022 |
| konfig | -0.00717895135532 | 0.00306843461773 |
| cid22_a25 | -0.000117395775546 | 0.00108325873757 |
| aic3 | 0.00152966858266 | 0.000893459553957 |

This comparison does not contribute to E25's registered verdict.

## Reported only: external sets versus the Rev4 control

The existing `external_sets.py` owner receives `control_root=/var/tmp/rev4-featpot/v2c` (the `--control-root` path); the arm view points to Rev4 weights and Rev5 external tables. The two roots' pair order and labels are checked by that owner. NITS/LIVE/MCIQA have 50 paired cells each (five folds × ten seeds), zero refused cells. The arm's prediction cache is fresh and separate from the control cache. All per-type/model/dimension metrics are retained in `external.json`.

| Set | E25 mean SROCC | Δ versus Rev4 | SE | Paired cells |
| --- | --- | --- | --- | --- |
| nits | 0.709847757817 | 1.40196425224e-06 | 1.42589452238e-06 | 50 |
| live | 0.927764399198 | -3.19845454728e-07 | 3.1603938316e-07 | 50 |
| mciqa | 0.249086531145 | -9.64963824471e-07 | 6.75436337749e-07 | 50 |

LIVE content-disjoint subset, using the existing fixed overlap exclusion: mean SROCC 0.917433953938, Δ 1.26090982522e-08, SE 1.26090982522e-08, 50 pairs. Overall LIVE includes known training-content overlap; the disjoint result is retained separately by the existing owner. These sets remain evaluation only.

## Execution and verification

Workspace rebased first with `jj rebase -r @ -d main@origin`; parent is registration 399acb59. The earlier R5STEER commits had already landed as squashes, so only the empty working revision was moved and main's landed files were retained. No worktree, push or sibling edit.

The only implementation change extends `scripts/rev4_featpot/e24_rev5.py`. The Rev4 control root and Rev5 retrained cells are untouched; an independent output view carries E25 records. An existing output root is refused to preserve evidence. Production arithmetic remains frozen at 60174678.

Two implementation attempts are preserved and marked failed. First: all 50 exact gates passed, then an already-dense bake was incorrectly passed to the trainer's Rev5 path, which refused re-densification (zero E25 predictions completed). Second: all 50 gates and held-out predictions/comparisons completed, then the external owner refused the generated view because its required keep_features.txt metadata was omitted. The final command passes the original bake through the trainer's own dense path, verifies identical dense bytes against the gate, and copies the pinned feature list. All 50 numerical prediction vectors and metric dictionaries from the second attempt equal the final run exactly. No weight, feature, metric, seed, rule or selection was changed in response to a score.

Final chain exit 0; all outputs present. Python compilation, script lint (811 runnable scripts) and CI-exact `just clippy` pass. The unchanged E24 decision helper reproduces its stored decisions exactly. `checks.json`, input/source/tool hashes and full-precision tables are committed; large per-cell prediction JSON, full gate receipts, models and logs are archived.

Replay with a fresh E25 output directory:

```bash
TMPDIR=/var/tmp/e25/tmp RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ~/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- \
  python3 -u scripts/rev4_featpot/e24_rev5.py e25 \
  --root /var/tmp/rev4-featpot/v2c5 --rev4-root /var/tmp/rev4-featpot/v2c \
  --out /var/tmp/e25/REPLAY_FRESH --jobs 4
```

Successful scratch root: `/var/tmp/e25/complete`. Durable raw archive: `/mnt/tower/output/zensim/e25-2026-10-04/`. Committed record: `benchmarks/e25_2026-10-04/`; `pointer.json` and `archive_inventory.tsv` pin archive bytes. Rev4/Rev5 source tables remain in their original roots and are hash-pinned in `inputs.tsv`; no table was extracted, rewritten or moved. No sealed/confirmatory population was read. `E25_DONE.md` is written last, after archive verification and the local commit.
