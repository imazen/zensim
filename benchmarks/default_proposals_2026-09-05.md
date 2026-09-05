# SDR + HDR default proposals under the 2026-09-05 ruling (2026-09-05)

**Proposals only. Nothing was installed** — `zensim/weights/` is untouched,
`ZensimProfile::D` and `ZensimProfile::BHdr` still resolve to exactly the bakes
they resolved to this morning. The user decides defaults.

**The ask, verbatim (2026-09-05):** *"ok, is there poor resolution compared to
ssim2? update and share thw gauntlet for what should be the. ew sdr and bdr"*,
then *"hdr"* — i.e. **"what should be the new SDR and HDR [defaults]"**. The
leading "ok" is read as accepting the two grading changes recorded in
[`dial_addressability_gate_2026-09-04.md`](dial_addressability_gate_2026-09-04.md)
§17: `resolvable` becomes the operative floor window, and `A1`-`A6` become
report-only. That reading is an INFERENCE and is reversible without a code
change; §17.1 says so in the same words.

---

## 1. SDR — the proposal is **keep Profile D**, and the ruling is why it can be

Graded on the 2026-09-05 FLOOR-DENSE **372 ladder** instrument (`4c3874a7…`,
9,593 distinct-setting rows) under the OPERATIVE rule, with the registered 372
negative-tail and identity probes. `✓`/`✗` is that codec against the **mentor's
own fraction on the same cells** (`peer_ssim2`, registered this pass).

| scorer | `avif-rav1e` | `avif-svt` | `jpeg` | `jxl` | `webp` | **A7r** | **CONTRACT** |
|---|--:|--:|--:|--:|--:|:--:|:--:|
| **`peer_ssim2` — THE BAR** | 0.6410 | 1.0000 | 0.6667 | 0.9615 | 1.0000 | — | PASS (6/6) |
| **Profile D — SHIPPED** | **0.6667 ✓** | **1.0000 ✓** | **0.6667 ✓** | **1.0000 ✓** | **1.0000 ✓** | **PASS** | **PASS (6/6)** |
| Profile D — previous (08-31) | 0.6667 ✓ | 1.0000 ✓ | 0.6667 ✓ | 1.0000 ✓ | 1.0000 ✓ | PASS | **FAIL** (`C5`) |
| Profile A (`v47_strict_qat_native`) | 0.3590 ✗ | 0.8462 ✗ | 0.5128 ✗ | 0.8462 ✗ | 1.0000 ✓ | FAIL | PASS |
| Profile B (shipped SDR ranker) | 0.1795 ✗ | 0.4359 ✗ | 0.5641 ✗ | 0.4231 ✗ | 0.9487 ✗ | FAIL | **FAIL** (`C3` `C4` `C5`) |

**Shipped Profile D is the only scorer measured this pass that passes BOTH
tiers** — headline verbatim: `SHIPPABLE (regression PASS + contract PASS)`. It
meets the mentor on `avif-svt`, `jpeg` and `webp`, and **exceeds** it on
`avif-rav1e` (0.6667 vs 0.6410) and `jxl` (1.0000 vs 0.9615).

**The ruling is load-bearing for that answer.** Under the retired `distinct`
window D failed `A7r` on jpeg by exactly one ladder (0.5128 vs 0.5385) — the
installed default was being failed on the axis at which it is in fact best.
`resolvable` grades only steps the mentor can separate, and the miss dissolves.

**Its two report-tier misses, for information** (both `report-only` since the
ruling — measured, printed, gating nothing): `A1` **99.99996372112122** against
the mentor's 100.0 (short by 3.6 × 10⁻⁵ — a single scalar property of D's own
spline at the identity point) and `A3` **93.88421311743264** against 93.9743354
(0.09 — a distribution-shape gap near the ceiling). Both are deterministic, not
noise: `ladder_floor_resolution_2026-09-05.md` §5 reproduced them bit-for-bit
across a fresh process. `A2` **−94.9697**, `A4` **−17.8086**, `A5` **194.9697**
and `A6` **111.6928** all clear the mentor.

### 1.1 There are no other candidates — measured, not assumed

Every one of the **97** reconstructible board cells was re-graded on the ladder
instrument of its own width under the operative rule (94 → 944, 3 → 372; full
coverage, 0 skipped). **All 97 fail `A7r`**: 48 miss all five codec bars, 33 miss
four, 13 miss three, 2 miss two, 1 misses one; exactly one reaches CONTRACT PASS.
**Cells that are CONTRACT-PASS and clear all five codec floors: zero.** Table:
`/mnt/v/output/zensim/gaddr-board-ladder-2026-09-05/ladder_regrade_summary.tsv`.

So the shortlist the ask calls for ("top 3") does not have three members, and
saying otherwise would be inventing a ranking. **Profile D is the only passer;
the nearest failers and the axis that fails them are:**

| cell | why it is not the proposal |
|---|---|
| **Profile D — previous** | identical `A7r` (5/5) — the two share ADD156 weights and differ only in the output spline — but **CONTRACT `C5`**: identity dial 96.1157, outside the registered `[97.5, 100]` band. A monotone spline moves range, not rank, which is exactly what these two demonstrate on shipped artifacts. |
| **Profile A** | CONTRACT PASS (6/6) but `A7r` **FAIL on 4 of 5 codecs** — `avif-rav1e` 0.3590 against a 0.6410 bar. |
| `Dpeaks_lam1em3` and the whole λ family | `ladder_instrument_2026-09-05.md` §9.4 measured all ten arms worse than the incumbent **on every codec at every λ**. The axis is exhausted, not under-tuned. |
| Profile B | fails both tiers on every codec; its per-codec `dial_min` is POSITIVE everywhere — the known collapsed negative tail. |

### 1.2 Rank, for completeness — and why it is not the deciding axis

Board composite + CID22 for the SDR cells (read from
`gauntlet.write_fairness_tsv`, the board's own owner; **all four are
`FAIR-NOTED` and ungroupable — no seed group, so `k` is blank and there is no
k-mean**; single draws, per the registered seed-group rule):

| board cell | tier | composite | CID22 (signed) |
|---|---|--:|--:|
| `b_sdr_linear_cid80_inclwinsor_dense_dial@cur372` (Profile B) | FAIR-NOTED | 0.8286 | **0.882117** |
| `Dpeaks_lam1em3` | FAIR-NOTED | 0.8252 | 0.871792 |
| **`d_id100_negrich@did100lane` (= shipped Profile D)** | FAIR-NOTED | **0.8242** | 0.863380 |
| `v47_strict_QAT_native@cur372` (Profile A) | FAIR-NOTED | 0.8220 | 0.866063 |

**Profile B still leads on rank and fails the dial on every codec; Profile D
trails it by 0.019 CID22 and is the only dial that works.** The dial and the
ranker are not the same decision — the gate record has said so since §14.6, and
this table is what that looks like with numbers.

⚠ **Shipped D is on the board only under a lane-scoped name.** `Profile D` has no
fulleval of its own; `d_id100_negrich@did100lane` carries **bake sha256
`921a8f677a225b01…`, which is byte-identical to
`zensim/weights/d_sdr_add156_id100_negrich_dial_2026-09-05.bin`** — verified by
hashing both. Anyone looking for "the shipped SDR dial" on the board will not
find it by name. Promoting it under its own name is a registered follow-up, not
done here.

---

## 2. HDR — the proposal is **keep `BHdr`**, and the evidence is thin on purpose

**G-ADDR on HDR is NOT MEASURED, and no instrument exists that could measure
it.** Confirmed three ways: no HDR dial grid exists on `/mnt/v`; the G-ADDR
registry's 25 rows are all SDR; and every `HDR944_*` cell's own emitted block
reads `regression NOT MEASURABLE (unregistered dial grid)`. **An HDR ladder
instrument is REGISTERED AS THE FOLLOW-UP**, with
[`docs/PLAN_LADDER_INSTRUMENT_2026-09-05.md`](../docs/PLAN_LADDER_INSTRUMENT_2026-09-05.md)
as the template — whose own §3 says "SDR only. HDR is a separate registered arm
and is not started here."

⚠ **`bhdr_linear_shaped_cvvdpmix@cur372` DOES carry a full G-ADDR verdict and it
must not be quoted as BHdr's HDR dial** — it was cut on the SDR 372 codec grid,
which the profile's own docs record as invalid for it (rank agreement 0.72,
unbounded extrapolation). It is an SDR-instrument grading of an HDR-only profile.

So the HDR proposal rests on the rank evidence that does exist: **UPIQ HDR
(n = 380: Narwaria 140 + Korshunov 240), pooled SROCC** —
`scripts/hdr/upiq_panel.py`, the owner of the §5 HDR bar.

| scorer | UPIQ pooled | narwaria | korshunov |
|---|--:|--:|--:|
| **`BHdr` — SHIPPED** (372 PU-linear, 11.8 KB) | **0.7536** | 0.7834 | 0.9175 |
| PU-SSIM (literature bar) | 0.7395 | — | — |
| `BHdr` previous (`anchored2`) | 0.7313 | 0.7757 | 0.9104 |
| **ssim2 integrated PU21 — the mentor** | **0.7044** | — | — |
| `HDR944_L1T1_s4004` — best single HDR944 seed | 0.7254 | 0.7419 | 0.9312 |
| `HDR944_L1T1_s4005_hfpack` (= frozen `CHdr`) | 0.6664 | 0.6434 | 0.9280 |

**Ranked honestly by SEED GROUP, not by best cell** (the registered
`seed-group-single-draw-2026-09-04` rule): `L1T2` k=3 mean **0.7058**, `L1T1`
k=7 mean **0.6797** (range 0.6472–0.7254), `R_t2` k=3 0.6408, `GH2a` k=2 0.6361,
`R_t1` k=3 0.6335, `GH2b` k=2 0.6317, `GH1` k=3 0.6176. **All 24 HDR944 cells:
mean 0.6564, max 0.7254.**

**`BHdr`'s 0.7536 sits above every one of the 24 HDR944 cells and above every arm
mean.** It is also the only HDR scorer above the mentor (+0.049 vs ssim2-PU) and
above the literature PU-SSIM bar. `CHdr` — the frozen HDR candidate-of-record —
loses to `BHdr` by **−0.0872, paired p = 0.0000 (B = 5000)** and fails both
registered HDR bars.

**The caveats are large and belong in the proposal, not under it:**

* **n = 380 and burned** (~21 looks). Total human-labelled HDR pairs on disk
  across every corpus is **1,855**; nothing reaches n ≥ 500.
* **`BHdr`'s own promotion was never significant.** Selection-adjusted maxT
  **p = 0.221**; family deltas vs the prior bake are 4 positive / 3 negative with
  median ≈ +0.002. The honest reading is "not established", not "a win".
* **`BHdr` overlaps its own instrument**: `hdr_v3mix` contains 7 of the 9 frozen
  HDR census scenes, so the census JUDGE is trained on the census. Registered as
  `hdr-instrument-v3mix-scene-overlap`. The HDR944 candidates are clean here.
* **Both HDR models train on a target that is 50 % SSIMULACRA2**; no HDR training
  row anywhere carries a human label.
* **Seed mining is falsified** as the HDR944 path: seed-rank split-half agreement
  **0.14** across 7 seeds, and the select-half REORDERS the population.
* `BHdr` carries a **user-accepted era-2 tiling exception** (misses the
  zero-tolerance composite clause by 3.2 × 10⁻⁶) — *"5/6 plus this exception"*,
  never 6/6.
* **The board carries NO HDR rank axis at all** — all 15 `rank.*` corpora are
  SDR — so every board composite on an HDR cell is a cross-domain SDR read. Do
  not rank HDR by the board.

**Proposal: keep `BHdr`.** Not because it is proven better — its own promotion
was not significant — but because it leads every measured HDR alternative on the
only HDR axis we have, and nothing on the board is a candidate to replace it.

---

## 3. Compare links (all four harness-verified)

Verified with `node scripts/v_next/gauntlet_render_check.js <html> --hash
'<fragment>' --expect-visible '<ids>' --expect-no-banner` — each reports
*"scoreboard holds exactly N row(s) in fragment order"* with no banner.

**SDR — fair board** (shipped D, Profile A, Profile B, the λ arm):
`http://192.168.50.44:3300/zensim/reports/summer_gauntlet_fair.html#compare=d_id100_negrich@did100lane,v47_strict_QAT_native@cur372,b_sdr_linear_cid80_inclwinsor_dense_dial@cur372,Dpeaks_lam1em3`

**HDR — fair board** (BHdr + the three top-composite VERIFIED-FAIR HDR arms):
`http://192.168.50.44:3300/zensim/reports/summer_gauntlet_fair.html#compare=bhdr_linear_shaped_cvvdpmix@cur372,HDR944_L1T1_s4005_hfpack,HDR944_GH2b_s4005_hfpack,HDR944R_t1_s4005_hfpack`

**SDR — all rows, with `peer_ssim2`:**
`http://192.168.50.44:3300/zensim/reports/summer_gauntlet.html#compare=d_id100_negrich@did100lane,v47_strict_QAT_native@cur372,b_sdr_linear_cid80_inclwinsor_dense_dial@cur372,Dpeaks_lam1em3,peer_ssim2`

**HDR — all rows**, adding the UPIQ-best single seed (LEGACY, so absent from the
fair board) and `peer_ssim2`:
`http://192.168.50.44:3300/zensim/reports/summer_gauntlet.html#compare=bhdr_linear_shaped_cvvdpmix@cur372,HDR944_L1T1_s4005_hfpack,HDR944_GH2b_s4005_hfpack,HDR944R_t1_s4005_hfpack,HDR944_L1T1_s4004,peer_ssim2`

Boards: fair **9.66 MB** (130 of 467 rows, cap 12 MB), all-rows **21.91 MB** (467
rows; the over-cap is pre-existing and reported, not introduced here). Both pass
`scripts/v_next/gauntlet_gates.sh` — `node --check` on every script block, the
DOM-shim render harness, and all six compare-fragment sub-gates.

**New board column: `floors ok`.** `n_pass / n_gradeable` codec families for
`A7r`, beside `G-ADDR p/f`, read from the owner's `measured.codec_floor` (nothing
re-derived). The hover carries every fraction, its mentor bar, the ladder count,
the per-codec state, the instrument, **and the RULE the window was cut under** —
because a `resolvable` fraction and a `distinct` one are different quantities and
the column would otherwise invite exactly the comparison the owner refuses.

## 4. Registered, not done

1. **An HDR ladder instrument + HDR G-ADDR registry rows.** Three prerequisites
   the SDR program did not face: a shipped HDR *reference dial* to pin bars
   against (none exists); an HDR identity probe (the 372 zero-vector identity
   does **not** extend to 944 — 190 of 944 slots are non-zero on `ref == dist`);
   and HDR ladder pixels plus an anchor whose target is not 50 % ssim2.
2. **Promote shipped Profile D onto the board under its own name.** It is
   currently reachable only as `d_id100_negrich@did100lane`.
3. **Mentor cell-tables for the three instruments that lack them** (944-2026-08-01,
   944-era2r4-foldapp2, postC-372). Without per-cell mentor truth the operative
   rule reads `A7r` NOT MEASURED there — 100 of 130 board cells today.
4. **The 16 cells still on pre-ruling tiering** (12 POOLS with unidentifiable
   probes, 4 `A3b_*` refused as wrong-regime reads).
