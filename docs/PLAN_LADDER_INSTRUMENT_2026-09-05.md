# PLAN — the ladder instrument rebuild (pre-registered 2026-09-05)

**Lane:** `claude-ladder-2026-09-05`, jj sibling workspace `~/work/zen/zensim--ladder`.
**Directive (user, verbatim):** *"do everything that needs to be done in order to fix
the ladder situation, including running fleet encodes - the latest git repo has a
faster zenav1-svt to use"*.
**Standing rule this serves (user, 2026-09-04, verbatim):** *"i care that the lowest
configurable settings per codec are representable, not that negative fifty is in that
specifically."*

**This document is PRE-REGISTERED**: §1-§7 (arms, gates, budgets, endgame) were written
and pushed BEFORE any encode ran. Results land in §8 and never edit §1-§7 — a gate that
moves after the measurement is not a gate.

---

## 0. What is actually broken, and what is already fixed

Prior lanes closed more of this than the brief assumed. Stating it precisely so this
lane does not re-do it:

| # | defect | status entering this lane |
|---|---|---|
| a | the dial grid could not be re-extracted at the runtime era (pixels deleted 2026-06-22) | **FIXED 2026-09-05** by `d_peaks_372_postC_2026-09-05.md` §1.4 — the surviving 2026-07-27 pixels are the RIGHT instrument (the `peer_ssim2` pin table is exactly those cells), and `dial_grid_372col_postC_2026-09-05.parquet` is extracted at HEAD |
| b | **jpeg's three lowest settings are byte-identical, so the grid never exercises "the lowest configurable settings"** | **OPEN — this lane** |
| c | **one AVIF backend only, at a pre-2026-09 zenavif** | **OPEN — this lane** |
| d | the D dial sits far below ssim2 at the JXL floor; the peaks arms invert on 4 JXL ladders | **OPEN — this lane re-measures on new pixels/encoders** |
| e | the spline anchor (`multiband_anchor_dial100`, safesyn) is pre-fix era with dead pixels | **OPEN — this lane** |

So the ladder situation that remains is an **ENCODER** problem, not an extraction-era
problem. The grid's q axis samples settings the encoders do not distinguish, and it
samples only one of the three AV1 encoders we ship.

---

## 1. MEASURED: the per-codec floor map (the finding that specifies the grid)

Run before writing this plan, because the grid cannot be specified without it.
3 references from the canonical 39, `--q-grid 0..30 step 1`, `zenmetrics sweep` (the
owner), metric ssim2. Two settings are the SAME setting when `encoded_bytes` **and**
ssim2 (6 dp) are identical. Artifacts + script:
`/mnt/v/output/zensim/ladder-2026-09-05/floorprobe/`.

| codec / backend | floor plateau (identical output) | first DISTINCT step | 3 lowest DISTINCT settings |
|---|---|---|---|
| `zenjpeg` | **q 0..10 — eleven settings, one output** | q = 11 | 0, 11, 12 |
| `zenwebp` | q = 0 only | q = 1 | 0, 1, 2 |
| `zenavif` / `zenravif` | q 0..1 | q = 2 | 0, 2, 3 |
| `zenavif` / **`svt-rs`** | q 0..1, then **pairwise ties** (2=3, 5=6, …) | q = 2 | 0, 2, 4 |
| `zenjxl` | **distance >= 25 — 25/26/30/40/50 identical** | d = 24 | 25, 24, 23 |

Three consequences, all load-bearing:

1. **The jpeg boundary is an encoder clamp, not content** — identical on all 3
   references. The old grid's bottom three steps (q 0 / 5 / 10) are ONE setting
   sampled three times, which is why the mentor's jpeg bar is `0.0000` and the
   A7r jpeg column is vacuous for every scorer.
2. **`svt-rs` needs ~step-2 sampling at the floor.** quality 0..100 maps onto QP
   0..63, so adjacent quality values collide irregularly (runs of 1 or 2). A fixed
   step cannot express "the lowest distinct settings" for it — **only
   deduplication can**.
3. **JXL's distance saturates at exactly 25.0**, so the old grid's floor was already
   correct; 26/30/40/50 buy nothing. Recorded so nobody "extends" it.

**Therefore the grid rule is DEDUPLICATION, not a per-codec step table.** Encode a
dense ladder, mark every step whose `(encode_sha)` equals its predecessor's as
`saturated=true`, keep it in the table, and let the floor rule read only DISTINCT
settings. This is codec-agnostic, survives an encoder change, and is the only form
that stays correct when a backend's quantizer mapping shifts under us.

---

## 2. Encoder pins (ARM 0)

| component | old | new |
|---|---|---|
| `zenav1-svt` | `2ca060f42` | **`2d75a105fe0b310bf586110951315f014e274fff`** (origin/main, 140 commits, pushed 2026-09-05 05:48 MDT) |
| `zenavif` | `9170d549` | `9170d549` + the rev bump (this lane) |
| `zenmetrics` | — | built at `e94284aa` with `--features sweep,avif-svt` |

**Speed claim is MEASURED, not assumed** (`benchmarks/svt_pin_speed_2026-09-05.md`):
old pin vs new pin, same image, same preset, same box, interleaved, min of N. "2x
faster" is the user's description of their own work in the encoder repo; this lane
reports what the AVIF still-encode path actually does across the two pins and does
not repeat a number it did not measure.

---

## 3. Grid specification (ARM 1)

**References:** the canonical 39 (`/mnt/v/output/zensim/dial-grid-pixels-2026-07-27/sources/`),
kept for continuity with the registered `peer_ssim2` pins.

**q axis — 66 steps**, floor-dense (this is the change):
```
0..30 step 1   (31)   <- NEW: guarantees 3 DISTINCT lowest settings for every codec
35..70 step 5  (8)
72..90 step 2  (10)
91..96 step 1  (6)
96.5,97,97.5,98,98.5,99,99.25,99.5,99.75,99.9,100  (11)
```
**JXL** keeps its `distance` knob ladder, with 24 and 22 added so the bottom three
distinct settings (25, 24, 23) exist.

**Ladder families — five, each a separate `(image, codec, backend)` ladder:**
`zenjpeg` · `zenwebp` · `zenjxl` · `zenavif backend=zenravif` (continuity with the old
grid's backend) · `zenavif backend=svt-rs` (**new**, the pin above).

**`zenavif backend=aom-rs` is DEFERRED, with a measured reason**, not skipped
silently: the `avif-aom` arm byte-verifies every cell against a cmake-built C libaom
oracle (`aom-sys-ref`), so it is both a build dependency on a C encoder and a
per-cell oracle cost. §8 records its measured per-cell time from a smoke encode; the
decision to defer is recorded either way. It is also the arm the imazen-only rule
most constrains — a C oracle in a *tuning-data* loop is exactly what that rule bars,
whereas port-validation inside the port repo is sanctioned.

**Size axis (REPORT-only this pass):** each reference additionally at 256 and 512 long
side via zenresize/Mitchell, declared in the same run, graded as a report. Native is
the primary instrument; the size axis is registered so the size-dependence of floor
representability is measurable rather than assumed.

**SDR only.** HDR is a separate registered arm and is not started here.

**Persistence (the rule the last grid died on):** every cell persists encoded bytes
content-addressed, the decoded PNG, `encode_ms`/`decode_ms`, and the encoder
version/commit. The 2026-07-27 grid persisted decoded PNGs only, and the 2026-05-29
grid persisted nothing — which is why its ladders are unreconstructible.

---

## 4. Execution (ARM 2) — and the honest fleet decision

The zenfleet mandate bars **hand-rolled** fleet orchestration. It does not require a
fleet for work that does not need one, and `zenmetrics sweep` is itself an owner (it
built the 2026-07-27 grid).

**Pre-registered decision rule, with the measured inputs already in hand:**
per-cell encode cost from §1's probe is `zenjpeg` 0.011 s, `zenwebp` 0.011 s,
`avif/zenravif` **0.387 s**, `avif/svt-rs` 0.043 s. The native grid is
39 x 66 x 4 + 39 x ~52 = ~12.3k cells, of which the rav1e leg dominates at ~17 min.

* **If the full program's measured encode + score + extract wall time is under ~2 h
  on this box, it runs LOCALLY through `zenmetrics sweep`** and the fleet is not
  stood up — with the measurement reported. Standing up a fleet costs a musl image
  rebuild with `avif-svt` (the current CPU tag does not advertise it), a
  `zenfleet-ctl` rebuild so the backend knob's `requires` capability is stamped at
  declare time, and a launch/reconcile/teardown cycle; that setup exceeds the job.
* **If it exceeds that, it goes through zenfleet** (`declare-encodes` -> first-cell
  gate -> `gap` -> workers -> `compact`) on the LAN tier only: r7900x (24c, idle,
  measured), mac (12c, idle, measured), tower (>=8 cores free, `--cpu-shares=256`,
  media has priority). **node-2 and node-3 are OFF** (both SSH accounts refused) and
  are not raised for this. No paid cloud.
* **Known fleet hazards, pre-registered so they cannot surprise:** (i) `JobKind::Feature`
  is **not wired** in the executor — features ride on `JobKind::Metric` with a
  zensim-variant metric name and return features *instead of* a score, so score jobs
  and feature jobs are separate declares; (ii) r7900x's `/tmp` is **still tmpfs**
  (masked, not yet rebooted) so `TMPDIR` must be set explicitly; (iii) any hand-launched
  container uses `--restart on-failure:5`, never `unless-stopped`.

**Metrics:** ssim2 is the mentor and is required. Every additional variant the CLI can
produce is persisted (butteraugli max/pnorms, dssim; cvvdp where a GPU node allows) —
incremental cost is microseconds, recovery cost is days.

**Features:** at HEAD, for both runtime instruments — 372 (`v1postc`) and 944
(`folded720append2pools`) — with `feature_set_id`, era and `build_commit` in the
manifests.

---

## 5. Instrument registration (ARM 3)

1. Pack as `dial_grid_372col_ladder_2026-09-05` and `dial_grid_944pools_ladder_2026-09-05`
   with `_MANIFEST.json` (rows, sha256, `build_commit`, `feature_set_id`, encoder pins,
   per-cell `saturated` flag).
2. Derive `peer_ssim2` pins on the new grids **through the owner**
   (`bake_verdict --dial-peer-scores ... --gaddr-json`), never re-implemented beside it.
3. Append — **never overwrite** — to `benchmarks/dial_addressability_floor_2026-09-04.json`:
   the new grid rows, and the per-codec floor bars including the two AVIF families as
   **distinct** codecs (`avif-zenravif`, `avif-svt-rs`).
4. The old instruments stay registered as retired-era reads. Nothing is rewritten.
5. Re-grade the fair board on the new instruments (`gaddr_board_regrade`,
   `promote_fulleval.py --graft-gaddr`), regenerate both boards, run
   `scripts/v_next/gauntlet_gates.sh`.

**Pre-registered expectation, so a surprise is legible:** jpeg's bar moves off
`0.0000` to a real number, because its bottom three settings become distinct. A codec
whose bar rises makes the gate STRICTER; that is the point, and no candidate is
re-graded against a bar derived from a different grid.

---

## 6. Anchor set (ARM 4)

The dial's calibration anchor (`multiband_anchor_dial100.parquet`, 2,000 safesyn rows)
is pre-fix era and its pixels are dead (`CLAUDE.md`: safesyn's `decoded_path` PNGs are
0/3000 present). A dial cannot be re-anchored on an anchor that cannot be rebuilt.

* **Sources:** imazen-26 canonical origins (png-v3 + manifest — the canonical copy,
  never `/mnt/v/imazen-26*` which is quarantined inspo).
* **Selection:** k-means on zenanalyze features, centroid-nearest member per cluster,
  ~60 references, content-class stratified — per the sweep discipline's
  cluster-don't-random rule.
* **Disjointness is a GATE:** `check_holdout_overlap --threshold 10` against CID22's
  49 validation references must return **0 hits**, or the set is rebuilt. CID22 is
  validation-only, always.
* Same ladders, same pipeline, same persistence as §3.
* **Target = ssim2, UNCLAMPED.** The shipped anchor's `max(ssim2, 0)` is exactly what
  collapsed Profile B's negative tail (147 genuinely-negative rows stored as 0, and
  `fit_spline_knots` collapsed the run to one bottom knot). Identity rows at 100.

---

## 7. Model side + the ship gate (ARM 5)

Three arms, each spline-only re-anchoring on the §6 anchor (rank must stay
byte-identical — a monotone spline cannot change rank, and that is asserted, not
assumed):

* **(i) D re-anchored** — the incumbent on a live anchor.
* **(ii) the peaks candidates** (`lam1em3`, `minus_f162`) re-anchored and re-graded on
  the NEW grid. The 4-ladder JXL inversion was measured on the OLD pixels and the OLD
  AVIF backend; whether it persists on the new JXL ladders is an open measurement.
  `d_peaks_jxl_floor_2026-09-05.md` §4 proved the inversion is in the WEIGHTS (8/8
  raw-level), so the pre-registered expectation is that it PERSISTS — recording that
  now so a "fixed by re-anchoring" result is correctly read as suspicious.
* **(iii)** the D-vs-ssim2 calibration gap at the JXL floor, reported before/after.

**SHIP GATE — all four, or nothing installs:**
1. G-ADDR **CONTRACT 6/6**.
2. **Floor representability >= the mentor's bar on EVERY codec family**, on the new
   instruments (this now includes a real jpeg bar and two AVIF families).
3. **CID22 >= today's D**, with the paired-bootstrap CI not excluding a gain.
4. **No regression axis lost** relative to today's D.

If an arm passes, it installs with the flip lane's full discipline
(`d_ship_flip_2026-09-05.md` §6: weights + manifest + `profile.rs` + the W4 two-control
protocol + tests/clippy/fmt/public-API zero delta + CHANGELOG/docs/annotations/ledger +
board). **If none passes, nothing installs and the record carries the numbers.**
Nothing is relaxed to make an arm fit.

---

## 8. RESULTS

Full record: [`../benchmarks/ladder_instrument_2026-09-05.md`](../benchmarks/ladder_instrument_2026-09-05.md).
§1-§7 above were not edited after pre-registration.

**Against the pre-registered gate (§7): NOTHING INSTALLS — no arm passes, including
the incumbent.** `ZensimProfile::D` is unchanged; `zensim/weights/` was never opened
for writing.

| pre-registered item | outcome |
|---|---|
| **ARM 0** encoder pins | DONE. `zenav1-svt` -> `2d75a105f` in zenavif `2ebca1b4` (on `origin/main`). Speed **MEASURED, not assumed: 1.498x** on summed `encode_ms`, 9/9 cells byte-identical — **not** the "2x" it was described as. |
| **ARM 1** grid spec | DONE, and the floor map (§1) is what specified it. 5 ladders x 39 refs, 66-step floor-dense q axis, dedup by encode hash. |
| **ARM 2** execution | LOCAL, by the pre-registered decision rule: measured ~28 min against a ~2 h threshold, so the fleet was not stood up. node-2/node-3 were **off**; r7900x and mac idle and left alone. |
| **ARM 3** registration | DONE at **both** widths (372 `4c3874a7…`, 944 `0e8e5fb7…`), append-only, bars derived through the owner. Board re-grade scoped and deferred with a census (§11a). |
| **ARM 4** anchor set | 32 k-means imazen-26 picks (12 content classes, 1,082-image population). **CID22 disjointness GATE PASSED: 0 hits at d <= 10**, closest d=19. |
| **ARM 5(i)** D re-anchored | **Provably cannot pass** — §9.2 measures all 19 failing jpeg ladders as RAW inversions (raw-vs-dial verdicts agree 39/39), and two shipped D bakes with identical weights and different splines have identical A7r. A monotone spline moves range, never rank. |
| **ARM 5(ii)** peaks arms on the new grid | Inversion **PERSISTS** and widens to three codecs — the pre-registered expectation. §9.4 then exhausts the whole λ family: every arm worse than the incumbent on every codec at every λ. |
| **ARM 5(iii)** JXL-floor calibration gap | MEASURED: mean **−8.55** at the jxl floor, **not** the 20-30 the brief assumed (§9.3) — and jpeg's floor gap has the **opposite sign**. |

**The pre-registered expectation that mattered.** §7 recorded, before measuring, that
the peaks arms' jxl inversion should PERSIST because `d_peaks_jxl_floor` §4 had
localised it to the weights — and that a "fixed by re-anchoring" result would be the
suspicious one. It persisted.

**What the program changed even though nothing shipped:** the instrument can now ask
the question the user's rule names. Shipped Profile D reads a clean A7r pass on every
older grid and **fails here on jpeg by one ladder** — a gap that was structurally
unmeasurable while jpeg's bar was a vacuous `0.0000` over three samples of one
setting. §9.5 localises the fix to `f93` (49.6 % of the move) and its neighbours.
