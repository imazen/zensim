# The ladder instrument — rebuilding the dial grid so each codec's lowest settings are REAL (2026-09-05)

**Lane:** `claude-ladder-2026-09-05`, jj sibling workspace `~/work/zen/zensim--ladder`.
**Pre-registration:** [`docs/PLAN_LADDER_INSTRUMENT_2026-09-05.md`](../docs/PLAN_LADDER_INSTRUMENT_2026-09-05.md),
pushed as `4a1047a4` **before any encode ran**. §1-§7 there are frozen; this file
is the measurement record.
**User rule this serves (2026-09-04, verbatim):** *"i care that the lowest
configurable settings per codec are representable, not that negative fifty is in
that specifically."*

---

## 1. The defect, restated as a measurement

`dial_addressability.rs`'s `A7r` asks whether a codec's **three lowest configurable
settings** still resolve on the dial. On every grid built before today, that
question was unanswerable for `jpeg`, and nobody could see it, because the grid
sampled q = 0 / 5 / 10 and **zenjpeg emits ONE bitstream for all of q 0..10**.

MEASURED here (`--q-grid 0..30 step 1`, 3 references, `zenmetrics sweep`, ssim2;
`/mnt/v/output/zensim/ladder-2026-09-05/floorprobe/`), where two settings are the
SAME setting when `encoded_bytes` **and** ssim2 to 6 dp are identical:

| codec / backend | floor plateau | first DISTINCT | 3 lowest DISTINCT |
|---|---|---|---|
| `zenjpeg` | **q 0..10 — eleven settings, one output** | q = 11 | 0, 11, 12 |
| `zenwebp` | q = 0 only | q = 1 | 0, 1, 2 |
| `zenavif` / `zenravif` | q 0..1 | q = 2 | 0, 2, 3 |
| `zenavif` / `svt-rs` | q 0..1, then **pairwise ties** (2=3, 5=6) | q = 2 | 0, 2, 4 |
| `zenjxl` | **distance >= 25** (25/26/30/40/50 identical) | d = 24 | 25, 24, 23 |

The jpeg boundary is identical on all three references, so it is an encoder clamp,
not content. And the incumbent's own grading shows the damage directly — shipped
Profile D on the postC grid reports jpeg `bottom_medians` **22.22 / 22.22 / 22.22**,
three identical numbers passing a bar of `0.0000`.

**Three consequences that shaped the design:**

1. **`svt-rs` cannot be expressed by any fixed step.** quality 0..100 maps onto
   QP 0..63, so adjacent quality values collide irregularly (runs of 1 or 2).
2. **JXL's floor is exactly 25.0.** Recorded so nobody "extends" it — 26/30/40/50
   buy nothing.
3. **Therefore the rule is DEDUPLICATION BY ENCODE HASH, not a per-codec step
   table** — codec-agnostic, and the only form that stays correct when a
   quantizer mapping shifts under us.

---

## 2. Encoder pins (ARM 0) — and the speed claim, measured

`zenav1-svt` `2ca060f42` -> **`2d75a105fe0b310bf586110951315f014e274fff`** (origin/main,
140 commits, pushed 2026-09-05 05:48 MDT); `zenavif` consumes it at `2ebca1b4`,
verified on `origin/main`.

**MEASURED on zenavif's own AVIF still-encode path** (3 images x q30/60/90, svt-rs,
`--no-score --jobs 1`, arms INTERLEAVED over 7 rounds, min per arm, quiet box):

| | old pin | new pin | speedup |
|---|--:|--:|--:|
| summed per-cell `encode_ms` | 6932.8 ms | 4627.9 ms | **1.498x** |
| wall clock | 7.067 s | 4.768 s | **1.482x** |
| bitstream identity | — | — | **9 / 9 cells IDENTICAL** |

The gain rises with quality (1.30-1.46x at q30 -> 1.58-1.61x at q90). Which rev each
arm linked was verified from the **build fingerprints**, not assumed — the old-pin
build was launched moments before the manifests were restored, and
`target-svtold/release/.fingerprint/zenav1-svt-encoder-*` referencing only
`…/2ca060f` is what proves cargo resolved before the restore.

**This is 1.50x, not the "2x" the encoder work was described as**, and this record
does not repeat a number it did not measure. It is a two-BUILD comparison, which the
perf discipline warns cannot be trusted below ~10%; the separation is 48% with
non-overlapping per-round spreads, so it clears that bar.
Record: `zenavif/benchmarks/svt_pin_speed_2026-09-05.md`.

---

## 3. The grid (ARM 1+2)

39 canonical references (unchanged, for continuity with the registered `peer_ssim2`
pins) x a **66-step floor-dense q grid** (0..30 step 1, 35..70 step 5, 72..90 step 2,
91..96 step 1, then the old grid's fractional top) x **five ladder families**.
Every cell persists the **encoded bitstream**, the decoded PNG, `encode_ms`,
`decode_ms`, and ssim2 + butteraugli (max + pnorm3) + dssim.

| leg | cells | wall | failures |
|---|--:|--:|--:|
| `jpeg` | 2,574 | 97 s | 0 |
| `webp` | 2,574 | 98 s | 0 |
| `avif_svt` | 2,574 | 124 s | 0 |
| `avif_rav1e` | 2,574 | 1,413 s | 0 |
| `jxl` | see §8 | | |

**Local, not fleet — a pre-registered decision rule, resolved by measurement.** The
plan set the threshold at ~2 h on this box. The whole native grid encodes and scores
in **~28 minutes**; standing up zenfleet for it would have cost a musl image rebuild
with `avif-svt` (the current CPU tag does not advertise it), a `zenfleet-ctl` rebuild
so the backend knob's `requires` capability is stamped at declare time, and a
launch/reconcile/teardown cycle — more setup than the job. `zenmetrics sweep` is
itself an owner (it built the 2026-07-27 grid), so this is not hand-rolled
orchestration. **node-2 and node-3 are powered off** (both SSH accounts refused);
r7900x and mac were idle and were left alone.

---

## 4. Deduplication — what the instrument drops, and why it is not a loss

Two tables come out. `ladder_grid_*_full.parquet` keeps **every** step with a
`saturated` flag, `encode_sha`, and every metric. `dial_grid_372col_ladder.parquet`
keeps only DISTINCT settings — which is what lets `dial_addressability.rs` stay
**unchanged**: its "bottom K steps" are then the bottom K *configurable* settings by
construction.

Measured on the three legs available when this was first validated:

| codec | cells | saturated | distinct | ladders |
|---|--:|--:|--:|--:|
| `avif-svt` | 2,574 | **936 (36.4 %)** | 1,638 | 39 |
| `jpeg` | 2,574 | 585 (22.7 %) | 1,989 | 39 |
| `webp` | 2,574 | 274 (10.6 %) | 2,300 | 39 |

**36 % of the svt-rs ladder is duplicate settings** — the quality->QP collision, and
the reason a fixed step could never have expressed its floor.

---

## 5. What the mentor's bars become

`peer_ssim2`'s own floor representability, previewed on the same three legs
(the registered values are derived through the owner — §8):

| codec | bar, NEW grid | bar, old grid |
|---|--:|--:|
| `avif-svt` | 0.9744 (38/39) | — (family did not exist) |
| `jpeg` | **0.5385 (21/39)** | **0.0000 (vacuous)** |
| `webp` | 1.0000 (39/39) | 1.0000 |

**jpeg's bar goes from a vacuous `0.0000` — which anything passes — to a real
`0.5385`.** That is the user's rule becoming enforceable: jpeg's lowest configurable
settings are now IN the instrument, and a candidate has to resolve them. The bar is
not 1.0 because ssim2 itself does not strictly increase across q 11/12/13 on 18 of
39 images; that is real encoder behaviour at a near-flat part of the curve, and
pinning to the mentor is exactly what keeps it honest.

---

## 6. Design facts verified against source, not assumed

* **`FloorMeasure::from_grid` is codec-name-agnostic** — it groups by
  `(image_id, codec)` from whatever strings the grid carries and sorts by `q`
  ascending. The codec names in `dial_addressability.rs` appear only in its tests.
  So `avif-svt` / `avif-rav1e` become two ladders with **zero production code
  change**.
* **The AVIF backend is a KNOB, not a codec** — `--codec zenavif --knob-grid
  '{"backend":["svt-rs"]}'`, feature `avif-svt`. There is no `zenavif-svt` codec
  string.
* **`--metric` is a repeated flag**, not comma-separated.
* **JXL's `q` column reuses the canonical grid's own (distance -> q) curve**
  (q=0 <-> d=25.0 … q=99.8 <-> d=0.05), read from that grid rather than re-derived,
  so the two instruments' jxl columns stay comparable.

---

## 7. Two process failures worth more than the code

* **Never edit a shell script while it is running.** bash reads a script
  incrementally by byte offset. Editing `build_ladder_grid.sh` at 12:23 while the
  12:12 run was inside its `avif_rav1e` leg made bash resume at a stale offset and
  die with `line 81: unexpected EOF while looking for matching '` — **after** rav1e
  finished and **before** jxl started, with no `COMPLETE` marker. The four finished
  legs' data is intact (already parsed and executing); only the jxl leg was lost and
  was re-run from a **frozen copy** of the script. The same hazard was correctly
  avoided for the chain script an hour earlier by killing and restarting it rather
  than editing in place — the lesson was available and not generalised.
* **Measure elapsed time; do not estimate it.** A misread clock turned 100 s of
  progress into an apparent 7 minutes and nearly triggered a full restart of the
  grid onto a reduced metric set. The check that resolved it — sampling the row
  count twice 20 s apart — cost 40 seconds and saved the run.

---

## 8. The instrument as built

| | value |
|---|---|
| instrument | `dial_grid_372col_ladder.parquet`, sha256 `4c3874a78c469e15…` |
| rows (DISTINCT settings) | **9,593** (the postC grid holds 4,424) |
| full archive (every step + `saturated` + `encode_sha` + every metric) | 11,921 rows |
| ladders | `jpeg` 39 · `webp` 39 · `avif-svt` 39 · `avif-rav1e` 39 · `jxl` **26** |

The canonical grid, for contrast, carries `avif` 35 / `jpeg` 22 / `jxl` 33 / `webp` 16 —
uneven per-codec coverage that this instrument does not inherit.

**Saturated (duplicate) settings removed per codec:**

| codec | cells | saturated | distinct |
|---|--:|--:|--:|
| `avif-svt` | 2,574 | **936 (36.4 %)** | 1,638 |
| `jpeg` | 2,574 | 585 (22.7 %) | 1,989 |
| `webp` | 2,574 | 274 (10.6 %) | 2,300 |
| `avif-rav1e` | 2,574 | **78 (3.0 %)** | 2,496 |
| `jxl` | 1,625 | 0 | 1,625 |

`avif-svt` and `avif-rav1e` differ by **12x** in duplicate rate on the same quality
axis — the clearest possible argument that a per-codec step table could not have
expressed either one, and that dedup had to key on the encoded bytes.

### 8.1 A defect this surfaced, filed: `imazen/jxl-encoder#101`

130 jxl cells failed to decode, and they were not noise: **13 odd-dimensioned
sources x the 10 largest distances**. Chasing it produced a precisely-bounded
encoder bug.

**At butteraugli distance >= 10.0, `jxl-encoder` writes a `SizeHeader` rounded UP to
even.** Read from the codestream's own header (a minimal `FF 0A` + LSB-first bit
parse, not from a decoder's buffer), a 513x769 source declares:

| distance | 8.0 | 8.5 | 9.0 | 9.5 | **9.9** | **10.0** | 10.5 | 11.0 |
|---|---|---|---|---|---|---|---|---|
| declared | 513x769 | 513x769 | 513x769 | 513x769 | **513x769** | **514x770** | 514x770 | 514x770 |

The threshold is exact. Both dimensions round independently (`769x513 -> 770x514`),
even-dimensioned sources are unaffected, and the 2026-07-27 run of the same sources
shows the identical signature — so it is **pre-existing, not introduced here**. The
decoder is correct; it reproduces what was signalled. A supporting oddity consistent
with a mode switch at that threshold: encoded size is **not monotone** across it
(d=10.0 gives 8,813 B against d=8.0's 8,635 B on the same image).

**This was diagnosable only because the run persisted encoded bytes.** No re-encode
was needed to find, bound, or file it.

Consequence for the instrument: those 13 ladders lost exactly their FLOOR cells, so
they are **excluded as truncated-floor** rather than graded on distance 8 — which
would have flattered jxl's bar. jxl therefore carries 26 ladders.

---

## 9. THE RESULT — the shipped dial fails a floor it used to pass

`peer_ssim2`'s bars on this instrument, and every candidate against them
(`ZL_ERA=ladder`, `scripts/dialgate_arms.sh score`):

| bake | `avif-rav1e` | `avif-svt` | `jpeg` | `jxl` | `webp` | A7r |
|---|--:|--:|--:|--:|--:|:--:|
| **peer_ssim2 — THE BAR** | 0.5385 | 1.0000 | **0.5385** | 0.9231 | 1.0000 | — |
| **Profile D — SHIPPED** | 0.5385 ✓ | 1.0000 ✓ | **0.5128 ✗** | 0.9615 ✓ | 1.0000 ✓ | **FAIL** |
| `lam1em3` | 0.2564 ✗ | 1.0000 ✓ | 0.4615 ✗ | 0.7308 ✗ | 1.0000 ✓ | FAIL |
| `Dpeaks` (`lam2em3`) | 0.2821 ✗ | 1.0000 ✓ | 0.3846 ✗ | 0.5769 ✗ | 1.0000 ✓ | FAIL |

**Shipped Profile D reads a clean A7r PASS on every prior grid and FAILS here, on
jpeg, by one ladder (20/39 against the mentor's 21/39).** That gap was structurally
invisible before: jpeg's bar was `0.0000` because its three lowest "settings" were
one setting sampled three times. This is the instrument doing the job it was built
for.

Its other two regression misses, both real but small:

| axis | shipped D | bar (mentor) |
|---|--:|--:|
| A1 ceiling — pooled dial max | 99.99996372 | **100.0** (ssim2 reaches exactly 100) |
| A3 robust ceiling — dial p95 | 93.8842 | **93.9743** |

A2/A4/A5/A6 all pass comfortably — this grid probes far deeper than the old one
(D reaches **-94.97** against the mentor's -64.33, reach 194.97 vs 164.33).

### 9.1 The JXL inversion PERSISTS — pre-registered expectation confirmed

`d_peaks_jxl_floor_2026-09-05.md` §4 measured the peaks arms' jxl failure as a
RAW-level inversion (8/8) and concluded the lever is in the weights. The plan
therefore pre-registered that it should **persist** on new pixels and a new encoder
era. It does, and worse: jxl 0.7308 / 0.5769 against a 0.9231 bar, plus new failures
on `avif-rav1e` and `jpeg`. A "fixed by re-anchoring" result here would have been
the suspicious one.

### 9.2 MEASURED: re-anchoring cannot fix this, so no spline-only arm can ship

Not argued from monotonicity — measured, by the same method §4 of the jxl-floor lane
used. `bake_dial_refit predict` on all 9,593 rows, with and without `--score-units`,
over shipped D's 39 jpeg ladders:

| | count |
|---|--:|
| RAW (pre-spline) bottom-3 not ordered | **19** |
| DIAL (post-spline) bottom-3 not ordered | **19** |
| ladders where raw and dial verdicts AGREE | **39 / 39** |

Every jpeg failure is already an inversion in the raw model. A monotone output
spline preserves rank by construction, so **no re-anchoring — on any anchor, with
any knot placement — can turn D's jpeg floor into a pass.** ARM 5's spline-only
levers can move A1 and A3 (range properties) but provably not A7r.

---

## 10. SHIP DECISION — nothing installs

The pre-registered gate (plan §7) required all four: contract 6/6, **floor
representability >= mentor on EVERY codec**, CID22 >= today's D with the CI not
excluding a gain, and no regression axis lost.

**No arm passes, including the incumbent.** `ZensimProfile::D` is unchanged and
`zensim/weights/` was not opened for writing.

That is a stronger outcome than an install would have been: the instrument now
detects a real, previously-unmeasurable gap in the shipped dial, and §9.2 says the
fix has to be in the **weights**, not the calibration. The registered next arm is a
fit that constrains adjacent-step ordering at the floor — the same lever
`d_peaks_jxl_floor` §7(a) registered for jxl, now with jpeg and `avif-rav1e`
evidence behind it and an instrument that can actually score it.

