# D-vs-ssim2 ladder inversions — census on both instruments, and the ten worst (2026-09-05)

> **⛔ ERA NOTE (added 2026-09-05): every inversion count, `mono_pct` and ladder-zone
> figure below is the `single` reading — every material backwards rung charged to the
> DIAL, including the rungs where the CODEC itself ran backwards.** Since the
> 2026-09-05 user ruling the operative reading is `agree`: a rung is charged to the
> ENCODER where ssim2 (≤ −0.5 pt) AND butteraugli-pnorm3 (≥ +0.05 distance) both call
> the higher setting worse. The two are different quantities — do not compare or
> average them. Reproduce these numbers exactly with
> `bake_verdict --inversion-truth single` (proven byte-identical to the pre-ruling
> binary). Rule + margins + the per-codec encoder table:
> [`inversion_truth_2026-09-05.md`](inversion_truth_2026-09-05.md); registry scope
> `inversion-counts-single-reference-pre-2026-09-05`.


**Report-only lane.** No rule, registry, default, weight, or spline changed. `zensim/src/profile.rs`
and `zensim/weights/` were not opened for writing. This answers the user's read of the fair
gauntlet's `d_id100_negrich@did100lane` cell ("it says inversions and corruptions are a problem")
for the **inversions** half — codec-loop ranking, not corruption robustness.

**Bake scored throughout:** `zensim/weights/d_sdr_add156_id100_negrich_dial_2026-09-05.bin`,
sha256 `921a8f677a225b01dd1030f805f8429e6e6100325e50e87d2c56bfd32a1acad1` — verified byte-identical
to the board cell's own `bake_sha256`, and a fresh `bake_verdict --full-json` run against the
board's own grid reproduces the board's `dial.zones` block **exactly** (`mono_pct`, `tied_pct`,
all 24 codec/class/zone cells, all 12 named `worst_ladders` — byte match, not approximate).

## 1. Which grid the board's dial block is on — and the quarantine question

`d_id100_negrich@did100lane.fulleval.json`'s `dial.zones.grid` names
**`/mnt/v/output/zensim/eval_panels_2026-05-29/dial_grid_372col_2026-05-29_quarantined_v2.parquet`**
(4,424 rows, 106 ladders) — the CANONICAL, already-quarantined grid (`zensim_validate::eval_roots`'
pin, the same grid `peer_ssim2`'s registered G-ADDR bars are measured on). It is not the raw
2026-05-29 grid and not the single-quarantine intermediate. Row/ladder counts, measured directly:

| grid | rows | ladders |
|---|--:|--:|
| `dial_grid_372col_2026-05-29.parquet` (raw, CORRUPT) | 4,817 | 115 |
| `..._quarantined.parquet` | 4,457 | 106 |
| `..._quarantined_v2.parquet` (**what D was scored on**) | 4,424 | 106 |

**The 9 w11/GPU-odd-dim ladders CLAUDE.md flags are not merely flagged — they are physically absent
from the file D was scored on** (115 → 106 ladders, all 9 dropped are `webp` except one `jpeg`):

```
0e53ea752da698d9_1022x818/webp   1a20ecb0c1b92466_1022x818/webp
9059ec43b26aa167_769x513/jpeg    9059ec43b26aa167_769x513/webp
a06b91d3d8419aad_513x769/webp    a9143f4b78fe5a13_513x769/webp
c37e9ae52fbab790_1022x818/webp   ef576c4ed599d75d72145a8f34b58ccb_1022x818/webp
f65a24b7e176eb47_1022x818/webp
```

The further `quarantined` → `quarantined_v2` step drops 33 more rows (4,457 → 4,424) at the SAME
106 ladders — the pre-fix JXL cells at butteraugli distance 0.025 documented in
`failure_profiles_2026-08-31.md` §5.1, individual rungs within surviving ladders, not whole ladders.

**Conclusion: none of D's counted board-grid inversions can sit on a quarantined ladder — it is
structurally impossible, since those ladders do not exist in the 4,424-row file the board cell was
computed from.** (Two of the dropped webp ladders' *image sources* — `a9143f4b78fe5a13` and
`f65a24b7e176eb47` — do reappear below in the §5 top-10 list, but only via their **jpeg** and
**avif-rav1e** ladders on the completely separate NEW ladder instrument, never via the dropped webp
ladder itself.)

## 2. Method — owner tools, cross-validated before trusting anything

`bake_verdict` computes D's dial zones natively (`--full-json`, `dial.zones`), but refuses to emit
JSON for a **peer** score (`--dial-peer-scores`) — the markdown `--output` report is the only
peer-mode surface, and it does not print named worst-ladders (JSON-only). To get ssim2's own
per-ladder classification (needed for the coincidence question), `zensim-validate/src/bin/
bake_verdict.rs`'s five-bucket rule (forward / material inversion / codec-saturated / flat-clamp /
sub-resolution; `MATERIAL_INV_PT=0.5`, `FEAT_EPS=1e-5`, `ZONE_EDGES=[50,85]`, zone by pair midpoint,
ladder-level zone by each POINT's own q) was read from source and ported to Python
(`~/tmp/dinv_ladder_reconstruct.py`), fed the grid's own stored features (for `codec_sat`) and both
scorers' raw per-cell predictions (`ZENSIM_DIAL_PRED_OUT` for D; the `dialcells_ssim2_*.tsv` files —
already exactly ssim2's raw per-cell scores — for the peer).

**Validated, not assumed:** the ported classifier's per-`(codec,zone)` aggregate counters
(`n_pairs`/`forward`/`inv_material`/`codec_sat`/`subres`/`flat`/`inv_strict`) were checked against
`bake_verdict`'s own `--full-json` output for D on **both** grids — **0 mismatches on every counter,
both grids** — and its 12 named `worst_ladders` (board grid) / new-instrument top-12 reproduce to
`1e-9` on `end_delta`/`worst_step`/`n_rungs`. Only after this passed was the same classifier trusted
to name ssim2's own inverted ladders (which `bake_verdict` cannot surface at all today — a real,
documented gap: peer mode has no per-ladder JSON, board-registered or not).

## 3. Full census — both instruments, D vs ssim2, per codec × zone

**Board grid** (`..._quarantined_v2.parquet`, 4,424 rows, avif/jpeg/jxl/webp, 106 ladders):

| codec | zone | pairs | D fwd | D mat-inv | **D inv%** | D codec-sat | D subres | D mag med/max | D ladders ≥1inv | D ends-bwd | **ssim2 inv%** | ssim2 mag max | ssim2 ladders ≥1inv | who inverts more |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|
| avif | q<50  | 280  | 278 | 1  | 0.357% | 0   | 1   | 2.27/2.27   | 1/35  | 0/35 | 0.000% | 0.0  | 0/35  | **D** |
| avif | q50-85| 315  | 295 | 10 | 3.175% | 0   | 10  | 3.75/6.99   | 6/35  | 0/35 | 0.950% | 20.9 | 2/35  | **D** |
| avif | q>=85 | 770  | 265 | 50 | 6.494% | 35  | 420 | 2.59/13.36  | 14/35 | 2/35 | 3.770% | 52.2 | 16/35 | **D** |
| jpeg | q<50  | 176  | 131 | 1  | 0.568% | 44  | 0   | 3.71/3.71   | 1/22  | 0/22 | 0.570% | 8.8  | 1/22  | ~tie (ssim2 +0.002pt) |
| jpeg | q50-85| 198  | 189 | 0  | 0.000% | 0   | 9   | —           | 0/22  | 0/22 | 0.000% | 0.0  | 0/22  | tie |
| jpeg | q>=85 | 484  | 207 | 3  | 0.620% | 110 | 164 | 1.29/1.98   | 1/22  | 0/22 | 0.000% | 0.0  | 0/22  | **D** |
| jxl  | all 3 zones | — | — | 0 | 0.000% | — | — | — | 0/23-33 | 0 | 0.000% | 0.0 | 0 | tie (both clean) |
| webp | q<50/q50-85 | — | — | 0 | 0.000% | — | — | — | 0/16 | 0 | 0.000% | 0.0 | 0/16 | tie |
| webp | q>=85 | 352  | 158 | 1  | 0.284% | 112 | 81  | 2.76/2.76   | 1/16  | 0/16 | 0.000% | 0.0  | 0/16  | **D** |

Pooled (`all`, every codec): **D worst zone is q>=85** — 1.933% inv rate (54/2794), 16/106 ladders
(15.1%) touched, **2/106 end backwards**. ssim2 pooled: q<50 0.14%, q50-85 0.37%, **q>=85 1.04%**
(29/2794, 16% ladders touched, 0 ending backwards). **D inverts more than ssim2 at every pooled
zone on the board grid** (2.0×, 3.4×, 1.9× respectively), and D is the only one of the two with
ladders that end backwards there (2 vs 0).

**New ladder instrument** (`dial_grid_372col_ladder.parquet`, 9,593 rows, 5 codec families incl.
both AVIF backends, 182 ladder-zone groups; `ladder_instrument_2026-09-05.md`):

| codec | zone | pairs | **D inv%** | D mag med/max | D ladders ≥1inv | D ends-bwd | **ssim2 inv%** | ssim2 mag max | ssim2 ladders ≥1inv | who inverts more |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|---|
| avif-rav1e | q<50   | 1287 | 2.486% | 1.09/4.54 | 16/39 | 0/39 | 3.500% | 6.2 | 20/39 (51%) | **ssim2** |
| avif-rav1e | q50-85 | 429  | 0.932% | 2.23/2.26 | 1/39  | 0/39 | 0.000% | 0.0 | 0/39  | **D** |
| avif-rav1e | q>=85  | 741  | 0.270% | 0.98/0.98 | 1/39  | 0/39 | 0.130% | 0.5 | 1/39  | **D** |
| avif-svt   | q<50   | 858  | 0.000% | —         | 0/39  | 0/39 | 0.117% | 0.6 | 1/39  | **ssim2** |
| avif-svt   | q50-85 | 429  | 0.466% | 1.38/1.38 | 1/39  | 0/39 | 0.000% | 0.0 | 0/39  | **D** |
| avif-svt   | q>=85  | 312  | 0.321% | 0.79/0.79 | 1/39  | 0/39 | 0.000% | 0.0 | 0/39  | **D** |
| jpeg       | q<50   | 936  | 1.923% | 1.37/6.05 | 11/39 | 0/39 | 5.128% | 12.1| 17/39 (44%) | **ssim2** |
| jpeg       | q50-85 | 429  | 0.000% | —         | 0/39  | 0/39 | 0.233% | 0.7 | 1/39  | **ssim2** |
| jpeg       | q>=85  | 585  | 0.171% | 1.98/1.98 | 1/39  | 0/39 | 0.000% | 0.0 | 0/39  | **D** |
| jxl        | q<50   | 208  | 0.000% | —         | 0/26  | 0/26 | 0.481% | 1.5 | 1/26  | **ssim2** |
| jxl        | q50-85/q>=85 | — | 0.000% | — | 0/26 | 0/26 | 0.000% | — | 0/26 | tie (both clean) |
| webp       | q<50   | 1326 | 0.377% | 0.55/2.25 | 3/39  | 0/39 | 0.452% | 2.3 | 4/39  | **ssim2** |
| webp       | q50-85 | 429  | 0.000% | —         | 0/39  | 0/39 | 0.466% | 2.8 | 1/39  | **ssim2** |
| webp       | q>=85  | 506  | 0.000% | —         | 0/39  | 0/39 | 0.000% | 0.0 | 0/39  | tie |

Pooled (`all`): **D worst zone flips to q<50** — 1.192% (55/4615), 30/182 ladders (16.5%) touched,
**0 ending backwards**; q>=85 becomes D's cleanest zone (0.136%, 3/182, 0 ending backwards). ssim2
pooled: q<50 **2.19%** (101/4615, 24% ladders), q50-85 0.16%, q>=85 0.03%. Cell tally over the 15
codec×zone rows above: **ssim2 inverts more in 7, D in 5, tie in 3.**

**Plain answer to "more or less, per codec and zone": on the board/canonical grid D inverts at or
above ssim2's rate in 11 of 12 codec×zone cells (strictly more in 6, essentially tied in the
other 5-6). On the new floor-dense instrument the picture inverts specifically at the q<50 floor:
ssim2 itself inverts MORE than D on avif-rav1e, avif-svt, jpeg, jxl and webp there — because that
is exactly where the mentor's own RD curve is measurably non-monotonic**
(`ladder_floor_resolution_2026-09-05.md` §3: 78%/67% of the mentor's own bottom-triplet failures on
jpeg/avif-rav1e are genuine, non-noise inversions, median 1.29/1.43 ssim2 points, at near-flat
bytes). D stays somewhat worse than ssim2 at q50-85/q>=85 on avif-rav1e, avif-svt and jpeg, but at
absolute rates ≤1% either way (1-2 events per cell against denominators of 300-900).

**Overall pooled `mono_pct`** (informational — this is a *rate*, not the same as the per-zone
picture above): board grid D=**0.98472** vs ssim2=**0.99244** (ssim2 more monotone, matches the
registered pin `benchmarks/dial_addressability_floor_2026-09-04.json` to 4 dp); new ladder
instrument D=**0.99313** vs ssim2=**0.98877** (D is now the MORE monotone of the two — a genuine
reversal, driven by the floor-dense instrument surfacing far more of the mentor's own q<50
non-monotonicity than the old grid ever sampled). **Note:** the task brief's stated
`peer_ssim2 0.9779` does not match either measured value; both numbers above were independently
reproduced through `bake_verdict --dial-peer-scores` and one matches the project's own registered
G-ADDR pin exactly, so `0.9779` looks stale or from a different reading and is not used here.

## 4. Coincidence — same ladders (encoder non-monotonicity) or different (D's own)?

Per-`(image, codec, zone)` group carrying ≥1 material inversion (`has_inv`), both scorers, same
grid, same classification rule:

| grid | D inverted groups | ssim2 inverted groups | **BOTH (coincide)** | D-only | ssim2-only |
|---|--:|--:|--:|--:|--:|
| board grid | 24 | 19 | **11 (46% of D's)** | 13 | 8 |
| new ladder instrument | 35 | 46 | **26 (74% of D's)** | 9 | 20 |

**Reading:** on the new, denser instrument, three-quarters of D's inverted ladder-zones are ALSO
inverted by ssim2 in the same (image, codec, zone) group — evidence for shared encoder
non-monotonicity rather than a D-specific defect, for most of the surface. The board grid's lower
coincidence rate (46%) is partly an artifact of its much sparser q<50 sampling (722 pairs vs 4,615
on the new instrument) giving ssim2 fewer chances to show its own floor wobbles there.

**D-only groups are not random** — on both instruments the single reference `090d19695a8b43c2_512sq`
recurs disproportionately (3 of 13 D-only board-grid groups: avif q50-85, avif q>=85, jpeg q>=85;
6 of 9 D-only ladder-instrument groups: avif-rav1e q50-85/q>=85, avif-svt q50-85/q>=85, jpeg
q<50/q>=85) — this ONE source is D's most reference-specific weak point, present across both grids
and across three codec families, independent of ssim2. It did not make the top-10-by-magnitude list
below (its worst individual steps are smaller than the ones that did), but it is the recurring name
worth a follow-up look if this becomes a real fix target.

**The board grid's 2 ladders that end backwards for D** (`00b13be94a4867dd_1022x818`/avif/q≥85,
end −5.29; `f65a24b7e176eb47_1022x818`/avif/q≥85, end −2.45) are the **same two images, same codec,
same zone** `failure_profiles_2026-08-31.md` §4.5 already named for ADD156 (−5.19/−2.47) — this is
a lineage-wide avif near-lossless weakness D inherited, not something new to D's spline.

## 5. D's ten worst material inversions on the new ladder instrument

Selected by `worst_step` magnitude over ALL 35 of D's material-inversion groups on the new
instrument (not the JSON's own top-12, which sorts by end-of-ladder direction first and truncates —
the fuller, independently-validated Python reconstruction was used to get the true top 10). **All
ten are in the `q<50` aggressive-compression zone** (6 jpeg, 4 avif-rav1e) and **none end their
ladder backwards** — every one is an internal wobble that the ladder recovers from by its top q.

| # | image | codec | step | D | ssim2 (truth) | bytes | ssim2 agrees? |
|--:|---|---|---|---|---|---|:--:|
| 1 | `f65a24b7e176eb47_1022x818` | jpeg | q12→q13 | 54.74→48.69 (Δ−6.05) | 61.15→49.01 (Δ**−12.14**) | 26,342→26,446 (+0.4%) | **yes** |
| 2 | `d01e6b7798bbe066_513x769` | jpeg | q0→q11 | 23.81→18.91 (Δ−4.90) | 27.54→20.05 (Δ**−7.49**) | 10,129→10,233 (+1.0%) | **yes** |
| 3 | `b2e6e2b5969eaf25_1022x818` | avif-rav1e | q2→q3 | −1.05→−5.59 (Δ−4.54) | −6.07→−10.59 (Δ**−4.52**) | 2,850→2,917 (+2.4%) | **yes** |
| 4 | `76c1e30469720c75_769x513` | avif-rav1e | q5→q6 | 26.81→22.85 (Δ−3.96) | 23.32→21.10 (Δ**−2.22**) | 1,496→1,482 (−0.9%) | **yes** |
| 5 | `68845bbc29306de5_769x513` | avif-rav1e | q3→q4 | −7.85→−11.43 (Δ−3.58) | −13.33→−13.09 (Δ**+0.24**) | 1,605→1,656 (+3.2%) | **NO — D-specific** |
| 6 | `5a9b3b963f852e20_512sq` | jpeg | q16→q17 | 49.20→45.87 (Δ−3.32) | 48.60→38.87 (Δ**−9.73**) | 6,603→6,854 (+3.8%) | **yes** |
| 7 | `a9143f4b78fe5a13_513x769` | jpeg | q16→q17 | 56.95→53.69 (Δ−3.25) | 52.68→43.84 (Δ**−8.84**) | 7,569→7,828 (+3.4%) | **yes** |
| 8 | `20f63bf11ab2c911_512sq` | jpeg | q0→q11 | 23.44→20.86 (Δ−2.58) | 23.65→16.39 (Δ**−7.26**) | 5,960→6,000 (+0.7%) | **yes** |
| 9 | `d01e6b7798bbe066_513x769` | avif-rav1e | q0→q2 | −17.29→−19.81 (Δ−2.53) | −5.71→−11.91 (Δ**−6.20**) | 1,966→1,953 (−0.7%) | **yes** |
| 10 | `b2e6e2b5969eaf25_1022x818` | jpeg | q12→q13 | 30.22→27.85 (Δ−2.37) | 31.31→23.58 (Δ**−7.73**) | 13,912→13,973 (+0.4%) | **yes** |

**9 of 10 are confirmed by ssim2 at the exact same step** — and in every one of those 9, ssim2's own
drop is LARGER than D's (median ssim2 Δ ≈ −7.5 vs D's median Δ ≈ −3.6), at flat-to-small byte
increases (−0.9% to +3.8%). This is the same signature `ladder_floor_resolution_2026-09-05.md`
documents for the mentor's own floor failures: real RD-curve non-monotonicity at a near-flat part
of the curve, not measurement noise on either side. **Only #5 is D-specific** — ssim2 is flat there
(+0.24, inside its own noise) while D reads a real −3.58 point drop.

### The visual page

`/mnt/v/output/zensim/ladder-2026-09-05/inversions/index.html` — for each of the 10, the reference
plus the two failing steps' full frame (Mitchell-downscaled, ≤512px) and a same-window native 1:1
crop (centered, sized to 40% of the image's short side — not hand-picked, so the choice cannot be
accused of cherry-picking a flattering or damning region), each labelled with q / ssim2 / D /
encoded bytes, plus a per-entry badge stating whether ssim2 independently confirms the drop.

- **Local:** `http://localhost:3300/zensim/ladder-2026-09-05/inversions/index.html` — curl `200`.
- **LAN:** `http://192.168.50.44:3300/zensim/ladder-2026-09-05/inversions/index.html` — curl `200`.

Tiles rendered **only** through `zensim-bench/examples/ladder_tile_gen` (`full`/`crop` modes —
`zenpng` decode/encode + `zenresize` Mitchell downscale; no foreign imaging tool touches a pixel),
per the imazen-only-codec-tools rule. 60 PNG tiles, 3.9 MB total, under `.../inversions/tiles/`.

## 6. A live, pre-existing build defect found and worked around (not fixed, not shipped)

Building `ladder_tile_gen` from a clean `zensim-bench` checkout currently fails on
**any** target: `zenjxl`'s Cargo.toml pins `jxl-encoder = "0.3.2"` (`^0.3.2`, i.e. `<0.4.0`), but
the local sibling `~/work/zen/jxl-encoder` has been at **`0.4.0`** since a genuine, already-pushed
release commit (`4363a3d5`, "release-prep(#76): ... 0.4.0 version bump") — this predates and is
unrelated to the concurrent issue-101 session active in that repo during this lane (confirmed via
`git log -S`, not assumed). `zensim-bench`'s own `[patch.crates-io]` path-patches `jxl-encoder` to
the local sibling, but Cargo only honours a patch when its version satisfies the dependent's
requirement string — `0.4.0` does not satisfy `^0.3.2`, so the patch is silently skipped and
resolution falls through to crates.io, where the same gap exists (published: 0.4.0/0.3.1/0.3.0,
none satisfy `^0.3.2` either). Confirmed this is universal, not workspace-specific: copying the
PRIMARY checkout's own (gitignored, previously-resolved) `zensim-bench/Cargo.lock` into this lane's
workspace and building `--locked` fails identically, because a `path` dependency's version is read
live from its manifest regardless of what the lock expects.

**Worked around locally, in `zensim-bench/Cargo.toml` only, and fully reverted before this lane
ends** (`diff` against the pre-edit copy is empty; `jj status` shows no changes): commented out the
`zenjxl` optional dependency and every feature/`[[example]]` block that names it
(`extract-omni`, `verify-jxl`, `verify-all`, `zen-decode-tests`, `hdr944-extract`,
`[[example]] hdr944_extract`), which drops the whole `zenjxl`→`jxl-encoder` edge from resolution —
`m3-fixtures` (what `ladder_tile_gen` needs: `zenpng`/`zenjpeg`/`zenresize`/`zenpixels`/
`zenpixels-convert`/`enough`) never touches `zenjxl`. Built clean in 20s. **Not a fix** — this repo
did not touch `zenjxl` or `jxl-encoder`, and the underlying pin staleness (someone needs to bump
`zenjxl`'s `jxl-encoder` requirement to accept `0.4.x`, or re-check what else changed under
semver-checks) is a real, currently-live defect worth the user's attention, flagged here rather than
silently worked around and left undocumented.

### Addendum (2026-09-05, later same day): FIXED — but the live defect was one level deeper than
### either of today's two reports found

USER-AUTHORIZED sibling-repo fix, verbatim "fix the zenjxl pin." Two separate things were true at
once, and each masked the other:

1. **`zenjxl`'s own pin was NOT actually broken on `main` — it had been fixed six days earlier**
   (`5e9b8793`, 2026-08-30, "fix(deps): jxl-encoder requirement 0.3.2 -> 0.4.0 + adapt to the 0.4.0
   API restructure," already pushed to `origin/main`). Both this lane's report above and the
   independent 2026-08-28-dated campaign-log mention of the same symptom
   (`balance_campaign_2026-08-28.md`) were reading a **stale local `~/work/zen/zenjxl` checkout** —
   8 commits behind `origin/main`, predating the fix — because nobody had run `jj git fetch` in
   that repo since before 2026-08-30. `git log -S` on the (stale) checkout correctly showed the
   fix commit didn't exist *there*, which is why both reports treated it as still-live; it existed
   on the remote the whole time. `jj git fetch && jj new main@origin` in `~/work/zen/zenjxl` is the
   entire fix for this half — zero new zenjxl code was needed.
2. **A SEPARATE, genuinely live defect was hiding behind #1**: `zensim-bench/Cargo.toml`'s own
   `[patch.crates-io]` table did not use a plain path patch for `jxl-encoder` — it pinned a specific
   git rev (`jxl-encoder = { git = "file:///home/lilith/work/zen/jxl-encoder", rev = "bfb880f9",
   package = "jxl-encoder" }`), added 2026-08-29 as a stopgap for the *original* version-mismatch
   window (before zenjxl's own fix existed), with an explicit "retire this the moment zenjxl accepts
   ^0.4" comment. zenjxl accepted `^0.4.0` the very next day (`5e9b8793`), but nobody retired the
   stopgap. By 2026-09-05 the stopgap was itself failing — `bfb880f9` predates jxl-encoder's version
   bump to 0.4.0 (it still declares itself `0.3.2`), so once zenjxl's requirement correctly reads
   `^0.4.0` (post-fetch), that pinned rev **no longer satisfies zenjxl's own requirement**, and
   `cargo update`/`cargo metadata` failed with the same-shaped error
   ("candidate versions found which didn't match: 0.3.2, 0.3.1, 0.3.0...") one version later. This
   is why fetching zenjxl alone was not sufficient to unblock `zensim-bench` — the real blocker by
   today had moved into zensim's own tree.

**Fix applied:** `zensim-bench/Cargo.toml`'s `jxl-encoder` patch reverted to a plain path patch
(`{ path = "../../jxl-encoder/jxl-encoder" }`, matching zenjxl's own convention and the other
sibling entries in the same table) — safe now because both sides' version strings agree (`0.4.0`).
No zenjxl code changed; `zenjxl`'s six-day-old fix (`5e9b8793`) was re-verified, not redone.

**Proof, all from an empty `zensim-bench/target/`:**
- `cargo update` (zensim-bench workspace root): resolves clean, 329 packages locked, zero errors —
  this is the operation that was actually failing (whole-graph resolution, independent of which
  features a given build enables).
- `cargo build -p zensim-bench --features training,zen-decode,verify-jxl` — rc=0. (Note:
  `zensim-bench` has no `src/`/lib/default-bin, so a bare `-p` build with no example/bin target
  compiles nothing beyond resolution — this confirms resolution but not compilation.)
- `cargo build -p zensim-bench --example extract_features_372col --features
  training,zen-decode,verify-jxl` — rc=0, genuinely compiles `zenjxl` (with `zencodec`+`decode`).
- `cargo build -p zensim-bench --example extract_features_372col --features
  training,zen-decode,verify-jxl,zenjxl/encode` — rc=0, additionally compiles **`jxl-encoder`
  itself** and zenjxl's `encode`-gated `mod encoding` in `src/codec.rs` (the module containing
  `wrap_codestream_with_metadata`, the local port that replaced `jxl_encoder::container::
  wrap_in_container` when 0.4.0 privatized it) — the actual code this pin gates, not just the
  dependency edge. Zero errors in all three build logs (`grep -c '^error'` = 0 on each).
- `zenjxl`'s own full `just ci` (fmt-check + 5 clippy feature variants + 9 feature-check
  build/test variants incl. `--all-features`) re-run clean after the fetch: **rc=0, 0 failures,
  0 warnings, 1189s.**

**Commits:** zenjxl `5e9b8793e71ce83802a81354b84a9b53e4aba986` (the original fix, 2026-08-30,
already on `origin/main` before this addendum) + `681021dacca7...` (this lane's CHANGELOG note
documenting the stale-checkout root cause, pushed 2026-09-05, verified
`git merge-base --is-ancestor` against `origin/main`). zensim: the `zensim-bench/Cargo.toml` patch
retirement, this commit.

**Status: FIXED**, not merely worked around — the `zenjxl` dependency edge is back in the graph
(unlike the §6 workaround above, which dropped it), and every build/example/test path that touches
`zenjxl`/`jxl-encoder` is exercised clean.

## 7. Reproduction

```sh
cargo build --release -p zensim-validate --bin bake_verdict

BV=target/release/bake_verdict
D=zensim/weights/d_sdr_add156_id100_negrich_dial_2026-09-05.bin
CANON=/mnt/v/output/zensim/eval_panels_2026-05-29/dial_grid_372col_2026-05-29_quarantined_v2.parquet
LADDER=/mnt/v/output/zensim/ladder-2026-09-05/instruments/dial_grid_372col_ladder.parquet
SSIM2_CANON=/mnt/v/output/zensim/ssim2-bar-2026-08-31/dialcells_ssim2_qv2grid.tsv
SSIM2_LADDER=/mnt/v/output/zensim/ladder-2026-09-05/instruments/dialcells_ssim2_ladder.tsv

# D's own zones (full-json), both grids
$BV --bake $D --dial-grid $CANON  --corpora cid22 --full-json /tmp/D_canon.json
$BV --bake $D --dial-grid $LADDER --corpora cid22 --full-json /tmp/D_ladder.json

# ssim2 peer zones (markdown only — JSON is refused in peer mode)
$BV --bake $D --dial-peer-scores peer_ssim2=$SSIM2_CANON  --dial-grid $CANON  --output /tmp/s2_canon.md
$BV --bake $D --dial-peer-scores peer_ssim2=$SSIM2_LADDER --dial-grid $LADDER --output /tmp/s2_ladder.md

# raw per-cell dumps (for the Python cross-validation + coincidence + top-10 pick)
ZENSIM_DIAL_PRED_OUT=/tmp/D_canon_raw.tsv  $BV --bake $D --dial-grid $CANON  --corpora cid22
ZENSIM_DIAL_PRED_OUT=/tmp/D_ladder_raw.tsv $BV --bake $D --dial-grid $LADDER --corpora cid22

# validated ported classifier + coincidence (0 mismatches vs the JSON above, both grids)
python3 ~/tmp/dinv_ladder_reconstruct.py --grid $CANON  --d-raw /tmp/D_canon_raw.tsv  --ssim2-raw $SSIM2_CANON  --d-json /tmp/D_canon.json  --class-table benchmarks/dial_grid_content_classes_2026-08-31.tsv --out-prefix /tmp/canon
python3 ~/tmp/dinv_ladder_reconstruct.py --grid $LADDER --d-raw /tmp/D_ladder_raw.tsv --ssim2-raw $SSIM2_LADDER --d-json /tmp/D_ladder.json --class-table benchmarks/dial_grid_content_classes_2026-08-31.tsv --out-prefix /tmp/ladder

# tiles + page (needs the temporary zenjxl-drop workaround in §6 to build ladder_tile_gen)
cargo build --release --manifest-path zensim-bench/Cargo.toml --example ladder_tile_gen --features m3-fixtures
python3 ~/tmp/gen_tiles.py
python3 ~/tmp/build_inversions_page.py
```

Analysis scripts (scratch, per policy, exact steps above): `~/tmp/dinv_analysis.py`,
`~/tmp/dinv_ladder_reconstruct.py`, `~/tmp/gen_tiles.py`, `~/tmp/build_inversions_page.py`.

Data: `/mnt/v/output/zensim/ladder-2026-09-05/dinv/` (`json/` — D's full-json both grids;
`md/` — ssim2 peer markdown both grids; `tsv/` — raw dumps, the validated per-cell/per-ladder
JSON reconstructions, the coincidence JSONs, `top10_full.json`). Visual page + tiles:
`/mnt/v/output/zensim/ladder-2026-09-05/inversions/`.
