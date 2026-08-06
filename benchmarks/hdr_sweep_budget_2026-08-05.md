# Multi-codec HDR sweep budget — encode speed test + metric cost + weekend N (2026-08-05)

Purpose: budget the registered multi-codec HDR still-image sweep
(N sources x 6-size ladder x 21 qualities x {zenav1-svt, zenjxl,
jpeg+gainmap-via-ultrahdr}) before committing fleet time. Per the sweep
discipline (`~/work/zen/CLAUDE.md` "Sweep / Calibration"): every cell class is
measured across tiny -> 4K-class sizes, `total = alpha + beta * pixels` is
fitted per quality point, and both terms are reported. No `-C
target-cpu=native` anywhere (runtime SIMD dispatch = what ships). Raw
per-cell data + harness: see Provenance.

## TL;DR

- **THE INVERSION — encode is minutes, scoring is hours.** For the whole
  canonical 76-source corpus, the registered 3-arm encode is **6.9 CPU-h,
  which is ~3-5 MINUTES of wall** once it fans out across the available CPU
  workers, while GPU metric scoring of the same cells is **11.5 GPU-h**
  (cvvdp on CPU + ssim2/butteraugli on the 5070; 16.1 GPU-h with all three
  on GPU, 5.8 GPU-h for ssim2 alone) — **serial on one GPU**. The sweep's
  wall-clock IS the metric queue; encode cost is a rounding error. Budget,
  schedule, and optimize the metric side; do not spend effort making
  encoders faster for this sweep.
- **Encode is cheap per source**: 3-arm encode of one source's full ladder
  x 21 q is **324 CPU-s (0.09 CPU-h)**; scoring those 378 cells with
  cvvdp+ssim2+butteraugli is **763 GPU-s (0.21 GPU-h)** — 2.4x in raw
  seconds, ~200x as a *wall-clock* constraint because encode parallelizes
  and GPU scoring does not.
- **Weekend N**: 283 sources with all three metrics GPU-serial; 396 with
  cvvdp routed to CPU workers (measured: the CPU cvvdp port costs the same
  per pair as the GPU path); 792 with the minimum set (cvvdp-CPU +
  ssim2-GPU). The whole canonical 76-source corpus = **~16 GPU-h + ~7
  CPU-h**: comfortably ONE day, not a weekend.
- **Per-MP encode cost (median across q)**: zenav1-svt p6 **742 ms/MP**
  (p10 127, p13 88), zenjxl e7 **451 ms/MP** (e1 80), jpeg+gainmap
  **120 ms/MP**; context: zenavif/zenrav1e HDR default (speed 4, 4:4:4
  identity) **3.6 s/MP** (s10: 365 ms/MP). zenav1-svt HdrFork mode costs
  1.3-2.1x its mainline.
- **The preset choice is free at sweep scale**: the fastest tier
  (p13+e1+uhdr, 70 s/source) is 4.6x cheaper than the quality tier
  (p6+e7+uhdr, 324 s/source), but weekend-N is metric-bound in every
  scenario — so run the quality tier unless encode ever moves to
  billed-per-CPU-hour infrastructure.

## Method

- **Sources**: 3 content-diverse refs from the canonical 76-source HDR grid
  (`/mnt/v/output/imazen-26-hdr-grid-2026-06-14`, 16-bit PQ PNG, cICP
  {1,16,0,1}): `1069` illuminated-castle-night, `1233`
  cathedral-ceiling-interior, `1493` sunset-over-sea. Real HDR pixels
  end-to-end; nothing synthetic.
- **Size ladder** (64-aligned native pyramid levels + one real-pixel 64x64
  center crop): 64x64 / 192x256 / 384x512 / 768x1024 / 1536x2048 /
  2304x3072 = 4,096 .. 7,077,888 px (sum 11.26 MP per source per q).
- **Quality grids, 21 points each**: zenav1-svt CQP qp {1,4,...,60,63}
  (qp0 = lossless is refused by the port); zenjxl / zenavif / ultrahdr
  generic quality {0,5,...,100}.
- **Encode paths are the canonical ones**: the jxl and avif arms replicate
  `zenmetrics-cli sweep/hdr.rs::encode_jxl_hdr / encode_avif_hdr` verbatim
  (16-bit PQ code values as `RGB16_BT2100_PQ` PixelSlice + source cICP
  metadata through the zencodec adapters; avif = 10-bit identity-matrix
  MC=0 GBR 4:4:4 via zenravif/zenrav1e at its default rav1e speed 4). The
  zenav1-svt arm feeds the port's native 10-bit entry
  (`try_encode_frame_420_hbd`, still-picture CQP, presets 6 and 10) with
  BT.2020nc limited-range 10-bit 4:2:0 converted from the PQ code values
  (SVT-AV1 is a 4:2:0 encoder; this is the shape an SVT-based AVIF HDR
  pipeline would use). The jpeg+gainmap arm drives `ultrahdr-rs`
  `Encoder` (HDR-only input: PQ -> linear f32 RGBA at 203 nits = 1.0,
  internal tonemap -> base JPEG q swept + gainmap JPEG q85, 1/4-scale
  gainmap — crate defaults).
- **Timing**: in-process `Instant` around the encode call (setup timed
  separately for svt; ~0 in all cells), untimed warmup per (bin,size),
  reps 5/3/3/3/2/2 by size tier, per-cell statistic = min across reps
  (additive-noise-robust), `RAYON_NUM_THREADS=1`, cpu_ms/wall recorded per
  cell (~=1.0 everywhere -> all arms single-threaded). Cells ran nice -n19
  ionice -c3 in two serial streams on an otherwise mostly idle 7950X.
- **Metric cost**: `zenmetrics score-pairs --hdr --gpu-runtime cuda`
  (2026-08-01 release binary, RTX 5070) over same-dims (ref,ref) PQ-PNG
  pairs; per-pair marginal = (wall_9rows - wall_1row)/8, so process +
  scorer-cache init is excluded. (ref,ref) is a cost proxy: per-pair work is
  content-independent; it includes 2x PNG16 decode per row where a real
  sweep cell decodes the ref once per source and the codec variant instead.

## Encode fits: total_ms = alpha + beta * pixels, per quality point

Full per-(codec,q) table: `hdr_sweep_budget_2026-08-05.tsv` (committed
alongside; columns codec/q/n_points/alpha_ms/beta_ms_per_mp/r2/alpha_bytes/
bpp/cpu_wall_ratio_med). Every row carries its own `n_points`: **18** =
full coverage (3 sources x 6 sizes) for all seven primary arms,
16-18 for the truncated `zenrav1e-sdefault` arm, and 2 for the
`svt-p6-fork` spot check — read `n_points` before quoting a row.

Summary across the 21-point q grids (beta in ms/MP, alpha in ms; alpha is
the OLS intercept — small and noisy (|alpha| <= ~80 ms) against
multi-second large cells; the measured 64x64 cells below are the practical
fixed floor, and svt pipeline setup itself measured ~0 ms):

| codec (arm) | beta med | beta min..max over q | alpha med | one 7.08 MP cell (med) |
|---|---|---|---|---|
| zenav1-svt p6, bd10 420 CQP | 742 | 538 (qp63) .. 1639 (qp1) | 20.5 | 4.79 s |
| zenav1-svt p10 | 127 | 90 (qp63) .. 197 (qp1) | 2.6 | 0.92 s |
| zenav1-svt p6 HdrFork (spot, 2 sizes x 5 qp) | ~900 | 1.3-2.1x mainline per cell | — | ~6.5 s (est) |
| zenjxl e7 (default) | 451 | 392 .. 490 (q-flat) | 7.8 | 3.12 s |
| jpeg+gainmap (ultrahdr, gm q85 s4) | 120 | 113 (q<=50) .. 185 (q100) | -1.4 | 0.85 s |
| **fastest tier** svt p13 | 88 | 63 .. 141 | 4.1 | 0.61 s |
| **fastest tier** jxl e1 | 80 | 78 .. 84 | -5.6 | 0.57 s |
| context: zenavif HDR s10 (444 identity) | 365 | 277 .. 530 | 13.9 | 2.40 s |
| context: zenavif HDR default (rav1e speed 4, 444) | 3568 | 2181 .. 5093 | 551 | 22.98 s |

The zenavif-default row is the canonical `sweep --hdr` avif arm as it ships
today: its per-encode fixed cost is ~0.55 s (rav1e context construction)
and it costs more per source than the three registered arms combined
(959 s vs 324 s). Its slow-tier grid is truncated (sources 1+2 complete,
source 3 small tiers only — superseded mid-run by the s10 fast arm when
the preset axis was extended); the s10 row has full 3-source coverage.

Per-size median cell cost (median of per-cell min over sources x q):

| codec | 64x64 | 192x256 | 384x512 | 768x1024 | 1536x2048 | 2304x3072 |
|---|---|---|---|---|---|---|
| svt-p6 | 3.2 ms | 45.0 | 153.5 | 554.5 | 2166.5 | 4793.5 |
| svt-p10 | 0.6 ms | 7.4 | 26.6 | 103.8 | 379.6 | 923.9 |
| jxl-e7 | 2.3 ms | 22.8 | 84.8 | 399.4 | 1360.2 | 3117.9 |
| uhdr | 0.6 ms | 6.2 | 22.5 | 91.1 | 373.6 | 854.5 |
| svt-p13 | 0.4 ms | 5.6 | 20.3 | 74.1 | 273.3 | 606.0 |
| jxl-e1 | 0.3 ms | 3.3 | 13.0 | 52.2 | 229.0 | 564.9 |
| avif-s10 | 2.1 ms | 23.0 | 83.7 | 295.6 | 1080.7 | 2403.4 |
| avif-sdefault | 16.1 ms | 248.8 | 884.7 | 3616.8 | 11610.3 | 22984.5 |

Quality-axis shape (beta ms/MP at sampled q; r2 of the size fit 0.89-0.997
throughout):

| q | svt-p6 | svt-p10 | | q | jxl-e7 | uhdr |
|---|---|---|---|---|---|---|
| qp1 | 1639 | 197 | | q0 | 473 | 117 |
| qp22 | 742 | 144 | | q25 | 439 | 120 |
| qp32 | 751 | 128 | | q50 | 458 | 117 |
| qp47 | 604 | 116 | | q75 | 394 | 135 |
| qp63 | 547 | 90 | | q100 | 462 | 185 |

AV1 cost rises steeply toward low qp (high quality): svt-p6 qp1 is 3.0x its
qp63 cost. JXL is q-flat (effort-dominated). JPEG+gainmap only grows at
q>=75 (entropy-coding volume).

### The preset decision, costed

Fastest tier vs quality tier, registered arm set, one source's full grid:

| tier | arms | encode s/source | ratio |
|---|---|---|---|
| quality | svt-p6 + jxl-e7 + uhdr | 324 s | 1.0x |
| fastest | svt-p13 + jxl-e1 + uhdr | 70 s | **4.6x cheaper** |

Per arm: p13 is 8.4x cheaper than p6; e1 is 5.6x cheaper than e7; avif s10
is 9.8x cheaper than the avif default.

**RECOMMENDATION: run the QUALITY tier (svt-p6 + jxl-e7 + uhdr).**
Rationale, in order:

1. **It is free.** Both tiers finish encoding the 76-source corpus in
   minutes of fleet wall (324 vs 70 CPU-s per source, against an 11.5 GPU-h
   metric queue). Saving 254 CPU-s/source saves nothing anyone waits for.
2. **The sweep's product is RD data.** A sweep exists to characterize
   quality-vs-bytes; the fastest presets ship worse RD points at the same
   nominal q, so a picker or dial trained on p13/e1 data is calibrated to a
   configuration nobody ships. Cheap-but-wrong is the expensive outcome.
3. **The 4.6x is real but purchasable elsewhere.** If encode ever becomes
   the constraint, the same factor is available by dropping the avif arm
   (which alone costs more than all three registered arms combined) rather
   than by degrading the arms that matter.

Switch to the fastest tier only when (a) the metric set shrinks to
CPU-only so encode actually becomes the bound, (b) encode moves to
billed-per-CPU-hour infrastructure, or (c) the run is a smoke/plumbing
test where RD fidelity is irrelevant.

Fixed **byte** overhead (alpha of bytes = alpha + bpp/8 * pixels, median
across q):

| codec | fitted fixed bytes (median over q) | note |
|---|---|---|
| svt-p6 / p10 | ~5.7 / 6.0 KB | OBU seq/frame headers + content floor (fit intercept, not literal header size) |
| jxl-e7 | ~10.3 KB | codestream + color-encoding boxes + content floor |
| uhdr | ~14.7 KB | TWO JPEGs (base+gainmap) + XMP container dir + ISO 21496-1 + ICC |
| zenavif s10 / default | ~17.6 / 11.5 KB | AVIF container + 444-identity content floor |

On a 64x64 tile the uhdr overhead alone is ~29 bpp — at tiny sizes the
container, not the content, is the bitrate. Any bitrate model for this
sweep MUST carry the `header_bytes + content_bpp * pixels` split (sweep
discipline: the intercept dominates at thumbnail sizes).

## Metric cost per pair (the score half of a sweep cell)

Marginal per-pair wall, RTX 5070 + 7950X, PU/hdr feeding per the validated
paths (`--hdr-transfer pu-rescale` default):

| metric | 0.20 MP | 0.79 MP | 3.15 MP | 7.08 MP | ~ms/MP |
|---|---|---|---|---|---|
| cvvdp-gpu | 52 ms | 199 ms | 932 ms | 2285 ms | ~320 |
| cvvdp (CPU native port) | 56 ms | 226 ms | 1017 ms | 2226 ms | ~315 |
| ssim2-gpu | 78 ms | 298 ms | 1181 ms | 2765 ms | ~390 |
| butteraugli-gpu | 80 ms | 286 ms | 1105 ms | 2841 ms | ~400 |
| iwssim (CPU) | 148 ms | 698 ms | 3341 ms | 7432 ms | ~1050 |
| zensim-gpu | — | — | — | — | disabled in build ("pending v2 GPU-port validation") |

Findings:

- **cvvdp-gpu == cvvdp-CPU per pair** (~0.32 s/MP both). The "GPU" pair cost
  is dominated by CPU-side prep (PNG16 decode + linear/PU feeding), not the
  kernel — so cvvdp scoring parallelizes across CPU fleet workers at the
  same per-pair cost; the 5070 is NOT the only place cvvdp can run.

> **ANNOTATION 2026-08-05 (hdr-corpus lane) — this row was CHALLENGED and is
> UPHELD; but the general silent-fallback hazard behind the challenge is REAL
> and is a defect.** Appended, not rewritten; the numbers above stand as
> measured.
>
> **The challenge.** cvvdp can silently fall back to CPU when the GPU is
> unavailable, so "cvvdp-gpu == cvvdp-CPU" might have been CPU-vs-CPU — which
> would make the equality vacuous and the 283→396 source uplift unfounded.
>
> **Verdict: the measurement above is GENUINELY GPU.** Independently re-run on
> 2026-08-05 with `nvidia-smi` sampled at 100 ms through each scoring run
> (RTX 5070, ~7.4 GB free). `cvvdp-gpu` allocates real device memory above
> baseline and shows nonzero GPU utilization at every size; `cvvdp` (CPU) shows
> a perfectly flat allocation. Both this doc's runs and the re-run passed
> `--gpu-runtime cuda`, which is an **explicit** backend request — and the code
> refuses to fall back on those (see below). Raw:
> `~/tmp/hdrcorpus/cvvdp_probe/results.tsv`.
>
> | metric | 0.79 MP | 3.15 MP | 7.08 MP |
> |---|---|---|---|
> | `cvvdp-gpu` Δ device mem | 321 MiB | 865 MiB | **1569 MiB** |
> | `cvvdp-gpu` peak GPU util | 2 % | 8 % | 10 % |
> | `ssim2-gpu` Δ device mem | 449 MiB | 1217 MiB | **2466 MiB** |
> | `ssim2-gpu` peak GPU util | 6 % | 13 % | 24 % |
> | `cvvdp` (CPU) Δ device mem | 0 MiB | 0 MiB | 0 MiB |
>
> **The mechanism is now measured, which strengthens the row's conclusion:**
> cvvdp-gpu peaks at only **2–10 % GPU utilization**. The device is nearly idle
> during a "GPU" cvvdp pair, which is exactly what CPU-prep-bound means. Routing
> cvvdp to CPU workers therefore remains sound.
>
> **Per-pair GPU memory ceiling (the operationally useful number).** Scaling is
> ~222 MiB/MP for cvvdp-gpu and ~348 MiB/MP for ssim2-gpu. At the corpus's
> largest tier (7.08 MP) that is ~1.6 GB and ~2.5 GB respectively, so **every
> size in the ladder fits on the 8 GB fleet cards** with headroom. No size tier
> is excluded from GPU scoring, and one pair of both metrics resident together
> (~4 GB) still fits.
>
> **THE REAL DEFECT — silent CPU fallback under the DEFAULT runtime.** With the
> GPU hidden (`CUDA_VISIBLE_DEVICES=""`):
>
> | invocation | exit | stdout |
> |---|---|---|
> | `--metric cvvdp-gpu --gpu-runtime cuda` | **1** | *(nothing)* — refuses, loudly |
> | `--metric cvvdp-gpu` (runtime defaults to `auto`) | **0** | `metric=cvvdp-gpu cvvdp_imazen_v0_0_1=10.000000` |
>
> The explicit path is safe and says so itself: *"explicit backend requests
> never fall back; use Backend::Auto for fallback"*. **But the default is
> `auto`**, and in `auto` a CPU-computed value is emitted under the **GPU column
> name** `cvvdp_imazen_v0_0_1` with exit 0 — indistinguishable downstream from a
> real GPU measurement. A sidecar built that way silently mislabels CPU numbers
> as GPU.
>
> **Standing rule for every scoring run and every fleet job: pass
> `--gpu-runtime cuda` explicitly.** Never rely on the default when the column
> is going to be recorded. Worth fixing in zenmetrics so the fallback can never
> be silent (either log the runtime actually used, or stamp the mode into the
> emitted column/row); until then the explicit flag is the mitigation.
>
> Caveat on the re-run: its wall times include process startup (~0.3–1.0 s) and
> so are NOT comparable to this table's marginal per-pair figures. It measured
> **mode and memory**, not cost — the cost table above is unchanged.
- ssim2 HDR remains GPU-only (no CPU HDR dispatch — the PLAN_HDR gap), so
  ssim2-gpu is the one metric pinned to the GPU box.
- iwssim requires min(W,H) >= 176: the 64x64 tier (and any tier below
  176px short side) cannot carry iwssim.
- The **minimum metric set for the cvvdp-mix target** is cvvdp + ssim2
  (~0.7 s/MP/pair combined); butteraugli-gpu adds ~0.4 s/MP and per the
  persistence discipline all cheap variants (butteraugli max+pnorm3 come
  from one compute) should be saved when the scorer is already warm.

## Encode vs metric: which dominates?

**Metric compute dominates.** Per source (6-size ladder x 21 q):

| stage | cost | fans out over |
|---|---|---|
| encode, 3 registered arms | 324 CPU-s (svt-p6 188 + jxl 106 + uhdr 30) | every CPU worker (>= 44) |
| score 3 arms x cvvdp-gpu+ssim2-gpu+butteraugli-gpu | 763 GPU-s | 1x RTX 5070 (serial) |
| optional + iwssim (CPU, min-dim>=176 so no 64x64 tier) | +732 CPU-s | CPU workers |

At corpus scale (76 sources) that is **6.9 CPU-h of encode — 3-5 minutes
of wall once distributed — against 11.5-16.1 GPU-h of scoring that cannot
be distributed past the one GPU.** Even in raw seconds metrics cost 2.4x
the encodes; as a wall-clock constraint the gap is ~200x. Two measured
levers move the bound:

1. **cvvdp routes to CPU at zero cost** — the native CPU cvvdp port's
   per-pair marginal equals the GPU path's (both ~0.32 s/MP; the pair cost
   is CPU-prep-bound). That alone lifts weekend N 283 -> 396.
2. ssim2 HDR is GPU-only (no CPU HDR dispatch — the known PLAN_HDR gap), so
   ssim2 stays the GPU floor: minimum set (cvvdp-CPU + ssim2-GPU) = 792
   sources/weekend.

Per the persistence discipline, butteraugli's max+pnorm3 come from one
compute and every scored variant lands in the sidecar — the marginal cost
of *saving* extra variants is zero; the cost above is the compute itself.

## Fleet note (descoped)

Per-node speed benchmarking was dropped from this deliverable by
direction; fleet work belongs to the corpus-build lane. Raw per-node
encode fits measured before the descope are archived (not a deliverable)
at `/mnt/v/output/zensim/hdrbudget-2026-08-05/nodes/`. Node state after
this session: node-2 was benched from its already-booted Ubuntu side and
flipped back to Windows; node-3 was NOT benched (its firmware boot order
has regressed to Windows-first so the PXE OS flag is never consulted —
repair per the fleet runbook at an idle window) and was left in Windows
with the worker flag cleared; lianli / i265 / ryzen5800xt had their bench
stage directories removed.

## Weekend budget

Assumptions: 60 h wall; local 7950X = 24 usable single-thread workers
(run-heavy's nproc-4), lianli 7900X = 20 workers; one RTX 5070; cells are
embarrassingly parallel via zenfleet (single-threaded encoders measured
cpu/wall ~= 1.0).

| scenario (metric routing) | CPU-h/source | GPU-h/source | N_cpu | N_gpu | **N weekend** |
|---|---|---|---|---|---|
| cvvdp+ssim2+butteraugli all on GPU | 0.090 | 0.212 | 29,315 | 283 | **283** |
| cvvdp on CPU; ssim2+butteraugli GPU | 0.152 | 0.151 | 17,376 | 396 | **396** |
| minimum set: cvvdp CPU + ssim2 GPU | 0.152 | 0.076 | 17,376 | 792 | **792** |

- **The canonical 76-source HDR corpus is NOT weekend-scale — it is
  day-scale**: 76 x 0.09 = 6.9 CPU-h encode + 76 x 0.212 = 16.1 GPU-h
  full-3-metric scoring. Encoding is an afternoon on one box.
- The N that fits a weekend is therefore bounded by metric routing, not by
  any encoder: **N ~= 283 (conservative, all-GPU scoring) to ~800 (minimum
  set)**. Growing the corpus 4-10x over the current 76 sources is feasible
  before the GPU becomes a real wall.
- Storage for encoded variants (persistence rule): ~46 MB/source for the 3
  registered arms (~13 GB at N=283) — trivial for R2/mnt-v.
- Adding zenavif (speed 4, 4:4:4) as a 4th arm costs more than the other
  three arms COMBINED (see context row) — if it joins, budget it
  explicitly or drop its speed.

## Caveats

- zenav1-svt is **not perf-tuned yet** (its own perf gate targets <= ~1.2x C
  and documents the baseline as well above that), and its native-10-bit
  entry requires 64-aligned dims + still/CQP. Numbers here are the port as
  of `4c5c1324`; they will improve.
- zenavif's HDR arm is 4:4:4 identity-matrix (no subsampling) at rav1e
  speed 4 — a heavier configuration than the svt arm's 4:2:0; the two AV1
  arms are different products, not an apples-to-apples codec race.
- The ultrahdr arm's tonemap+gainmap+2xJPEG cost is included in its
  encode number; base-quality is the swept knob, gainmap q85 scale-4 fixed.
- Metric marginals were measured against PQ-PNG (ref,ref) rows: real sweep
  cells decode the codec variant instead (jxl/avif decode-back), and decode
  the ref once per source rather than per cell. Treat metric numbers as
  +-20%-class estimates, not exact.
- Single box (7950X, WSL2) + one 5070; lianli worker count comes from its
  roster (`nproc`=24), and the cvvdp-on-CPU-worker routing scenario assumes
  the locally measured per-pair cvvdp cost per worker (not measured on
  lianli).
- The dev box measured itself while carrying two concurrent bench streams
  plus resident agent load; per-cell min-across-reps bounds the inflation
  but these single-thread numbers are conservative (a quiet box runs
  faster).
- Wall-clock contention: two encode streams + metric runs shared the box;
  per-cell min-across-reps + cpu_ms cross-check absorb this (the fit input
  is the min), but tails in the raw TSV include some inflated walls.

## Provenance

`hdr_sweep_budget_2026-08-05.meta` (commits of every encoder worktree, host,
grids, harness description). Raw per-cell TSV + metric timing TSV + the
harness sources + fit/budget scripts:
`/mnt/v/output/zensim/hdrbudget-2026-08-05/` (sha256s in the .meta).
