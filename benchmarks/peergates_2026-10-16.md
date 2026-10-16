# Peer gates — 2026-10-16

**Neither candidate establishes butteraugli-level spatial steering. Matched
human-rank evidence against CVVDP-FHD is incomplete.** No acceptance bar changed.
G-RD moved to the separate shipgates-sol lane. No human-label file was opened.

Original diff-410/diff+frag-820 hashes are `7cf4cfd9`/`89cb4861`.
New steering uses those original D12 captures. Other E34 engineering results
below belong to its freshly fitted controls `458eba66`/`2df72ab0`, separately
identified from the original bytes. W-C is the closest overall E34 arm; L-A
alone passes steering but fails rank eligibility and spatial cost.

## Same-panel steering

135 unchanged cases, 55,515 base/independent block-repair forwards per peer.
Maps sum over identical clipped raster blocks; score gains use the same repairs.
Candidates use encoder-facing `refinement_gain`; peers use native diffmaps.
Spearman uses the unchanged Rust coherence owner. Every cell below is the count
with per-case rho >= .70; 135/135 is the current absolute bar. No undefined cases.

| Map | Fixed diff-410 judge | Butteraugli judge | SSIMULACRA2 judge |
|---|---:|---:|---:|
| diff-410 | 135 PASS | 44 FAIL | 39 FAIL |
| diff+frag-820 | 105 FAIL | 60 FAIL | 46 FAIL |
| butteraugli | 46 FAIL | 131 FAIL | 42 FAIL |
| SSIMULACRA2 | 50 FAIL | 70 FAIL | 84 FAIL |

Against butteraugli's own gains, the candidates trail its map by **87 and 71
cases**. Their mean rho is .537876/.602947 versus butteraugli's .923514.
Against the SSIMULACRA2 judge, diff+frag-820 beats butteraugli by four cases
and .086921 mean rho; diff-410 trails by three cases. The fixed diff-410 judge
favors diff-410's map. Own-judge success alone does not establish steering
against an independent metric. Butteraugli itself misses the 135/135 bar.

| Registered gate / bar | diff-410 | diff+frag-820 | Best E34 context | Butteraugli | SSIMULACRA2 | CVVDP-FHD |
|---|---|---|---|---|---|---|
| M2 >= .99, all cases | 131/135 FAIL | 130/135 FAIL | L-A 135/135 PASS | N/A: no exact gradient | N/A: no exact gradient | N/A: scoring reference |
| Own M3f >= .70, all cases | 135/135 PASS | 135/135 PASS | L-A 135/135 PASS | 131/135 FAIL | 84/135 FAIL | not measured; outside steering role |
| Joint G-STEER | 131/135 FAIL | 130/135 FAIL | W-C 133/132/130 FAIL; L-A 135/135/135 PASS | M2 N/A; own-map FAIL | M2 N/A; own-map FAIL | N/A |

Butteraugli's four failures are b64: broad-199 .503497, broad-202 .356643,
broad-206 .377622 (origin8206, scalar settings10/13/17, 256x144), and
broad-223 .669003 (origin8384, setting13, 192x256). Candidate misses are all
M2-only: diff-410 broad-10-b32 .965054, broad-10-b64 .938235,
broad-76-b64 .988235, broad-136-b64 .967647; diff+frag-820 broad-31-b64
.727273, broad-34-b32 .951802, broad-34-b64 .972028, broad-202-b32
.966604, broad-202-b64 .860140. [Every failing case, content identity and
M2/M3f value](peergates_2026-10-16_failures.md) accompanies this report.

## Scoring and engineering gates

CVVDP mapping was frozen before scoring: **Q=10*JOD**, unclipped;
materiality .5Q=.05JOD. Standard/ladder are 4,424/9,593 rows; negative probes
2,000; identity38. Table entries use standard/ladder order. `NM` means not
measured at the stated pin, never PASS. SSIMULACRA2 dial cells below are
historical registered0.8.2 results, separate from the54fabca9 native-map pin.

| Gate / bar | diff-410 E34 control | diff+frag-820 E34 control | W-C | Butter historical a10, ladder only | SSIM2 historical, ladder only | CVVDP-FHD |
|---|---|---|---|---|---|---|
| C1 mono >=.93 | .994210/.986399 PASS | .995368/.988205 PASS | .995137/.990012 PASS | .982999 PASS | .991606 PASS | .999305/.999787 PASS |
| C2 ties <=.05 | 0/0 PASS | 0/0 PASS | 0/0 PASS | 0 PASS | 0 PASS | .031496/.008288 PASS |
| C3 negative fraction >0 | .6095 PASS | .654 PASS | .548 PASS | NM: no probe scores | 1 PASS | 0 FAIL |
| C4 minimum <0 | -103.9396 PASS | -122.5700 PASS | -82.3141 PASS | NM: no probe scores | -770.6197 PASS | 20.02761 FAIL |
| C5 exact100 identity | 620/620 PASS | 620/620 PASS | 620/620 PASS | 38/38 PASS | 38/38 PASS | 38/38 PASS |
| C6 above identity =0 | 0/0 PASS | 0/0 PASS | 0/0 PASS | 0 PASS | 0 PASS | 0/0 PASS |
| G-DIAL p5<=25,p95>=85,mono>=.93 | 12.094/96.019/.994; -14.121/94.597/.986 PASS | 11.108/96.501/.995; -14.047/94.830/.988 PASS | 12.412/95.977/.995; -11.097/94.463/.990 PASS | 39.423/98.020/.983 FAIL | -16.252/93.974/.992 PASS | 77.280/100/.999;71.247/99.957/1.000 FAIL |
| G-ADDR CONTRACT C1-C6 | 6/6 PASS | 6/6 PASS | NM: no separate owner report | 4 measured PASS,2 NM: INCOMPLETE | 6/6 PASS | 4/6: FAIL C3,C4 |
| G-ADDR REGRESSION, registered ladder | INCOMPLETE: mentor binding | INCOMPLETE: mentor binding | INCOMPLETE: mentor binding | four codec-floor failures FAIL | 5/5 PASS | 5/5 PASS |

C1/C2/C5/C6 test monotonicity, ties, identity and scores above identity.
C3/C4 and G-DIAL test negative reach and the numerical dial range. E34 controls
pass these gates. The reference-peer outcomes above determine which absolute
requirements could reasonably be reconsidered; no gate was weakened here.

CVVDP beats both controls on material monotonicity but has more ties; all pass
those bars. It fails negative reach and misses p5 by52.280004/46.246632Q.
Thus neither absolute negative-tail nor low-p5 range is met by the scoring
reference. Identity and above-identity remain hard reference-supported bars.
The fresh CVVDP scope is38 identities, not E34's broader620 tier checks.

Registered G-ADDR resolvable window, mentor margin.5: CVVDP represented ladders
rav1e28/39 versus25/39, svt39/39 versus39/39, JPEG32/39 versus26/39,
JXL26/26 versus25/26, WebP39/39 versus39/39: all PASS. Butter historical
fractions10/39,37/39,30/39,14/26,36/39 fail four corresponding bars.
The standard grid's REGRESSION is unregistered, explicitly incomplete.
Ladder `single`/two-reference `agree` reports agree on verdicts; standard
has no two-reference attribution binding. Historical butter sensitivities
Q=100-a*d, a5/10/20: mono .991287/.982999/.973648; p5
69.711412/39.422825/-21.154350; p95 99.010247/98.020493/96.040987.
G-DIAL fails/fails/passes; all three retain four codec-floor failures.
No new butter standard-grid or negative-probe scoring was run.

| Other gate / bar | diff-410 | diff+frag-820 | W-C / E34 context | Reference-peer evidence |
|---|---|---|---|---|
| N1/N2: all24 >=99 | minima99.3891/99.5291 PASS | 99.6415/99.7035 PASS | 99.5521/99.6176 PASS | NM for peers |
| N3: >=122/144 | 127 PASS | 130 PASS | 131 PASS | NM for peers |
| K1-K3 valid packed spline | 1,000,058 probes PASS | same PASS | same PASS | N/A: model representation |
| K4: zero floor hits, seven populations | 0 PASS | 0 PASS | 0 PASS | N/A: model representation |
| K5: repeated C1/C3/C4/C6/G-DIAL | PASS | PASS | PASS | N/A: model representation |
| Scalar p95 <=50/200ms, dev, 1024/2048 square | QUAL-A22.24/83.09 PASS | QUAL-A22.38/83.68 PASS | dev NM; r7900x31.2265/131.4524 | peer p95 not measured here |
| Spatial ratio <=3, same sizes | 3.070612/3.040718 FAIL | 2.676826/2.711418 PASS | 2.831437/2.720669 PASS | new map cost NM: quiet gate refused |
| Incremental map RSS <=196608/589824KiB | 157056/576264 PASS | 157256/577216 PASS | 157088/575756 PASS | new peer map RSS NM |
| Complete input/serving/HDR, corruption, G-TARGET | INCOMPLETE | INCOMPLETE | INCOMPLETE | not measured on these peers |

The two original candidates' current map/RSS values arrived from the parallel
[ship-gates measurement](shipgates_2026-10-16.md), committed `febb0698`, after
the phase1 report froze. These r7900x D12 results supersede its older QUAL-A
map failures: diff+frag-820 now passes both map-cost cells and both candidates
pass memory. Diff-410 exceeds the ratio bar by .070612/.040718.
W-C's relative scalar runtime passes every E34 cell. Other map ratios at
1024/2048 square: W-A2.67112/3.09367, K-A3.10368/3.12804,
L-A3.11259/3.02966, K-C2.76369/3.07233; each fails at least one <=3 cell.
W-A also fails the v3/1024 relative-runtime bar: paired difference CI
[810313,1061501]ns versus +2% allowance691638ns.

The corruption failures belong to historical D plus its September8 companion,
not either candidate: honest lowering20/304 versus0; all-honest22/326=6.75%
versus <=1%; q20 ordering4877/5353=91.11% versus >=99%. No current composed
candidate or peer corruption result closes that gap. Model-only native/WASM
checks pass, but complete serving and HDR remain incomplete.
Historical detection5330/5353=99.57% passes95%; real-bug312/317=98.42%
passes90%; non-inert swaps120/120,120/120,117/117 pass100%.

G-TARGET is NM for current candidates/peers: one-shot median/p95/large-undershoot
bars are <=2/8/5%; two-shot <=1/3/5%; three-shot <=.5/1/1%, maximum<=3.
Disposition coverage must be100%, with witnessed attainability and actual shot
cost reported. Complete serving/HDR requires packed Rust-surface identity and
declared-precision pixel/cache/component parity over every supported input;
model-only checks leave that gate incomplete.

Existing SPEEDQ3 warm whole-call **medians**, dev v4x/one thread, 1024/2048
square: butteraugli0.9.3 85.5748/395.2502ms; SSIMULACRA2 0.8.2
66.6047/289.0917ms. These are not p95 or map costs. Three spaced quiet probes
all failed load1<2, so no new quiet timing was admitted. W-C's existing timing
receipt is `rc=0 702s | peak-RSS0.58GiB | min-avail20215MiB | peak-load8.15`.
New diff-410/diff+frag-820 timing receipts: `rc=0 83s | peak-RSS0.58GiB |
min-avail23083MiB | peak-load1.07` / `rc=0 81s | peak-RSS0.58GiB |
min-avail23078MiB | peak-load1.48`.
SPEEDQ3 uses individual `/usr/bin/time -v` process RSS and validated quiet-worker
receipts; its collector exit code was unavailable after recovery.

## Existing human-rank evidence

| Dataset | diff-410 | diff+frag-820 | CVVDP-FHD existing | Matched? |
|---|---:|---:|---:|---|
| KADID | .883153 (7869) | .889299 (7869) | .862469 (5000) | No |
| TID2013 | .894175 (3000) | .897616 (3000) | .877461 (1440) | No |
| KonFiG | .820340 (756) | .828234 (756) | .838991 triplet accuracy | No |
| CID22-A25 | .877025 (2192) | .878061 (2192) | NM at FHD | No |
| Four-source mean signed SROCC | .868673 | .873303 | NM | INCOMPLETE |

Candidate ten-seed composites reproduce E33's deltas versus control:
-.000888198 and +.003741264. CVVDP's KonFiG327-stimulus/10782-triplet accuracy
cannot be substituted for candidate signed SROCC. Different populations and
missing FHD CID22 prevent a valid winner or G-RANK verdict. W-C passes E34's
own eligibility/guard comparison; L-A fails (overall delta-.0226126;
KADID/TID/KonFiG deltas-.0169534/-.0350605/-.0371706 versus >=-.005).

## Draft gates for owner decision

1. **Steering:** freeze the panel, repair definition and all judges; require
   candidate pass count and mean per-case rho no lower than butteraugli under
   each same judge, with content guards. Both candidates fail the butteraugli
   judge comparison above. Keep exact-gradient M2 as a separate zensim
   correctness requirement; peers cannot satisfy it by a map proxy.
2. **Scoring:** require noninferiority to CVVDP-FHD on the registered four-source
   human composite, matched populations, paired uncertainty and every dataset/
   content guard. Freeze allowances before assessment. Current evidence is
   incomplete; unrelated display settings do not fill missing cells.
3. **Absolute gates:** consider peer-referencing the all135 steering count and
   numerical range/negative-tail requirements where the reference fails.
   Preserve identity, monotonicity, ties and above-identity limits where the
   peer passes. Keep cost, memory, serving, corruption and native RD as
   independent requirements. These are proposals, not adopted thresholds.

## Reproduction and evidence

Butteraugli0.9.3 published VCS `6e3a7bd2`, pnorm3 plus native diffmap;
SSIMULACRA2 map `54fabca919230767ba0944cbec0a37d7eb3a2a7a`, fused native map
with scalar/map score-bit parity. CVVDP uses retained CPU executable SHA256
`ca154a6a39f910c9cba21422eb9b238463661bfa2d4a8d8f10407904c83e4cd1`,
standard_fhd: 1920x1080,24inch,.6m,200cd/m2,1000:1,250lux;37.843ppd.
Embedded display entry is pinned; reflected-light defaults come from the
earlier report. Retained source-state observation `06b15665446e` is not a
reconstructed full dependency closure. Existing human-rank figures use their
own E2b build pin, separately recorded in that report.

Evidence on dev: `/mnt/v/output/zensim/peergates-2026-10-16/`, mirrored at
`/mnt/tower/output/zensim/peergates-2026-10-16/`; SHA256SUMS pins plans, inputs,
sources, binaries, maps, per-block results, owner gate JSON and full logs.
`steering-summary.json` has the full matrix; `phase1-sources.json` pins source
reports. The first96 steering cases survived a failed case-only capture join;
the corrected `(case,block)` transport preserved them exactly and completed39
remaining cases. Both attempts and the correction receipt are retained.

Sources: [E34](e34_results_2026-10-15.md), [D12](e34_d12_2026-10-15.json),
[QUAL-A](qual_a_2026-10-10.md), [E33](e33_result_summary_2026-10-10.md),
[CVVDP-FHD rank](rev4_e2b_cvvdp_display_2026-09-23.md),
[scalar timing](rev5_speedq3_2026-10-08.md),
[cost comparison](costcmp_2026-10-09.md),
[corruption](canonical_corruption_d228_2026-09-08.md).
Local measurement-tool clippy/fmt and script checks pass. The manually
triggered CI run38977865933 failed in an unchanged script test because numpy
was missing; no green CI claim is made. Later source CI was manually triggered
as run38980708462; its live result is separate from the measured gate verdicts.
