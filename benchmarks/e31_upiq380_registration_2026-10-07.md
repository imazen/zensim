# E31 — UPIQ-380 human HDR supervision for by_v2fy

Status: final registration proposal for coordinator review; not registered or
launched by this lane. Round-two admission fixes and all-pair re-extraction
are recorded below. The earlier five-source draft is superseded. Missing
execution pins and the label-provenance owner disposition block fitting.
No serving profile or production qualification changes here.

Question: does adding human HDR supervision from UPIQ-380 improve pooled
cross-image HDR ranking of by_v2fy while preserving its SDR behavior?

## Why this experiment

E27's hp4 passed the SDR guard and improved pooled HDR-VDP-3, but failed pooled
CVVDP noninferiority. Its frozen summary records hp4 pooled CVVDP delta
-0.0889260400361078, SE 0.002981641513570055. E26/E27's within-reference teacher
agreement (0.998) and pooled agreement (0.828, as recorded in E29's registration)
do not supply a human resolution of cross-image order. E29's teacher consensus
experiment remains a separate study; its outcome will not select E31's weight.

Owner decision D3 (DATA_SPLITS exposure ledger 2026-10-07) re-designates exactly
UPIQ's 380 HDR conditions as T2 TRAIN. All other UPIQ remains T0. The dataset's
own README and HDR image-directory filenames establish 140 Narwaria conditions
on 10 references and 240 Korshunov conditions on 20 references. An explicit
metadata allowlist precedes all label/pixel access. The original mixed UPIQ
subjective/objective CSVs and SDR image payloads are not ingestion inputs.
Primary source: [UPIQ dataset project](https://www.cl.cam.ac.uk/research/rainbow/projects/upiq/).

## Frozen proposed arms and training

One primary arm, `uh4`, versus E30's complete four-source nA3 arm, seed paired. Base: `sel:59f0bbc2f290@h32:H128:cv16:cf98`, head N, exact by_v2fy
420 feature IDs. Keep the E30 D1-compliant SDR recipe, SafeSyn/CID22 teachers, coverage
leg, preprocessing, seed streams and all other fit settings unchanged.
No E26/E27/E29 teacher HDR leg is added to either E31 arm. No second human weight
or loss variant is searched.

`uh4` adds the UPIQ-380 fit slice as one pooled rank-only human HDR group:
pairs across references *within this leg*, target order from human JOD,
trainer group mode `rank` as in E27 hp4. Nominal weight 4 in E26's fixed
acceptance-weight convention: actual group weight
`4 / mean_ref(1 - 1/n_ref)`, computed solely from fit membership sizes.
Do not cross-pool UPIQ targets with SDR/teacher legs. The fit has nine 14-condition references and seventeen 12-condition
references: mean acceptance is 0.9207875457875457 and the fixed actual weight
is 4.34410740924913. The transport and receipt must pin it; no labels choose it.

Target representation, fixed without consulting its distribution:
`human_score = 100 + 10 * human_JOD`, no clipping, fitted offset, fitted scale,
quantile map or normalization. UPIQ defines reference JOD as zero and higher
JOD as better quality; this monotone representation places that reference at
100. The rank-only loss uses its order, not an asserted cross-corpus cardinal
calibration. Preserve raw JOD alongside the target in the training table.

Use the current production native HDR walk via `research::extract_hdr` at
`FormulaRevision::Rev5`, the same owner used by E26/E27, with explicit 420 IDs.
UPIQ's EXRs supply BT.709 absolute linear f32 nits (`HdrEncoding::Linear`),
through zenexr and the established `upiq-exr-bt709-nits-v1` contract. E26/E27's
PQ ingress and UPIQ's linear ingress share the production HDR/PU processing;
never relabel linear EXRs as PQ or reuse the old u8-shell/Rev1/944 tables.
Absent slots stay NaN, features remain f64 until the existing trainer's f32
cast. Slot-subset identity is explicit; no family identity is inferred from
width. Record binary, build, native decoder and original input hashes.

Reference split, frozen before label conversion:
`int(SHA256(original reference EXR bytes),16) % 5 == 0` -> TRAIN development;
all other references -> TRAIN fit. Every rendition of a reference inherits
its byte-hash split. Development is report-only on epoch 119, never an
independent test, early-stopping input, hyperparameter selector or calibration
set. Its payload is excluded from the fit transport. Duplicate original-byte
reference hashes cannot cross the split; perceptual near-duplicate independence
has not been established. No split is rebalanced after seeing counts or labels.

D1 excludes AIC-3 from fitting, development, preprocessing, packing,
evaluation, control discovery and every new read. Never enumerate AIC-family
cells to discover controls. Four LODO folds (kadid, tid2013, konfig,
cid22_a25) × seeds 0–9: exactly 40 primary `uh4` cells. Source roles remain
D1's KADID TRAIN+SELECT, TID2013, KonFiG TRAIN+VAL and CID22-A25.
Preprocessing fits only each fold's training rows. KonJND BPG is not a human
label leg. The same UPIQ fit slice is available to every fold; these folds
are not four independent HDR tests. 120 epochs × 50,000 draws, final zero-based
epoch 119, H128/N, 1 CPU and 6 GiB per cell. No checkpoint or seed selection.
The exact dense prediction gate and receipt-bound Rev5 stamp apply identically
to arm and control; historical trainer provenance is not silently qualified.

E31 launches only after all 40 E30 nA3 cells are complete and pinned.
Control is that nA3 arm, not E30's E24 comparison control. See
[e30_four_source_registration_2026-10-07.md](e30_four_source_registration_2026-10-07.md)
and the exact-parity rule in
[e32_palette_registration_2026-10-07.md](e32_palette_registration_2026-10-07.md).
This lane has not opened E30 result/model bytes or certified completion.
Before fitting, freeze every source/seed's result bytes, final-119 bake,
feature list, program/binary and fit/data receipt SHA-256. Reuse requires
identical labels, roles, ordered rows, inherited feature bits, teacher/coverage
inputs, scaler/transform procedure, baseline loss weights, optimizer,
sampler/RNG/seed streams, epochs and fit program/binary identity. The added
UPIQ transport must prove the control projection unchanged. Strict-admission
or program changes that alter baseline fitting invalidate reuse. If parity
cannot be established, the coordinator registers one complete fresh matched
40-cell control before any E31 fit/assessment, after E30 completion. Freeze
the decision and parity audit before outcomes; do not mix old/new cells or
choose controls from results. Historical five-source E24 results cannot
substitute for this matched four-source baseline.

## Assessment and limits

Use E32's exact paired reduction, with ten inferential seed units, each
containing all four folds. For source k and seed s, delta(s,k) is arm signed
SROCC minus paired control signed SROCC under the existing score orientation.
Set d_s = (delta(s,kadid)+delta(s,tid2013)+delta(s,konfig)+
delta(s,cid22_a25))/4; Mean Delta = mean_s(d_s), and
SE = sample_sd(d_s, ddof=1)/sqrt(10). Equal source weights are fixed.
Do not treat 40 folds as independent units or combine source SEs in quadrature.
For W2, independently for arm and control within each KADID and TID2013
source/seed panel, average the three lowest signed distortion-type SROCCs.
Subtract paired control W2, average these two source deltas within seed,
then compute the ten-seed mean and sample_sd(ddof=1)/sqrt(10). Worst-three
membership may differ between arm and control; undefined distortion SROCCs
cannot be dropped. This is E32's seed reduction, not historical E21 quadrature.

E21 numerical guards are unchanged: Mean Delta >= -0.002; each source's
ten-seed mean delta >= -0.005; W2 mean delta > -2 paired-seed SE.
Missing cells, nonfinite metrics, undefined SE or zero-SE degeneracy mean
INCOMPLETE, never a pass. Report every seed/source, W1/W2, prediction tails
and failures. Seed variability measures optimization uncertainty under this
fixed design, not independent content-population sampling. The signed panel
scorer is `zensim-validate/src/panel.rs`, re-exporting `zenstats::panel` at
zenmetrics revision `3e16cb4550a7b47e534774ba5d48f9c08c7fc6c9` (workspace
Cargo.toml pin). Freeze the future assessment program/source/binary hashes
and a paired-reduction fixture before launch; this proposal does not claim
that execution owner already exists. No sign/search or post-result rule change.
E32's +0.002/p<0.05 adoption rule is not an E31 requirement. These SDR guards
are neither an HDR improvement rule nor a shipping adoption decision.

The UPIQ fit/development panels report pooled signed SROCC, per-study pooled
signed SROCC, within-reference SROCC and raw scatter geometry. They describe
training/development behavior and cannot establish generalization or adopt an
HDR shipping model. No UPIQ test panel is permitted. HDR-VDP-3 is UPIQ-calibrated;
its agreement with a UPIQ-trained student is not independent human evidence.

Remaining valid external HDR reads are the **frozen** HDR-VDC and
AVT-VQDB-UHD-1-HDR human-video reads, with stored frame extraction/display
contracts, original roles, exposed-study history and source hashes retained.
These may provide report-only transfer evidence for a frozen E31 candidate:
pooled and per-reference/per-study signed rank and raw geometry at the
registered video/frame aggregation. No new ffmpeg extraction, UPIQ-SDR probe
fit, baseline refit, weight choice or checkpoint selection is authorized.
Existing historical analysis scripts that load the mixed UPIQ CSV or fit the
UPIQ-SDR probe must not be invoked wholesale; a future assessment must use
the frozen stored frame inputs and the current Rust candidate scoring owner.
Neither HDR teacher metric becomes an independent human test merely because
its stored validation rows were not fit. Synthetic hdr_v3mix teacher panels
remain separately labeled report-only metric agreement.

E31 can describe UPIQ TRAIN/development cross-image ranking and paired
SDR preservation. It **cannot conclude independent still-image HDR human
accuracy, a calibrated product dial, or shipping adoption** until the planned
independent Squintly HDR study exists, with qualified HDR displays, untouched
sources and a separately preregistered paired decision rule. No numerical
adoption threshold is invented for that future study here.

## Required execution freeze before any E31 fit

Freeze ingestion receipt, fit table/manifest/key hashes, exact input inventory,
reference split, computed weight, control cells/bakes, numerical producer,
program/image/manifest pins and runtime caps. Admit source/split/role/member
and row-key bindings before any human table hash/read, using E28's admission
ordering. Include zero-open refusal tests for foreign UPIQ portions, VAL/dev
transport, wrong pins and extra members. The actual pinned extraction CLI must refuse incomplete/ambiguous extraction
options before any label-reader access and enforce exact canonical by_v2fy
IDs before image access. Freeze passing actual-binary negative controls for
malformed options and same-width/different-ID admissions as well as the
member/role/pin cases. The exact list source is
`costset2_2026-10-03.candidate_ids.json`, SHA-256
`0a6a20dc356acef3bef9deffc411f03189813e8b924fddcf7b22f7efea6b9f17`.

The legacy HDR-only derivative producer commit remains null and
`qualified_provenance=false`. No agreement with the original mixed UPIQ CSV
has been established. Before fitting, recover admissible producer provenance
without T0 reads or record an explicit owner disposition of this gap; do not
silently qualify it. The bounded recovery search is recorded in
[upiq380_label_provenance_2026-10-07.md](upiq380_label_provenance_2026-10-07.md).

Every fleet smoke must run through the image's **actual `fit-cell-exec` CLI**,
including executor staging/extraction, inventory verification and FIT_ROOT
symlink creation, with the real declared job and argv. Run a bounded result
smoke and a real-argv first-epoch check before scaling. Verify exact pack delta,
artifact persistence and actual jobset caps through the existing zenfleet owners.
No training, image build/publication, R2 upload or fleet enqueue is part of this
ingestion/proposal lane.

## Round-two ingestion and extraction pins

Canonical version: `/mnt/v/output/zensim/upiq380-rev5-r2-2026-10-07/`.
Mirror: `/mnt/tower/output/zensim-upiq380-rev5-r2-2026-10-07/`.
Extractor source `a7625c14c6b83f689515b178aa7aab52bdf35c77`; binary SHA-256
`41dffd5e3f1e18c7cbf83955d4c6a71ec03f789d825ae140e49aadc6a28ddad9`.
Admission SHA-256
`a64c1dba3640e623f485b7f267120c5fd8fb48fb7bba4d78226f90ef460df454`.
Decoder zenexr `109a9ec367278a93751f7701f8536358e5d3f5cd`.
The historical 1825-slot f64 transport remains explicit; the newer palette
slots are not E31 inputs. Exact 420 ordered IDs are pinned above.

| Artifact | SHA-256 |
|---|---|
| features.tsv | `d03c78e01f68b92bd81f1866b7344690a082d7746171158e1effa68865f07b59` |
| features.tsv.manifest.json | `40adf1dddd1de3da1f50479da6a9e8f59a05307e45576735d9585c10e61eba7d` |
| upiq380_fit.parquet | `7f09debedc591e7dd3494846ada9fe0c93b01779918b8358e2cf8053f5f1a6c4` |
| upiq380_fit.keys.parquet | `c47ca1c12d1e8e206464938884ff9d05baa8274785826060604d16caf0bff49e` |
| upiq380_fit.parquet.manifest.json | `2da346bb17e08a4a63aae9ed89b159b36edd331199a9dd1c42b4a05b53ac939e` |
| upiq380_development.parquet | `bb56912e9b8d45542b1efdd8f7df506efcee939990e06cb909a816ecd42a0243` |
| upiq380_development.keys.parquet | `a6e524f808146c2bee170a27e1bbbdca05327d2fb00555625b3b3530bb53a069` |
| upiq380_development.parquet.manifest.json | `51f8adb58707993e3f12bda52c246cb28b30a07ccd68fb7ebd6c864a5f34538f` |

All 380 fresh pairs (159,600 requested values) and both Parquets/key tables
are byte-identical to the reviewed version. New producer manifests and
receipts carry the new source/binary pins; prior artifacts remain immutable.
The syscall audit matches exactly 410 approved HDR EXRs and zero extraction
label opens. Seven Python and seven Rust tests pass; all 20 actual-binary
refusals have zero dataset payload opens. The prior frozen binary reproduces
three reviewed failure scenarios using synthetic paths only. See
`REEXTRACTION_PARITY_PASS.json`, `VERIFY_PASS.json`, `DATA_READS.json`,
`PRODUCER_PINS.json` and `validation/binary-refusals/` in the canonical version.
These pins qualify this ingestion audit; they do not fill future training,
control, image or assessment manifest entries or authorize launch.
