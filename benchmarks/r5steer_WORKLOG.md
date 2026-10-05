MISSING: none for the requested panels, comparisons, data TSV, provenance and verified tower archive.

# R5STEER worklog — 2026-10-04

Authorized work order: `~/tmp/zensim-paper/rev4/R5STEER_brief.md`. Workspace
`~/work/zen/zensim--r5steer`, base `c83d22cd` (`main@origin` at workspace creation),
bookmark `quarantine/codex/r5steer`. Local commits only. Development diagnostic;
no model selection or product qualification.

## Fixed setup and provenance

The Rev4 reference is the correction banner in `jpegsteer_2026-10-04.md`,
`jpegsteer_cf98_2026-10-04.tsv`, and the full-precision reports in
`/var/tmp/neighsteer/fppanels/` and `/var/tmp/neighsteer/fpbroad/`. COSTSET
strict-route rows are excluded from this comparison.

Rev5 sources are E24 `v2c5/cells/{sel:59f0bbc2f290,set:v2+basic}@h32:H128:cv16:cf98__N/without_kadid_s{0,1,2}/refit/last.bin`.
Each selected-bake hash matches its fleet receipt. Canonical
`scripts/rev4_featpot/v2_common.py::dense_bake` executes `bake_dial_refit densify`
and requires its BIT-IDENTICAL gate. `bake_stamp_revision` then stamps 5 without
requantizing. Repeating the same canonical last-bake/densify/Rev4-stamp recipe
matches all six existing Rev4 reference bakes byte for byte.

Seed indices 0/1/2 mean initializer seeds 1101/1103/1107 and sample streams
101/100000101/200000101. Filenames preserve the historical panel labels
5101/5103/5107, which are labels rather than the initializer seeds.

The scoring binary is a pinned copy of the brief's existing build at 70a5066e:
`28e22ee289fa859da84a29a12639e01115de354c41095f4f16d37b84c7d7c8ec`.
Its fingerprint includes `custom-profiles,feature-regime-v2,candidate-profiles`.
The diff from 70a5066e to the workspace base contains no Rust source changes.
`RAYON_NUM_THREADS=1`, prepared steering on, formula revision 5 for candidates
and 4 for controls. Exact panels additionally set `ZENSIM_NEIGHBOUR_EXACT=1`;
plain panels explicitly clear it. Repair alpha/source, finite-moment,
steering-bin and per-feature diagnostic overrides are cleared.

KADID JPEG: I01/I21/I41/I61, distortion 10, levels 03/05, block 8, three
individual seeds per arm. Owner: the same level-03 pairs, block 32. Broad:
the existing TRAIN-origin 24-pair pixel register, blocks 8/16/32/64, uniform
three-seed ensembles per arm. The existing `run_broad.py` already accepts
`STEERCHECK_REV=5`; no owner change was necessary. Gates remain M2 >= 0.99 and
M3f >= 0.70; unavailable or unsupported refinement is never a pass.

Four single-thread scoring workers run through
`TMPDIR=/var/tmp/r5steer/tmp ~/work/zen/scripts/run-heavy --mem 16G --jobs 8 --`.
The wrapper reports an enforced 16G systemd memory cap. Other steering jobs
and an unrelated zensim test were observed; this record makes no timing claim.
Scratch is `/var/tmp/r5steer`; no sealed-directory contents are used, and no
human labels, dataset tables, fitting or calibration are needed for this run.

## Verification

CI-exact `just clippy` passed (exit 0; only the pre-existing dependency future
incompatibility notice). `just lint-scripts` passed, 811 scripts. The three
lane orchestration scripts pass Python compilation. Model preparation and
pixel-hash/block-count checks pass on all 624 comparison rows (312 matched revision pairs).

The by_v2fy owner regression (I61_10_03, seed index 2, block 32) repeats
byte-identically: M2 0.8876312291457559, M3f 0.8470519219813906; Rev4 reference
M2 0.999877926376041, M3f 0.9718264248704663. The repeat report is retained.

## Measured results

The plain 8×8 KADID pass counts are unchanged at Rev5: by_v2fy 12/12 at
level 03 and 10/12 at level 05; v2 + basic 12/12 and 9/12. Neighbour-exact
passes 12/12 in all four arm/level groups at both revisions.

The by_v2fy owner loses one pass (12/12 → 11/12), the reproduced I61 seed-2
case above. V2 + basic stays 12/12. Broad by_v2fy falls 92/96 → 85/96
(nine pass-to-fail transitions, two fail-to-pass); v2 + basic falls
91/96 → 90/96 (five pass-to-fail, four fail-to-pass). All Rev5 broad failures
are M2 misses; every broad M3f remains above 0.70. These are diagnostic
regressions; E24's registered quality rule is unchanged.

Plain KADID and owner rows pair individual seed indices; broad rows pair the
same image/block with each revision's uniform three-seed ensemble. Paired
ΔM3f is the median of casewise differences, not the difference of medians.
Full-precision values and transitions are in the TSVs.

| Arm | Panel | Rev4 pass | Rev4 M3f median (min) | Rev4 M2 min | Rev5 pass | Rev5 M3f median (min) | Rev5 M2 min | Paired ΔM3f median |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| by_v2fy | KADID JPEG L03 b8 | 12/12 | 0.912610 (0.905888) | 0.999490 | 12/12 | 0.918087 (0.881939) | 0.997876 | +0.000828 |
| by_v2fy | KADID JPEG L05 b8 | 10/12 | 0.835368 (0.549251) | 0.999860 | 10/12 | 0.817367 (0.646098) | 0.999745 | -0.002845 |
| by_v2fy | Owner L03 b32 | 12/12 | 0.970330 (0.958198) | 0.998851 | 11/12 | 0.970652 (0.847052) | 0.887631 | -0.002344 |
| by_v2fy | KADID exact L03 b8 | 12/12 | 0.979155 (0.960621) | 0.999490 | 12/12 | 0.982519 (0.952432) | 0.997876 | +0.002669 |
| by_v2fy | KADID exact L05 b8 | 12/12 | 0.968227 (0.938923) | 0.999860 | 12/12 | 0.961028 (0.930507) | 0.999745 | -0.002659 |
| by_v2fy | Broad all | 92/96 | 0.966283 (0.843782) | 0.971930 | 85/96 | 0.961268 (0.814273) | 0.955059 | -0.002941 |
| by_v2fy | Broad b8 | 24/24 | 0.923406 (0.843782) | 0.992690 | 23/24 | 0.925596 (0.814273) | 0.989559 | -0.007617 |
| by_v2fy | Broad b16 | 24/24 | 0.965135 (0.875115) | 0.992242 | 23/24 | 0.964273 (0.829759) | 0.972474 | -0.001416 |
| by_v2fy | Broad b32 | 24/24 | 0.984862 (0.946658) | 0.996030 | 21/24 | 0.982399 (0.872194) | 0.955059 | -0.002841 |
| by_v2fy | Broad b64 | 20/24 | 0.993562 (0.907182) | 0.971930 | 18/24 | 0.992092 (0.860140) | 0.958824 | -0.001471 |
| v2 + basic | KADID JPEG L03 b8 | 12/12 | 0.929867 (0.893199) | 0.999234 | 12/12 | 0.917852 (0.883514) | 0.999221 | -0.008328 |
| v2 + basic | KADID JPEG L05 b8 | 9/12 | 0.807284 (0.640174) | 0.999836 | 9/12 | 0.806205 (0.656000) | 0.999852 | -0.012230 |
| v2 + basic | Owner L03 b32 | 12/12 | 0.975888 (0.939923) | 0.998660 | 12/12 | 0.969956 (0.938373) | 0.996882 | -0.004330 |
| v2 + basic | KADID exact L03 b8 | 12/12 | 0.976936 (0.954162) | 0.999234 | 12/12 | 0.976411 (0.941719) | 0.999221 | -0.000361 |
| v2 + basic | KADID exact L05 b8 | 12/12 | 0.959426 (0.934814) | 0.999836 | 12/12 | 0.965353 (0.941994) | 0.999852 | +0.005600 |
| v2 + basic | Broad all | 91/96 | 0.968626 (0.822500) | 0.951049 | 90/96 | 0.967904 (0.732050) | 0.870588 | -0.002377 |
| v2 + basic | Broad b8 | 23/24 | 0.922730 (0.822500) | 0.987533 | 24/24 | 0.930728 (0.804404) | 0.994314 | -0.002641 |
| v2 + basic | Broad b16 | 23/24 | 0.960215 (0.892663) | 0.988982 | 24/24 | 0.960392 (0.892097) | 0.992584 | -0.002779 |
| v2 + basic | Broad b32 | 24/24 | 0.982470 (0.949571) | 0.995899 | 22/24 | 0.985523 (0.907051) | 0.949863 | +0.000717 |
| v2 + basic | Broad b64 | 21/24 | 0.990621 (0.895105) | 0.951049 | 20/24 | 0.987125 (0.732050) | 0.870588 | -0.004412 |

## Individual-seed pass counts

| Arm | Panel | Seed index | Rev4 | Rev5 |
|---|---|---:|---:|---:|
| by_v2fy | kadidjpeg L03 | 0 | 4/4 | 4/4 |
| by_v2fy | kadidjpeg L03 | 1 | 4/4 | 4/4 |
| by_v2fy | kadidjpeg L03 | 2 | 4/4 | 4/4 |
| by_v2fy | kadidjpeg L05 | 0 | 3/4 | 3/4 |
| by_v2fy | kadidjpeg L05 | 1 | 3/4 | 4/4 |
| by_v2fy | kadidjpeg L05 | 2 | 4/4 | 3/4 |
| by_v2fy | owner L03 | 0 | 4/4 | 4/4 |
| by_v2fy | owner L03 | 1 | 4/4 | 4/4 |
| by_v2fy | owner L03 | 2 | 4/4 | 3/4 |
| by_v2fy | kadid_exact L03 | 0 | 4/4 | 4/4 |
| by_v2fy | kadid_exact L03 | 1 | 4/4 | 4/4 |
| by_v2fy | kadid_exact L03 | 2 | 4/4 | 4/4 |
| by_v2fy | kadid_exact L05 | 0 | 4/4 | 4/4 |
| by_v2fy | kadid_exact L05 | 1 | 4/4 | 4/4 |
| by_v2fy | kadid_exact L05 | 2 | 4/4 | 4/4 |
| v2 + basic | kadidjpeg L03 | 0 | 4/4 | 4/4 |
| v2 + basic | kadidjpeg L03 | 1 | 4/4 | 4/4 |
| v2 + basic | kadidjpeg L03 | 2 | 4/4 | 4/4 |
| v2 + basic | kadidjpeg L05 | 0 | 3/4 | 3/4 |
| v2 + basic | kadidjpeg L05 | 1 | 3/4 | 3/4 |
| v2 + basic | kadidjpeg L05 | 2 | 3/4 | 3/4 |
| v2 + basic | owner L03 | 0 | 4/4 | 4/4 |
| v2 + basic | owner L03 | 1 | 4/4 | 4/4 |
| v2 + basic | owner L03 | 2 | 4/4 | 4/4 |
| v2 + basic | kadid_exact L03 | 0 | 4/4 | 4/4 |
| v2 + basic | kadid_exact L03 | 1 | 4/4 | 4/4 |
| v2 + basic | kadid_exact L03 | 2 | 4/4 | 4/4 |
| v2 + basic | kadid_exact L05 | 0 | 4/4 | 4/4 |
| v2 + basic | kadid_exact L05 | 1 | 4/4 | 4/4 |
| v2 + basic | kadid_exact L05 | 2 | 4/4 | 4/4 |

## Artifacts and replay

`benchmarks/r5steer_2026-10-04.tsv` contains 624 full-precision measured rows;
`benchmarks/r5steer_2026-10-04/paired.tsv` contains 312 matched revision pairs.
`INPUTS.json` pins model/source/dense/stamped hashes, fleet receipts, paired
seed streams and pixel-file hashes. `REV4_MODEL_VERIFY.json` records all six
byte-identical Rev4 reconstructions. `summary.json` and `table.md` hold panel
aggregates. `pointer.json` and `archive_inventory.json` pin durable artifacts.

Every per-block JSON, including all files over 30 KB, is stored at
`/mnt/tower/output/zensim/r5steer-2026-10-04/` in separate revision/panel
directories. 1202 files, 248214740 bytes, were re-read and
SHA-256 verified, including 624 comparison reports, one repeated owner report,
12 revision-specific models and logs/provenance. Inventory SHA-256:
`1110a41c62250e5a0219af93f8e569fc7170f5985903340feaf93471e4557310`. Raw per-block JSONs are not committed.

Replay:

```bash
TMPDIR=/var/tmp/r5steer/tmp ~/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- python3 -u benchmarks/r5steer_2026-10-04/run.py
TMPDIR=/var/tmp/r5steer/tmp ~/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- python3 benchmarks/r5steer_2026-10-04/summarize.py
TMPDIR=/var/tmp/r5steer/tmp ~/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- python3 benchmarks/r5steer_2026-10-04/archive.py
```

The existing scoring owner is unchanged. Checks refuse mismatched models,
pixels, blocks or seed streams; the archive refuses differing existing bytes.
The scoring and summary runs completed with exit 0. Archive replay also
checks existing bytes rather than overwriting them. Python compilation and
CI-exact clippy pass; script lint reports all 811 scripts runnable.
No encoder RD, runtime/performance or held-out quality qualification is claimed.
The DONE report is written after local commit and final verification.
