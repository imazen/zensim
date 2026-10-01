# signedfeat worklog

Lane: Claude Sonnet, workspace `zensim--signedfeat`, base `main@origin` = `c393de29`. Times MT.

## Phase 1 (2026-10-01 02:38–02:55 MT): inventory + proposal, no feature code

Source-reading only; no pixel, label or `_sealed` read; nothing built or run. Findings and the decision list:
`/home/lilith/tmp/zensim-paper/rev4/SIGNEDFEAT_decisions.md`.

Commands behind the claims (all read-only):
- layout / signal tables: `grep -n "^pub(crate) static\|BLOCKS" zensim/src/feature_defs.rs`; read `feature_defs.rs:1491-2835`.
- arithmetic: `feature_v2.rs:1703-1740` (bounded_sim / excess / pair), `:4676-4687` (hf_gain/loss/mag_loss), `:7098-7107` (contrast pair),
  `:7939-8035` (csfw + global), `:5080-5147` (C8 chromaticity + gradient bank), `:5510-5518` (bandvis);
  `streaming.rs:609-615`; `hf_gain_form.rs:153-214,321-382`; `feature_v2/restore_cuts.rs:1-232`.
- extractor family subset: `zensim-bench/examples/extract_features_372col.rs:129-143,276-283,426-470,518-520`; `research.rs:1332-1445`.
- sidecar precedent: `scripts/restore_cuts/bank_sidecar.py` (FAMILIES map), `/home/lilith/tmp/zensim-paper/rev4/RESTORE_CUTS_DONE.md`.
- Cost numbers quoted are from `benchmarks/restore-cuts_cost_2026-09-24.md` via that report (contended, descriptive); every cost figure
  for the new families is a structural ESTIMATE, labelled so.

## Phase 2 (2026-10-01, from ~03:00 MT): chunk 0 gate, S1 `texgain`, S2 `satsign`

Coordinator decisions: `/home/lilith/tmp/zensim-paper/rev4/SIGNEDFEAT_decisions.md` (bottom). Bookmark `quarantine/claude/signedfeat`.

- **Chunk 0** (existing mapdev/z1max tier parity at Rev4): `zensim/tests/signedfeat_tier_parity.rs` via `just signedfeat-tier-parity`
  (inputs `scripts/signedfeat/prep_parity_pairs.py`, index with sha256 of every input). 24 pair-sizes x 10 token permutations x 288 slots =
  69,120 cells, 0 differing; Rev3 negative control diverges. Logs `/var/tmp/signedfeat/chunk0.log`, `chunk0_rev3.log`. No canonicalisation needed.
- **S2 measurements** (`cargo run --release -p zensim --example signedfeat_sat_probe -- /var/tmp/signedfeat/satprobe`, TRAIN reference pixels
  from `scripts/signedfeat/prep_sat_probe_images.py`, 22 images, 4,325,376 px; log `/var/tmp/signedfeat/satprobe.log`):
  gray ramp Xc in +-8.3e-7, Bc = 0.155954 on every level, m = 0.177271 (spread 2.0e-7 = 3.5e-7 of the median); pooled median m 0.5794,
  q01 0.1188, q99 7.5065. Gates passed; frozen: C_SAT = 1, GMSBANK_CS_C[2], C8 centring.
- **Implementation** 2dfe0880: tokens, registry (`TEXGAIN_SIGNALS`, `SATSIGN_SIGNALS`, blocks, widths 1837/1853), plan/layout flags, walk
  wiring, side-pass accumulation in `feature_v2/restore_cuts.rs`.
- **Gates** (all outputs under `/var/tmp/signedfeat/`):
  - before/after full vectors: `scripts/signedfeat/compare_dumps.py` over `signedfeat_dump_vector` dumps, baseline = tree at `c393de29`
    (overlay copy `/var/tmp/signedfeat/base_src`, target `target-base`), 24 real pair-sizes, Rev1/2/3/4: 43,800 cells each, 0 differing (`dumps.out`).
  - `tests/signedfeat_families.rs` (prefix independence on every tier, serial/MT8, tight/strided, layout independence, identity zero,
    semantic controls): pass.
  - tier parity of the new slots at Rev4 on the same 24 pair-sizes: 6,720 cells, 0 differing (`chunk1_tier.log`).
  - f64 mirror `scripts/signedfeat/numpy_mirror.py` over `signedfeat_plane_dump` (9 cases): texgain max rel 1.3e-5, median 2.4e-7;
    satsign max 2.7e-12, median 1.4e-16; wrong-definition control rejected (`mirror.log`).
  - cost: `scripts/signedfeat/cost_fit.py` over zenbench raw files `cost*_mt*.zenbench`; CONTENDED (box load ~21), Rev3, one core (ST) / 4 cores.
- **Neutral-axis centring** (coordinator 04:03 MDT): `satsign` uses `Bn = B - 0.55 - cbrt(K_B0)`; re-run: probe (gray ramp m <= 2.2e-5, pooled median m 0.5636),
  mirror (satsign max rel 1.06e-12), tier parity (3,840 cells, 0 differing), families test, f0-f1824 identity at Rev4 (0 of 43,800). Logs `satprobe2.log`, `mirror2.log`, `tier2.log`.
- **Extraction cost correction**: `signedfeat_plan_probe` shows a family request still plans 1,637 of 1,853 slots (nested layout chain); dense layout returns 28 values at the same cost.
