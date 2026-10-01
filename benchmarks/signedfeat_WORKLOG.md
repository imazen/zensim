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
