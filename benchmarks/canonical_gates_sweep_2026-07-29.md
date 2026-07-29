# Canonical-source gate sweep (task #11 chunk 2, 2026-07-29)

`scripts/canonical_corpus/canonical_filter_gates.py` run over **33 canonical
sources** → per-source verdict JSONs at `/mnt/v/output/zensim/canonical-gates/`
(each with build_commit + input sha256 + per-gate provenance). Logs:
`~/tmp/gates_sweep_{small,big}_2026-07-29.log`.

## Result: 0 hard failures across the inventory

- **misjoin gate (§3.18 detectors, via the owner `audit_metric_columns.py`):
  PASS on all 33** — including the 2026-07-15-fixed kadid/tid train legs (the
  fix is verifiably still clean), safesyn 196k, both 700k kadis canonicals,
  tbig_924_200k (208k), kadis negrich 266k + negrich-924 167k, and all 11
  ext924 legs.
- **jxl_zone (§3.20 two-clause)**: PASS or structurally N/A everywhere. The
  HF near-lossless pair (`hf_nearlossless_{train,val}`) initially FAILED a
  distance-only assert — a **gate false positive**, not contamination: that
  corpus samples d<0.03 BY DESIGN and is the POST-fix rebuild (2026-07-06
  06:35, after `eeb52735` 06:09; `jxl_nearlossless_corpus_2026-07-06.pointer.md`).
  The gate now implements the full two-clause test (distance AND pre-fix
  provenance) with `--jxl-postfix` attesting documented post-fix sources.
  Lesson recorded in the gate's doc comment: the date clause is load-bearing.
- **poison inventory**: the 8 documented TARGET-SHAPE-poison columns recorded
  per source (present in the c0521 family; absent from the 924-era legs,
  which carry raw targets only). Columns are never dropped — trainer
  manifests must simply not select them as MSE targets.
- **winsor bounds (§3.19)**: [p0.1, p99.9] recorded for every feature column
  of every source (372/380/387/924-wide as applicable) as guard provenance.

Coverage: c0521 train+val (10 legs), canonical-2026-07-15 (hf pair +
kadis_negrich), kadis-924 trio (50k slice, negrich 167k, 700k), kadis-700k-gpu
canonical, tbig_924_200k, the 924 dial (2026-07-28 file) + corruption grids,
the quarantined 372 dial grid, all 11 ext924 legs. NOT yet gated: bigcodec
mm6 1.56M + the 21 bigcodec 924 split views + hdr_v3mix (paths need
confirmation against DATA_PROVENANCE) — the remaining chunk-2 tail, plus
manifest wiring + R2/Tower mirrors (chunk 3).

## Tail sweep + mirrors (same day)

- **bigcodec tail**: mm6 (1.56M) + ALL 21 bc924 split views PASS misjoin;
  views' jxl_zone PASS on the structural q-mapping basis. **56 sources total,
  0 hard failures** (`_MANIFEST.json` in the gates dir indexes all verdicts).
- **tbig_924_full (25.6 GB, 5.7M rows)**: whole-table read OOMs (errno 12) —
  recorded as `misjoin: COVERED-BY-VIEWS` (its rows are the union of the 21
  individually-passing views, keyed by encode_sha) with a schema-level
  poison/jxl verdict; a single-pass full-table audit needs a streaming mode
  in `audit_metric_columns.py` (deferred, recorded — never silently skipped).
- **Mirrors**: `/mnt/tower/output/zensim/canonical-gates-2026-07-29/` +
  `s3://zentrain/canonical-gates-2026-07-29/` (57 objects verified) +
  local `/mnt/v/output/zensim/canonical-gates/`.
- Still open for #11: hdr_v3mix raw-parquet provenance (its Gram derivative
  is sha-attested via the #68 chain), per-source-dir `_MANIFEST.json` gate
  pointers, winsor for the 25.6 GB table (needs the streaming mode).
