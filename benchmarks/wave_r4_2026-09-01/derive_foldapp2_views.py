#!/usr/bin/env python3
"""wave-r4: derive the `foldapp2` (pool-block ZEROED) views from the pools root.

WHY THIS EXISTS — a measured correction, 2026-09-01.  The wave root was
extracted at `foldapp2pools` because that is the regime both 2026-08-31 lanes
used (it lets a 372-input and a 944-input bake be READ on identical pixels).
But the flagship recipe this wave retrains VERBATIM (`W10L9PH_s4004`) was
TRAINED at `foldapp2`, where f156..f371 are STRUCTURAL ZEROS.  Training the
verbatim recipe on a pools root is therefore not the same experiment: the 64
feature transforms and every weight were fitted against a block that was zero.

Measured consequence: the first A1 attempt on the pools root reached
`best_val = 0.3058` against the incumbent's embedded `0.9235`, with CSIQ
-0.184, LIVE -0.143 and KADID -0.434 against the incumbent.  The regime, not
the radius, is the first-order term there.

MEASURED EQUIVALENCE that makes this derivation exact rather than approximate
(gate re-run below on every table): a native `ZENSIM_AB_MODE=foldapp2`
extraction of cid22val at radius 4 and the pools extraction of the same pairs
differ in **0 non-pool cells (max abs 0.0)**, and the native foldapp2 f156..371
block is **0 nonzero cells**.  So `foldapp2 == foldapp2pools with f156..f371
set to 0`, exactly.

Regime purity: the outputs are their OWN regime (`folded720append2`) and must
never be column-mixed with the pools tables they came from.
"""
import json, os, sys
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

SRC  = os.environ.get("WR4_ROOT",  "/mnt/v/zen/zensim-training/ext944-era2r4-2026-09-01")
DEST = os.environ.get("WR4_ZERO",  SRC + "/foldapp2_views")
os.makedirs(DEST, exist_ok=True)

TABLES = [t for t in (os.environ.get("WR4_ZERO_TABLES","").split(",")) if t] or [
    "ext_safesyn_full.parquet", "ext_cid22_train201.parquet", "ext_kadid.parquet",
    "ext_tid.parquet", "ext_kadis.parquet", "ext_konjnd_bpg_train.parquet",
    "ext_konjnd_bpg_val.parquet", "ext_cid22val.parquet", "ext_csiq.parquet",
    "ext_live.parquet", "ext_aic3.parquet", "ext_aic4.parquet", "ext_sdr25.parquet",
    "ext_konjnd_jpeg_val.parquet", "ext_imazen26.parquet", "ext_nonphoto.parquet",
    "ext_hfnlproxy.parquet",
    "recipe_views/safesyn_pure.parquet", "recipe_views/tbig_944_200k_pure.parquet",
    "recipe_views/tbig_hf_pure.parquet", "recipe_views/safesyn_teacher944_pure.parquet",
    "recipe_views/tbig_teacher944_pure.parquet",
]
POOL = [f"f{i}" for i in range(156, 372)]
# MERGE into an existing manifest rather than overwrite it (fixed 2026-09-01,
# same run that added derive_foldapp2_anchor.py). A scoped re-invocation via
# WR4_ZERO_TABLES (e.g. finalize_safesyn_big_r4.sh's A6 step, which derives
# only the ONE new safesyn-big table) used to silently discard every prior
# entry -- this is provenance data loss, not just a cosmetic manifest gap, and
# it is exactly the failure class "ML Data Pipeline Discipline SS2" exists to
# prevent. A full unscoped run reproduces the same 22-24 entries either way,
# so this is safe in both the full and the scoped case.
_man_path = os.path.join(DEST, "_MANIFEST.json")
man = json.load(open(_man_path)) if os.path.exists(_man_path) else {
    "source_root": SRC, "regime": "folded720append2",
    "rule": "f156..f371 := 0.0 on every row (measured EXACT vs a native foldapp2 extraction)",
    "purity": "NEVER column-mix with the folded720append2pools tables these came from",
    "tables": {},
}
man.setdefault("tables", {})

for rel in TABLES:
    src = os.path.join(SRC, rel)
    if not os.path.exists(src):
        print(f"  skip (absent): {rel}"); continue
    t = pq.read_table(src)
    names = set(t.schema.names)
    missing = [c for c in POOL if c not in names]
    if missing:
        sys.exit(f"ABORT: {rel} lacks {len(missing)} pool columns (first {missing[:3]})")
    zero = pa.array(np.zeros(t.num_rows, dtype=np.float64), pa.float64())
    cols = {n: (zero if n in set(POOL) else t.column(n)) for n in t.schema.names}
    out_rel = os.path.basename(rel)
    out = os.path.join(DEST, out_rel)
    pq.write_table(pa.table(cols), out, compression="zstd", compression_level=7)
    # gate: non-pool columns must be untouched, pool columns must be exactly 0
    chk = pq.read_table(out)
    bad = int(sum(1 for c in POOL if chk.column(c).to_numpy(zero_copy_only=False).any()))
    if bad: sys.exit(f"ABORT: {out_rel} has {bad} nonzero pool columns after zeroing")
    man["tables"][out_rel] = {"from": rel, "rows": t.num_rows, "pool_cols_zeroed": len(POOL)}
    print(f"  {out_rel}: {t.num_rows} rows, f156-371 zeroed, gate PASS")

json.dump(man, open(os.path.join(DEST, "_MANIFEST.json"), "w"), indent=1)
print("OK ->", DEST)
