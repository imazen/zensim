#!/usr/bin/env python3
"""wave-r4: derive the flagship recipe's training views on the radius-4 root.

The W10L9PH_s4004 recipe consumes 11 --group legs, but only a few are distinct
EXTRACTIONS; the rest are key-filters or target-swaps of those.  This script
reproduces each derived view against the wave-r4 features so the retrain is a
pure feature swap: identical rows, identical targets, only f0..f943 change.

  safesyn_pure         <- ext_safesyn_full        (identity; era-1 manifest: 0 dropped)
  tbig_944_200k_pure   <- ext_tbig200k, filtered to the era-1 view's keys
  tbig_hf_pure         <- ext_tbig200k, filtered to the era-1 view's keys
  *_teacher944_pure    <- those rows + the era-1 teacher's target column

MEASURED FACTS THIS RELIES ON, each re-asserted at run time (never assumed):
  * `v2_ab_extract` emits rows in INPUT TSV ORDER.  Verified 2026-09-01 on a
    2,000-row tbig probe: ref stem 2000/2000, human_score 2000/2000.
  * `pairs_tbig_png.tsv` is row-aligned with `tbig_944_200k.parquet`.  Verified
    on all 208,169 rows across THREE independent columns (ref_basename,
    human_score, encoded-filename stem): 208169/208169 each.
That alignment is what lets `encoded_filename` be attached positionally; the
extractor does not emit it, and `(ref_basename, human_score)` is NOT a key
(measured: 111 duplicate pairs in a 2,000-row probe).

Any gate failure ABORTS.  Nothing is padded, truncated or best-effort joined.
"""
import csv, json, os, sys
import pyarrow as pa, pyarrow.parquet as pq

R4    = os.environ.get("WR4_ROOT",  "/mnt/v/zen/zensim-training/ext944-era2r4-2026-09-01")
ERA1  = "/mnt/v/zen/zensim-training/sdr-pure-2026-08-28"
TBIG1 = "/mnt/v/zen/zensim-training/tbig_944_200k.parquet"
TBTSV = "/mnt/v/zen/zensim-training/wlin7-pools944-2026-08-30/pairs/pairs_tbig_png.tsv"
DEST  = os.environ.get("WR4_VIEWS", os.path.join(R4, "recipe_views"))
os.makedirs(DEST, exist_ok=True)

def die(m): sys.exit("ABORT: " + m)
def load(p, columns=None):
    if not os.path.exists(p): die(f"missing {p}")
    return pq.read_table(p, columns=columns)
def stem(x): return x.rsplit(".", 1)[0]

manifest = {"root": R4, "era1_mirror": ERA1, "gates": {}, "views": {}}

def write(t, name, expect_rows, meta):
    if t.num_rows != expect_rows:
        die(f"{name} has {t.num_rows} rows, expected {expect_rows}")
    pq.write_table(t, os.path.join(DEST, name), compression="zstd", compression_level=7)
    print(f"  wrote {name}: {t.num_rows} x {t.num_columns}")
    manifest["views"][name] = dict(meta, rows=expect_rows)

# ---------------- 1. safesyn: identity, with a row-alignment gate -------------
print("safesyn_pure ...")
ss  = load(os.path.join(R4, "ext_safesyn_full.parquet"))
ss1 = load(os.path.join(ERA1, "safesyn_pure.parquet"), columns=["ref_basename", "human_score"])
if ss.num_rows != ss1.num_rows: die(f"safesyn rows {ss.num_rows} vs era-1 {ss1.num_rows}")
a, b = ss.column("ref_basename").to_pylist(), ss1.column("ref_basename").to_pylist()
bad = sum(1 for x, y in zip(a, b) if x != y)
manifest["gates"]["safesyn_row_alignment"] = f"{len(a)-bad}/{len(a)} ref_basename match"
if bad: die(f"safesyn ref_basename disagrees on {bad}/{len(a)} rows")
write(ss, "safesyn_pure.parquet", 111068,
      {"from": "ext_safesyn_full.parquet", "rule": "identity (era-1 manifest: 0 dropped)"})

# ---------------- 2. tbig: attach encoded_filename positionally ---------------
print("tbig views ...")
tb  = load(os.path.join(R4, "ext_tbig200k.parquet"))
tb1 = load(TBIG1, columns=["ref_basename", "human_score", "encoded_filename"])
if tb.num_rows != tb1.num_rows: die(f"tbig rows {tb.num_rows} vs era-1 {tb1.num_rows}")

# gate A: my extraction is row-aligned with the pairs TSV (extractor order)
tsv = list(csv.DictReader(open(TBTSV), delimiter="\t"))
if len(tsv) != tb.num_rows: die(f"tbig pairs TSV {len(tsv)} vs extraction {tb.num_rows}")
rb, hs = tb.column("ref_basename").to_pylist(), tb.column("human_score").to_pylist()
def rstem(p):
    x = os.path.basename(p); return x[:-4] if x.lower().endswith(".png") else x
okr = sum(1 for i, t in enumerate(tsv) if rstem(t["ref_path"]) == rb[i])
okh = sum(1 for i, t in enumerate(tsv) if abs(float(t["human_score"]) - hs[i]) < 1e-9)
manifest["gates"]["tbig_extraction_vs_pairs_tsv"] = f"ref {okr}/{len(tsv)}, human {okh}/{len(tsv)}"
if okr != len(tsv) or okh != len(tsv): die("tbig extraction is not in pairs-TSV order")

# gate B: the pairs TSV is row-aligned with the era-1 keyed table
rb1, hs1, ef1 = (tb1.column(c).to_pylist() for c in ("ref_basename", "human_score", "encoded_filename"))
g1 = sum(1 for i, t in enumerate(tsv) if os.path.basename(t["ref_path"]) == rb1[i])
g2 = sum(1 for i, t in enumerate(tsv) if abs(float(t["human_score"]) - hs1[i]) < 1e-9)
g3 = sum(1 for i, t in enumerate(tsv) if stem(os.path.basename(t["dist_path"])) == stem(ef1[i]))
manifest["gates"]["pairs_tsv_vs_era1_tbig"] = f"ref {g1}/{len(tsv)}, human {g2}/{len(tsv)}, encoded-stem {g3}/{len(tsv)}"
if not (g1 == g2 == g3 == len(tsv)): die("pairs TSV is not row-aligned with tbig_944_200k.parquet")

tb = tb.append_column("encoded_filename", pa.array(ef1, pa.string()))
key_idx = {k: i for i, k in enumerate(ef1)}
for era1_name, rows in [("tbig_944_200k_pure.parquet", 192714), ("tbig_hf_pure.parquet", 11941)]:
    want = load(os.path.join(ERA1, era1_name), columns=["encoded_filename"]).column(0).to_pylist()
    idx = [key_idx[k] for k in want if k in key_idx]
    if len(idx) != len(want): die(f"{era1_name} key join covered {len(idx)}/{len(want)}")
    write(tb.take(idx), era1_name, rows,
          {"from": "ext_tbig200k.parquet", "rule": f"encoded_filename in era-1 {era1_name}"})

# ---------------- 3. teacher legs: NOT DONE HERE ------------------------------
# The teacher target-swap has an OWNER: scripts/canonical_corpus/build_teacher944.py
# `--graft-from` mode, which carries `human_score` from an era-1 twin onto a
# feature table at a new regime under its own G-T row-identity gate. Duplicating
# it here would be exactly the second-implementation the repo's no-duplication
# rule forbids, so this script stops at the feature views and emits the two
# commands the caller must run:
print("teacher legs -> run the OWNER (build_teacher944.py --graft-from):")
for base, era1_teacher, out_name in [
    ("safesyn_pure.parquet",       "safesyn_teacher944_pure.parquet", "safesyn_teacher944_pure.parquet"),
    ("tbig_944_200k_pure.parquet", "tbig_teacher944_pure.parquet",    "tbig_teacher944_pure.parquet"),
]:
    cmd = ("  python3 scripts/canonical_corpus/build_teacher944.py"
           f" --graft-from {os.path.join(ERA1, era1_teacher)}"
           f" --graft-features {os.path.join(DEST, base)}"
           f" --out {os.path.join(DEST, out_name)}")
    print(cmd)
    manifest["views"][out_name] = {"from": base, "target_from": f"era-1 {era1_teacher}",
                                   "rule": "build_teacher944.py --graft-from (OWNER)",
                                   "produced_by": "caller", "rows": None}

with open(os.path.join(DEST, "_MANIFEST.json"), "w") as fh:
    json.dump(manifest, fh, indent=1)
print("OK ->", DEST)
for k, v in manifest["gates"].items(): print(f"  gate {k}: {v}")
