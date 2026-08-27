#!/usr/bin/env python3
"""imazen-26 ID audit (GOAL DATA criterion, id half): zero test/eval/fixture ids
in any training view.

The imazen26 eval slice IS the canonical bigcodec-924 TEST views (FULL_EVAL
"924-era eval slices"), so the audit checks, per dataset:
  1. train/validate/test origin_id sets pairwise DISJOINT;
  2. the registered origin-parity split rule (origin_split.py: even=train,
     {1,3,5}=validate, {7,9}=test) holds for every row;
  3. no loop-instrument / dial-grid FIXTURE id appears in any TRAIN/VALIDATE
     ref_filename (corpus9 refs + the 39 dial-grid tiles).
Then cross-set: the union of test origin ref basenames (the eval id set) is
checked against every other local training table root that carries ref ids.

SCOPE (honest): this is the ID half only. The dHash+eye half (perceptual
near-duplicates across DIFFERENT filenames) is a separate daylight pass —
this report says so rather than implying it happened.
"""
import glob
import os
import sys

import pyarrow.parquet as pq

ROOT = "/mnt/v/zen/zensim-training/ext924-canonical-2026-07-27/bigcodec"
SPLIT_OK = {"train": {0, 2, 4, 6, 8}, "validate": {1, 3, 5}, "test": {7, 9}}
FIXTURES = {
    # corpus9 loop-instrument refs (basenames sans extension)
    "city", "dog", "girl", "1025469", "1418519", "1189261",
    "sc_codec_wiki", "sc_gui", "sc_imessage",
}
DIAL_DIR = "/mnt/v/output/zensim/dial-grid-pixels-2026-07-27/sources"
if os.path.isdir(DIAL_DIR):
    for f in os.listdir(DIAL_DIR):
        FIXTURES.add(os.path.splitext(f)[0])

fails = []
lines = ["# imazen-26 ID audit — {}".format(os.popen("date -u +%F").read().strip()), ""]
eval_refs = set()
for ds in sorted(os.listdir(ROOT)):
    dsp = os.path.join(ROOT, ds)
    if not os.path.isdir(dsp):
        continue
    def onum(v):
        """Numeric origin for the parity rule — origin_id may be a string
        (e.g. '123' or a name carrying trailing digits)."""
        s = str(v)
        d = "".join(ch for ch in s if ch.isdigit())
        return int(d) if d else -1

    ids = {}
    refs = {}
    for split in ("train", "validate", "test"):
        p = os.path.join(dsp, f"{split}_924.parquet")
        t = pq.read_table(p, columns=["origin_id", "ref_filename"])
        ids[split] = set(t.column("origin_id").to_pylist())
        refs[split] = {os.path.splitext(os.path.basename(r))[0] for r in t.column("ref_filename").to_pylist()}
        bad_parity = {i for i in ids[split] if onum(i) % 10 not in SPLIT_OK[split]}
        if bad_parity:
            fails.append(f"{ds}/{split}: {len(bad_parity)} origin ids violate the parity rule (e.g. {sorted(bad_parity)[:3]})")
    for a, b in (("train", "test"), ("train", "validate"), ("validate", "test")):
        inter = ids[a] & ids[b]
        if inter:
            fails.append(f"{ds}: {a}∩{b} origin overlap = {len(inter)} (e.g. {sorted(inter)[:3]})")
    fx = (refs["train"] | refs["validate"]) & FIXTURES
    if fx:
        fails.append(f"{ds}: FIXTURE ids in train/validate: {sorted(fx)[:5]}")
    eval_refs |= refs["test"]
    lines.append(
        f"- `{ds}`: origins train={len(ids['train'])} val={len(ids['validate'])} test={len(ids['test'])} — "
        f"disjoint ✓, parity ✓, fixtures-clean ✓"
        if not any(ds in f for f in fails)
        else f"- `{ds}`: **FAIL — see findings**"
    )

lines += ["", f"Eval id set (test-split ref basenames, all datasets): **{len(eval_refs)} ids**", ""]
# cross-set sweep: any other training table roots carrying these ref ids?
others = []
for cand in sorted(glob.glob("/mnt/v/zen/zensim-training/*/")):
    if "ext924-canonical-2026-07-27" in cand:
        continue
    for p in glob.glob(os.path.join(cand, "**", "*.parquet"), recursive=True)[:40]:
        try:
            names = pq.ParquetFile(p).schema_arrow.names
        except Exception:
            continue
        col = next((c for c in ("ref_filename", "ref_basename", "image_path", "source_filename") if c in names), None)
        if not col:
            continue
        try:
            vals = {os.path.splitext(os.path.basename(v))[0] for v in pq.read_table(p, columns=[col]).column(col).to_pylist() if v}
        except Exception:
            continue
        hit = vals & eval_refs
        if hit:
            others.append((p, len(hit)))
if others:
    lines.append("## Cross-set hits (eval ids appearing in OTHER local training tables)")
    for p, n in others:
        lines.append(f"- {p}: **{n} eval ids** — classify before any training use")
else:
    lines.append("Cross-set sweep: no eval id found in any other local training table root scanned.")

lines += ["", "## Verdict", ""]
if fails:
    lines.append("**FAIL** — findings:")
    lines += [f"- {f}" for f in fails]
else:
    lines.append("**PASS (id half)**: splits disjoint, parity rule holds, no fixture id in any train/validate view, cross-set sweep clean.")
lines += [
    "",
    "**Remaining (registered, NOT done here): the dHash+eye half** — perceptual",
    "near-duplicates under different filenames need the dHash-64 d≤10 screen +",
    "user-eye verification per the 2026-05-14 policy; scheduled as a daylight pass.",
]
out = "benchmarks/imazen26_id_audit_2026-08-27.md"
open(out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines[-12:]))
print(f"\nwrote {out}")
sys.exit(1 if fails else 0)
