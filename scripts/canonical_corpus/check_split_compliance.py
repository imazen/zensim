#!/usr/bin/env python3
"""check_split_compliance.py — assert train-side content never appears on an eval surface.

Split-policy-v2 enforcement owner (registered 2026-08-29, balance campaign).
For every --group parquet (or every `--group` in a fulleval's embedded
zentrain.repro argv), extract CONTENT ids per family and intersect with every
registered eval surface of the same family. Any overlap is a hard error unless
the (group, surface) pair is a registered GUARD (train==val by design, e.g.
the kadid/tid full-set integrity rows) — those print WARN.

Content-id rules (family -> extraction):
  imazen26  leading numeric origin stem of ref_basename (origin_split family)
  kadis     source_id column
  kadid/tid I<number> ref prefix
  konjnd    first integer in ref_basename (src number)
  cid22     ref stem before first '_'
T0 corpora (aic3/aic4/sdr25/csiq/live) may NEVER train: matching a train path
against their names is an immediate error.

Usage:
  check_split_compliance.py --from-fulleval <bake.fulleval.json>
  check_split_compliance.py --group <path.parquet> [--group ...]
Exit 0 = compliant (WARNs allowed), 1 = violation.
"""
import argparse, json, re, sys, os
import pyarrow.parquet as pq

EXT944 = "/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01"
ROOT372 = "/mnt/v/zen/zensim-training/2026-05-15-full-features"

T0_NAMES = ("aic3", "aic4", "sdr25", "csiq", "live", "upiq")

# (family, surface-label, path). Surfaces are SELECT/TERMINAL eval content.
SURFACES = [
    ("imazen26", "ext_imazen26@944(select)", f"{EXT944}/ext_imazen26.parquet"),
    ("imazen26", "ext_nonphoto@944(select)", f"{EXT944}/ext_nonphoto.parquet"),
    ("imazen26", "ext_hfnlproxy@944(select)", f"{EXT944}/ext_hfnlproxy.parquet"),
    ("imazen26", "ext_hfnlproxy@372(terminal)", f"{ROOT372}/ext_hfnlproxy.parquet"),
    ("kadid", "kadid_select", f"{EXT944}/ext_kadid_select_2026-08-29.parquet"),
    ("kadid", "kadid_terminal", f"{EXT944}/ext_kadid_terminal_2026-08-29.parquet"),
    ("tid", "tid_select", f"{EXT944}/ext_tid_select_2026-08-29.parquet"),
    ("tid", "tid_terminal", f"{EXT944}/ext_tid_terminal_2026-08-29.parquet"),
    ("konjnd", "konjnd_jpeg(select+terminal)", f"{EXT944}/ext_konjnd_jpeg_val.parquet"),
    ("cid22", "cid22_49ref_gold", f"{ROOT372}/cid22_features_372col_2026-05-15.parquet"),
]

# (train-path-substring, surface-label) pairs that are registered train==val
# guards: report WARN, not error. Everything else overlapping is an ERROR.
GUARDS = [
    ("ext_kadid.parquet", "kadid_select"), ("ext_kadid.parquet", "kadid_terminal"),
    ("ext_tid.parquet", "tid_select"), ("ext_tid.parquet", "tid_terminal"),
    ("konjnd_dense", "konjnd_jpeg(select+terminal)"),
]

def family_of(path):
    b = os.path.basename(path).lower()
    if "kadis" in b: return "kadis"
    if "kadid" in b: return "kadid"
    if "tid" in b: return "tid"
    if "konjnd" in b or "konfig" not in b and "kon" == b[:3]: return "konjnd"
    if "cid22" in b: return "cid22"
    if any(k in b for k in ("tbig", "bigcodec", "hf_pure", "hfnl", "imazen", "nonphoto", "avif")): return "imazen26"
    return None  # safesyn/teacher/hdrmix etc: no registered surface family

def ids_of(path, family):
    pf = pq.ParquetFile(path)
    cols = pf.schema_arrow.names
    if family == "kadis":
        col = "source_id" if "source_id" in cols else "source_filename"
        vals = pq.read_table(path, columns=[col]).column(0).to_pylist()
        if col == "source_id":
            return {str(int(v)) for v in vals}
        return {str(m.group(1)) for v in vals if (m := re.search(r"(\d+)", str(v)))}
    col = "ref_basename" if "ref_basename" in cols else cols[0]
    vals = pq.read_table(path, columns=[col]).column(0).to_pylist()
    out = set()
    for v in vals:
        s = str(v)
        if family in ("kadid", "tid"):
            m = re.match(r"[iI](\d+)", s)
        elif family == "cid22":
            out.add(s.split("_")[0]); continue
        else:
            m = re.search(r"(\d+)", s)
        if m: out.add(m.group(1).lstrip("0") or "0")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", action="append", default=[])
    ap.add_argument("--from-fulleval")
    a = ap.parse_args()
    groups = list(a.group)
    if a.from_fulleval:
        d = json.load(open(a.from_fulleval))
        argv = (d.get("repro") or {}).get("argv") or []
        for tok in argv:
            if tok.count(":") >= 2 and ".parquet" in tok:
                path = tok.split(":")[1]
                w = tok.split(":")[2]
                try: trains = float(w) > 0
                except ValueError: trains = True
                if trains: groups.append(path)
    if not groups:
        print("no train groups found", file=sys.stderr); return 2
    surf_ids = {}
    for fam, label, path in SURFACES:
        if os.path.exists(path):
            surf_ids[label] = (fam, ids_of(path, fam))
    rc = 0
    for g in groups:
        if not os.path.exists(g):
            print(f"MISSING  {g} (train input not on disk — cannot audit)"); rc = 1; continue
        b = os.path.basename(g).lower()
        if any(t in b for t in T0_NAMES):
            print(f"ERROR    {b}: T0 corpus in a TRAIN group"); rc = 1; continue
        fam = family_of(g)
        if fam is None:
            print(f"ok       {b}: no registered eval-surface family (train-only estate)")
            continue
        gids = ids_of(g, fam)
        if fam == "kadis":
            # The kadis surface is a RULE, not a file: source_id%10 in {8,9}
            # is the reserved val+test estate (DATA_SPLITS.md 2b).
            bad = {i for i in gids if int(i) % 10 >= 8}
            if bad:
                print(f"ERROR    {b} vs kadis %10 in {{8,9}} (reserved val+test): "
                      f"{len(bad)} of {len(gids)} train sources violate (e.g. {sorted(bad, key=int)[:5]})")
                rc = 1
            else:
                print(f"ok       {b} vs kadis %10 rule: all {len(gids)} sources are train-side (<8)")
            continue
        matched = False
        for label, (sfam, sids) in surf_ids.items():
            if sfam != fam: continue
            matched = True
            ov = gids & sids
            if not ov:
                print(f"ok       {b} vs {label}: 0 overlap ({len(gids)} vs {len(sids)} ids)")
            elif any(gs in g for gs, sl in GUARDS if sl == label):
                print(f"WARN     {b} vs {label}: {len(ov)} shared ids — REGISTERED GUARD (train==val by design)")
            else:
                print(f"ERROR    {b} vs {label}: {len(ov)} shared content ids (e.g. {sorted(ov)[:5]})")
                rc = 1
        if not matched:
            print(f"NOSURF   {b}: family {fam!r} has no registered surface — "
                  f"add one to SURFACES before this family gains an eval role")
    print("=>", "VIOLATIONS FOUND" if rc else "COMPLIANT (guards WARN-only)")
    return rc

if __name__ == "__main__":
    sys.exit(main())
