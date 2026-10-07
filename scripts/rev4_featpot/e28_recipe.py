"""E28 preparation, admitted dataset legs and fixed grid. Never launches jobs.

POTENTIAL — ceiling, not a model score. Binding registration and pre-fit pins
are benchmarks/e28_*_2026-10-07.*. Training/prediction remain with v2 owners.
"""
import argparse
import collections
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from v2_common import REPO, SOURCE_ORDER, sha, human_dev, table_path

PIN = REPO / "benchmarks/e28_teacher_pin_2026-10-07.json"
GROUPING = REPO / "benchmarks/e28_nm_grouping_2026-10-07.json"


def read_pin():
    pin = json.loads(PIN.read_text())
    if pin["schema"] != "e28-teacher-pin-v1" or pin["registration_commit"] != "82c9af81":
        raise ValueError("E28 leg pin identity mismatch")
    return pin


def spec(arm):
    if arm not in ("s2o", "s2m"):
        raise ValueError(arm)
    return f"sel:59f0bbc2f290@h32:H128:cv16:cf98:{arm}"


def checked(rec):
    path = table_path(rec)
    if sha(path) != rec["sha256"] or sha(Path(f"{path}.manifest.json")) != rec["manifest_sha256"]:
        raise ValueError(f"E28 table changed: {path}")
    return path


def training_groups(arm, heldout, legs, original, human_weight):
    """Keep R1 teachers/coverage; replace mixed human leg with isolated datasets."""
    pin = read_pin()
    for teacher in pin["arms"][arm]["teachers"]:
        if legs[teacher] != pin["teachers"][teacher]:
            raise ValueError(f"E28 teacher differs from pre-fit pin: {teacher}")
    groups = [g for g in original if g[0] not in ("human", "human_development")]
    sources = [s for s in SOURCE_ORDER if s != heldout and s in pin["arms"][arm]["human_members"]]
    counts = {}
    refs = {}
    for source in sources:
        path = checked(legs[f"e28_{arm}_{source}"]["fit"])
        refs[source] = pq.read_table(path, columns=["ref_basename"])["ref_basename"].to_pylist()
        counts[source] = collections.Counter(refs[source])
    eligible = {s: sum(n >= 2 for n in c.values()) for s, c in counts.items()}
    nrefs = sum(eligible.values())
    acceptance = sum(1-1/n for c in counts.values() for n in c.values() if n >= 2) / nrefs
    human_effective = human_weight / acceptance
    records = {}
    for source in sources:
        rec = legs[f"e28_{arm}_{source}"]
        fit, dev = checked(rec["fit"]), checked(rec["dev"])
        tw = human_effective * eligible[source] / nrefs
        groups += [(source, fit, tw, 0, "withinref,rank"),
                   (f"{source}_development", dev, 0, eligible[source]/nrefs, "withinref,rank")]
        records[source] = dict(fit=rec["fit"], dev=rec["dev"], train_weight=tw)
    return groups, dict(arm=arm, teacher_pin_sha256=sha(PIN), legs=records,
                       pooled_rank_share=pin["pooled_rank_share"], pooled_pearson_weight=pin["pooled_pearson_weight"],
                       konfig_deviation=pin["konfig_deviation"], coverage=pin["arms"][arm]["coverage"])


def prepare(source: Path, out: Path):
    """Derive legs from admitted Rev5 full tables, using pinned member sets and R1 reference dev split."""
    from v2_teacher import write_curated
    pin = read_pin()
    receipt_path = source / "wide/main/real/receipt.json"
    if sha(receipt_path) != pin["source_receipt_sha256"]:
        raise ValueError("E28 preparation source changed")
    if out.exists():
        raise ValueError("E28 root must be fresh")
    shutil.copytree(source / "wide", out / "wide")
    vdir = out / "wide/main/real"
    receipt = json.loads(receipt_path.read_text())
    if receipt["formula_revision"] != 5:
        raise ValueError("E28 requires Rev5")
    receipt["legs"].pop("hdr", None)
    # No confirmation tables were copied by this LODO-only source.
    if (out / "wide/confirm").exists():
        raise ValueError("E28 preparation source contains confirmation tables")
    for arm, config in pin["arms"].items():
        for dataset, members in config["human_members"].items():
            full = receipt["legs"][dataset]["full"]
            table = out / full["rel"]
            if sha(table) != full["sha256"]:
                raise ValueError(f"source table changed: {dataset}")
            keys_path = vdir / f"{dataset}.keys.parquet"
            if sha(keys_path) != receipt["legs"][dataset]["keys_sha256"]:
                raise ValueError("E28 source keys changed")
            keys = pq.read_table(keys_path)
            frame = keys.to_pandas()
            row = pq.read_table(table, columns=["ref_basename", "human_score"]).to_pandas()
            if row.ref_basename.tolist() != frame.ref_basename.tolist():
                raise ValueError("E28 key/table row identity differs")
            selected = frame.member_set.isin(members).to_numpy()
            dev = np.array([human_dev(str(r)) for r in frame.ref_basename])
            rec = {}
            for part, mask in (("fit", selected & ~dev), ("dev", selected & dev)):
                if not mask.any():
                    raise ValueError(f"empty E28 leg {arm}/{dataset}/{part}")
                dest = vdir / f"e28_{arm}_{dataset}_{part}.parquet"
                write_curated(table, dest, mask, row.human_score.to_numpy())
                kp = dest.with_suffix(".keys.parquet")
                pq.write_table(keys.filter(pa.array(mask)), kp, compression="zstd")
                declaration = json.loads(Path(f"{dest}.manifest.json").read_text())
                declaration.update(study="E28", label="POTENTIAL — ceiling, not a model score",
                                   member_sets=members, teacher_pin_sha256=sha(PIN), keys_sha256=sha(kp),
                                   source_table_sha256=full["sha256"], split=part,
                                   split_rule="sha256(ref_basename) mod 5 == 0 is dev", build_commit=pin["registration_commit"])
                Path(f"{dest}.manifest.json").write_text(json.dumps(declaration, indent=1)+"\n")
                rec[part] = dict(rel=str(dest.relative_to(out)), sha256=sha(dest),
                                 manifest_sha256=sha(Path(f"{dest}.manifest.json")), keys_sha256=sha(kp),
                                 rows=int(mask.sum()), references=int(frame.loc[mask,"ref_basename"].nunique()))
            receipt["legs"][f"e28_{arm}_{dataset}"] = rec
    receipt["e28_teacher_pin_sha256"] = sha(PIN)
    (vdir / "receipt.json").write_text(json.dumps(receipt, indent=1)+"\n")
    print(json.dumps(dict(root=str(out), receipt_sha256=sha(vdir/"receipt.json"),
                         legs={k:v for k,v in receipt["legs"].items() if k.startswith("e28_")})),flush=True)


def grid(root, program_sha, data_sha, out):
    from e21_cheap_recipe import columns
    cells = [dict(name=f"{spec(a)}__N/without_{s}_s{i}", argv=["v2_lodo_mlp.py", "--spec", spec(a), "--head", "N",
             "--heldout", s, "--seed-index", str(i), "--root", str(root), "--columns", ",".join(map(str,columns("by_v2fy")))])
             for a in ("s2o", "s2m") for s in SOURCE_ORDER for i in range(10)]
    out.write_text(json.dumps(dict(program_sha=program_sha,data_sha=data_sha,cells=cells),indent=1)+"\n")
    print(json.dumps(dict(cells=len(cells),specs={a:spec(a) for a in ("s2o","s2m")})))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest="cmd",required=True)
    p=sub.add_parser("prepare");p.add_argument("--source",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p=sub.add_parser("grid");p.add_argument("--root",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--program-sha",required=True);p.add_argument("--data-sha",required=True)
    args=ap.parse_args()
    if args.cmd=="prepare":prepare(args.source,args.out)
    else:grid(args.root,args.program_sha,args.data_sha,args.out)


if __name__=="__main__":main()
