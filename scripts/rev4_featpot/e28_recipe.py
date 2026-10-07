"""E28 preparation, admitted dataset legs and fixed grid. Never launches jobs.

POTENTIAL — ceiling, not a model score. Binding registration and pre-fit pins
are benchmarks/e28_*_2026-10-07.*. Training/prediction remain with v2 owners.
"""
import argparse
import collections
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from v2_common import REPO, SOURCE_ORDER, sha, human_dev, table_path

PIN = REPO / "benchmarks/e28_teacher_pin_2026-10-07.json"
GROUPING = REPO / "benchmarks/e28_nm_grouping_2026-10-07.json"
ADMISSION = REPO / "benchmarks/e28_admission_inventory_2026-10-07.json"
KEY_COLUMNS = ["pair_key", "source_row_id", "ref_basename", "member_set"]


def admission_pin():
    policy = json.loads(ADMISSION.read_text())
    if policy["schema"] != "e28-approved-inputs-v1" or policy["teacher_pin_sha256"] != sha(PIN):
        raise ValueError("E28 approved input inventory identity mismatch")
    return policy


def bound_record(rec, policy):
    """Admit a record's names and metadata before its label-bearing checksum."""
    rel = rec.get("rel")
    files = policy["prepared_files"]
    if "path" in rec or rel not in files or rec.get("sha256") != files[rel]:
        raise ValueError("E28 record is outside the approved input inventory")
    manifest_rel = rel + ".manifest.json"
    if rec.get("manifest_sha256") != files.get(manifest_rel):
        raise ValueError("E28 manifest is outside the approved input inventory")
    path = table_path(rec)
    declaration = json.loads(Path(f"{path}.manifest.json").read_text())
    if sha(Path(f"{path}.manifest.json")) != files[manifest_rel]:
        raise ValueError("E28 manifest changed")
    if declaration.get("table_sha256") != files[rel] or declaration.get("formula_revision") != 5:
        raise ValueError("E28 table declaration binding mismatch")
    return path, declaration


def human_declaration(rec, arm, source, split, policy, pin):
    if arm not in pin["arms"] or source not in pin["arms"][arm]["human_members"]:
        raise ValueError("E28 unregistered human population")
    expected = f"wide/main/real/e28_{arm}_{source}_{split}.parquet"
    if rec.get("rel") != expected or split not in ("fit", "dev"):
        raise ValueError("E28 arm/source/split record mismatch")
    path, d = bound_record(rec, policy)
    source_rel = f"wide/main/real/{source}.parquet"
    source_keys = f"wide/main/real/{source}.keys.parquet"
    keys_rel = expected.replace(".parquet", ".keys.parquet")
    required = dict(study="E28", arm=arm, source=source, split=split,
                    role="TRAIN" if split == "fit" else "DEV",
                    registration_commit=pin["registration_commit"], teacher_pin_sha256=sha(PIN),
                    member_sets=pin["arms"][arm]["human_members"][source],
                    split_rule="sha256(ref_basename) mod 5 == 0 is dev",
                    source_table_sha256=policy["source_files"][source_rel],
                    source_manifest_sha256=policy["source_files"][source_rel+".manifest.json"],
                    source_keys_sha256=policy["source_files"][source_keys],
                    source_label_free_row_keys_sha256=policy["source_key_identities"][source],
                    keys_sha256=policy["prepared_files"][keys_rel])
    if any(d.get(k) != v for k,v in required.items()) or rec.get("keys_sha256") != required["keys_sha256"]:
        raise ValueError("E28 human population/split/role/source/key declaration mismatch")
    if not d.get("build_commit") or not d.get("row_keys_sha256") or not d.get("row_selection_sha256"):
        raise ValueError("E28 human provenance is incomplete")
    if rec.get("rows") != d.get("rows_kept"):
        raise ValueError("E28 human row count binding mismatch")
    return path, d


def human_keys(path, d, source, split, pin):
    """Only approved, label-free keys may be decoded during admission."""
    from v2_teacher import row_keys_sha
    kp = path.with_suffix(".keys.parquet")
    if sha(kp) != d["keys_sha256"]:
        raise ValueError("E28 label-free key file changed")
    keys = pq.read_table(kp)
    if keys.column_names != KEY_COLUMNS or row_keys_sha(keys) != d["row_keys_sha256"]:
        raise ValueError("E28 label-free row-key binding mismatch")
    if keys.num_rows != d["rows_kept"]:
        raise ValueError("E28 label-free key row count differs")
    if not set(keys["member_set"].to_pylist()) <= set(pin["arms"][d["arm"]]["human_members"][source]):
        raise ValueError("E28 label-free keys contain a forbidden population")
    if any(human_dev(str(ref)) != (split == "dev") for ref in keys["ref_basename"].to_pylist()):
        raise ValueError("E28 label-free reference split differs")
    return keys


def admit_humans(arm, heldout, legs):
    """Two-phase preflight: every manifest, then label-free keys; no table opens."""
    pin, policy = read_pin(), admission_pin()
    if arm not in pin["arms"] or heldout not in SOURCE_ORDER:
        raise ValueError("E28 unregistered arm/fold")
    declarations = {}
    for source in pin["arms"][arm]["human_members"]:
        if source == heldout:
            continue
        for split in ("fit", "dev"):
            rec = legs[f"e28_{arm}_{source}"][split]
            declarations[source,split] = human_declaration(rec,arm,source,split,policy,pin)
    keys = {k:human_keys(path,d,k[0],k[1],pin) for k,(path,d) in declarations.items()}
    for source in {s for s,_ in declarations}:
        if set(keys[source,"fit"]["ref_basename"].to_pylist()) & set(keys[source,"dev"]["ref_basename"].to_pylist()):
            raise ValueError("E28 fit/dev references overlap")
    return declarations, keys


def source_inventory(source, policy):
    """Refuse forbidden directories before opening even a source receipt."""
    for base, dirs, _ in os.walk(source, followlinks=False):
        for name in dirs:
            if Path(base,name).is_symlink() or name.lower().startswith(("confirm", "hdr", "_sealed")):
                raise ValueError("E28 preparation source contains forbidden confirmation/HDR directory")
    paths = {}
    for rel in policy["source_files"]:
        path = source / rel
        if not rel.startswith("wide/") or ".." in Path(rel).parts or any(p.is_symlink() for p in [path,*path.parents]) or not path.is_file():
            raise ValueError("E28 preparation inventory path missing or unsafe")
        paths[rel] = path
    # All metadata is admitted before hashing/copying any label-bearing file.
    for rel,path in paths.items():
        if rel.endswith(".json") and sha(path) != policy["source_files"][rel]:
            raise ValueError("E28 preparation approved metadata changed")
    return paths


def read_pin():
    if sha(PIN) != "ddbc9c22cd08f545f76c50dd60947062b285d937df862f61153ff1685d9bc599":
        raise ValueError("E28 pre-fit teacher pin changed")
    pin = json.loads(PIN.read_text())
    if pin["schema"] != "e28-teacher-pin-v1" or pin["registration_commit"] != "82c9af81":
        raise ValueError("E28 leg pin identity mismatch")
    return pin


def spec(arm):
    if arm not in ("s2o", "s2m"):
        raise ValueError(arm)
    return f"sel:59f0bbc2f290@h32:H128:cv16:cf98:{arm}"


def checked(rec, *, arm=None, source=None, split=None):
    policy = admission_pin()
    if arm is None:
        path, d = bound_record(rec, policy)
        if d.get("study") == "E28":
            raise ValueError("E28 human record requires explicit arm/source/split admission")
    else:
        path, d = human_declaration(rec,arm,source,split,policy,read_pin())
        human_keys(path,d,source,split,read_pin())
    if sha(path) != rec["sha256"]:
        raise ValueError(f"E28 table changed: {path}")
    return path


def training_groups(arm, heldout, legs, original, human_weight):
    """Keep R1 teachers/coverage; replace mixed human leg with isolated datasets."""
    pin = read_pin()
    for teacher in pin["arms"][arm]["teachers"]:
        if legs[teacher] != pin["teachers"][teacher]:
            raise ValueError(f"E28 teacher differs from pre-fit pin: {teacher}")
    declarations, keys = admit_humans(arm, heldout, legs)
    groups = [g for g in original if g[0] not in ("human", "human_development")]
    sources = [s for s in SOURCE_ORDER if s != heldout and s in pin["arms"][arm]["human_members"]]
    counts = {}
    refs = {}
    for source in sources:
        refs[source] = keys[source,"fit"]["ref_basename"].to_pylist()
        counts[source] = collections.Counter(refs[source])
    eligible = {s: sum(n >= 2 for n in c.values()) for s, c in counts.items()}
    nrefs = sum(eligible.values())
    acceptance = sum(1-1/n for c in counts.values() for n in c.values() if n >= 2) / nrefs
    human_effective = human_weight / acceptance
    records = {}
    for source in sources:
        rec = legs[f"e28_{arm}_{source}"]
        fit = checked(rec["fit"],arm=arm,source=source,split="fit")
        dev = checked(rec["dev"],arm=arm,source=source,split="dev")
        tw = human_effective * eligible[source] / nrefs
        groups += [(source, fit, tw, 0, "withinref,rank"),
                   (f"{source}_development", dev, 0, eligible[source]/nrefs, "withinref,rank")]
        records[source] = dict(fit=rec["fit"], dev=rec["dev"], train_weight=tw)
    return groups, dict(arm=arm, teacher_pin_sha256=sha(PIN), legs=records,
                       pooled_rank_share=pin["pooled_rank_share"], pooled_pearson_weight=pin["pooled_pearson_weight"],
                       konfig_deviation=pin["konfig_deviation"], coverage=pin["arms"][arm]["coverage"])


def prepare(source: Path, out: Path):
    """Derive legs from admitted Rev5 full tables, using pinned member sets and R1 reference dev split."""
    from v2_teacher import write_curated, row_keys_sha
    pin = read_pin()
    policy = admission_pin()
    paths = source_inventory(source, policy)
    producer = subprocess.check_output(["jj", "log", "-r", "@-", "--no-graph", "-T", "commit_id"],
                                       cwd=REPO, text=True).strip()
    committed = subprocess.check_output(["jj", "file", "show", "-r", producer,
                                        "scripts/rev4_featpot/e28_recipe.py"], cwd=REPO)
    if committed != Path(__file__).read_bytes():
        raise ValueError("commit the E28 preparation owner before deriving its data views")
    receipt_path = source / "wide/main/real/receipt.json"
    if sha(receipt_path) != pin["source_receipt_sha256"]:
        raise ValueError("E28 preparation source changed")
    if out.exists():
        raise ValueError("E28 root must be fresh")
    for rel,path in paths.items():
        if sha(path) != policy["source_files"][rel]:
            raise ValueError("E28 preparation approved member changed")
    for rel,path in paths.items():
        dest = out / rel
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest)
    vdir = out / "wide/main/real"
    receipt = json.loads(receipt_path.read_text())
    if receipt["formula_revision"] != 5:
        raise ValueError("E28 requires Rev5")
    receipt["legs"].pop("hdr", None)
    for arm, config in pin["arms"].items():
        for dataset, members in config["human_members"].items():
            full = receipt["legs"][dataset]["full"]
            table = out / full["rel"]
            if sha(table) != full["sha256"]:
                raise ValueError(f"source table changed: {dataset}")
            keys_path = vdir / f"{dataset}.keys.parquet"
            if sha(keys_path) != receipt["legs"][dataset]["keys_sha256"]:
                raise ValueError("E28 source keys changed")
            keys = pq.read_table(keys_path, columns=KEY_COLUMNS)
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
                                   arm=arm, source=dataset, role="TRAIN" if part=="fit" else "DEV",
                                   member_sets=members, teacher_pin_sha256=sha(PIN), keys_sha256=sha(kp),
                                   row_keys_sha256=row_keys_sha(keys.filter(pa.array(mask))),
                                   source_keys_sha256=receipt["legs"][dataset]["keys_sha256"],
                                   source_label_free_row_keys_sha256=row_keys_sha(keys),
                                   source_table_sha256=full["sha256"], split=part,
                                   split_rule="sha256(ref_basename) mod 5 == 0 is dev", build_commit=producer,
                                   registration_commit=pin["registration_commit"])
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
