"""E29 adapters for the existing strict LODO, preparation and assessment owners."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import rankdata

from v2_common import acceptance_weight, sha
from v2_human_role import PRODUCTION_SOURCES, preflight_recipe

TEACHER = "deb70e775b043a578c77e0c3ff27960ebfa936e9d901497f9d74389e6ce9fbce"
SOURCE_TABLE = "1d66d371b85733d7a4fde4964688284e7ac01d69707c62e0b7c4888af0a9b0c3"
SOURCE_MANIFEST = "66154e1760fb9724a6021897b39e9950a811eeb1ad846daf2677a63a34d93b0e"
SOURCE_KEYS = "b55a4ec95dac7f4c060f910c7a6dabf2a1d7c6245efc81f6a36e90ed33d09553"


def consensus(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or len(a) < 2 or not np.isfinite([a, b]).all():
        raise ValueError("invalid teacher vectors")
    return ((rankdata(a, method="average") - 1) + (rankdata(b, method="average") - 1)) / (2 * (len(a) - 1))


def agreement_pairs(a, b, refs):
    """Lexicographic unordered pairs; raw JOD threshold, no fitted transform."""
    a, b, refs = np.asarray(a), np.asarray(b), np.asarray(refs)
    if a.shape != b.shape or a.shape != refs.shape or not np.isfinite([a, b]).all():
        raise ValueError("invalid agreement population")
    for i in range(len(a) - 1):
        da, db = a[i] - a[i + 1:], b[i] - b[i + 1:]
        valid = (refs[i] != refs[i + 1:]) & (np.abs(da) >= .05) & (np.abs(db) >= .05) & (np.sign(da) == np.sign(db))
        for j in np.flatnonzero(valid) + i + 1:
            yield [i, int(j)]


def admit_metadata(path, d):
    from e21_cheap_recipe import columns
    from v2_teacher import row_keys_sha, key_path
    if (d.get("study") != "E29" or d.get("role") != "train" or d.get("rows") != 7390
            or d.get("population") != "agree-only" or d.get("formula_revision") != 5
            or d.get("teacher_sha256") != TEACHER or d.get("requested_ids") != columns("by_v2fy")
            or d.get("arm") not in ("hb4", "hc4") or not d.get("build_commit")
            or d.get("source_table_sha256") != SOURCE_TABLE or d.get("source_keys_sha256") != SOURCE_KEYS
            or d.get("source_manifest_sha256") != SOURCE_MANIFEST
            or d.get("source_bank_feature_set_id") is not None
            or d.get("target_transform") != ("pooled-midrank-Borda-[0,1]" if d.get("arm") == "hb4" else "score=10*q_jod;no-clipping")):
        raise ValueError("E29 refuses foreign/HDR VAL population or provenance")
    keys = pq.read_table(key_path(path))
    if (keys.column_names != ["row_id", "role", "agree", "ref_basename"] or keys.num_rows != 7390
            or set(keys["role"].to_pylist()) != {"train"} or set(keys["agree"].to_pylist()) != {True}
            or len(set(keys["row_id"].to_pylist())) != 7390 or row_keys_sha(keys) != d.get("row_keys_sha256")):
        raise ValueError("E29 label-free keys differ from TRAIN agreement population")
    if sha(key_path(path)) != d["keys_sha256"]:
        raise ValueError("E29 key pin changed")


def preflight(root, heldout, arm):
    """All D1 metadata/key populations precede HDR or SDR payload hashing."""
    preflight_recipe(root, root / "human_role_decision.json", heldout)
    r = json.loads((root / "wide/main/real/receipt.json").read_text())["legs"]["hdr_consensus"][arm]
    path = root / r["fit"]["rel"]
    mpath = Path(f"{path}.manifest.json")
    if sha(mpath) != r["fit"]["manifest_sha256"]:
        raise ValueError("E29 manifest pin changed")
    d = json.loads(mpath.read_text())
    if d.get("arm") != arm:
        raise ValueError("E29 arm mismatch")
    admit_metadata(path, d)
    return path, d


def hdr_group(record, keep, recipe):
    from v2_common import V2
    arm = recipe["hdr_consensus"]
    path = V2 / record[arm]["fit"]["rel"]
    d = json.loads(Path(f"{path}.manifest.json").read_text())
    admit_metadata(path, d)
    if sha(path) != d["table_sha256"] or keep != d["requested_ids"]:
        raise ValueError("E29 table/read set changed")
    refs = pq.read_table(path, columns=["ref_basename"])["ref_basename"].to_pylist()
    return ("hdr", path, acceptance_weight(4, refs), 0, "rank"), d


def trainer_options(groups, arm):
    path = next(p for n, p, _, _, _ in groups if n == "hdr")
    d = json.loads(Path(f"{path}.manifest.json").read_text())
    admit_metadata(path, d)
    options = ["--hdr-consensus-research"]
    if arm == "hc4":
        pairs = path.parent / d["pair_list"]
        if pairs.parent != path.parent or sha(pairs) != d["pair_list_sha256"]:
            raise ValueError("E29 pair list changed")
        options += ["--rank-pair-list", f"hdr:{pairs}", "--no-sample-coverage"]
    return options


def complete_cells(results, control, pins_path):
    """No bank or label read before complete, frozen final-119 paired fits."""
    from e30_four_source import SPEC
    if pins_path is None or not pins_path.is_file():
        raise ValueError("INCOMPLETE: frozen complete E30/fresh matched-control pins absent")
    pins = json.loads(pins_path.read_text())
    if pins.get("study") != "E29" or pins.get("control_choice") not in ("exact-E30-nA3", "fresh-matched"):
        raise ValueError("INCOMPLETE: unregistered E29 baseline")
    records = {}
    for label, base, spec in [("control", control, SPEC), ("hb4", results, SPEC + ":hb4"),
                              ("hc4", results, SPEC + ":hc4")]:
        records[label] = []
        for source in PRODUCTION_SOURCES:
            for seed in range(10):
                cell = base / "cells" / f"{spec}__N/without_{source}_s{seed}"
                rp, bake = cell / "result.json", cell / "refit/last.bin"
                if not rp.is_file() or not bake.is_file():
                    raise ValueError(f"INCOMPLETE: {label}/{source}/s{seed} missing")
                r = json.loads(rp.read_text())
                if (r.get("execution_contract") != "registered-fit" or r.get("spec") != spec
                        or r.get("heldout") != source or r.get("seed_index") != seed
                        or r.get("head") != "N" or r.get("epochs") != 120 or r.get("pairs_per_epoch") != 50000
                        or r.get("selection", {}).get("selected_epoch") != 119
                        or sha(bake) != r.get("selected_bake_sha256")):
                    raise ValueError("INCOMPLETE: E29 final-cell identity/budget differs")
                if label == "control" and pins["cells"].get(f"{source}_s{seed}") != {
                        "result_sha256": sha(rp), "bake_sha256": sha(bake)}:
                    raise ValueError("INCOMPLETE: frozen E29 baseline changed")
                records[label].append((source, seed, cell, r))
    return records


def prepare(source, hdr, out, build_commit):
    """Fresh root, copying only the frozen D1 file inventory and approved HDR TRAIN."""
    import shutil
    from v2_teacher import row_keys_sha, key_path, selection_sha
    from e21_cheap_recipe import columns
    if out.exists() or not build_commit or len(build_commit) != 40:
        raise ValueError("E29 preparation needs a fresh root and committed owner")
    for fold in (None, *PRODUCTION_SOURCES):
        preflight_recipe(source, source / "human_role_decision.json", fold)
    mpath = Path(f"{hdr}.manifest.json")
    if sha(mpath) != SOURCE_MANIFEST:
        raise ValueError("original E26 TRAIN manifest changed")
    original = json.loads(mpath.read_text())
    if (original.get("role") != "train" or original.get("population") != "agree-only"
            or original.get("rows") != 7390 or original.get("teacher_sha256") != TEACHER
            or original.get("requested_ids") != columns("by_v2fy")):
        raise ValueError("E29 original TRAIN population refused")
    # Project only role/identity columns before opening targets/features.
    identity = pq.read_table(key_path(hdr), columns=["row_id", "role", "agree", "ref_path"])
    if (identity.num_rows != 7390 or set(identity["role"].to_pylist()) != {"train"}
            or set(identity["agree"].to_pylist()) != {True} or len(set(identity["row_id"].to_pylist())) != 7390):
        raise ValueError("original HDR keys are not TRAIN agreement-only")
    if sha(hdr) != SOURCE_TABLE or sha(key_path(hdr)) != SOURCE_KEYS:
        raise ValueError("original E26 TRAIN payload changed")
    frozen = json.loads((source / "wide/frozen.json").read_text())
    receipt = json.loads((source / "wide/main/real/receipt.json").read_text())
    # No directory walk or historical/AIC source discovery.
    files = {"wide/frozen.json", "wide/keep_lists.json", "wide/main/real/receipt.json", *frozen["auxiliary_files"]}
    for rec in receipt["legs"].values():
        for split in ("full", "fit", "dev"):
            if split in rec:
                rel = rec[split]["rel"]
                files.update((rel, rel + ".manifest.json", str(key_path(Path(rel)))))
    for rel in sorted(files):
        if any(x in rel.lower() for x in ("aic", "confirm", "sealed", "hdr", "terminal", "sdr25")):
            raise ValueError("unauthorized source transport member")
    from v2_lodo_mlp import strict_training_groups
    groups = [(str(i), source / rel, 1., 0., "withinref,rank")
              for i, rel in enumerate(sorted(files)) if rel.endswith(".parquet") and not rel.endswith(".keys.parquet")]
    strict_training_groups(groups, source / "human_role_decision.json")
    # metadata/key populations above are admitted before any source copy.
    for rel in sorted(files):
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / rel, dst)
    keys = pq.read_table(key_path(hdr))
    if not {"hdrvdp3_q_jod", "cvvdp_jod"} <= set(keys.column_names):
        raise ValueError("corrected two-teacher TRAIN authority missing")
    a, b = [keys[c].to_numpy() for c in ("hdrvdp3_q_jod", "cvvdp_jod")]
    table = pq.read_table(hdr)
    refs = table["ref_basename"].to_pylist()
    if refs != keys["ref_path"].to_pylist() or not np.isfinite([a, b]).all():
        raise ValueError("teacher/reference join differs")
    label_free = identity.rename_columns(["row_id", "role", "agree", "ref_basename"])
    base = out / "wide/main/real"
    receipt["legs"]["hdr_consensus"] = {}
    for arm in ("hb4", "hc4"):
        path = base / f"hdr_{arm}.parquet"
        target = consensus(a, b) if arm == "hb4" else 10 * a
        t = table.set_column(table.schema.get_field_index("human_score"), "human_score", pa.array(target))
        pq.write_table(t, path, compression="zstd")
        pq.write_table(label_free, key_path(path), compression="zstd")
        d = {**original, "study": "E29", "arm": arm, "build_commit": build_commit,
             "native_producer_build_commit": original["build_commit"],
             "source_table_sha256": SOURCE_TABLE, "source_keys_sha256": SOURCE_KEYS,
             "source_manifest_sha256": SOURCE_MANIFEST, "table_sha256": sha(path),
             "keys_sha256": sha(key_path(path)), "row_keys_sha256": row_keys_sha(label_free),
             "row_selection_sha256": selection_sha(range(7390)),
             "immutable_input_roots": [*frozen["admission_view"]["immutable_input_roots"], str(out.resolve()), "/var/tmp/rev4-featpot/v2e29"],
             "target_transform": "pooled-midrank-Borda-[0,1]" if arm == "hb4" else "score=10*q_jod;no-clipping"}
        if arm == "hc4":
            pairs = base / "hdr_hc4.pairs.json"
            count = 0
            with pairs.open("w") as stream:
                stream.write("[")
                for pair in agreement_pairs(a, b, refs):
                    if count:
                        stream.write(",")
                    stream.write(json.dumps(pair, separators=(",", ":")))
                    count += 1
                stream.write("]\n")
            if not count:
                raise ValueError("empty hc4 pair population")
            d.update(pair_list=pairs.name, pair_list_sha256=sha(pairs), pairs=count, agreement_threshold=.05)
            frozen["auxiliary_files"][str(pairs.relative_to(out))] = sha(pairs)
        sidecar = Path(f"{path}.manifest.json")
        sidecar.write_text(json.dumps(d, indent=1) + "\n")
        receipt["legs"]["hdr_consensus"][arm] = {"fit": {"rel": str(path.relative_to(out)),
            "sha256": sha(path), "manifest_sha256": sha(sidecar), "keys_sha256": sha(key_path(path)), "rows": 7390}}
        for member in (path, sidecar, key_path(path)):
            frozen["auxiliary_files"][str(member.relative_to(out))] = sha(member)
    rp = base / "receipt.json"
    rp.write_text(json.dumps(receipt, indent=1) + "\n")
    frozen["wide_receipts"]["main/real"] = sha(rp)
    frozen["build_commit"] = build_commit
    (out / "wide/frozen.json").write_text(json.dumps(frozen, indent=1) + "\n")
    return {"build_commit": build_commit, "source_root": str(source), "prepared_root": str(out),
            "folds": list(PRODUCTION_SOURCES), "arms": receipt["legs"]["hdr_consensus"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", choices=("prepare", "preflight"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--hdr", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--build-commit")
    a = p.parse_args()
    if a.cmd == "prepare":
        print(json.dumps(prepare(a.root, a.hdr, a.out, a.build_commit)))
    else:
        for fold in PRODUCTION_SOURCES:
            for arm in ("hb4", "hc4"):
                preflight(a.root, fold, arm)
        print(json.dumps({"status": "PASS", "scope": "metadata/key preflight; no assessment"}))


if __name__ == "__main__":
    # Clean environment works without caller-provided PYTHONPATH.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
