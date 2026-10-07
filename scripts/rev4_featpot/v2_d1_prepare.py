"""Prepare fresh D1 tables with existing admission, curation and transport owners.

No AIC payload is visited. Historical tables and frozen roots remain immutable.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pyarrow.parquet as pq

from v2_common import SOURCES, TEACHERS, sha, refuse_immutable_output
from v2_human_role import PRODUCTION_SOURCES, MEMBER_SOURCE, decision_record, human_declaration, human_keys
from v2_teacher import key_path, row_keys_sha, write_curated
from lib.assessment_identity import safe_path


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def table_record(root, table):
    d = json.loads(Path(f"{table}.manifest.json").read_text())
    keys = pq.read_table(key_path(table))
    return {"rel": str(table.relative_to(root)), "sha256": sha(table),
            "manifest_sha256": sha(Path(f"{table}.manifest.json")), "keys_sha256": sha(key_path(table)),
            "rows": keys.num_rows, "references": len(set(keys["ref_basename"].to_pylist()))}


def prepare(source, original, bank, stage, out, decision_path, fleet_root):
    source, original, bank, stage, out, fleet_root = map(safe_path, (source, original, bank, stage, out, fleet_root))
    receipt_path = original / "wide/main/real/receipt.json"
    decision = decision_record(decision_path, sha(receipt_path))
    if sha(original / "wide/frozen.json") != decision["source_frozen_sha256"]:
        raise ValueError("D1 original frozen receipt changed")
    roots = [p.resolve() for p in (source, original, bank)]
    for dest in (stage, out):
        refuse_immutable_output(dest, roots)
        if dest.exists():
            raise ValueError("D1 preparation needs fresh stage and output directories")
    if stage.resolve() in out.resolve().parents or out.resolve() in stage.resolve().parents or stage == out:
        raise ValueError("stage and output must be disjoint")
    old = json.loads(safe_path(source / "wide/main/real/receipt.json").read_text())
    if old["admission_view"]["source_receipt_sha256"] != decision["source_receipt_sha256"]:
        raise ValueError("admitted view does not bind the D1 original receipt")
    # Explicit allowlist: never enumerate table files or read a five-source union.
    selected = [(s, "full", old["legs"][s]["full"]) for s in PRODUCTION_SOURCES]
    selected += [(s, split, old["legs"][s][split]) for s in TEACHERS for split in ("fit", "dev")]
    selected += [("human_all", split, old["legs"]["human_without_aic3"][split]) for split in ("fit", "dev")]
    checked = []
    for name, split, rec in selected:
        p = safe_path(source / rec["rel"])
        sp, kp = safe_path(Path(f"{p}.manifest.json")), safe_path(key_path(p))
        d = json.loads(sp.read_text())
        if d.get("data_role_decision_required"):
            human_declaration(d, decision)
        if sha(sp) != rec["manifest_sha256"] or sha(p) != rec["sha256"] or sha(kp) != d["keys_sha256"]:
            raise ValueError("D1 source table/declaration/key pin changed")
        keys = pq.read_table(kp)
        if d.get("human_sources"):
            human_keys(keys, d)
        if row_keys_sha(keys) != d["row_keys_sha256"]:
            raise ValueError("D1 source key order changed")
        for member, pin in d["bank_manifest_sha256"].items():
            if member not in {*MEMBER_SOURCE, *(t[0] for t in TEACHERS.values())}:
                raise ValueError("D1 declaration includes an unapproved bank member")
            if sha(safe_path(bank / member / "_MANIFEST.json")) != pin:
                raise ValueError("D1 original bank binding changed")
        checked.append((name, split, p, d))
    # Temporary byte-identical approved inputs allow write_curated to preserve
    # source roots while deriving the four LODO masks outside the final root.
    for root in (stage, out):
        (root / "wide/main/real").mkdir(parents=True)
    for name, split, src, d in checked:
        stem = name if split == "full" else f"{name}_{split}"
        for root in (stage, out):
            dst = root / f"wide/main/real/{stem}.parquet"
            shutil.copyfile(src, dst)
            shutil.copyfile(key_path(src), key_path(dst))
            declaration = {**d, "data_role_decision_sha256": sha(decision_path),
                           "immutable_input_roots": [str(p) for p in (*roots, stage.resolve(), out.resolve(), fleet_root)]}
            declaration.pop("admission_root", None)
            dump(Path(f"{dst}.manifest.json"), declaration)
    legs = {}
    for name, split, _, _ in checked:
        p = out / f"wide/main/real/{name if split == 'full' else name + '_' + split}.parquet"
        legs.setdefault(name, {})[split] = table_record(out, p)
        if name in TEACHERS:
            for field in ("bounds",):
                if field in old["legs"][name]:
                    legs[name][field] = old["legs"][name][field]
    parity = []
    for heldout in PRODUCTION_SOURCES:
        name = f"human_without_{heldout}"
        legs[name] = {}
        for split in ("fit", "dev"):
            src = stage / f"wide/main/real/human_all_{split}.parquet"
            keys = pq.read_table(key_path(src))
            keep = np.array([MEMBER_SOURCE[m] != heldout for m in keys["member_set"].to_pylist()])
            target = pq.read_table(src, columns=["human_score"])["human_score"].to_numpy()
            dst = out / f"wide/main/real/{name}_{split}.parquet"
            # Final root is a protected future input, but is not an input to this
            # derivation yet; the stage declaration protects its actual inputs.
            stage_decl = json.loads(Path(f"{src}.manifest.json").read_text())
            stage_decl["immutable_input_roots"] = [str(p) for p in (*roots, stage.resolve())]
            dump(Path(f"{src}.manifest.json"), stage_decl)
            record = write_curated(src, dst, keep, target)
            if record["targets_changed"]:
                raise ValueError("D1 derivation changed targets")
            d = json.loads(Path(f"{dst}.manifest.json").read_text())
            d["human_sources"] = [s for s in PRODUCTION_SOURCES if s != heldout]
            d["row_selection"] = [r for r in d["row_selection"] if r["source"] != heldout]
            d["curation_mask_sha256"] = record["row_selection_sha256"]
            d["row_selection_sha256"] = hashlib.sha256(json.dumps(d["row_selection"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            d["immutable_input_roots"] = [str(p) for p in (*roots, stage.resolve(), out.resolve(), fleet_root)]
            dump(Path(f"{dst}.manifest.json"), d)
            # Full feature/target parity, including NaN payload bits, against the
            # same selected ordered rows. This is a preparation gate, not a fit.
            before, after = pq.read_table(src).filter(__import__('pyarrow').array(keep)), pq.read_table(dst)
            for col in after.column_names:
                if col.startswith("f") or col == "human_score":
                    a, b = before[col].to_numpy(), after[col].to_numpy()
                    if a.dtype != b.dtype or a.tobytes() != b.tobytes():
                        raise ValueError(f"D1 selected values changed: {name}/{split}/{col}")
                elif before[col].to_pylist() != after[col].to_pylist():
                    raise ValueError("D1 row order changed")
            legs[name][split] = table_record(out, dst)
            parity.append({"leg": name, "split": split, "rows": int(keep.sum()), "all_columns_bit_equal": True})
    from e15_coverage import admit_pool
    admit_pool(original / "e15", out / "e15", immutable_roots=(*roots, fleet_root))
    shutil.copyfile(source / "wide/keep_lists.json", out / "wide/keep_lists.json")
    shutil.copyfile(decision_path, out / "human_role_decision.json")
    (out / "source_bindings").mkdir()
    shutil.copyfile(receipt_path, out / "source_bindings/receipt.json")
    shutil.copyfile(original / "wide/frozen.json", out / "source_bindings/frozen.json")
    view = {"schema": "rev5-recipe-admission-v1", "source_root": str(original.resolve()), "bank_root": str(bank.resolve()),
            "source_receipt_sha256": decision["source_receipt_sha256"], "source_frozen_sha256": decision["source_frozen_sha256"],
            "immutable_input_roots": [str(p) for p in (*roots, stage.resolve(), out.resolve(), fleet_root)],
            "human_role_decision_sha256": sha(decision_path)}
    receipt = {**old, "legs": legs, "required_legs": list(legs), "complete": True, "admission_view": view,
               "bank": {k: v for k, v in old["bank"].items() if k in {*MEMBER_SOURCE, *(t[0] for t in TEACHERS.values())}}}
    dump(out / "wide/main/real/receipt.json", receipt)
    auxiliary = ["human_role_decision.json", "source_bindings/receipt.json", "source_bindings/frozen.json",
                 "e15/coverage_pool.parquet", "e15/coverage_pool.keys.parquet", "e15/coverage_pool.parquet.manifest.json"]
    dump(out / "wide/frozen.json", {"schema": "rev5-recipe-admission-freeze-v1", "admission_view": view,
         "wide_receipts": {"main/real": sha(out / "wide/main/real/receipt.json")},
         "keep_lists_sha256": sha(out / "wide/keep_lists.json"), "auxiliary_files": {p: sha(out / p) for p in auxiliary}})
    return {"schema": "shippath-d1-preparation-v1", "sources": list(PRODUCTION_SOURCES), "aic_payloads_read": False,
            "decision_sha256": sha(decision_path), "frozen_sha256": sha(out / "wide/frozen.json"), "derived_parity": parity,
            "copied_table_payloads": len(checked), "all_copied_table_sha256_unchanged": True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "original", "bank", "stage", "out", "decision", "fleet-root", "report"):
        p.add_argument(f"--{name}", type=Path, required=True)
    a = p.parse_args()
    dump(a.report, prepare(a.source, a.original, a.bank, a.stage, a.out, a.decision, a.fleet_root))


if __name__ == "__main__":
    main()
