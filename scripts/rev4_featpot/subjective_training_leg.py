"""Receipt-bound unnamed subjective leg. Metadata/key admission precedes payloads.

The authority digest comes from the independently frozen program/argv, never
from the leg being admitted. This module contains no population identities.
"""
import hashlib
import json
from pathlib import Path
import math


def safe(path):
    path = Path(path)
    for p in (path.absolute(), path.resolve()):
        for part in p.parts:
            token = part.lower()
            if any(s in token for s in ("_sealed", "holdout", "terminal")) or token.startswith("labels__"):
                raise ValueError("protected subjective input ancestry")
    return path


def sha(path):
    return hashlib.sha256(safe(path).read_bytes()).hexdigest()


def document(path, expected=None):
    path = safe(path)
    data = path.read_bytes()
    if expected is not None and hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("subjective metadata/key pin changed")
    return json.loads(data)


def admit(authority_path, authority_sha256, cell_id, groups, selected_ids):
    if len(authority_sha256) != 64:
        raise ValueError("independently pinned authority required")
    a = document(authority_path, authority_sha256)
    if (a.get("schema") != "subjective-training-authority-v1" or a.get("state") != "approved"
            or a.get("artifact_visibility") != "private" or a.get("selection") != "fixed-final-epoch"):
        raise ValueError("subjective authority not admitted")
    c = a["cells"][cell_id]
    if c["feature_ids"] != selected_ids or c["formula_revision"] != 5 or c["tier"] != "v3":
        raise ValueError("subjective feature/producer contract differs")
    if len(groups) != len(c["groups"]) or len({g[0] for g in groups}) != len(groups):
        raise ValueError("complete unique cell group inventory required")
    records = {r["name"]:r for r in c["groups"]}
    pending = []
    # First ALL paths, then ALL manifests, then keys. Never target-bearing data.
    for name, path, tw, vw, mode in groups:
        r = records[name]
        if (str(safe(path).resolve()) != str(safe(r["path"]).resolve())
                or not all(math.isfinite(v) for v in (tw,vw))
                or (tw,vw,mode) != (r["train_weight"],r["val_weight"],r["mode"])):
            raise ValueError("subjective cell group path/mode/weight mismatch")
        safe(str(path)+".manifest.json")
        safe(r["keys_path"])
        pending.append((r, Path(str(path)+".manifest.json")))
    declarations = [(r, document(p,r["manifest_sha256"])) for r,p in pending]
    admitted = {}
    for r,d in declarations:
        if not r.get("subjective",False):
            continue  # Existing strict D1/oracle owner remains mandatory.
        if (d.get("schema") != "subjective-training-leg-v1"
                or d.get("loss") != "within-source-rank" or d.get("nominal_weight") != 8
                or d.get("target_orientation") != "quality" or d.get("target_transform") != "negate-pointwise"
                or d.get("target_column") != "human_score" or d.get("selection") != "fixed-final-epoch"
                or d.get("formula_revision") != 5 or d.get("tier") != "v3"
                or d.get("feature_ids") != selected_ids or d.get("producer_commit") != c["producer_commit"]
                or r["mode"] != "withinref,rank" or r["val_weight"] != 0
                or r["train_weight"] <= 0 or d.get("table_sha256") != r["table_sha256"]):
            raise ValueError("subjective role/loss/producer/target binding differs")
        admitted[r["name"]] = (r,d)
    if len(admitted) != 1:
        raise ValueError("authority requires one aggregated unnamed subjective leg")
    for r,d in admitted.values():
        keys = document(r["keys_path"],r["keys_sha256"])
        if len(keys) != r["rows"] or d["rows"] != r["rows"] or not keys:
            raise ValueError("subjective ordered key count differs")
        observations = set()
        source_counts = {}
        references = {}
        for i,key in enumerate(keys):
            if (set(key) != {"observation", "source", "family", "reference", "role", "tier", "row"}
                    or key["row"] != i or key["role"] != "fit" or key["tier"] != "T2"
                    or key["source"] not in c["sources"] or key["family"] in c["excluded_families"]
                    or key["observation"] in observations
                    or any(not isinstance(key[k], str) or not key[k] for k in ("observation","source","family","reference"))):
                raise ValueError("subjective source/observation/order/role mismatch")
            if references.setdefault(key["source"],key["reference"]) != key["reference"]:
                raise ValueError("subjective source must have one pairing reference")
            observations.add(key["observation"])
            source_counts[key["source"]] = source_counts.get(key["source"],0)+1
        if (source_counts != r["source_counts"] or any(n < 2 for n in source_counts.values())
                or set(source_counts) != set(c["sources"])
                or len(set(references.values())) != len(references)):
            raise ValueError("subjective source population differs from authority")
        expected = 8/(sum(1-1/source_counts[k] for k in sorted(source_counts))/len(source_counts))
        if r["train_weight"] != expected:
            raise ValueError("subjective nominal-eight acceptance correction differs")
    return {name:d for name,(_,d) in admitted.items()}


def payloads(admitted, groups):
    """Called only after every group passed native/wrapper metadata admission."""
    for name,path,_,_,_ in groups:
        if name in admitted and sha(path) != admitted[name]["table_sha256"]:
            raise ValueError("subjective payload pin changed")
