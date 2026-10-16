"""Label-free family masks and fixed factorial fit inventory. No execution."""
import argparse
import hashlib
import json
from pathlib import Path
from v2_common import confirm_seeds, seeds

D1 = ("kadid", "tid2013", "konfig", "cid22_a25")
ARMS = ("control", "p", "h", "ph")


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def build(inventory):
    if inventory.get("schema") != "source-family-inventory-v1" or not inventory.get("overlap_audit_sha256"):
        raise ValueError("pre-existing cross-leg overlap audit required")
    sources = inventory["sources"]
    if len(sources) != 28 or len({s["id"] for s in sources}) != 28:
        raise ValueError("exact 28-source inventory required")
    families = [f for s in sources for f in s["families"]]
    if not all(s["families"] for s in sources) or len(set(families)) != len(families):
        raise ValueError("unique complete source family closure required")
    legs = inventory["legs"]
    if not legs or "calibration" not in legs or "preprocessing" not in legs:
        raise ValueError("all training legs, scaler/preprocessing and calibration keys required")
    for rows in legs.values():
        for row in rows:
            if set(row) != {"observation","family","role"} or not all(isinstance(v,str) and v for v in row.values()):
                raise ValueError("label-free family-key schema required")
    if "subjective_rank" not in legs or not inventory.get("reusable_groups"):
        raise ValueError("subjective and reusable Control/P group inventory required")
    for name in inventory["reusable_groups"]:
        if any(row["family"] in set(families) for row in legs[name]):
            raise ValueError("Control/P full-data reuse leaks a held-out family")
    for source in sources:
        if not any(row["family"] in source["families"] for row in legs["subjective_rank"]):
            raise ValueError("subjective source missing from family keys")
    masks = []
    for source in sources:
        excluded=set(source["families"])
        groups={name:[i for i,row in enumerate(rows) if row["family"] not in excluded] for name,rows in legs.items()}
        # Both metadata and exclusion closure are pinned; derived data must bind
        # these exact ordered indices and inherit every source-family role.
        masks.append(dict(source=source["id"],excluded_families=sorted(excluded),groups=groups,
                          mask_sha256=digest(groups),parent_inventory_sha256=digest(inventory)))
    cells=[]
    maps=[]
    for family in ("A","C"):
        for arm in ARMS:
            for heldout in D1:
                for seed in range(10):
                    init,sample=seeds(heldout,seed)
                    cells.append(dict(name=f"{family}-{arm}__N/without_{heldout}_s{seed}",family=family,arm=arm,
                        route="d1",heldout=heldout,seed_index=seed,init_seed=init,sample_seed=sample))
            for seed in range(10):
                init,sample=confirm_seeds(seed)
                cells.append(dict(name=f"{family}-{arm}__N/full_s{seed}",family=family,arm=arm,route="full",
                                  seed_index=seed,init_seed=init,sample_seed=sample))
        for mi,mask in enumerate(masks):
            for seed in range(3):
                init,sample=confirm_seeds(seed) # No held-out-source seed rotation.
                for arm in ARMS:
                    if arm in ("h","ph"):
                        name=f"{family}-{arm}__N/loso_{mi:02}_s{seed}"
                        cells.append(dict(name=name,family=family,arm=arm,route="loso",heldout=mask["source"],
                            mask_sha256=mask["mask_sha256"],seed_index=seed,init_seed=init,sample_seed=sample))
                    else:
                        name=f"{family}-{arm}__N/full_s{seed}"
                    maps.append(dict(family=family,arm=arm,source=mask["source"],seed_index=seed,fit=name,
                                     role="fold-local-diagnostic-only",mask_sha256=mask["mask_sha256"]))
    if len(cells)!=736 or len({c["name"] for c in cells})!=736 or len(maps)!=672:
        raise ValueError("fixed cell/map inventory drift")
    return dict(schema="source-loso-plan-v1",cells=cells,maps=maps,masks=masks,
                inventory_sha256=digest(inventory),labels_read=False)


def fit_spec(plan,program_sha,data_sha,contract,contract_sha,root,destination):
    if any(len(pin)!=64 for pin in (program_sha,data_sha,contract_sha)):
        raise ValueError("program/data/contract hashes required")
    cells=[]
    for c in plan["cells"]:
        argv=["subjective_cell.py","--contract",str(contract),"--contract-sha256",contract_sha,
              "--cell-id",c["name"],"--root",str(root),"--dest",str(Path(destination)/c["name"]),
              "--strict-admission","--train-only"]
        if c["route"]=="full": argv.append("--pack-production")
        cells.append(dict(name=c["name"],argv=argv))
    # Feed directly to zenfleet-ctl declare-fits; no custom queue/claims/IDs.
    return dict(program_sha=program_sha,data_sha=data_sha,cells=cells)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory",required=True,type=Path)
    p.add_argument("--inventory-sha256",required=True)
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    raw=args.inventory.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=args.inventory_sha256:
        raise ValueError("source inventory pin differs")
    result=build(json.loads(raw))
    with args.out.open("x") as stream: stream.write(json.dumps(result,sort_keys=True)+"\n")


if __name__=="__main__": main()
