"""One frozen, training-only cell through the existing trainer/packing owners.

No discovery, assessment, scheduling or label conversion. All paths/recipes
are supplied by an independently pinned private execution contract.
"""
import argparse
import json
from pathlib import Path

from subjective_training_leg import document, safe, sha
from subjective_loso import build


def preread(contract, pin, cell_id):
    packet=document(contract,pin)
    if packet.get("schema") != "registered-training-cells-v1":
        raise ValueError("registered training contract required")
    document(packet["wide_receipt_path"],packet["wide_receipt_sha256"])
    document(packet["frozen_path"],packet["frozen_sha256"])
    inventory=document(packet["inventory_path"],packet["inventory_sha256"])
    plan=build(inventory)
    logical=next(c for c in plan["cells"] if c["name"]==cell_id)
    cell=packet["cells"][cell_id]
    if any(cell[k] != v for k,v in logical.items()) or cell["selected_epoch"]!=119:
        raise ValueError("registered cell/seed/mask binding differs")
    records=cell["groups"]
    if len({r["name"] for r in records})!=len(records):
        raise ValueError("duplicate registered group")
    for r in [*records,cell["calibration"]]:
        safe(r["path"])
        safe(str(r["path"])+".manifest.json")
    declarations=[(r,document(str(r["path"])+".manifest.json",r["manifest_sha256"]))
                  for r in [*records,cell["calibration"]]]
    if logical["route"]=="loso":
        from v2_teacher import selection_sha
        mask=next(m for m in plan["masks"] if m["source"]==logical["heldout"])
        for r,d in declarations:
            indices=mask["groups"][r["mask_key"]]
            if (d.get("row_selection_sha256")!=selection_sha(indices)
                    or d.get("rows_kept")!=len(indices)
                    or d.get("parent_inventory_sha256")!=plan["inventory_sha256"]):
                raise ValueError("LOSO materialized input omits source-family mask binding")
    has_h=logical["arm"] in ("h","ph")
    if sum(r["name"]=="subjective_rank" for r in records)!=int(has_h):
        raise ValueError("registered arm/subjective group differs")
    if any(r["nominal_weight"] != 8 for r in records if r["name"]=="subjective_rank"):
        raise ValueError("registered subjective weight differs")
    palette=[r for r in records if r.get("kind")=="palette" and r["train_weight"]>0]
    if bool(palette)!=(logical["arm"] in ("p","ph")):
        raise ValueError("registered arm/palette group differs")
    if palette and abs(sum(r["nominal_weight"] for r in palette)-2)>1e-12:
        raise ValueError("registered palette total nominal mass differs")
    return packet,cell


def execute(args):
    # Import the canonical owners only after contract and all metadata/masks
    # pass. This command is never invoked by the declaration builder.
    packet,cell=preread(args.contract,args.contract_sha256,args.cell_id)
    import os
    if os.environ.get("ZENSIM_MAX_TIER")!="v3":
        raise ValueError("registered cell requires the v3 tier cap")
    if bool(args.pack_production)!=(cell["route"]=="full"):
        raise ValueError("registered packing route differs")
    from v2_lodo_mlp import train_and_select
    from v2_common import recipe_of, sha
    groups=[(r["name"],Path(r["path"]),r["train_weight"],r["val_weight"],r["mode"]) for r in cell["groups"]]
    spec="sel:3b7bd5ebe929@h32:H128:cv16:cf98"+ (":fx1" if cell["family"]=="C" else "")
    recipe=recipe_of(spec)
    authority=None
    if cell["arm"] in ("h","ph"):
        authority=(Path(packet["subjective_authority"]),packet["subjective_authority_sha256"],
                   args.cell_id,cell["feature_ids"])
    if cell["formula_revision"]!=5 or cell["tier"]!="v3" or cell["epochs"]!=120 or cell["pairs_per_epoch"]!=50000:
        raise ValueError("registered full-budget recipe differs")
    from v2_lodo_mlp import strict_training_groups, strict_group_input_roots
    from v2_common import refuse_immutable_output
    strict_training_groups(groups,Path(packet["data_role_decision"]),subjective_authority=authority)
    refuse_immutable_output(args.dest,strict_group_input_roots(groups))
    if args.dest.exists() and any(args.dest.iterdir()):
        raise ValueError("registered cell requires fresh outputs")
    args.dest.mkdir(parents=True,exist_ok=True)
    bake,curve,selection=train_and_select(groups,cell["init_seed"],cell["sample_seed"],1825,
        Path(cell["keep_file"]),"N",args.dest,recipe,strict_admission=True,
        data_role_decision=Path(packet["data_role_decision"]),subjective_authority=authority)
    packed={}
    if cell["route"]=="full":
        from v2_production_pack import pack_production
        packed=pack_production(bake,args.dest,Path(cell["calibration"]["path"]),e33_output_stage=True)
    result=dict(schema="registered-training-cell-v1",training_only=True,execution_contract="registered-fit",
        train_weights={r["name"]:r["train_weight"] for r in cell["groups"]},
        epochs=120,pairs_per_epoch=50000,wide_receipt_sha256=sha(packet["wide_receipt_path"]),
        frozen_sha256=sha(packet["frozen_path"]),data_role_decision_sha256=sha(packet["data_role_decision"]),
        selected_bake=str(bake),selected_bake_sha256=sha(bake),selection=selection,dev_curve=curve,
        cell_id=args.cell_id,contract_sha256=args.contract_sha256,seed_index=cell["seed_index"],
        init_seed=cell["init_seed"],sample_seed=cell["sample_seed"],**packed)
    (args.dest/"result.json").write_text(json.dumps(result,sort_keys=True)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract",type=Path,required=True)
    parser.add_argument("--contract-sha256",required=True)
    parser.add_argument("--cell-id",required=True)
    parser.add_argument("--pack-production",action="store_true")
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--dest",type=Path,required=True)
    parser.add_argument("--strict-admission",action="store_true",required=True)
    parser.add_argument("--train-only",action="store_true",required=True)
    args=parser.parse_args()
    execute(args)


if __name__=="__main__":main()
