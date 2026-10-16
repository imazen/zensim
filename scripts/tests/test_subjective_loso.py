from pathlib import Path
import sys
import unittest
import json
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"rev4_featpot"))
import subjective_loso as loso
from v2_common import confirm_seeds


class Loso(unittest.TestCase):
    def fixture(self):
        sources=[dict(id=f"source-{i}",families=[f"root-{i}",f"crop-{i}"]) for i in range(28)]
        rows=[dict(observation=f"{i}-{j}",family=f"{prefix}-{i}",role="fit")
              for i in range(28) for j,prefix in enumerate(("root","crop"))]
        return dict(schema="source-family-inventory-v1",overlap_audit_sha256="a"*64,sources=sources,
                    reusable_groups=["inherited","palette","calibration","preprocessing"],
                    legs={name:(rows if name=="subjective_rank" else [r|dict(family="unrelated-"+r["family"]) for r in rows])
                          for name in ("subjective_rank","inherited","palette","calibration","preprocessing")})

    def test_exact_grid_seed_pairing_and_reuse(self):
        plan=loso.build(self.fixture())
        self.assertEqual(len(plan["cells"]),736)
        self.assertEqual(len(plan["maps"]),672)
        self.assertEqual(sum(c["route"]=="loso" for c in plan["cells"]),336)
        self.assertEqual(sum(c["route"]=="d1" for c in plan["cells"]),320)
        self.assertEqual(sum(c["route"]=="full" for c in plan["cells"]),80)
        for c in plan["cells"]:
            if c["route"]=="loso":
                self.assertEqual((c["init_seed"],c["sample_seed"]),confirm_seeds(c["seed_index"]))
        reused=[m for m in plan["maps"] if m["arm"] in ("control","p")]
        self.assertEqual(len(reused),336)
        self.assertTrue(all("/full_s" in m["fit"] for m in reused))

    def test_all_leg_crop_scaler_calibration_exclusions(self):
        inventory=self.fixture()
        plan=loso.build(inventory)
        for mask in plan["masks"]:
            for name,indices in mask["groups"].items():
                self.assertEqual(len(indices),54 if name=="subjective_rank" else 56)
                self.assertTrue(all(inventory["legs"][name][i]["family"] not in mask["excluded_families"] for i in indices))
        inventory["sources"][0]["families"].append("root-1")
        with self.assertRaises(ValueError): loso.build(inventory)
        inventory=self.fixture()
        inventory["legs"]["calibration"][0]["family"]="crop-0"
        with self.assertRaisesRegex(ValueError,"reuse leaks"): loso.build(inventory)

    def test_neutral_transport_is_complete_and_declarative(self):
        plan=loso.build(self.fixture())
        spec=loso.fit_spec(plan,"a"*64,"b"*64,"contract.json","c"*64,"data","results/cells")
        self.assertEqual(len(spec["cells"]),736)
        self.assertTrue(all(c["argv"][0]=="subjective_cell.py" and "--train-only" in c["argv"] for c in spec["cells"]))
        self.assertFalse(plan["labels_read"])

    def test_real_cell_preread_refuses_wrong_materialized_masks(self):
        import subjective_cell as cell_owner
        from subjective_training_leg import sha
        from v2_teacher import selection_sha
        with tempfile.TemporaryDirectory() as scratch:
            root=Path(scratch)
            inventory=self.fixture()
            inventory_path=root/"inventory.json";inventory_path.write_text(json.dumps(inventory))
            plan=loso.build(inventory)
            logical=next(c for c in plan["cells"] if c["route"]=="loso")
            mask=plan["masks"][0]
            def record(name,key):
                path=root/(name+".parquet")
                path.write_bytes(b"unread invented payload")
                declaration=Path(str(path)+".manifest.json")
                declaration.write_text(json.dumps(dict(row_selection_sha256=selection_sha(mask["groups"][key]),
                    rows_kept=len(mask["groups"][key]),parent_inventory_sha256=plan["inventory_sha256"])))
                return dict(name=name,path=str(path),manifest_sha256=sha(declaration),mask_key=key,
                            train_weight=8.,val_weight=0.,mode="withinref,rank",nominal_weight=8)
            group=record("subjective_rank","subjective_rank")
            calibration=record("calibration","calibration")
            value=logical|dict(selected_epoch=119,groups=[group],calibration=calibration)
            frozen=root/"frozen.json";frozen.write_text("{}")
            packet=dict(schema="registered-training-cells-v1",inventory_path=str(inventory_path),
                        inventory_sha256=sha(inventory_path),wide_receipt_path=str(frozen),
                        wide_receipt_sha256=sha(frozen),frozen_path=str(frozen),frozen_sha256=sha(frozen),
                        cells={logical["name"]:value})
            path=root/"contract.json";path.write_text(json.dumps(packet))
            cell_owner.preread(path,sha(path),logical["name"])
            declaration=Path(group["path"]+".manifest.json")
            d=json.loads(declaration.read_text());d["rows_kept"]+=1;declaration.write_text(json.dumps(d))
            group["manifest_sha256"]=sha(declaration);path.write_text(json.dumps(packet))
            with self.assertRaisesRegex(ValueError,"materialized input"):
                cell_owner.preread(path,sha(path),logical["name"])


if __name__=="__main__": unittest.main()
