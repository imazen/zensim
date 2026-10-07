"""E32 strict consumer refusals and named-column reads use synthetic data only."""
import copy
import json
import os
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import test_shippath2_admission as fixtures
import e32_palette as owner
import v2_common as common
import v2_lodo_mlp as fit
from e28_short_parity import normalized_control_repro


class PaletteTraining(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.RecipeAdmissionTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.f.admit()
        self.path = self.f.out / "wide/main/real/safesyn_fit.parquet"
        self.sidecar = Path(f"{self.path}.manifest.json")
        d = json.loads(self.sidecar.read_text())
        self.old_sha = common.sha(self.path)
        table = pq.read_table(self.path)
        for i in range(1853):
            if f"f{i}" not in table.column_names:
                table = table.append_column(f"f{i}", pa.array([float("nan")]*len(table), type=pa.float32()))
        # Populate the inherited primary inputs; this fixture's other slots are NaN.
        for i in owner.ARM_IDS[:-42]:
            index = table.schema.get_field_index(f"f{i}")
            table = table.set_column(index, f"f{i}", pa.array([i/512]*len(table), type=pa.float32()))
        for i in reversed(owner.PALETTE_IDS):
            table = table.append_column(f"palette_f{i}", pa.array([i/512]*len(table), type=pa.float32()))
        numeric=[f"f{i}" for i in range(1853)]
        table=table.select([n for n in table.column_names if n not in numeric and not n.startswith("palette_f")]
                           +numeric+[f"palette_f{i}" for i in reversed(owner.PALETTE_IDS)])
        pq.write_table(table, self.path, compression="zstd")
        self.d = {**d, "feature_set_id":owner.FEATURE_SET_ID, "table_sha256":common.sha(self.path),
            "research_palette":{**{k:owner.CONTRACT[k] for k in ("schema", "inherited_feature_set_id",
                "palette_feature_set_id", "feature_ids_sha256", "bank_manifest_sha256", "instrument_manifest_sha256",
                "producer_binary_sha256", "build_commit", "serving_allowed", "cast")},
                "columns":{str(i):f"palette_f{i}" for i in owner.PALETTE_IDS},
                "inherited_table_sha256":self.old_sha,"member_sets":["safesyn"],
                "role":"TRAIN-oracle-fit","key_domain":"member-pair-observation"}}
        self.save(self.d)

    def save(self, d):
        self.sidecar.write_text(json.dumps(d))

    def test_strict_owner_and_named_finite_reads(self):
        result=fit.strict_training_groups([("safesyn",self.path,1,0,"withinref,both")])
        self.assertEqual(result[0]["table_sha256"],common.sha(self.path))
        cols=owner.feature_columns(self.path,owner.ARM_IDS)
        self.assertEqual(cols[-42:],[f"palette_f{i}" for i in owner.PALETTE_IDS])
        common.refuse_nonfinite_kept([self.path],owner.ARM_IDS)
        # Physical f1825 remains an absent legacy auxiliary, not the new input.
        self.assertTrue(pq.read_table(self.path,columns=["f1825"])["f1825"].to_numpy()[0] != 1825/512)

    def test_real_trainer_accepts_named_research_projection(self):
        binary=Path(os.environ["SHIPPATH_TRAINER"])
        self.assertTrue(binary.is_file(),"build the trainer before running the E32 gates")
        keep=self.f.root/"keep.txt";keep.write_text("\n".join(map(str,owner.ARM_IDS))+"\n")
        out=self.f.root/"palette.bin"
        argv=[str(binary),"--group",f"synthetic:{self.path}:1:0:withinref,both",
              "--target-column","human_score","--target-scale","1","--hidden","128",
              "--epochs","2","--pairs-per-epoch","128","--init-seed","1101","--sample-seed","101",
              "--pair-sampling","uniform","--max-features","1867","--keep-features",str(keep),
              "--mse-weight","1","--early-stop-patience","0","--out-dtype","f32",
              "--no-auto-eval","--nonneg-distance","--out",str(out)]
        result=subprocess.run(argv,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        decoded=json.loads(subprocess.check_output([str(binary.parent/"examples/inspect_qualified_checkpoint"),str(out)],text=True))
        self.assertEqual(decoded["feature_set_id"],owner.FEATURE_SET_ID)
        self.assertEqual(decoded["repro"]["keep_features_n"],462)
        self.assertIs(decoded["repro"]["table_admission"]["serving_allowed"],False)

    def test_refusals_open_no_label_payload(self):
        original=Path.open
        opens=[]
        def opened(path,*args,**kwargs):
            if path.suffix==".parquet" and not path.name.endswith(".keys.parquet"):
                opens.append(str(path));raise AssertionError("early label-bearing payload open")
            return original(path,*args,**kwargs)
        for key,bad in [("role","TERMINAL"),("role","TRAIN-oracle-development"),
                        ("serving_allowed",0),("member_sets",["jpeg-aic-heldout"]),
                        ("palette_feature_set_id","palette_v1"),("instrument_manifest_sha256","0"*64)]:
            with self.subTest(key=key):
                d=copy.deepcopy(self.d);d["research_palette"][key]=bad;self.save(d)
                with patch.object(Path,"open",opened),self.assertRaises(ValueError):
                    fit.strict_training_groups([("safesyn",self.path,1,0,"withinref,both")])
                self.assertEqual(opens,[])

    def test_late_bad_declaration_refuses_before_first_payload(self):
        late = self.path.with_name("late.parquet")
        d = copy.deepcopy(self.d)
        d["research_palette"]["role"] = "TERMINAL"
        Path(f"{late}.manifest.json").write_text(json.dumps(d))
        original = Path.open
        opens = []
        def opened(path, *args, **kwargs):
            if path.suffix == ".parquet" and not path.name.endswith(".keys.parquet"):
                opens.append(str(path))
                raise AssertionError("early label-bearing payload open")
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", opened), self.assertRaises(ValueError):
            fit.strict_training_groups([("safesyn", self.path, 1, 0, "withinref,both"),
                                        ("late", late, 1, 0, "withinref,both")])
        self.assertEqual(opens, [])

    def test_swapped_map_and_positional_fallback_refused(self):
        d=copy.deepcopy(self.d);d["research_palette"]["columns"]["1825"]="f1825"
        with self.assertRaisesRegex(ValueError,"map"):
            owner.admit_declaration(d)

    def test_subset_or_reordering_refused(self):
        for keep in (owner.ARM_IDS[:-1],list(reversed(owner.ARM_IDS))):
            with self.assertRaisesRegex(ValueError,"462"):
                owner.feature_columns(self.path,keep)

    def test_nonfinite_named_primary_refused(self):
        table=pq.read_table(self.path)
        i=table.schema.get_field_index("palette_f1866")
        for bad in (float("nan"),None):
            with self.subTest(value=bad):
                altered=table.set_column(i,"palette_f1866",pa.array([bad]*len(table),type=pa.float32()))
                pq.write_table(altered,self.path,compression="zstd")
                self.d["table_sha256"]=common.sha(self.path);self.save(self.d)
                with self.assertRaisesRegex(ValueError,"palette_f1866.*not finite"):
                    common.refuse_nonfinite_kept([self.path],owner.ARM_IDS)

    def test_recipe_has_only_registered_difference(self):
        recipe=common.recipe_of("sel:59f0bbc2f290@h32:H128:cv16:cf98")
        receipt={"research_palette":owner.CONTRACT,"width":1867}
        owner.admit_recipe(receipt,recipe,owner.ARM_IDS,"N",True,True)
        bad={**recipe,"hidden":64}
        with self.assertRaisesRegex(ValueError,"unchanged E30"):
            owner.admit_recipe(receipt,bad,owner.ARM_IDS,"N",True,True)

    def test_recipe_preflight_checks_all_projected_keys_and_split_roles(self):
        import v2_d1_prepare as prep
        import v2_human_role as roles
        import v2_teacher as teacher
        # D1 preparation starts from a fresh, unmodified admitted fixture.
        self.f=fixtures.RecipeAdmissionTests()
        self.f.setUp();self.addCleanup(self.f.doCleanups);self.f.admit()
        decision=self.f.root/"decision.json"
        self.f.json(decision,{"schema":"shippath-human-role-decision-v1",
            "decision_id":"SHIPPATH-human-production-role","state":"approved","decided_by":"TEST",
            "allowed_use":"qualified-recipe-training","sources":list(roles.PRODUCTION_SOURCES),
            "ledger_commit":roles.LEDGER_COMMIT,"source_receipt_sha256":common.sha(self.f.wide/"receipt.json"),
            "source_frozen_sha256":common.sha(self.f.source/"wide/frozen.json")})
        root=self.f.root/"projected-d1"
        prep.prepare(self.f.out,self.f.source,self.f.bank,self.f.root/"stage",root,
                     decision,self.f.root/"logical")
        receipt_path=root/"wide/main/real/receipt.json"
        receipt=json.loads(receipt_path.read_text())
        receipt.update(width=1867,research_palette=owner.CONTRACT)
        paths=[]
        for name,leg in receipt["legs"].items():
            for split in ("fit","dev"):
                if split in leg:
                    paths.append((root/leg[split]["rel"],
                        ("D1-" if name.startswith("human_") else "TRAIN-oracle-")+
                        ("fit" if split=="fit" else "development")))
        pool=root/"e15/coverage_pool.parquet"
        keys=pq.read_table(teacher.key_path(pool))
        keys=keys.append_column("__index_level_0__",pa.array(list(range(len(keys))),type=pa.int64()))
        pq.write_table(keys,teacher.key_path(pool))
        paths.append((pool,"TRAIN-ordinal"))
        for path,role in paths:
            sp=Path(f"{path}.manifest.json");d=json.loads(sp.read_text())
            keys=pq.read_table(teacher.key_path(path))
            if role.startswith("TRAIN-oracle") and set(keys["member_set"].to_pylist())=={"cid22"}:
                keys=keys.set_column(keys.schema.get_field_index("member_set"),"member_set",
                                     pa.array(["cid22_train"]*len(keys)))
                pq.write_table(keys,teacher.key_path(path))
            d.update(feature_set_id=owner.FEATURE_SET_ID,keys_sha256=common.sha(teacher.key_path(path)),
                     row_keys_sha256=teacher.row_keys_sha(keys))
            d["research_palette"]={**copy.deepcopy(self.d["research_palette"]),"role":role,
                "member_sets":(["coverage_pool"] if role=="TRAIN-ordinal" else sorted(set(keys["member_set"].to_pylist()))),
                "key_domain":("coverage-selection-ordinal" if role=="TRAIN-ordinal" else "member-pair-observation"),
                "inherited_table_sha256":d["table_sha256"]}
            self.f.json(sp,d)
        for leg in receipt["legs"].values():
            for split in ("fit","dev"):
                if split in leg:leg[split]["manifest_sha256"]=common.sha(Path(f'{root/leg[split]["rel"]}.manifest.json'))
        self.f.json(receipt_path,receipt)
        def seal():
            fp=root/"wide/frozen.json";f=json.loads(fp.read_text())
            f["wide_receipts"]["main/real"]=common.sha(receipt_path)
            for rel in f["auxiliary_files"]:f["auxiliary_files"][rel]=common.sha(root/rel)
            self.f.json(fp,f)
        seal()
        original_open,original_read=Path.open,pq.read_table
        def guard(path):
            if str(path).endswith(".parquet") and not str(path).endswith(".keys.parquet"):
                raise AssertionError("preflight opened a label-bearing payload")
        def opened(path,*args,**kwargs):guard(path);return original_open(path,*args,**kwargs)
        def read(path,*args,**kwargs):guard(path);return original_read(path,*args,**kwargs)
        with patch.object(Path,"open",opened),patch.object(pq,"read_table",read):
            roles.preflight_recipe(root,root/"human_role_decision.json","kadid")
        human=receipt["legs"]["human_without_kadid"]["fit"]
        hsp=Path(f'{root/human["rel"]}.manifest.json')
        original=json.loads(hsp.read_text());bad=copy.deepcopy(original)
        bad["research_palette"]["role"]="D1-development"
        self.f.json(hsp,bad);human["manifest_sha256"]=common.sha(hsp)
        self.f.json(receipt_path,receipt);seal()
        with patch.object(Path,"open",opened),patch.object(pq,"read_table",read),self.assertRaisesRegex(ValueError,"split"):
            roles.preflight_recipe(root,root/"human_role_decision.json","kadid")
        self.f.json(hsp,original);human["manifest_sha256"]=common.sha(hsp)
        self.f.json(receipt_path,receipt);seal()
        # Poison the last population while keeping byte/ordered-key pins self-consistent.
        keys=pq.read_table(teacher.key_path(pool))
        keys=keys.append_column("member_set",pa.array(["kadid_terminal"]*len(keys)))
        pq.write_table(keys,teacher.key_path(pool))
        sp=Path(f"{pool}.manifest.json");d=json.loads(sp.read_text())
        d.update(keys_sha256=common.sha(teacher.key_path(pool)),row_keys_sha256=teacher.row_keys_sha(keys))
        self.f.json(sp,d);seal()
        with patch.object(Path,"open",opened),patch.object(pq,"read_table",read),self.assertRaisesRegex(ValueError,"coverage"):
            roles.preflight_recipe(root,root/"human_role_decision.json","kadid")


class ControlParityMetadata(unittest.TestCase):
    def repro(self):
        return {"argv":["/old/train", "--group", "human:/old/table:32:0:withinref,rank",
                        "--out", "/old/best", "--keep-features", "/old/keep"],
                "inputs":[{"path":"/old/table", "sha256":"a"*64, "train_w":32}],
                "table_admission":{"tables":[{"path":"/old/table",
                    "source":"/old/table: stored table declaration", "formula_revision":5}]},
                "sample_seed":101, "init_seed":1101, "epochs":120, "pairs_per_epoch":50000}

    def test_only_locations_normalize(self):
        before=self.repro()
        after=json.loads(json.dumps(before).replace("/old/", "/new/"))
        after["timestamp_epoch"]=999
        self.assertEqual(normalized_control_repro(before), normalized_control_repro(after))

    def test_hash_weight_seed_budget_and_admission_remain_compared(self):
        before=self.repro()
        for branch,key,value in [("inputs","sha256","b"*64),("inputs","train_w",31),
                                 ("table_admission","formula_revision",4),
                                 (None,"sample_seed",102),(None,"epochs",119)]:
            with self.subTest(key=key):
                after=copy.deepcopy(before)
                if branch=="inputs":after[branch][0][key]=value
                elif branch=="table_admission":after[branch]["tables"][0][key]=value
                else:after[key]=value
                self.assertNotEqual(normalized_control_repro(before), normalized_control_repro(after))


if __name__=="__main__":unittest.main()
