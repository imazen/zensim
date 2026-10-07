"""E32 strict consumer refusals and named-column reads use synthetic data only."""
import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import test_shippath2_admission as fixtures
import e32_palette as owner
import v2_common as common
import v2_lodo_mlp as fit


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
        pq.write_table(table, self.path)
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

    def test_refusals_open_no_label_payload(self):
        original=Path.open
        opens=[]
        def opened(path,*args,**kwargs):
            if path.suffix==".parquet" and not path.name.endswith(".keys.parquet"):
                opens.append(str(path));raise AssertionError("early label-bearing payload open")
            return original(path,*args,**kwargs)
        for key,bad in [("role","TERMINAL"),("member_sets",["jpeg-aic-heldout"]),
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
        table=table.set_column(i,"palette_f1866",pa.array([float("nan")]*len(table),type=pa.float32()))
        pq.write_table(table,self.path)
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


if __name__=="__main__":unittest.main()
