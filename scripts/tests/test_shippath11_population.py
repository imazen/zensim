"""Population refusal precedes all table payload opens, including freeze auxiliaries."""
import contextlib
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import pyarrow as pa
import pyarrow.parquet as pq
import test_shippath2_admission as fixtures
import v2_d1_prepare as prep
import v2_lodo_mlp as fit
from v2_human_role import PRODUCTION_SOURCES, LEDGER_COMMIT


class PopulationOrder(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.RecipeAdmissionTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.f.admit()
        self.decision = self.f.root / "d1.json"
        self.f.json(self.decision, {"schema":"shippath-human-role-decision-v1",
            "decision_id":"SHIPPATH-human-production-role", "state":"approved", "decided_by":"TEST",
            "allowed_use":"qualified-recipe-training", "sources":list(PRODUCTION_SOURCES),
            "ledger_commit":LEDGER_COMMIT, "source_receipt_sha256":prep.sha(self.f.wide/"receipt.json"),
            "source_frozen_sha256":prep.sha(self.f.source/"wide/frozen.json")})
        self.opens = []

    @contextlib.contextmanager
    def tripwire(self):
        original_open, original_read = Path.open, pq.read_table
        def guard(p):
            if str(p).endswith(".parquet") and not str(p).endswith(".keys.parquet"):
                self.opens.append(str(p))
                raise AssertionError("table payload opened before population refusal")
        def opened(p, *args, **kwargs):
            guard(p)
            return original_open(p, *args, **kwargs)
        def read(p, *args, **kwargs):
            guard(p)
            return original_read(p, *args, **kwargs)
        with patch.object(Path,"open",opened), patch.object(pq,"read_table",read):
            yield
        self.assertEqual(self.opens, [])

    def poison(self, bank=False):
        p = self.f.out/"wide/main/real/human_without_aic3_fit.parquet"
        sp = Path(f"{p}.manifest.json")
        d = json.loads(sp.read_text())
        if bank:
            d["bank_manifest_sha256"]["jpeg-aic-heldout"] = "0"*64
        else:
            kp = prep.key_path(p)
            keys = pq.read_table(kp)
            members = keys["member_set"].to_pylist()
            members[0] = "jpeg-aic-heldout"
            keys = keys.set_column(keys.schema.get_field_index("member_set"), "member_set", pa.array(members))
            pq.write_table(keys,kp)
            d["keys_sha256"] = prep.sha(kp)
            d["row_keys_sha256"] = prep.row_keys_sha(keys)
        self.f.json(sp,d)
        receipt_path = self.f.out/"wide/main/real/receipt.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["legs"]["human_without_aic3"]["fit"]["manifest_sha256"] = prep.sha(sp)
        self.f.json(receipt_path,receipt)
        return p

    def test_strict_cli_aic_fold_opens_no_payload(self):
        argv = ["fit", "--strict-admission", "--train-only", "--heldout", "aic3", "--spec", "R0",
                "--head", "N", "--seed-index", "0", "--dest", str(self.f.root/"dest")]
        with self.tripwire(), patch.object(sys,"argv",argv), patch.object(fit,"V2",self.f.out):
            with self.assertRaisesRegex(ValueError,"AIC"):
                fit.main()
        self.assertFalse((self.f.root/"dest").exists())

    def test_lower_owner_aic_key_opens_no_payload(self):
        p = self.poison()
        with self.tripwire(), self.assertRaisesRegex(ValueError,"AIC"):
            fit.strict_training_groups([("safesyn",self.f.out/"wide/main/real/safesyn_fit.parquet",1,0,"rank"),
                                        ("human",p,1,0,"rank")],self.decision)

    def test_prepare_aic_key_opens_no_payload(self):
        self.poison()
        with self.tripwire(), self.assertRaisesRegex(ValueError,"AIC"):
            prep.prepare(self.f.out,self.f.source,self.f.bank,self.f.root/"stage",self.f.root/"new",
                         self.decision,Path("/var/tmp/rev4-featpot/test-d1"))
        self.assertFalse((self.f.root/"stage").exists())

    def test_prepare_aic_bank_opens_no_payload(self):
        self.poison(bank=True)
        with self.tripwire(), self.assertRaisesRegex(ValueError,"AIC"):
            prep.prepare(self.f.out,self.f.source,self.f.bank,self.f.root/"stage",self.f.root/"new",
                         self.decision,Path("/var/tmp/rev4-featpot/test-d1"))
