"""D1 derivation and registered cell shape; synthetic inputs only."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pyarrow.parquet as pq
import test_shippath2_admission as fixtures
from v2_human_role import PRODUCTION_SOURCES, LEDGER_COMMIT, preflight_recipe
import v2_d1_prepare as prep
import v2_production_pack as pack
from e30_four_source import grid, SPEC


class D1Preparation(unittest.TestCase):
    def test_four_source_derivation_and_portable_archive_auxiliaries(self):
        fixture = fixtures.RecipeAdmissionTests()
        fixture.setUp()
        try:
            fixture.admit()
            decision = fixture.root / "d1.json"
            fixture.json(decision, {"schema": "shippath-human-role-decision-v1", "decision_id": "SHIPPATH-human-production-role",
                "state": "approved", "decided_by": "TEST ONLY", "allowed_use": "qualified-recipe-training",
                "sources": list(PRODUCTION_SOURCES), "ledger_commit": LEDGER_COMMIT,
                "source_receipt_sha256": prep.sha(fixture.wide / "receipt.json"),
                "source_frozen_sha256": prep.sha(fixture.source / "wide/frozen.json")})
            out = fixture.root / "d1"
            original_open = Path.open
            forbidden = [fixture.out / f"wide/main/real/{stem}.parquet"
                         for stem in ("aic3", "human_all_fit", "human_all_dev")]
            def tripwire(path, *args, **kwargs):
                if path in forbidden:
                    raise AssertionError("AIC payload opened")
                return original_open(path, *args, **kwargs)
            with patch.object(Path, "open", tripwire):
                report = prep.prepare(fixture.out, fixture.source, fixture.bank, fixture.root / "stage", out,
                                      decision, Path("/var/tmp/rev4-featpot/v2d1"))
            self.assertEqual(len(report["derived_parity"]), 8)
            self.assertFalse(report["aic_payloads_read"])
            preflight_recipe(out, out / "human_role_decision.json")
            for source in PRODUCTION_SOURCES:
                preflight_recipe(out, out / "human_role_decision.json", source)
            with self.assertRaisesRegex(ValueError, "AIC"):
                preflight_recipe(out, out / "human_role_decision.json", "aic3")
            receipt = json.loads((out / "wide/main/real/receipt.json").read_text())
            self.assertNotIn("aic3", receipt["legs"])
            from v2c_pack import members_for
            members = members_for(out, "lodo", [("main", "real")])
            self.assertIn("human_role_decision.json", members)
            self.assertIn("e15/coverage_pool.parquet", members)
            self.assertFalse(any("aic3" in name for name in members))
        finally:
            fixture.doCleanups()

    def test_registered_grids_keep_four_folds_and_final_epoch_recipe(self):
        root, dest = Path("/var/tmp/rev4-featpot/v2d1"), Path("/var/tmp/rev4-featpot/e30-results/cells")
        cells = grid(root, dest, [13, 14])
        self.assertEqual(len(cells), 40)
        self.assertEqual({c["argv"][c["argv"].index("--heldout")+1] for c in cells}, set(PRODUCTION_SOURCES))
        for c in cells:
            self.assertIn("--strict-admission", c["argv"])
            self.assertIn(SPEC, c["argv"])
            self.assertNotIn("aic3", c["argv"])
        production = grid(root, dest, [13, 14], True)
        self.assertEqual(len(production), 3)
        self.assertEqual([c["argv"][c["argv"].index("--seed-index")+1] for c in production], ["0", "1", "2"])
        self.assertTrue(all("--pack-production" in c["argv"] for c in production))

    def test_canonical_pack_uses_explicit_train_anchor_after_densify(self):
        calls = []
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with patch.object(pack, "strict_training_groups", side_effect=lambda g: calls.append("admit")), \
                 patch.object(pack, "dense_bake", side_effect=lambda b, d: calls.append("densify") or root / "dense.bin"), \
                 patch.object(pack, "run", side_effect=lambda c, l: calls.append(c)), patch.object(pack, "sha", return_value="test"):
                pack.pack_production(root / "last.bin", root, root / "cid22_fit.parquet")
            self.assertEqual(calls[:2], ["admit", "densify"])
            cmd = calls[2]
            self.assertEqual(cmd[cmd.index("--dtype")+1], "f16")
            self.assertEqual(cmd[cmd.index("--anchor")+1], str(root / "cid22_fit.parquet"))
            self.assertNotIn("--historical-replay", cmd)


if __name__ == "__main__":
    unittest.main()
