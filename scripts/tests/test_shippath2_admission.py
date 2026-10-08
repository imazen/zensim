"""Full-recipe byte/row identity and strict-route negative controls, synthetic only."""
import contextlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rev4_featpot"))
import e15_coverage as coverage
import v2_lodo_mlp as fit
import v2_teacher as teacher
import v2c_wide as owner
from v2_common import SOURCE_ORDER, SOURCES, TEACHERS, human_dev
from v2_human_role import PRODUCTION_SOURCES, LEDGER_COMMIT


class RecipeAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="shippath2-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.bank, self.out = [self.root / p for p in ("source", "bank", "view")]
        self.wide = self.source / "wide/main/real"
        self.wide.mkdir(parents=True)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        binary = self.root / "extractor"
        binary.write_bytes(b"synthetic extractor binding")
        profile = owner.rev5_profile("rev5_localwin", "basic+peaks+v2@w1825/rev5_localwin#36c3f3af",
                                    owner.sha(binary), "b" * 40, 1853)
        self.stack.enter_context(patch.object(owner, "PROFILE", profile))
        self.receipt = {"schema": owner.SCHEMA, "family": "main", "variant": "real", "formula_revision": 5,
                        "feature_set_id": profile.feature_set_id, "era": profile.era, "extras": [], "bank": {},
                        "legs": {}, "complete": True, "width": 1853}
        for name in (*dict.fromkeys(n for s in SOURCE_ORDER for n in SOURCES[s]), *(v[0] for v in TEACHERS.values())):
            directory = self.bank / name
            directory.mkdir(parents=True)
            for filename in ("features.parquet", "keys.parquet"):
                (directory / filename).write_bytes(b"synthetic bank receipt")
            manifest = {"set": name, "schema": profile.schema, "feature_width": owner.WIDTH,
                        "formula_revision": "Rev5", "era_label": profile.era, "feature_set_id": profile.feature_set_id,
                        "binary_sha256": profile.binary, "build_commit": profile.build, "dtype": "float64",
                        "requested_slot_ranges": [list(r) for r in profile.requested], "input_contract": "legacy-rgb8",
                        "chunks": [{"extractor_manifest": {"producer_binary_sha256": profile.binary,
                                    "feature_set_id": profile.feature_set_id, "formula_revision": "5",
                                    "populated_feature_ids": [i for lo, hi in profile.requested for i in range(lo, hi)]}}]}
            self.json(directory / "_MANIFEST.json", manifest)
            self.receipt["bank"][name] = {"manifest_sha256": owner.sha(directory / "_MANIFEST.json"),
                                         "features_sha256": owner.sha(directory / "features.parquet"),
                                         "keys_sha256": owner.sha(directory / "keys.parquet")}
        frames = {}
        for name in (*SOURCE_ORDER, *TEACHERS):
            refs = [f"{name}|ref{i // 2}" for i in range(24)]
            frame = pd.DataFrame({"pair_key": [f"{name}|pair{i}" for i in range(24)], "source_row_id": range(24),
                                  "ref_basename": refs, "member_set": SOURCES[name][0] if name in SOURCES else ("cid22_train" if name == "cid22" else name), "target": np.arange(24, dtype=float)})
            frames[name] = frame
            leg = {}
            for split in (("full",) if name in SOURCES else ("fit", "dev")):
                stem = name if split == "full" else f"{name}_{split}"
                leg[split] = self.table(stem, frame)
                kp = self.wide / f"{stem}.keys.parquet"
                frame.to_parquet(kp, index=False)
                if split == "full":
                    leg["keys_sha256"] = owner.sha(kp)
                else:
                    leg[split]["keys_sha256"] = owner.sha(kp)
            self.receipt["legs"][name] = leg
        for name in ("human_all", *(f"human_without_{h}" for h in SOURCE_ORDER)):
            members = [h for h in SOURCE_ORDER if name != f"human_without_{h}"]
            self.receipt["legs"][name] = {}
            for split in ("fit", "dev"):
                parts = [f.loc[[human_dev(r) == (split == "dev") for r in f.ref_basename]]
                         for h in members for f in [frames[h]]]
                self.receipt["legs"][name][split] = self.table(f"{name}_{split}", pd.concat(parts, ignore_index=True))
        self.json(self.source / "wide/keep_lists.json", {"schema": "rev4-featpot-v2-keeplists-v2", "specs": {}})
        self.freeze()
        pooldir = self.source / "e15"
        (pooldir / "extract").mkdir(parents=True)
        keys = pa.table({"ladder": ["light|r0"] * 2 + ["noise|r1"] * 2 + ["new|r2"] * 2,
                         "family": ["light"] * 2 + ["noise"] * 2 + ["new"] * 2,
                         "source_filename": ["r0"] * 2 + ["r1"] * 2 + ["r2"] * 2,
                         "type": ["16"] * 2 + ["11"] * 2 + ["lbw"] * 2,
                         "severity_level": [1, 2] * 3, "severity": [1., 2.] * 3, "sign": [1.] * 6, "__index_level_0__": list(range(6))})
        pq.write_table(keys, pooldir / "coverage_pool.keys.parquet")
        pool = pa.table({"ref_basename": keys["ladder"], "human_score": [-1., -2., -1., -2., -1., -2.],
                         "f0": np.arange(6, dtype=np.float32), "f719": [float("nan")] * 6})
        pq.write_table(pool, pooldir / "coverage_pool.parquet")
        pq.write_table(keys.drop(["ladder"]), pooldir / "selection.parquet")
        self.json(pooldir / "coverage_pool.parquet.manifest.json",
                  {"source_bank_feature_set_id": profile.feature_set_id, "formula_revision": 5})
        self.json(pooldir / "extract/features.csv.manifest.json", {"feature_set_id": profile.feature_set_id,
                  "producer_binary_sha256": profile.binary, "era_label": profile.era, "formula_revision": "5",
                  "populated_feature_ids": [i for lo, hi in profile.requested for i in range(lo, hi)]})
        poolsha, keysha = owner.sha(pooldir / "coverage_pool.parquet"), owner.sha(pooldir / "coverage_pool.keys.parquet")
        self.json(pooldir / "coverage_pool.manifest.json", {"rows": 6, "sha256": poolsha, "keys_sha256": keysha,
                  "extractor_sha256": profile.binary, "era": profile.era, "formula_revision": 5})
        self.stack.enter_context(patch.multiple(teacher, POOL_SHA_REV5=poolsha, POOL_KEYS_SHA=keysha))
        self.stack.enter_context(patch.object(coverage.e14, "extractor", return_value={"sha": profile.binary,
            "era": profile.era, "requested": profile.requested, "bin": binary, "build": profile.build}))

    @staticmethod
    def json(path, value):
        path.write_text(json.dumps(value))

    def table(self, stem, frame):
        path = self.wide / f"{stem}.parquet"
        pq.write_table(pa.table({"ref_basename": frame.ref_basename, "human_score": frame.target,
                                "f0": np.arange(len(frame), dtype=np.float32), "f719": [float("nan")] * len(frame)}), path)
        sp = Path(f"{path}.manifest.json")
        self.json(sp, {"source_bank_feature_set_id": owner.PROFILE.feature_set_id, "formula_revision": 5})
        return {"rel": str(path.relative_to(self.source)), "sha256": owner.sha(path), "manifest_sha256": owner.sha(sp),
                "rows": len(frame), "references": frame.ref_basename.nunique()}

    def freeze(self):
        path = self.wide / "receipt.json"
        self.json(path, self.receipt)
        self.json(self.source / "wide/frozen.json", {"wide_receipts": {"main/real": owner.sha(path)},
            "keep_lists_sha256": owner.sha(self.source / "wide/keep_lists.json")})

    def admit(self):
        owner.admit_recipe(self.bank, self.source, self.out)

    def group(self, stem):
        return [(stem, self.out / f"wide/main/real/{stem}.parquet", 1., 0., "withinref,rank")]

    def test_full_view_preserves_bytes_keys_and_pending_roles(self):
        self.admit()
        tables = list((self.out / "wide/main/real").glob("*.parquet"))
        tables = [p for p in tables if not p.name.endswith(".keys.parquet")]
        self.assertEqual(len(tables), 21)
        for dest in tables:
            self.assertEqual(owner.sha(dest), owner.sha(self.wide / dest.name))
            keys = pq.read_table(teacher.key_path(dest))
            self.assertNotIn("target", keys.column_names)
            d = json.loads(Path(f"{dest}.manifest.json").read_text())
            self.assertEqual(d["row_keys_sha256"], teacher.row_keys_sha(keys))
            if dest.name.startswith("human_"):
                self.assertEqual(d["data_role_decision_required"], "SHIPPATH-human-production-role")
        from v2_common import load_frozen
        frozen, _ = load_frozen(self.out, training_only=True)
        self.assertEqual(frozen["schema"], "rev5-recipe-admission-freeze-v1")
        with self.assertRaisesRegex(ValueError, "unexpected schema"):
            load_frozen(self.out)
        with self.assertRaisesRegex(ValueError, "fresh"):
            self.admit()

    def test_changed_human_selection_refused_before_output(self):
        kp = self.wide / "kadid.keys.parquet"
        frame = pq.read_table(kp).to_pandas()
        frame.loc[0, "ref_basename"] = "wrong-order"
        frame.to_parquet(kp, index=False)
        self.receipt["legs"]["kadid"]["keys_sha256"] = owner.sha(kp)
        self.freeze()
        with self.assertRaisesRegex(ValueError, "selection/order"):
            self.admit()
        self.assertFalse(self.out.exists())

    def test_changed_pool_refused_before_output(self):
        (self.source / "e15/coverage_pool.parquet").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "pinned"):
            self.admit()
        self.assertFalse(self.out.exists())

    def test_cf98_preserves_values_and_derivation_metadata(self):
        self.admit()
        selected, record = teacher.coverage_leg(0x98, self.root, admitted_root=self.out)
        source = pq.read_table(self.source / "e15/coverage_pool.parquet")
        expected = source.take(pa.array([0, 1, 4, 5]))
        actual = pq.read_table(selected)
        for c in actual.column_names:
            if c.startswith("f"):
                self.assertTrue(np.array_equal(actual[c].to_numpy().view(np.uint32), expected[c].to_numpy().view(np.uint32)))
            else:
                self.assertEqual(actual[c].to_pylist(), expected[c].to_pylist())
        self.assertEqual(record["row_selection_sha256"], teacher.selection_sha([0, 1, 4, 5]))
        self.assertEqual(record["targets_changed"], 0)
        fit.strict_training_groups([("coverage", selected, 1., 0., "withinref,rank")])

    def test_pending_human_decision_refuses_before_trainer_or_output(self):
        self.admit()
        dest = self.root / "run"
        with self.assertRaisesRegex(ValueError, "PENDING"):
            fit.train_and_select(self.group("human_all_fit"), 1101, 101, 1853, self.root / "keep", "N", dest,
                                 strict_admission=True)
        self.assertFalse(dest.exists())

    def test_fixture_decision_must_bind_original_receipt_and_sources(self):
        self.admit()
        decision = {"schema": "shippath-human-role-decision-v1", "decision_id": "SHIPPATH-human-production-role",
                    "state": "approved", "decided_by": "SYNTHETIC TEST ONLY", "allowed_use": "qualified-recipe-training",
                    "sources": list(PRODUCTION_SOURCES), "ledger_commit": LEDGER_COMMIT,
                    "source_receipt_sha256": owner.sha(self.wide / "receipt.json")}
        p = self.root / "fixture-only-decision.json"
        self.json(p, decision)
        records = fit.strict_training_groups(self.group("human_without_aic3_fit"), p)
        self.assertEqual(records[0]["data_role_decision_sha256"], owner.sha(p))
        decision["source_receipt_sha256"] = "wrong"
        self.json(p, decision)
        with self.assertRaisesRegex(ValueError, "PENDING"):
            fit.strict_training_groups(self.group("human_all_fit"), p)

    def test_aic_family_decisions_and_tables_refused_before_payload(self):
        self.admit()
        good = {"schema": "shippath-human-role-decision-v1", "decision_id": "SHIPPATH-human-production-role",
                "state": "approved", "decided_by": "SYNTHETIC TEST ONLY", "allowed_use": "qualified-recipe-training",
                "sources": list(PRODUCTION_SOURCES), "ledger_commit": LEDGER_COMMIT,
                "source_receipt_sha256": owner.sha(self.wide / "receipt.json")}
        p = self.root / "fixture-only-decision.json"
        self.json(p, good)
        original_sha = fit.sha
        def tripwire(path):
            if str(path).endswith(".parquet"):
                raise AssertionError("human payload opened before refusal")
            return original_sha(path)
        with patch.object(fit, "sha", side_effect=tripwire):
            with self.assertRaisesRegex(ValueError, "AIC"):
                fit.strict_training_groups(self.group("human_all_fit"), p)
            for forbidden in ("aic3", "aic4", "jpeg-aic", "sdr25", "AIC-3"):
                self.json(p, {**good, "sources": [*PRODUCTION_SOURCES, forbidden]})
                with self.assertRaisesRegex(ValueError, "AIC"):
                    fit.strict_training_groups(self.group("human_without_aic3_fit"), p)

    def test_strict_and_historical_argv_and_tamper_refusal(self):
        self.admit()
        groups = self.group("safesyn_fit")
        argv = fit.train_command(groups, 1101, 101, 1853, self.root / "keep", "N", self.root / "model")
        self.assertIn("--historical-replay", argv)
        with patch.dict(fit.os.environ, {"REV4_V2_BIN_DIR": str(fit.TRAINER.parent)}):
            strict = fit.train_command(groups, 1101, 101, 1853, self.root / "keep", "N", self.root / "model", strict_admission=True)
        i = argv.index("--historical-replay")
        self.assertEqual(argv[:i] + argv[i + 2:], strict)
        teacher.key_path(groups[0][1]).write_bytes(b"changed keys")
        with self.assertRaisesRegex(ValueError, "key file changed"):
            fit.train_command(groups, 1101, 101, 1853, self.root / "keep", "N", self.root / "model", strict_admission=True)

    def test_curated_source_tamper_and_destination_reuse_refused(self):
        self.admit()
        src = self.group("safesyn_fit")[0][1]
        target = pq.read_table(src, columns=["human_score"])["human_score"].to_numpy()
        keep = np.ones(len(target), dtype=bool)
        dest = self.root / "curated.parquet"
        teacher.write_curated(src, dest, keep, target)
        with self.assertRaisesRegex(ValueError, "fresh"):
            teacher.write_curated(src, dest, keep, target)
        src.write_bytes(b"changed table")
        with self.assertRaises(Exception):
            teacher.write_curated(src, self.root / "bad.parquet", keep, target)
        self.assertFalse((self.root / "bad.parquet").exists())

    def test_resolved_sealed_alias_is_refused_before_any_output(self):
        alias = self.root / "alias"
        alias.symlink_to(self.root / "_sealed", target_is_directory=True)
        with self.assertRaises(PermissionError):
            owner.admit_recipe(self.bank, alias, self.out)
        self.assertFalse(self.out.exists())

    def immutable_snapshot(self):
        return {str(p): owner.sha(p) for root in (self.source, self.bank, self.out)
                for p in root.rglob("*") if p.is_file()}

    def test_admission_outputs_refuse_source_bank_and_resolved_aliases(self):
        before = self.immutable_snapshot()
        alias = self.root / "bank-alias"
        alias.symlink_to(self.bank, target_is_directory=True)
        for root in (self.source, self.bank, alias):
            dest = root / "forbidden-admission"
            with self.subTest(root=root):
                with self.assertRaisesRegex(ValueError, "immutable input root"):
                    owner.admit_recipe(self.bank, self.source, dest)
                self.assertFalse(dest.exists())
                self.assertEqual(before, self.immutable_snapshot())

    def test_old_admission_without_bank_binding_is_refused_before_output(self):
        self.admit()
        rp = self.out / "wide/main/real/receipt.json"
        receipt = json.loads(rp.read_text())
        receipt["admission_view"].pop("bank_root")
        self.json(rp, receipt)
        frozen_path = self.out / "wide/frozen.json"
        frozen = json.loads(frozen_path.read_text())
        frozen["admission_view"] = receipt["admission_view"]
        frozen["wide_receipts"]["main/real"] = owner.sha(rp)
        self.json(frozen_path, frozen)
        dest = self.root / "safe-destination"
        with self.assertRaisesRegex(ValueError, "bank_root; regenerate"):
            fit.strict_output_preflight(self.out, dest)
        self.assertFalse(dest.exists())

    def test_pool_admission_refuses_original_instrument_and_symlink(self):
        before = self.immutable_snapshot()
        alias = self.root / "source-alias"
        alias.symlink_to(self.source, target_is_directory=True)
        for root in (self.source, alias):
            dest = root / "forbidden-pool"
            with self.assertRaisesRegex(ValueError, "immutable input root"):
                coverage.admit_pool(self.source / "e15", dest)
            self.assertFalse(dest.exists())
            self.assertEqual(before, self.immutable_snapshot())

    def test_strict_cli_outputs_and_scratch_refuse_all_input_roots(self):
        import os
        import v2_common as common
        import v2_confirm_fit as confirm
        self.admit()
        before = self.immutable_snapshot()
        alias = self.root / "source-alias"
        alias.symlink_to(self.source, target_is_directory=True)
        safe_scratch = self.root / "safe-scratch"
        safe_scratch.mkdir()
        for module in (fit, confirm):
            for root in (self.source, self.bank, self.out, alias):
                for scratch_case in (False, True):
                    forbidden = root / f"forbidden-{module.__name__}-{scratch_case}"
                    dest = self.root / "safe-dest" if scratch_case else forbidden
                    scratch = root if scratch_case else safe_scratch
                    argv = ["test", "--root", str(self.out), "--strict-admission", "--train-only",
                            "--dest", str(dest), "--spec",
                            f"sel:{common.selection_id([0])}@h32:H128:cv16:cf98", "--columns", "0",
                            "--head", "N", "--seed-index", "0"]
                    if module is fit:
                        argv += ["--heldout", "tid2013"]
                    with self.subTest(module=module.__name__, root=root, scratch=scratch_case), \
                         patch.object(sys, "argv", argv), patch.object(common, "V2", self.out), \
                         patch.object(fit, "V2", self.out), patch.object(confirm, "V2", self.out), \
                         patch.dict(os.environ, {"TMPDIR": str(scratch)}), \
                         patch.object(fit, "run", side_effect=AssertionError("trainer must not run")):
                        with self.assertRaisesRegex(ValueError, "immutable input root"):
                            module.main()
                        self.assertFalse(dest.exists())
                        self.assertFalse(forbidden.exists())
                        self.assertEqual(before, self.immutable_snapshot())

    def test_derived_coverage_and_curated_outputs_refuse_all_input_roots(self):
        self.admit()
        before = self.immutable_snapshot()
        src = self.group("safesyn_fit")[0][1]
        target = pq.read_table(src, columns=["human_score"])["human_score"].to_numpy()
        alias = self.root / "view-alias"
        alias.symlink_to(self.out, target_is_directory=True)
        for root in (self.source, self.bank, self.out, alias):
            with self.subTest(root=root):
                with self.assertRaisesRegex(ValueError, "immutable input root"):
                    teacher.coverage_leg(0x98, root, admitted_root=self.out)
                with self.assertRaisesRegex(ValueError, "immutable input root"):
                    teacher.write_curated(src, root / "forbidden-curated.parquet",
                                          np.ones(len(target), dtype=bool), target)
                with (self.assertRaisesRegex(ValueError, "immutable input root"),
                      patch.object(fit, "run", side_effect=AssertionError("trainer must not run"))):
                    fit.train_and_select(self.group("safesyn_fit"), 1103, 101, 1853,
                        self.root / "keep", "N", root, strict_admission=True)
                self.assertEqual(before, self.immutable_snapshot())

    def test_safe_cli_outputs_still_refuse_pending_human_role(self):
        import os
        import v2_common as common
        import v2_confirm_fit as confirm
        self.admit()
        before = self.immutable_snapshot()
        scratch = self.root / "safe-scratch"
        scratch.mkdir()
        original_read = pq.read_table

        def label_free_human_read(path, *args, **kwargs):
            if Path(path).name.startswith(("human_", *SOURCE_ORDER)):
                columns = kwargs.get("columns")
                self.assertIsNotNone(columns)
                self.assertFalse(set(columns) & {"target", "human_score"})
            return original_read(path, *args, **kwargs)

        for module in (fit, confirm):
            dest = self.root / f"safe-{module.__name__}"
            argv = ["test", "--root", str(self.out), "--strict-admission", "--train-only",
                    "--dest", str(dest), "--spec", f"sel:{common.selection_id([0])}@h32:H128:cv16:cf98",
                    "--columns", "0", "--head", "N", "--seed-index", "0"]
            if module is fit:
                argv += ["--heldout", "tid2013"]
            with (patch.object(sys, "argv", argv), patch.object(common, "V2", self.out),
                  patch.object(fit, "V2", self.out), patch.object(confirm, "V2", self.out),
                  patch.dict(os.environ, {"TMPDIR": str(scratch)}),
                  patch.object(pq, "read_table", side_effect=label_free_human_read),
                  patch.object(fit, "run", side_effect=AssertionError("trainer must not run"))):
                with self.assertRaisesRegex(ValueError, "PENDING SHIPPATH-human-production-role"):
                    module.main()
                self.assertFalse(dest.exists())
                self.assertEqual(before, self.immutable_snapshot())
                self.assertEqual(list(scratch.iterdir()), [])

    def test_last_epoch_wins_even_when_development_best_is_earlier(self):
        self.admit()
        dest = self.root / "cell"
        dest.mkdir()

        def synthetic_run(argv, log):
            (dest / "refit/best.bin").write_bytes(b"earlier best weights")
            (dest / "ckpt/ckpt_epoch002.bin").write_bytes(b"last epoch weights")
            log.write_text("\n".join(f"epoch {i} | val(geomean3)={v:.4f}" for i, v in enumerate([.9, .5, .1])))
            self.assertNotIn("--historical-replay", argv)

        with patch.multiple(fit, EPOCHS=3, LOG_EVERY=1), patch.object(fit, "run", side_effect=synthetic_run):
            bake, _, selected = fit.train_and_select(self.group("safesyn_fit"), 1101, 101, 1853,
                self.root / "keep", "N", dest, strict_admission=True)
        self.assertEqual(selected["selected_epoch"], 2)
        self.assertEqual(selected["best_epoch_by_curve"], 0)
        self.assertEqual(bake.read_bytes(), b"last epoch weights")


if __name__ == "__main__":
    unittest.main()
