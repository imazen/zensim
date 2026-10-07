"""Exercise the actual E26 admission/packer owners on synthetic receipts; never fit."""
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/rev4_featpot"))
import e21_cheap_recipe as e21
import v2_common
import v2_teacher
import v2c_pack
import v2_lodo_mlp
import e24_rev5


class HdrLegAdmission(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR", "/var/tmp"), prefix="e26-fixture-")
        self.root = Path(self.tmp.name)
        self.previous = v2_common.V2
        v2_common.V2 = self.root
        self.path = self.root / "wide/main/real/hdr_fit.parquet"
        self.path.parent.mkdir(parents=True)
        self.cols = e21.columns("by_v2fy")
        self.keys = [dict(row_id=i, role="train", agree=True, ref_path=f"fixture://train/ref{i % 495}",
                          source_family=str(i % 33), hdrvdp3_q_jod=-2.0 if i == 0 else 8.0+i/100000)
                     for i in range(7390)]
        self.manifest = dict(study="E26", role="train", population="agree-only", rows=7390,
                             teacher_sha256=v2_teacher.HDR_TRAIN_SHA,
                             formula_revision=5, target_transform=v2_teacher.HDR_TRANSFORM, requested_ids=self.cols)
        self.table = pa.table({"ref_basename": [k["ref_path"] for k in self.keys],
                               "human_score": [10*k["hdrvdp3_q_jod"] for k in self.keys]})
        self.write()

    def tearDown(self):
        v2_common.V2 = self.previous
        self.tmp.cleanup()

    def write(self):
        pq.write_table(self.table, self.path)
        pq.write_table(pa.Table.from_pylist(self.keys), self.path.with_suffix(".keys.parquet"))
        manifest_path = Path(f"{self.path}.manifest.json")
        manifest_path.write_text(json.dumps(self.manifest))
        keys_sha = v2_common.sha(self.path.with_suffix(".keys.parquet"))
        self.record = {"fit": dict(rel=str(self.path.relative_to(self.root)), sha256=v2_common.sha(self.path),
                                  manifest_sha256=v2_common.sha(manifest_path), keys_sha256=keys_sha),
                       "keys_sha256": keys_sha}

    def test_actual_loader_preserves_negative_and_requires_exact_join(self):
        path, rec = v2_teacher.hdr_leg(self.record, self.cols)
        self.assertEqual(path, self.path)
        self.assertEqual(pq.read_table(path)["human_score"][0].as_py(), -20.0)
        self.assertEqual(rec["target_transform"], v2_teacher.HDR_TRANSFORM)
        self.keys.reverse()
        self.write()
        with self.assertRaisesRegex(ValueError, "row order"):
            v2_teacher.hdr_leg(self.record, self.cols)

    def test_e27_actual_groups_keep_population_weight_target_and_change_only_loss(self):
        groups = {}
        for token, mode in [("hd4", "withinref,rank"), ("hp4", "rank"), ("ha4", "withinref,both")]:
            recipe = v2_common.recipe_of(e24_rev5.CONTROL+":"+token)
            group, admitted = v2_lodo_mlp.hdr_training_group(self.record, self.cols, recipe)
            self.assertEqual(group[4], mode)
            self.assertEqual(group[3], 0)  # no HDR development contribution
            self.assertEqual(admitted["target_transform"], v2_teacher.HDR_TRANSFORM)
            command = v2_lodo_mlp.train_command([group], 1101, 101, 1825,
                self.root / "keep.txt", "N", self.root / "model.bin", recipe)
            self.assertEqual(command[command.index("--group")+1].rsplit(":", 1)[1], mode)
            self.assertEqual(command[command.index("--mse-weight")+1], "1")
            groups[token] = group[:4]
        self.assertEqual(groups["hd4"], groups["hp4"])
        self.assertEqual(groups["hd4"], groups["ha4"])
        self.assertEqual(pq.read_table(self.path)["human_score"][0].as_py(), -20.0)

    def test_e27_token_bounds_duplicates_and_forbidden_val_fail_closed(self):
        for token in ["hp16", "ha16", "hp0", "ha0", "hp4:ha4", "hd4:hp4", "hp4:hd4"]:
            with self.subTest(token=token), self.assertRaises(ValueError):
                v2_common.recipe_of(e24_rev5.CONTROL+":"+token)
        self.manifest["role"] = "val"
        self.write()
        for token in ["hp4", "ha4"]:
            self.assert_no_payload_read(self.record, lambda: v2_lodo_mlp.hdr_training_group(
                self.record, self.cols, v2_common.recipe_of(e24_rev5.CONTROL+":"+token)))

    def test_actual_loader_refuses_role_revision_membership_transform_and_pins(self):
        pristine = copy.deepcopy(self.manifest)
        for name, value in [("study", "other"), ("role", "val"), ("population", "all"),
                            ("formula_revision", 4), ("rows", 7391),
                            ("target_transform", "clip"), ("teacher_sha256", "0"*64),
                            ("requested_ids", self.cols[:-1])]:
            with self.subTest(name=name):
                self.manifest = dict(pristine, **{name: value})
                self.write()
                self.assert_no_payload_read(self.record)
        self.manifest = pristine
        for field, value in [("role", "val"), ("agree", False), ("row_id", 1)]:
            original = self.keys[0][field]
            with self.subTest(key=field):
                self.keys[0][field] = value
                self.write()
                with self.assertRaises(ValueError):
                    v2_teacher.hdr_leg(self.record, self.cols)
            self.keys[0][field] = original
        self.write()
        for part in ["sha256", "manifest_sha256"]:
            bad = copy.deepcopy(self.record)
            bad["fit"][part] = "0"*64
            with self.assertRaisesRegex(ValueError, "changed"):
                v2_teacher.hdr_leg(bad, self.cols)
        bad = copy.deepcopy(self.record)
        bad["dev"] = bad["fit"]
        self.assert_no_payload_read(bad)

    def assert_no_payload_read(self, record, loader=None):
        opened = []
        original = io.open
        forbidden = {self.path, self.path.with_suffix(".keys.parquet")}

        def tripwire(file, *args, **kwargs):
            if not isinstance(file, int) and Path(file) in forbidden:
                opened.append(str(file))
                raise AssertionError("forbidden payload opened before admission")
            return original(file, *args, **kwargs)

        with patch("io.open", side_effect=tripwire), self.assertRaises(ValueError):
            (loader or (lambda: v2_teacher.hdr_leg(record, self.cols)))()
        self.assertEqual(opened, [])

    def test_forbidden_val_fixture_has_zero_payload_opens_in_loader_and_packer(self):
        self.path = self.path.with_name("forbidden_val.parquet")
        self.manifest.update(role="val", rows=3900, population="all")
        self.write()
        self.assert_no_payload_read(self.record)
        (self.root / "wide/keep_lists.json").write_text('{}')
        receipt = dict(schema="rev4-featpot-v2c-wide-v1", family="main", variant="real", complete=True,
                       formula_revision=5, legs={"hdr": self.record})
        (self.path.parent / "receipt.json").write_text(json.dumps(receipt))
        self.assert_no_payload_read(self.record, lambda: v2c_pack.members_for(self.root, "lodo", [("main", "real")]))

    def test_actual_packer_refuses_confirmation_and_retains_fit_keys(self):
        wide = self.root / "wide"
        (wide / "keep_lists.json").write_text('{}')
        receipt = dict(schema="rev4-featpot-v2c-wide-v1", family="main", variant="real", complete=True,
                       formula_revision=5, legs={"hdr": self.record})
        (self.path.parent / "receipt.json").write_text(json.dumps(receipt))
        members = v2c_pack.members_for(self.root, "lodo", [("main", "real")])
        self.assertIn("wide/main/real/hdr_fit.keys.parquet", members)
        self.assertFalse(any("val" in p for p in members))
        (wide / "frozen.json").write_text("{}")  # Synthetic marker: reach the HDR packing guard.
        for kind in ["confirm", "all"]:
            with self.assertRaisesRegex(ValueError, "LODO only"):
                v2c_pack.members_for(self.root, kind, [("main", "real")])


class RegisteredHdrDecisions(unittest.TestCase):
    def panels(self, gains):
        return {label: [dict(source=source, seed=seed,
                     within_reference=dict(hdrvdp3=.8+(seed-4.5)*.0001*(label != 'control'),
                                           cvvdp=.7+(seed-4.5)*.0001*(label != 'control')),
                     pooled=dict(hdrvdp3=.6+gain+(seed-4.5)*.0001*(label != 'control'),
                                 cvvdp=.5+(seed-4.5)*.0001*(label != 'control')))
                 for source in e24_rev5.SOURCE_ORDER for seed in e24_rev5.SEEDS]
                for label, gain in dict(control=0., **gains).items()}

    def test_e27_larger_pooled_gain_wins_and_e26_hd4_is_report_only(self):
        cells = self.panels(dict(hp4=.01, ha4=.02, e26_hd4=.2))
        arms, adopt = e24_rev5.hdr_arm_decisions(cells, dict(hp4=dict(as_good=True), ha4=dict(as_good=True)), 'e27')
        self.assertEqual(adopt, 'ha4')
        self.assertEqual(set(arms), {'hp4', 'ha4'})
        self.assertTrue(all(a['passes'] for a in arms.values()))
        cells['hp4'][0]['seed'] = 9
        with self.assertRaisesRegex(ValueError, 'pairing'):
            e24_rev5.hdr_arm_decisions(cells, {}, 'e27')

    def test_e27_every_teacher_and_panel_is_a_gate_and_sdr_is_required(self):
        sdr = dict(hp4=dict(as_good=True), ha4=dict(as_good=True))
        for panel in ['pooled', 'within_reference']:
            for teacher in ['hdrvdp3', 'cvvdp']:
                cells = self.panels(dict(hp4=.01, ha4=0.))
                for c in cells['hp4']: c[panel][teacher] -= .03
                with self.subTest(panel=panel, teacher=teacher):
                    arms, adopt = e24_rev5.hdr_arm_decisions(cells, sdr, 'e27')
                    self.assertIsNone(adopt)
                    self.assertFalse(arms['hp4']['passes'])
        cells = self.panels(dict(hp4=.01, ha4=0.))
        arms, adopt = e24_rev5.hdr_arm_decisions(cells, dict(hp4=dict(as_good=False), ha4=dict(as_good=True)), 'e27')
        self.assertIsNone(adopt)
        self.assertFalse(arms['hp4']['passes'])
        cells['hp4'][0]['pooled']['cvvdp'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            e24_rev5.hdr_arm_decisions(cells, sdr, 'e27')

    def test_e27_positive_gain_below_two_paired_se_does_not_pass(self):
        cells = self.panels(dict(hp4=.000001, ha4=0.))
        arms, adopt = e24_rev5.hdr_arm_decisions(cells,
            dict(hp4=dict(as_good=True), ha4=dict(as_good=True)), 'e27')
        stats = arms['hp4']['pooled']['hdrvdp3']
        self.assertGreater(stats['delta'], 0.)
        self.assertGreater(stats['se'], 0.)
        self.assertLess(stats['delta'], 2*stats['se'])
        self.assertFalse(arms['hp4']['passes'])
        self.assertIsNone(adopt)

    def test_e26_still_uses_within_and_lowest_weight(self):
        cells = self.panels(dict(hd4=-.2, hd16=-.1))
        for label in ['hd4', 'hd16']:
            for c in cells[label]: c['within_reference']['hdrvdp3'] += .01
        arms, adopt = e24_rev5.hdr_arm_decisions(cells, dict(hd4=dict(as_good=True), hd16=dict(as_good=True)), 'e26')
        self.assertEqual(adopt, 'hd4')
        self.assertIn('pooled_report_only', arms['hd4'])


if __name__ == "__main__":
    unittest.main()
