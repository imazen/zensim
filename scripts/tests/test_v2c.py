#!/usr/bin/env python3
"""v2-canon tooling (CANONTAB lane): builder pieces, the sealed-path guard, the confirmatory read on synthetic labels.

    python3 -m unittest scripts/tests/test_v2c.py

The confirmatory read is tested ONLY on synthetic labels: a planted effect it must find, a null it must not, and the
orientation rule it must apply. Needs the `panel` binary (ZEN_PANEL_BIN, default the fleet v8 panel).
"""
import builtins
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

TMP = Path.home() / "tmp"  # never /tmp
TMP.mkdir(exist_ok=True)
tempfile.tempdir = str(TMP)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rev4_featpot"))
os.environ.setdefault("ZEN_PANEL_BIN", "/var/tmp/fitv2/bin-v2/panel")
import v2_common  # noqa: E402
import v2c_wide as w  # noqa: E402


def f64_bank_set(bank: Path, name: str, rows: int, seed: int, identical=(2,), extra_manifest=None):
    """A synthetic Rev4 bank set: keys.parquet, features.parquet (float64, f0..f1824), _MANIFEST.json."""
    rng = np.random.default_rng(seed)
    d = bank / name
    d.mkdir(parents=True)
    keys = pd.DataFrame({
        "pair_key": [hashlib.sha256(f"{name}{i}".encode()).hexdigest() for i in range(rows)],
        "row_id": np.arange(rows, dtype=np.int64),
        "ref_group": [f"ref{i % 3}" for i in range(rows)],
        "pixels_identical": [i in identical for i in range(rows)], "n_stimuli": np.ones(rows, dtype=np.int32)})
    pq.write_table(pa.Table.from_pandas(keys, preserve_index=False), d / "keys.parquet")
    feats = {"pair_key": keys.pair_key, "row_id": keys.row_id}
    values = rng.normal(size=(rows, w.WIDTH)) * 1e3 + rng.normal(size=(rows, w.WIDTH)) * 1e-9  # not f32-exact
    feats.update({f"f{i}": values[:, i] for i in range(w.WIDTH)})
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(feats), preserve_index=False), d / "features.parquet")
    manifest = {"set": name, "schema": "rev4-featbank-r4-v1", "rows": rows, "feature_width": w.WIDTH,
                "build_commit": w.CANON_BUILD, "binary_sha256": w.CANON_BINARY, "formula_revision": "Rev4",
                "era_label": w.CANON_ERA, "feature_set_id": w.CANON_FEATURE_SET_ID, "dtype": "float64",
                "formula_revision_eras": ["tiercanon", "c3negfold"], "keys_sha256": w.sha(d / "keys.parquet"),
                "features_parquet_sha256": w.sha(d / "features.parquet"), **(extra_manifest or {})}
    (d / "_MANIFEST.json").write_text(json.dumps(manifest))
    return values


class Guard(unittest.TestCase):
    def test_sealed_paths_refused(self):
        for bad in ("/var/tmp/rev4-featbank/_sealed", "/x/_sealed/y/labels.parquet", "_sealed_extra/z"):
            with self.assertRaises(PermissionError):
                w.safe_path(bad)
        w.safe_path("/var/tmp/rev4-featbank-r4/aic4/keys.parquet")

    def test_bank_file_names(self):
        with self.assertRaises(ValueError):
            w.bank_file(Path("/b"), "aic4", "labels__human.parquet")
        with self.assertRaises(ValueError):
            w.bank_file(Path("/b"), "_sealed", "keys.parquet")
        self.assertEqual(w.bank_file(Path("/b"), "aic4", "keys.parquet"), Path("/b/aic4/keys.parquet"))

    def test_manifest_pins_refused(self):
        with tempfile.TemporaryDirectory() as t:
            bank = Path(t)
            f64_bank_set(bank, "aic4", 6, 1, extra_manifest={"era_label": "tiercanon"})
            with self.assertRaises(ValueError):
                w.load_bank_set(bank, "aic4", [])


class Builders(unittest.TestCase):
    def test_permutation_matches_restore_data_owner(self):
        import restore_data
        rng = np.random.default_rng(3)
        refs = np.repeat([f"r{i}" for i in range(4)], [9, 7, 5, 6])
        keys = np.array([f"k{i // 2 if i % 5 == 0 else i}" for i in range(len(refs))])
        keys = np.where(np.arange(len(refs)) % 7 == 0, np.roll(keys, 1), keys)  # shared keys inside a reference
        vals = rng.normal(size=(len(refs), 6)).astype(np.float32)
        frame = pd.DataFrame({"ref_basename": refs, "pair_key": keys, **{f"c{j}": vals[:, j] for j in range(6)}})
        owner = frame.copy()
        restore_data._permute_within_reference(owner, [f"c{j}" for j in range(6)], np.ones(len(frame), dtype=bool),
                                               np.random.default_rng(20260931))
        ours = w.permute_within_reference(refs, keys, vals, np.ones(len(refs), dtype=bool), np.random.default_rng(20260931))
        self.assertTrue(np.array_equal(owner[[f"c{j}" for j in range(6)]].to_numpy(), ours))
        self.assertFalse(np.array_equal(vals, ours))

    def test_oracle_columns_are_distance_oriented_and_seeded(self):
        y = np.linspace(0, 100, 400) ** 1.1
        bounds, y01, cols = w.oracle_columns(y, 3)
        self.assertLess(np.corrcoef(cols["oracle_hi"], y01)[0, 1], -0.7)
        self.assertLess(np.corrcoef(cols["oracle_lo"], y01)[0, 1], np.corrcoef(cols["oracle_hi"], y01)[0, 1] + 1)
        rng = np.random.default_rng(v2_common.ORACLE_SEED_BASE + 3)
        sd = float(np.std(y01))
        expect = ((1.0 - y01) + rng.normal(0.0, v2_common.ORACLE_SIGMA["oracle_lo"] * sd, len(y))).astype(np.float32)
        self.assertTrue(np.array_equal(cols["oracle_lo"], expect))

    def test_confirm_tables_exact_cast_zero_labels_and_never_touch_sealed(self):
        seen = []
        real_open, real_file, real_read, real_write = builtins.open, pq.ParquetFile, pq.read_table, pq.write_table
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            bank = t / "bank"
            values = {n: f64_bank_set(bank, n, 9, i, identical=(4,)) for i, n in enumerate(("aic4", "csiq"))}
            sealed = bank / "_sealed"  # a trap beside the bank: any access is a failure
            sealed.mkdir()
            (sealed / "labels.parquet").write_text("trap")

            def read_text_spy(self, *a, **k):
                seen.append(str(self))
                with real_open(self) as stream:
                    return stream.read()

            def spy(real):
                def inner(path, *a, **k):
                    seen.append(str(path))
                    return real(path, *a, **k)
                return inner
            with mock.patch.object(builtins, "open", spy(real_open)), mock.patch.object(pq, "ParquetFile", spy(real_file)), \
                    mock.patch.object(pq, "read_table", spy(real_read)), mock.patch.object(pq, "write_table", spy(real_write)), \
                    mock.patch.object(Path, "read_text", read_text_spy):
                rec = w.build_confirm(bank, t / "out", ["main"], [], None, sets=("aic4", "csiq"))
            self.assertTrue(seen)
            self.assertFalse([p for p in seen if "_sealed" in p], seen)
            for name in ("aic4", "csiq"):
                tab = rec["sets"][name]["tables"]["main"]["real"]
                got = pq.read_table(t / "out" / tab["rel"]).to_pandas()
                keys = pq.read_table((t / "out" / tab["rel"]).parent / f"{name}.keys.parquet").to_pandas()
                self.assertEqual(len(got), 8)                         # the identical key is dropped
                self.assertTrue((got.human_score == 0).all())
                self.assertNotIn("target", keys.columns)
                keep = np.arange(9) != 4
                want = values[name][keep].astype(np.float32)
                have = got[[f"f{i}" for i in range(w.WIDTH)]].to_numpy(np.float32)
                self.assertTrue(np.array_equal(want.view(np.uint32), have.view(np.uint32)))
                self.assertFalse(np.array_equal(values[name][keep], want.astype(np.float64)))  # the cast is real

    def test_extra_families_widen_main_and_aux_zero(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            bank = t / "bank"
            f64_bank_set(bank, "aic4", 7, 5, identical=())
            keys = pq.read_table(bank / "aic4" / "keys.parquet").to_pandas()
            side = pd.DataFrame({"pair_key": keys.pair_key[::-1].to_numpy(),
                                 **{f"f{1825 + j}": np.arange(7, dtype=np.float32) + j for j in range(3)}})
            (t / "aic4").mkdir()
            pq.write_table(pa.Table.from_pandas(side, preserve_index=False), t / "aic4" / "texgain.parquet")
            extras = w.parse_extras([f"texgain=1825:3:{t}/{{set}}/texgain.parquet"])
            self.assertEqual(w.total_width(extras), 1828)
            peers = t / "peers"
            peers.mkdir()
            pq.write_table(pa.Table.from_pandas(pd.DataFrame({"pair_key": keys.pair_key, "gmsd": np.arange(7.0),
                                                              "gmsm": np.arange(7.0) * 2}), preserve_index=False),
                           peers / "aic4.parquet")
            rec = w.build_confirm(bank, t / "out", ["main", "aux"], extras, peers, variants=["real", "p2"], sets=("aic4",))
            main = pq.read_table(t / "out" / rec["sets"]["aic4"]["tables"]["main"]["real"]["rel"]).to_pandas()
            self.assertEqual(main.shape[1], 2 + 1828)
            by_key = dict(zip(keys.pair_key[::-1], range(7)))   # sidecar rows were reversed: join is by key, not position
            self.assertEqual([by_key[k] for k in keys.pair_key], main.f1825.astype(int).tolist())
            aux = pq.read_table(t / "out" / rec["sets"]["aic4"]["tables"]["aux"]["real"]["rel"]).to_pandas()
            self.assertTrue((aux[["f1825", "f1826", "f1827"]] == 0).all().all())
            self.assertEqual(aux.f944.tolist(), list(map(float, range(7))))
            self.assertTrue((aux.f946 == 0).all())                # oracle columns need a label: none here
            perm = pq.read_table(t / "out" / rec["sets"]["aic4"]["tables"]["main"]["p2"]["rel"]).to_pandas()
            self.assertTrue((perm.f5 == main.f5).all())           # bank columns untouched by the control
            self.assertEqual(sorted(perm.f1825), sorted(main.f1825))  # extras permuted jointly within reference
            self.assertNotEqual(perm.f1825.tolist(), main.f1825.tolist())

    def test_extra_arm_keep_lists(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            (root / "wide").mkdir()
            (root / "wide" / "extra_arms.json").write_text(json.dumps({
                "schema": "rev4-featpot-v2c-extra-arms-v1", "width": 1830, "arms": {"texgain": [1825, 1826]}}))
            with mock.patch.object(v2_common, "V2", root):
                self.assertEqual(v2_common.parse_spec("texgain~p2"), ("texgain", 2))
                fam, var, keep = v2_common.arm_columns("texgain~p2")
                self.assertEqual((fam, var, keep[-2:]), ("main", "p2", [1825, 1826]))
                self.assertIn("texgain~p3", v2_common.all_specs())
            with mock.patch.object(v2_common, "V2", root / "nope"):
                with self.assertRaises(ValueError):
                    v2_common.parse_spec("texgain")

    def test_root_override_from_argv_and_env(self):
        self.assertEqual(v2_common._root_override(["x", "--root", "/a/b"]), "/a/b")
        self.assertEqual(v2_common._root_override(["x", "--root=/c"]), "/c")
        with mock.patch.dict(os.environ, {"REV4_V2_ROOT": "/e"}):
            self.assertEqual(v2_common._root_override(["x"]), "/e")
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("REV4_V2_ROOT", None)
            self.assertIsNone(v2_common._root_override(["x"]))


class TrainRecipe(unittest.TestCase):
    def test_train_command_is_the_v2_recipe(self):
        import v2_lodo_mlp as lodo
        cmd = lodo.train_command([("safesyn", Path("/t/s.parquet"), 1.5, 0, "withinref,both"),
                                  ("human_development", Path("/t/h.parquet"), 0, 1.0, "withinref,rank")],
                                 1101, 101, 1825, Path("/k.txt"), "N", Path("/o/best.bin"))
        expect = [str(lodo.TRAINER), "--group", "safesyn:/t/s.parquet:1.5:0:withinref,both",
                  "--group", "human_development:/t/h.parquet:0:1.0:withinref,rank",
                  "--target-column", "human_score", "--target-scale", "1", "--hidden", "32", "--epochs", "120",
                  "--pairs-per-epoch", "50000", "--init-seed", "1101", "--sample-seed", "101", "--pair-sampling",
                  "uniform", "--max-features", "1825", "--keep-features", "/k.txt", "--mse-weight", "1",
                  "--early-stop-patience", "0", "--val-policy", "mean", "--val-aggregate", "geomean3", "--out-dtype",
                  "f32", "--log-every", "1", "--no-auto-eval", "--historical-replay", v2_common.REPLAY,
                  "--out", "/o/best.bin", "--nonneg-distance"]
        self.assertEqual(cmd, expect)
        self.assertNotIn("--nonneg-distance", lodo.train_command([], 1, 2, 3, Path("k"), "F", Path("o")))

    def test_grid_shape(self):
        import v2c_grid
        cells = v2c_grid.cells(["c1", "c3"], 2.0, "/r")
        self.assertEqual(len(cells), (1 + 2 * 4) * 2 * 10)
        self.assertEqual(len({c["name"] for c in cells}), len(cells))
        self.assertEqual(cells[0], {"name": "r0@h2__N/full_s0", "argv": [
            "v2_confirm_fit.py", "--spec", "r0@h2", "--head", "N", "--seed-index", "0", "--root", "/r"]})
        self.assertIn("c3~p3@h2__F/full_s9", {c["name"] for c in cells})


class PinBuilder(unittest.TestCase):
    def test_shortlist_rule(self):
        import v2c_pin
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)

            def rec(arm, head, excess, v1sources, regress=()):
                src = {f"s{i}": {"excess": e, "v1_pass": i < v1sources} for i, e in enumerate(excess)}
                (d / f"{arm}_{head}.json").write_text(json.dumps({"status": "OK", "sources": src, "regressions": list(regress)}))
            rec("c1", "N", [0.02] * 5, 2)             # eligible, mean 0.02
            rec("c1", "F", [0.02] * 5, 2)             # ties c1/N: N first
            rec("c2", "N", [0.05] * 5, 1)             # only 1 V1 source: ineligible
            rec("c3", "N", [0.09] * 5, 3, ["s0"])     # regression: ineligible
            rec("c4", "N", [0.03] * 5, 4)             # eligible, largest
            got, prov = v2c_pin.shortlist(d, ["c1", "c2", "c3", "c4"])
            self.assertEqual([(e["arm"], e["head"]) for e in got], [("c4", "N"), ("c1", "N"), ("c1", "F")])
            self.assertEqual(len(prov), 5)
            many = ["all", "csfw", "c7", "p1", "b1", "b1s", "c8n", "rall"]
            for k, arm in enumerate(many):            # more than 6 eligible: capped at six
                rec(arm, "N", [0.001 * k] * 5, 2)
            self.assertEqual(len(v2c_pin.shortlist(d, [*many, "c1", "c4"])[0]), 6)


class Freeze(unittest.TestCase):
    def test_freeze_pins_receipts_and_blocks_rebuilds(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            wide = root / "wide"
            for fam in w.FAMILIES:
                for var in w.VARIANTS:
                    (wide / fam / var).mkdir(parents=True)
                    (wide / fam / var / "receipt.json").write_text(json.dumps({"complete": True, "width": 1825, "feature_set_id": "x"}))
            sets = {n: {"tables": {f: {v: {} for v in w.VARIANTS} for f in w.FAMILIES}} for n in w.CONFIRM_SETS}
            (wide / "confirm").mkdir()
            (wide / "confirm" / "receipt.json").write_text(json.dumps({"width": 1825, "sets": sets}))
            (wide / "keep_lists.json").write_text("{}")
            (wide / "verify.json").write_text(json.dumps({"all_ok": False}))
            with self.assertRaises(ValueError):                              # an unclean verify cannot be frozen
                w.freeze(root)
            (wide / "verify.json").write_text(json.dumps({"all_ok": True}))
            with self.assertRaises(ValueError):                              # no frozen.json yet
                v2_common.load_frozen(root)
            w.freeze(root)
            record, digest = v2_common.load_frozen(root)
            self.assertEqual(len(digest), 64)
            with self.assertRaises(ValueError):                              # no rebuild into a frozen root
                w.build_confirm(Path("/nonexistent"), root, ["main"], [], None)
            (wide / "main" / "p1" / "receipt.json").write_text(json.dumps({"complete": True, "width": 1825, "feature_set_id": "y"}))
            with self.assertRaises(ValueError):                              # any receipt change after the freeze is caught
                v2_common.load_frozen(root)


class Labels(unittest.TestCase):
    """The label adapter (v2c_labels): by (ref_path, dist_path), never by row_id; strict accounting."""

    def keys(self, n=5):
        return pd.DataFrame({"pair_key": [f"k{i}" for i in range(n)], "ref_path": [f"/r/{i // 2}.png" for i in range(n)],
                             "dist_path": [f"/d/{i}.png" for i in range(n)], "ref_pixels_sha256": [f"r{i // 2}" for i in range(n)],
                             "dist_pixels_sha256": [f"d{i}" for i in range(n)], "n_stimuli": 1, "pixels_identical": False})

    def rows(self, keys, extra=()):
        base = pd.DataFrame({"ref_path": keys.ref_path, "dist_path": keys.dist_path, "label": np.arange(len(keys), dtype=float)})
        base = pd.concat([base, pd.DataFrame(extra, columns=["ref_path", "dist_path", "label"])], ignore_index=True)
        base["file_row"] = np.arange(len(base))
        return base

    def test_maps_by_paths_not_row_order(self):
        import v2c_labels as L
        keys = self.keys()
        rows = self.rows(keys).iloc[::-1].reset_index(drop=True)          # source order is not bank order
        got, acct = L.adapt(rows, keys)
        self.assertEqual(dict(zip(got.pair_key, got.label)), {f"k{i}": float(i) for i in range(5)})
        self.assertEqual(acct["rows_used"], 5)

    def test_unmatched_rows_must_be_identical_stimuli(self):
        import v2c_labels as L
        keys = self.keys()
        with self.assertRaises(ValueError):                               # a stray row that is no identical stimulus
            L.adapt(self.rows(keys, [("/r/9.png", "/d/x.png", 1.0)]), keys)
        keys.loc[1, "pixels_identical"] = True
        keys.loc[1, "n_stimuli"] = 3                                      # 2 more stimuli on the identical key, not matched by path
        got, acct = L.adapt(self.rows(keys, [("/r/0.png", "/d/a.png", 7.0), ("/r/0.png", "/d/b.png", 8.0)]), keys)
        self.assertEqual(sorted(got.pair_key), ["k0", "k2", "k3", "k4"])
        self.assertEqual(acct["rows_on_identical_keys_or_unmatched_identical"], 3)

    def test_collapsed_stimuli_need_the_pixel_table(self):
        import v2c_labels as L
        keys = self.keys()
        keys.loc[2, "n_stimuli"] = 2
        rows = self.rows(keys, [("/r/1.png", "/d/alias.png", 9.0)])
        with self.assertRaises(ValueError):
            L.adapt(rows, keys)
        got, acct = L.adapt(rows, keys, None, {"/r/1.png": "r1", "/d/alias.png": "d2"})
        self.assertEqual(sorted(got.label[got.pair_key == "k2"]), [2.0, 9.0])
        self.assertEqual(acct["matched_by_pixel_hash"], 1)

    def test_select_rule_and_via_pairs(self):
        import v2c_labels as L
        keys = self.keys()
        rows = self.rows(keys, [("/r/other.png", "/d/o.png", 5.0)])
        got, acct = L.adapt(rows, keys, {"ref_path_in": sorted(set(keys.ref_path))})
        self.assertEqual((len(got), acct["unselected_rows"]), (5, 1))
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            pairs = pd.DataFrame({"ref_path": keys.ref_path, "dist_path": keys.dist_path, "row_id": range(5)})
            pairs.to_csv(t / "pairs.tsv", sep="\t", index=False)
            pd.DataFrame({"row_id": range(5), "score": [10, 11, 12, 13, 14]}).to_csv(t / "labels.csv", index=False)
            spec = {"path": str(t / "labels.csv"), "sha256": v2_common.sha(t / "labels.csv"), "format": "csv", "label_col": "score",
                    "via_pairs": {"path": str(t / "pairs.tsv"), "sha256": v2_common.sha(t / "pairs.tsv"), "on": [["row_id", "row_id"]]}}
            got, _ = L.adapt(L.load_label_rows(spec), keys)
            self.assertEqual(got.label.tolist(), [10.0, 11.0, 12.0, 13.0, 14.0])
            spec["sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                L.load_label_rows(spec)

    def test_open_sets_reproduce_admitted_labels_exactly(self):
        import v2c_labels as L
        for name in ("aic3", "kadid_select"):                              # OPEN sets only; never a sealed one
            self.assertTrue(L.validate(name)["exact"], name)
        with self.assertRaises(SystemExit):
            L.validate("cid22_b")


class ConfirmRead(unittest.TestCase):
    PRIMARY = ("cid22_b", "aic4", "csiq", "mcljci")
    SETS = ("cid22_b", "aic4", "konjnd_jpeg_select", "konjnd_jpeg_terminal", "csiq", "mcljci")
    FILES = {"cid22_b": ("cid22val_pairs_ab.tsv", "human_score"), "aic4": ("aic4_pairs.tsv", "human_score"),
             "konjnd_jpeg_select": ("konjnd_jpeg_val_pairs.tsv", "human_score"), "konjnd_jpeg_terminal": ("konjnd_jpeg_val_pairs.tsv", "human_score"),
             "csiq": ("csiq_pairs.tsv", "human_score"), "mcljci": ("mcljci_labels.csv", "jnd_dist")}
    DISTORTION = ("aic4", "konjnd_jpeg_select", "konjnd_jpeg_terminal", "mcljci")  # label-file orientation, fixed independently
    REFS, PER_REF = 30, 8
    B = 200

    @classmethod
    def setUpClass(cls):
        import v2_compare
        import v2_confirm_read as cr
        cls.cmp, cls.cr = v2_compare, cr
        cls.saved = (v2_compare.BOOT_B, v2_compare.SEED_DRAWS, dict(v2_compare._ref_draws), dict(v2_compare._rendered))
        v2_compare.BOOT_B = cls.B
        v2_compare.SEED_DRAWS = np.random.default_rng(v2_common.BOOT_SEED + 1).integers(0, 10, size=(cls.B, 10))
        v2_compare._ref_draws.clear()
        v2_compare._rendered.clear()

    @classmethod
    def tearDownClass(cls):
        cmp = cls.cmp
        cmp.BOOT_B, cmp.SEED_DRAWS = cls.saved[0], cls.saved[1]
        cmp._ref_draws.clear(), cmp._ref_draws.update(cls.saved[2])
        cmp._rendered.clear(), cmp._rendered.update(cls.saved[3])
        cmp.REF_SEEDS.clear()

    def build_tree(self, root: Path, planted: dict, weight=None, heads=("N",), shortlist=None):
        """Synthetic canon root + bank + labels + pin. planted: arm -> noise sd, or {set: sd, None: default sd}."""
        cr = self.cr
        rng = np.random.default_rng(11)
        width = 948
        for rel in ("wide/confirm", "bank", "labels"):
            (root / rel).mkdir(parents=True, exist_ok=True)
        receipt = {"schema": "rev4-featpot-v2c-confirm-v1", "width": width, "feature_set_id": "x", "sets": {}}
        truth, labels = {}, {}
        for name in self.SETS:
            n = self.REFS * self.PER_REF
            keys = pd.DataFrame({"pair_key": [f"{name}-{i}" for i in range(n)], "row_id": np.arange(n),
                                 "ref_group": [f"ref{i // self.PER_REF}" for i in range(n)],
                                 "ref_path": [f"/{name}/ref{i // self.PER_REF}.png" for i in range(n)],
                                 "dist_path": [f"/{name}/d{i}.png" for i in range(n)],
                                 "ref_pixels_sha256": ["r"] * n, "dist_pixels_sha256": [f"p{i}" for i in range(n)],
                                 "n_stimuli": np.ones(n, dtype=np.int32), "pixels_identical": [i == 3 for i in range(n)]})
            (root / "bank" / name).mkdir()
            pq.write_table(pa.Table.from_pandas(keys, preserve_index=False), root / "bank" / name / "keys.parquet")
            nonid = keys.loc[~keys.pixels_identical].reset_index(drop=True)
            ck = pd.DataFrame({"pair_key": nonid.pair_key, "row_id": nonid.row_id, "ref_basename": nonid.ref_group, "member_set": name})
            tables = {}
            for variant in ("real", "p1", "p2", "p3"):
                d = root / "wide" / "confirm" / "main" / ("" if variant == "real" else variant)
                d.mkdir(parents=True, exist_ok=True)
                pq.write_table(pa.Table.from_pandas(ck, preserve_index=False), d / f"{name}.keys.parquet")
                added = rng.normal(size=(len(ck), width - 944))
                tab = pd.DataFrame({"ref_basename": ck.ref_basename, "human_score": 0.0,
                                    **{f"f{i}": added[:, i - 944] if i >= 944 else 0.0 for i in range(width)}})
                if name == "konjnd_jpeg_select" and variant != "real":
                    tab = tbl_real                                     # singleton references: the permutation is the identity
                pq.write_table(pa.Table.from_pandas(tab, preserve_index=False), d / f"{name}.parquet")
                if variant == "real":
                    tbl_real = tab
                tables[variant] = {"rel": str((d / f"{name}.parquet").relative_to(root)), "sha256": v2_common.sha(d / f"{name}.parquet"),
                                   "manifest_sha256": "m", "keys_sha256": v2_common.sha(d / f"{name}.keys.parquet")}
            receipt["sets"][name] = {"rows": len(nonid), "tables": {"main": tables}}
            q = rng.normal(size=len(nonid))
            truth[name] = q
            sign = -1.0 if name in self.DISTORTION else 1.0
            fname, col = self.FILES[name]
            (root / "labels" / name).mkdir()
            lab = pd.DataFrame({"ref_path": keys.ref_path, "dist_path": keys.dist_path, col: 0.0})
            lab.loc[~keys.pixels_identical.to_numpy(), col] = sign * (2.0 * q + 3.0)   # the identical key keeps a dummy label row
            lp = root / "labels" / name / fname
            lab.to_csv(lp, sep="," if fname.endswith(".csv") else "\t", index=False)
            labels[name] = {"path": str(lp), "sha256": v2_common.sha(lp), "format": "csv" if fname.endswith(".csv") else "tsv",
                            "ref_col": "ref_path", "dist_col": "dist_path", "label_col": col, "select": None}
        (root / "wide" / "confirm" / "receipt.json").write_text(json.dumps(receipt))
        (root / "wide" / "keep_lists.json").write_text("{}")
        wide = {}
        for v in ("real", "p1", "p2", "p3"):
            (root / "wide" / "main" / v).mkdir(parents=True, exist_ok=True)
            (root / "wide" / "main" / v / "receipt.json").write_text(json.dumps({"v": v}))
            wide[f"main/{v}"] = v2_common.sha(root / "wide" / "main" / v / "receipt.json")
        frozen = {"schema": v2_common.FROZEN_SCHEMA, "wide_receipts": wide, "extra_arms_sha256": None,
                  "confirm_receipt_sha256": v2_common.sha(root / "wide" / "confirm" / "receipt.json"),
                  "keep_lists_sha256": v2_common.sha(root / "wide" / "keep_lists.json")}
        (root / "wide" / "frozen.json").write_text(json.dumps(frozen))
        frozen_sha = v2_common.sha(root / "wide" / "frozen.json")
        panel = os.environ["ZEN_PANEL_BIN"]
        binaries = {"zensim_mlp_train": "t", "bake_dial_refit": "b", "panel": v2_common.sha(Path(panel))}
        prog, data = "a" * 64, "b" * 64
        w = weight
        specs = {"r0": 1.0}
        for a, sd in planted.items():
            specs[a] = sd
            specs.update({f"{a}~p{k}": 1.0 for k in (1, 2, 3)})
        for spec, sd in specs.items():
            variant = f"p{spec.partition('~p')[2]}" if "~p" in spec else "real"
            full = self.cr.full_spec(spec, w)
            for head in heads:
                for seed in range(10):
                    preds = {}
                    for name in self.SETS:
                        s = sd.get(name, sd.get(None, 1.0)) if isinstance(sd, dict) else sd
                        preds[name] = {"rows": len(truth[name]), "pred": (truth[name] + rng.normal(0, s, len(truth[name]))).tolist(),
                                       "table_sha256": receipt["sets"][name]["tables"]["main"][variant]["sha256"],
                                       "keys_sha256": receipt["sets"][name]["tables"]["main"][variant]["keys_sha256"]}
                    d = cr.cell_path(root, full, head, seed)
                    d.mkdir(parents=True)
                    (d / "result.json").write_text(json.dumps({
                        "schema": "rev4-featpot-v2c-confirm-cell-v1", "spec": full, "head": head, "seed_index": seed,
                        "family": "main", "variant": variant, "eval_variant": variant, "wide_receipt_sha256": wide[f"main/{variant}"],
                        "confirm_receipt_sha256": frozen["confirm_receipt_sha256"], "keep_lists_sha256": frozen["keep_lists_sha256"],
                        "frozen_sha256": frozen_sha, "binaries": binaries, "predictions": preds,
                        "human_nominal_weight": 0.5 if w is None else w}))
                    (d / "fleet_receipt.json").write_text(json.dumps({"program_sha": prog, "data_sha": data}))
        prov = root / "prov.json"
        prov.write_text("{}")
        pin = {"schema": cr.PIN_SCHEMA, "reference": "r0", "candidates": sorted(planted), "heads": ["N", "F"],
               "shortlist": shortlist or [{"arm": a, "head": h} for a in sorted(planted) for h in heads], "human_weight": weight,
               "frozen_sha256": frozen_sha, "wide_receipts": wide, "confirm_receipt_sha256": frozen["confirm_receipt_sha256"],
               "keep_lists_sha256": frozen["keep_lists_sha256"], "binaries": binaries, "program_sha": prog, "data_sha": data,
               "code": {k: v2_common.sha(p) for k, p in cr.CODE_FILES.items()},
               "shortlist_provenance": {"calibration": {"path": str(prov), "sha256": v2_common.sha(prov)}, "compare": {}},
               "labels": labels}
        pin_path = root / "pin.json"
        pin_path.write_text(json.dumps(pin))
        return pin_path

    def run_read(self, root: Path, planted: dict, **kw):
        cr = self.cr
        pin_path = self.build_tree(root, planted, **kw)
        out, ledger = root / "read.json", root / "ledger.md"
        with mock.patch.object(v2_common, "V2", root), mock.patch.object(cr, "V2", root):
            cr.main(["--confirmatory-read", "--pin", str(pin_path), "--root", str(root), "--bank", str(root / "bank"), "--out", str(out),
                     "--ledger", str(ledger), *[x for a in planted for x in ("--arm", a)]])
        return json.loads(out.read_text()), ledger

    def test_planted_effect_found_null_not_orientation_and_r22_konjnd(self):
        with tempfile.TemporaryDirectory() as t:
            res, ledger = self.run_read(Path(t), {"planted": 0.55, "nullarm": 1.0})
            by = {e["arm"]: e for e in res["primary"]}
            self.assertTrue(by["planted"]["confirmed"], by["planted"])
            self.assertEqual(by["planted"]["verdict"], "confirmed; V3 dial gates next")
            self.assertFalse(by["nullarm"]["confirmed"])
            self.assertEqual(sorted(by["planted"]["per_set_excess"]), sorted(self.PRIMARY))   # KonJND is not in the mean
            kon = by["planted"]["konjnd_select"]
            self.assertGreater(kon["delta"], 0.03)                                          # KonJND reports delta vs R0
            self.assertFalse(kon["regression"])
            self.assertGreater(kon["r0_mean"], 0.5)                                         # distortion labels negated: signed SROCC > 0
            sec = res["secondary_v1_v2"]["planted_N"]
            self.assertTrue(sec["V1"] and sec["V2"])
            self.assertNotIn("nullarm_F", res["secondary_v1_v2"])                          # shortlisted entries only
            self.assertEqual(res["provenance"]["perm_identity_share"]["konjnd_jpeg_select"], [1.0, 1.0, 1.0])
            self.assertLess(max(res["provenance"]["perm_identity_share"]["cid22_b"]), 0.1)
            self.assertIn("pending, update after read", ledger.read_text())
            self.assertIn("R2.2", ledger.read_text())
            self.assertEqual(res["provenance"]["per_set"]["csiq"]["accounting"]["rows_used"], self.REFS * self.PER_REF - 1)

    def test_konjnd_regression_vetoes_an_otherwise_confirmed_entry(self):
        with tempfile.TemporaryDirectory() as t:
            res, _ = self.run_read(Path(t), {"vetoed": {None: 0.55, "konjnd_jpeg_select": 3.0}})
            e = res["primary"][0]
            self.assertTrue(e["holm_significant"])
            self.assertTrue(e["konjnd_select"]["regression"])
            self.assertFalse(e["confirmed"])
            self.assertIn("regresses", e["verdict"])

    def test_primary_holm_keeps_planted_rejects_nulls(self):
        arms = {"planted": 0.6, "nullA": 1.0, "nullB": 1.0, "nullC": 1.0}
        with tempfile.TemporaryDirectory() as t:
            res, _ = self.run_read(Path(t), arms)
        by = {e["arm"]: e for e in res["primary"]}
        self.assertTrue(by["planted"]["confirmed"], by["planted"])
        self.assertLess(by["planted"]["p_one_sided"], 0.05 / 4)
        for n in ("nullA", "nullB", "nullC"):
            self.assertFalse(by[n]["confirmed"], by[n])

    def test_weight_suffixed_cells_and_free_head_labels(self):
        with tempfile.TemporaryDirectory() as t:
            res, _ = self.run_read(Path(t), {"planted": 0.55}, weight=2.0, heads=("N", "F"))
            by = {e["head"]: e for e in res["primary"]}
            self.assertEqual(by["N"]["verdict"], "confirmed; V3 dial gates next")
            self.assertEqual(by["F"]["verdict"], "confirmed under the free head (the N entry is also confirmed)")
        with tempfile.TemporaryDirectory() as t:
            res, _ = self.run_read(Path(t), {"planted": 0.55}, heads=("N", "F"), shortlist=[{"arm": "planted", "head": "F"}])
            self.assertEqual(res["primary"][0]["verdict"], "confirmed under the free head only: helps a free head")

    def test_independent_bootstrap_streams_per_set(self):
        with tempfile.TemporaryDirectory() as t:
            self.run_read(Path(t), {"planted": 0.55})
        d = self.cmp._ref_draws
        self.assertFalse(np.array_equal(d["cid22_b"][0], d["aic4"][0]))           # same sizes, different streams
        self.assertEqual(self.cmp.REF_SEEDS["aic4"], [v2_common.BOOT_SEED, 1])

    def test_holm_step_down(self):
        cr = self.cr
        self.assertEqual(cr.holm([0.001, 0.03, 0.04, 0.5, 0.6, 0.7]), [True, False, False, False, False, False])
        self.assertEqual(cr.holm([0.03, 0.5, 0.6, 0.7, 0.8, 0.9]), [False] * 6)     # lone nominally significant entry is rejected
        self.assertEqual(cr.holm([0.008, 0.012, 0.9]), [True, True, False])
        self.assertEqual(cr.holm([0.02, 0.5]), [True, False])
        self.assertEqual(cr.holm([0.03, 0.5]), [False, False])

    def test_weak_lucky_entry_fails_holm_end_to_end(self):
        cr = self.cr
        # p just under 0.05 for an entry on a list of six: nominally significant, not Holm-significant
        entries = [{"arm": f"a{i}", "head": "N", "status": "OK", "p_one_sided": p, "regressions": [], "konjnd_select": {"regression": False}}
                   for i, p in enumerate([0.03, 0.4, 0.5, 0.6, 0.7, 0.9])]
        cr.verdicts(entries)
        self.assertFalse(any(e["confirmed"] for e in entries))
        self.assertEqual(entries[0]["verdict"], "not confirmed")

    def exposed(self, ledger: Path) -> bool:
        return ledger.exists()

    def test_refusals_leave_no_ledger_and_open_no_label(self):
        cr = self.cr
        def attempt(tamper):
            with tempfile.TemporaryDirectory() as t:
                root = Path(t)
                pin_path = self.build_tree(root, {"planted": 0.55})
                tamper(root, pin_path)
                opened = []
                real_load = cr.v2c_labels.load_label_rows
                with mock.patch.object(v2_common, "V2", root), mock.patch.object(cr, "V2", root), \
                        mock.patch.object(cr.v2c_labels, "load_label_rows", lambda *a, **k: (opened.append(1), real_load(*a, **k))[1]):
                    with self.assertRaises(cr.Refusal):
                        cr.main(["--confirmatory-read", "--pin", str(pin_path), "--root", str(root), "--bank", str(root / "bank"),
                                 "--out", str(root / "o.json"), "--ledger", str(root / "l.md"), "--arm", "planted"])
                self.assertFalse((root / "l.md").exists())
                self.assertEqual(opened, [])

        def cell(mutator, spec="r0"):
            def f(root, pin_path):
                path = self.cr.cell_path(root, spec, "N", 3) / "result.json"
                rec = json.loads(path.read_text())
                mutator(rec)
                path.write_text(json.dumps(rec))
            return f

        attempt(cell(lambda r: r["binaries"].update(panel="other")))
        attempt(cell(lambda r: r.update(variant="real", eval_variant="real"), "planted~p1"))   # control scored on the real tables
        attempt(cell(lambda r: r["predictions"]["csiq"].update(pred=r["predictions"]["csiq"]["pred"] + [0.0])))   # longer vector
        attempt(cell(lambda r: r["predictions"]["aic4"].update(table_sha256="0" * 64)))
        attempt(cell(lambda r: r["predictions"]["aic4"].update(keys_sha256="0" * 64)))
        attempt(cell(lambda r: r.update(frozen_sha256="0" * 64)))
        attempt(lambda root, pin: __import__("shutil").rmtree(self.cr.cell_path(root, "planted", "N", 4)))     # missing cell
        attempt(lambda root, pin: (self.cr.cell_path(root, "r0", "N", 0) / "fleet_receipt.json").unlink())    # fleet receipt mandatory
        attempt(lambda root, pin: (root / "labels" / "csiq" / "csiq_pairs.tsv").write_text("tampered"))        # label sha256

        def pin_edit(mutator):
            def f(root, pin_path):
                pin = json.loads(pin_path.read_text())
                mutator(pin)
                pin_path.write_text(json.dumps(pin))
            return f
        attempt(pin_edit(lambda p: p.pop("program_sha")))
        attempt(pin_edit(lambda p: p["code"].update({"rev4_featpot/v2_compare.py": "0" * 64})))
        attempt(pin_edit(lambda p: p["labels"]["aic4"].update(label_col="other")))             # orientation key unlisted
        attempt(pin_edit(lambda p: p["labels"]["csiq"].update(path="/x/_sealed/csiq_pairs.tsv")))

    def test_cell_names_must_match_the_weight(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            pin_path = self.build_tree(root, {"planted": 0.55}, weight=2.0)
            pin = json.loads(pin_path.read_text())
            pin["human_weight"] = None                                                     # cells are @h2: a bare read finds none
            pin_path.write_text(json.dumps(pin))
            with mock.patch.object(v2_common, "V2", root), mock.patch.object(self.cr, "V2", root):
                with self.assertRaises(self.cr.Refusal):
                    self.cr.main(["--confirmatory-read", "--pin", str(pin_path), "--root", str(root), "--bank", str(root / "bank"),
                                  "--out", str(root / "o.json"), "--ledger", str(root / "l.md"), "--arm", "planted"])

    def test_panel_must_be_pinned(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            pin_path = self.build_tree(root, {"planted": 0.55})
            with mock.patch.object(v2_common, "V2", root), mock.patch.object(self.cr, "V2", root), \
                    mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("ZEN_PANEL_BIN")
                try:
                    with self.assertRaises(self.cr.Refusal):
                        self.cr.main(["--confirmatory-read", "--pin", str(pin_path), "--root", str(root), "--bank", str(root / "bank"),
                                      "--out", str(root / "o.json"), "--ledger", str(root / "l.md"), "--arm", "planted"])
                finally:
                    os.environ["ZEN_PANEL_BIN"] = "/var/tmp/fitv2/bin-v2/panel"


if __name__ == "__main__":
    unittest.main()
