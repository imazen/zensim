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


class ConfirmRead(unittest.TestCase):
    SETS = ("cid22_b", "aic4", "konjnd_jpeg_select", "konjnd_jpeg_terminal", "csiq", "mcljci")
    REFS, PER_REF = 30, 8
    B = 200

    @classmethod
    def setUpClass(cls):
        import v2_compare
        cls.cmp = v2_compare
        cls.saved = (v2_compare.BOOT_B, v2_compare.SEED_DRAWS, dict(v2_compare._ref_draws), dict(v2_compare._rendered))
        v2_compare.BOOT_B = cls.B
        v2_compare.SEED_DRAWS = np.random.default_rng(v2_common.BOOT_SEED + 1).integers(0, 10, size=(cls.B, 10))
        v2_compare._ref_draws.clear()
        v2_compare._rendered.clear()
        import v2_confirm_read as cr
        cls.patches = [mock.patch.object(cr, "HEADS", ("N",)), mock.patch.object(v2_compare, "HEADS", ("N",))]
        for p in cls.patches:
            p.start()

    @classmethod
    def tearDownClass(cls):
        for p in cls.patches:
            p.stop()
        cmp = cls.cmp
        cmp.BOOT_B, cmp.SEED_DRAWS = cls.saved[0], cls.saved[1]
        cmp._ref_draws.clear(), cmp._ref_draws.update(cls.saved[2])
        cmp._rendered.clear(), cmp._rendered.update(cls.saved[3])

    def build_tree(self, root: Path, planted: dict):
        """Synthetic confirm tree: arms r0 + the keys of `planted` (noise sd per arm) + their three permuted controls."""
        import v2_confirm_read as cr
        rng = np.random.default_rng(11)
        (root / "wide" / "confirm" / "main").mkdir(parents=True)
        (root / "wide" / "keep_lists.json").write_text("{}")
        receipt = {"schema": "rev4-featpot-v2c-confirm-v1", "width": 1825, "feature_set_id": "x", "sets": {}}
        labels, truth = {}, {}
        for name in self.SETS:
            n = self.REFS * self.PER_REF
            keys = pd.DataFrame({"pair_key": [f"{name}-{i}" for i in range(n)], "row_id": np.arange(n),
                                 "ref_basename": [f"ref{i // self.PER_REF}" for i in range(n)], "member_set": name})
            kp = root / "wide" / "confirm" / "main" / f"{name}.keys.parquet"
            pq.write_table(pa.Table.from_pandas(keys, preserve_index=False), kp)
            receipt["sets"][name] = {"rows": n, "tables": {"main": {"real": {
                "rel": str(kp.with_name(f"{name}.parquet").relative_to(root)), "keys_sha256": v2_common.sha(kp)}}}}
            q = rng.normal(size=n)
            truth[name] = q
            # the label file's own orientation, fixed here independently of the module's declaration table
            sign = -1.0 if name in ("aic4", "konjnd_jpeg_select", "konjnd_jpeg_terminal", "mcljci") else 1.0
            lab = pd.DataFrame({"pair_key": [*keys.pair_key, "dropped-identical"],
                                "score": [*(sign * (2.0 * q + 3.0)), 0.5]})
            lp = root / f"{name}.labels.parquet"
            pq.write_table(pa.Table.from_pandas(lab, preserve_index=False), lp)
            labels[name] = {"path": str(lp), "column": "score", "join": "pair_key"}
        (root / "wide" / "confirm" / "receipt.json").write_text(json.dumps(receipt))
        wide = {f"main/{v}": f"wide-{v}" for v in ("real", "p1", "p2", "p3")}
        pin = {"schema": cr.PIN_SCHEMA, "reference": "r0", "candidates": sorted(planted), "heads": ["N"],
               "shortlist": [{"arm": a, "head": "N"} for a in sorted(planted)],
               "wide_receipts": wide, "confirm_receipt_sha256": v2_common.sha(root / "wide" / "confirm" / "receipt.json"),
               "keep_lists_sha256": v2_common.sha(root / "wide" / "keep_lists.json"),
               "binaries": {"zensim_mlp_train": "t", "bake_dial_refit": "b", "panel": "p"}, "local": True}
        for spec, sd in {"r0": 1.0, **{a: s for a, s in planted.items()},
                         **{f"{a}~p{k}": 1.0 for a in planted for k in (1, 2, 3)}}.items():
            variant = f"p{spec.partition('~p')[2]}" if "~p" in spec else "real"
            for head in ("N",):
                for seed in range(10):
                    preds = {name: {"pred": (truth[name] + rng.normal(0, sd, len(truth[name]))).tolist()}
                             for name in self.SETS}
                    d = cr.cell_path(root, spec, head, seed)
                    d.mkdir(parents=True)
                    (d / "result.json").write_text(json.dumps({
                        "schema": "rev4-featpot-v2c-confirm-cell-v1", "spec": spec, "head": head, "seed_index": seed,
                        "family": "main", "variant": variant, "wide_receipt_sha256": wide[f"main/{variant}"],
                        "confirm_receipt_sha256": pin["confirm_receipt_sha256"], "keep_lists_sha256": pin["keep_lists_sha256"],
                        "binaries": pin["binaries"], "predictions": preds}))
        pin_path = root / "pin.json"
        pin["labels"] = labels
        pin_path.write_text(json.dumps(pin))
        return pin_path, pin_path

    def run_read(self, root: Path, planted: dict, extra=()):
        import v2_confirm_read as cr
        pin_path, lj = self.build_tree(root, planted)
        out = root / "read.json"
        ledger = root / "ledger.md"
        with mock.patch.object(v2_common, "V2", root), mock.patch.object(cr, "V2", root):
            cr.main(["--confirmatory-read", "--pin", str(pin_path), "--root", str(root),
                     "--out", str(out), "--ledger", str(ledger), *[x for a in planted for x in ("--arm", a)], *extra])
        return json.loads(out.read_text()), ledger

    def test_planted_effect_found_null_not_and_orientation_applied(self):
        with tempfile.TemporaryDirectory() as t:
            res, ledger = self.run_read(Path(t), {"planted": 0.55, "nullarm": 1.0})
            hit, miss = res["secondary_v1_v2"]["planted_N"], res["secondary_v1_v2"]["nullarm_N"]
            self.assertEqual(hit["status"], "OK")
            self.assertTrue(hit["V1"] and hit["V2"], hit["verdict"])
            self.assertGreaterEqual(len(hit["v1_sources"]), 4)
            self.assertNotIn("konjnd_jpeg_terminal", hit["sources"])    # the terminal split never counts
            for s in hit["sources"].values():
                self.assertGreater(s["delta"], 0.03)                     # orientation applied: improvement is positive
                self.assertGreater(s["r0_mean"], 0.5)                    # distortion labels were negated
            self.assertFalse(miss["V1"], miss["verdict"])
            self.assertEqual(miss["regressions"], [])
            self.assertIn("konjnd_jpeg_terminal", res["sanity_guard"]["planted_N"])
            self.assertIn("pending, update after read", ledger.read_text())

    def test_primary_holm_keeps_planted_rejects_nulls(self):
        arms = {"planted": 0.6, "nullA": 1.0, "nullB": 1.0, "nullC": 1.0}
        with tempfile.TemporaryDirectory() as t:
            res, _ = self.run_read(Path(t), arms)
        by = {e["arm"]: e for e in res["primary"]}
        self.assertTrue(by["planted"]["confirmed"], by["planted"])
        self.assertLess(by["planted"]["p_one_sided"], 0.05 / 4)
        self.assertGreater(by["planted"]["mean_excess"], 0.05)
        for n in ("nullA", "nullB", "nullC"):
            self.assertFalse(by[n]["confirmed"], by[n])
            self.assertEqual(by[n]["verdict"], "not confirmed")

    def test_holm_step_down(self):
        import v2_confirm_read as cr
        self.assertEqual(cr.holm([0.001, 0.03, 0.04, 0.5, 0.6, 0.7]), [True, False, False, False, False, False])
        # a lone nominally significant entry (p < 0.05) on a list of six is NOT kept: it must beat 0.05 / 6
        self.assertEqual(cr.holm([0.03, 0.5, 0.6, 0.7, 0.8, 0.9]), [False] * 6)
        self.assertEqual(cr.holm([0.008, 0.012, 0.9]), [True, True, False])  # 0.008<=.05/3, 0.012<=.05/2, 0.9>.05
        self.assertEqual(cr.holm([0.02, 0.5]), [True, False])                # m=2: smallest needs <= 0.025
        self.assertEqual(cr.holm([0.03, 0.5]), [False, False])

    def test_label_join_expands_stimuli_ignores_dropped_and_refuses_uncovered(self):
        import v2_confirm_read as cr
        keys = pd.DataFrame({"pair_key": ["a", "b", "c"], "row_id": [0, 1, 2], "ref_basename": ["r0", "r0", "r1"]})
        with tempfile.TemporaryDirectory() as t:
            lp = Path(t) / "l.parquet"
            lab = pd.DataFrame({"pair_key": ["a", "b", "b", "c", "gone"], "v": [1.0, 2.0, 3.0, 4.0, 9.0]})
            pq.write_table(pa.Table.from_pandas(lab, preserve_index=False), lp)
            out = cr.load_labels({"path": str(lp), "column": "v"}, "aic4", keys)      # aic4 is distortion-oriented
            self.assertEqual(out.pred_row.tolist(), [0, 1, 1, 2])
            self.assertEqual(out.y_quality.tolist(), [-1.0, -2.0, -3.0, -4.0])
            self.assertEqual(out.attrs["label_rows_not_predicted"], 1)
            out = cr.load_labels({"path": str(lp), "column": "v"}, "cid22_b", keys)    # quality-oriented: kept as is
            self.assertEqual(out.y_quality.tolist(), [1.0, 2.0, 3.0, 4.0])
            pq.write_table(pa.Table.from_pandas(lab.iloc[:3], preserve_index=False), lp)
            with self.assertRaises(cr.Refusal):                                       # key "c" has no label
                cr.load_labels({"path": str(lp), "column": "v"}, "cid22_b", keys)

    def test_refusals(self):
        import v2_confirm_read as cr
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            pin_path, lj = self.build_tree(root, {"planted": 0.55})
            base = ["--pin", str(pin_path), "--root", str(root), "--out", str(root / "o.json"),
                    "--ledger", str(root / "l.md"), "--arm", "planted"]
            with mock.patch.object(v2_common, "V2", root), mock.patch.object(cr, "V2", root):
                with self.assertRaises(cr.Refusal):                       # no acknowledgement flag
                    cr.main(base)
                with self.assertRaises(cr.Refusal):                       # candidate list differs from the pin
                    cr.main(["--confirmatory-read", *base, "--arm", "extra"])
                with self.assertRaises(cr.Refusal):                       # labels must name every set
                    bad = json.loads(pin_path.read_text())
                    bad["labels"].pop("csiq")
                    pin_path.write_text(json.dumps(bad))
                    cr.main(["--confirmatory-read", *base])
                self.assertFalse((root / "l.md").exists())                # nothing recorded, nothing read

    def test_cell_hash_mismatch_refused(self):
        import v2_confirm_read as cr
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            pin_path, lj = self.build_tree(root, {"planted": 0.55})
            cell = cr.cell_path(root, "r0", "N", 3) / "result.json"
            rec = json.loads(cell.read_text())
            rec["binaries"]["panel"] = "other"
            cell.write_text(json.dumps(rec))
            with mock.patch.object(v2_common, "V2", root), mock.patch.object(cr, "V2", root):
                with self.assertRaises(cr.Refusal):
                    cr.main(["--confirmatory-read", "--pin", str(pin_path), "--root", str(root),
                             "--out", str(root / "o.json"), "--ledger", str(root / "l.md"), "--arm", "planted"])


if __name__ == "__main__":
    unittest.main()
