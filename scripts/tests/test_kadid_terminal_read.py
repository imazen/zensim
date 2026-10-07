"""Synthetic D2 orchestration with real Rust panel; no dataset labels."""

import builtins
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rev4_featpot"))
import kadid_terminal_read as owner


class TerminalRead(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.home() / "tmp")
        self.root = Path(self.temp.name)
        self.ledger = self.root / "DATA_SPLITS.md"
        self.ledger.write_text("Synthetic exposure ledger\n")
        self.journal = self.root / "spent.json"
        self.output = self.root / "result.json"
        self.labels = self.root / "labels__synthetic.tsv"
        self.receipt = self.root / "receipt.json"
        self.authorization = self.root / "authorization.json"
        self.opens = []
        panel = Path(os.environ["ZEN_PANEL_BIN"]).resolve()
        self.assertTrue(panel.is_file(), "caller must provide real Rust panel")
        self.b = (
            self.root
            / "zensim/weights/b_sdr_linear_cid80_inclwinsor_dense_dial_byid_2026-09-06.bin"
        )
        self.b.parent.mkdir(parents=True)
        self.b.write_bytes(b"synthetic model fixture")
        self.inspector = self.root / "inspector"
        contract = json.loads(owner.CONTRACT.read_text())
        metadata = dict(
            qualified_provenance=True,
            formula_revision=5,
            checkpoint_epoch="119",
            admitted_tables=7,
            pair_sampling="uniform",
            repro=dict(
                epochs=120,
                requested_epochs=120,
                pairs_per_epoch=50000,
                init_seed=1101,
                sample_seed=101,
                keep_features_n=420,
                inputs=[
                    dict(name=v["name"], sha256=v["table_sha256"])
                    for v in contract["routes"]["production"]
                ],
            ),
        )
        self.inspector.write_text(
            "#!/usr/bin/env python3\nimport json\nprint(json.dumps("
            + repr(metadata)
            + "))\n"
        )
        self.inspector.chmod(0o755)
        self.registration = self.root / "registration.md"
        self.registration.write_text("synthetic D2 registration")
        self.population = self.root / "population.tsv"
        self.predictions = self.root / "predictions.tsv"
        self.write_population()
        self.prediction_receipt = self.root / "predictions.json"
        self.qualification = self.root / "qualification.json"
        self.record = dict(
            schema="kadid-terminal-final-model-v1",
            design_line=owner.DESIGN,
            population_rows=2000,
            bootstrap_resamples=10000,
            bootstrap_seed=44001,
            orientation="quality",
            registration_sha256=owner.sha(self.registration),
            code_sha256={k: owner.sha(p) for k, p in owner.CODE.items()},
            human_sources=["kadid", "tid2013", "konfig", "cid22_a25"],
            fit_jobset="fitv2d1-20261007",
            selected_epoch=119,
            seed_index=0,
            population=self.spec(self.population),
            predictions=self.spec(self.predictions),
            panel=self.spec(panel),
            scorer=self.spec(self.inspector),
            inspector=self.spec(self.inspector),
            models={m: self.spec(self.b) for m in owner.MODELS},
            composition={"primary": self.spec(self.b)},
            bindings=[],
            labels=dict(
                self.spec(self.labels),
                format="tsv",
                ref_col="ref_path",
                dist_col="dist_path",
                label_col="human_score",
            ),
        )
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        self.repo_patch = patch.object(owner, "REPO", self.root)
        self.reg_patch = patch.object(owner, "REGISTRATION", self.registration)
        self.repo_patch.start()
        self.reg_patch.start()
        self.pin()

    def tearDown(self):
        self.repo_patch.stop()
        self.reg_patch.stop()
        self.temp.cleanup()

    def spec(self, path):
        return {"path": str(path), "sha256": owner.sha(path)}

    def write_population(self, reverse=False):
        with (
            self.population.open("w") as p,
            self.predictions.open("w") as s,
            self.labels.open("w") as l,
        ):
            pw, sw, lw = (csv.writer(f, delimiter="\t") for f in (p, s, l))
            pw.writerow(
                (
                    "source_row_id",
                    "pair_key",
                    "ref_basename",
                    "distortion_type",
                    "ref_path",
                    "dist_path",
                    "ref_pixels_sha256",
                    "dist_pixels_sha256",
                )
            )
            sw.writerow(("source_row_id", *owner.MODELS))
            lw.writerow(("ref_path", "dist_path", "human_score"))
            for i in range(2000):
                y = (i % 100) + (i // 100) / 100
                pw.writerow(
                    (
                        i,
                        f"key{i // 2}",
                        f"ref{i // 100}",
                        f"type{i % 5}",
                        f"r{i // 100}",
                        f"d{i}",
                        "a" * 64,
                        "b" * 64,
                    )
                )
                sw.writerow((i, -y if reverse else y, y, y))
                lw.writerow((f"r{i // 100}", f"d{i}", y))

    def pin(self):
        self.record["predictions"] = self.spec(self.predictions)
        self.prediction_receipt.write_text(
            json.dumps(
                dict(
                    schema="kadid-terminal-surface-predictions-v1",
                    labels_read=False,
                    surface="zensim::BakeScorer / Zensim::codec_target",
                    **{
                        k: self.record[k]
                        for k in (
                            "population",
                            "predictions",
                            "scorer",
                            "models",
                            "composition",
                            "bindings",
                        )
                    },
                )
            )
        )
        self.qualification.write_text(
            json.dumps(
                dict(
                    composition=self.record["composition"],
                    model_sha256=self.record["models"]["production"]["sha256"],
                    gates={
                        g: {"state": "pass", "artifact": self.spec(self.registration)}
                        for g in owner.GATES
                    },
                )
            )
        )
        self.record["prediction_receipt"] = self.spec(self.prediction_receipt)
        self.record["qualification"] = self.spec(self.qualification)
        self.receipt.write_text(json.dumps(self.record))
        subprocess.run(["git", "add", "receipt.json"], cwd=self.root, check=True)
        subprocess.run(
            [
                "git",
                "commit",
                "--allow-empty",
                "-qm",
                "test: synthetic pre-read receipt",
            ],
            cwd=self.root,
            check=True,
        )
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=self.root, text=True
        ).strip()
        self.auth = dict(
            schema="kadid-terminal-authorization-v1",
            authorize_once=True,
            design_line=owner.DESIGN,
            receipt_sha256=owner.sha(self.receipt),
            coordinator_message="Synthetic test authorization only",
            ledger=str(self.ledger),
            journal=str(self.journal),
            pre_read_commit=commit,
            receipt_repo_path="receipt.json",
        )
        self.authorization.write_text(json.dumps(self.auth))

    def run_read(self):
        return owner.execute(
            self.receipt, self.authorization, self.ledger, self.journal, self.output
        )

    def no_open(self, action):
        original = io.open
        builtin = builtins.open

        def guard(real):
            def checked(file, *args, **kwargs):
                if (
                    not isinstance(file, int)
                    and Path(file).resolve() == self.labels.resolve()
                ):
                    self.opens.append(str(file))
                    raise AssertionError("terminal sentinel opened")
                return real(file, *args, **kwargs)

            return checked

        with patch("io.open", guard(original)), patch("builtins.open", guard(builtin)):
            with self.assertRaises(
                (
                    ValueError,
                    FileNotFoundError,
                    PermissionError,
                    subprocess.CalledProcessError,
                )
            ):
                action()
        self.assertEqual(self.opens, [])

    def test_missing_authorization(self):
        self.authorization.unlink()
        self.no_open(self.run_read)
        self.assertFalse(self.journal.exists())

    def test_wrong_receipt_hash(self):
        self.auth["receipt_sha256"] = "0" * 64
        self.authorization.write_text(json.dumps(self.auth))
        self.no_open(self.run_read)

    def test_uncommitted_pin(self):
        self.receipt.write_text(self.receipt.read_text() + "\n")
        self.auth["receipt_sha256"] = owner.sha(self.receipt)
        self.authorization.write_text(json.dumps(self.auth))
        self.no_open(self.run_read)

    def test_changed_input_and_missing_gate(self):
        self.predictions.write_text(self.predictions.read_text() + "\n")
        self.no_open(self.run_read)
        self.pin()
        q = json.loads(self.qualification.read_text())
        q["gates"]["G-RD"]["state"] = "blocked"
        self.qualification.write_text(json.dumps(q))
        self.record["qualification"] = self.spec(self.qualification)
        self.receipt.write_text(json.dumps(self.record))
        # Re-pin a failed gate without replacing its contents via pin().
        subprocess.run(["git", "add", "receipt.json"], cwd=self.root, check=True)
        subprocess.run(
            ["git", "commit", "-qm", "test: failed gate pin"], cwd=self.root, check=True
        )
        self.auth.update(
            receipt_sha256=owner.sha(self.receipt),
            pre_read_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=self.root, text=True
            ).strip(),
        )
        self.authorization.write_text(json.dumps(self.auth))
        self.no_open(self.run_read)

    def test_symlink_source_and_binding(self):
        for slot in ("scorer", "bindings"):
            link = self.root / f"{slot}.link"
            link.symlink_to(self.labels)
            spec = {"path": str(link), "sha256": self.record["labels"]["sha256"]}
            if slot == "bindings":
                self.record[slot] = [spec]
            else:
                self.record[slot] = spec
            self.pin()
            self.no_open(self.run_read)

    def test_spent_journal_without_output(self):
        self.journal.write_text("{}")
        self.no_open(self.run_read)

    def test_wrong_population_budget_and_seed(self):
        for key, value in (
            ("population_rows", 1952),
            ("bootstrap_resamples", 9999),
            ("bootstrap_seed", -1),
            ("seed_index", 3),
            ("selected_epoch", 1),
        ):
            saved = self.record[key]
            self.record[key] = value
            self.pin()
            self.no_open(self.run_read)
            self.record[key] = saved

    def test_short_decoded_model_refuses(self):
        self.inspector.write_text(
            self.inspector.read_text().replace("'epochs': 120", "'epochs': 2")
        )
        self.record["inspector"] = self.spec(self.inspector)
        self.record["scorer"] = self.spec(self.inspector)
        self.pin()
        self.no_open(self.run_read)

    def test_malformed_label_adapter_refuses(self):
        self.record["labels"]["format"] = "unsupported"
        self.pin()
        self.no_open(self.run_read)

    def test_holdout_gate_binding_refuses(self):
        protected = self.root / "holdout"
        protected.mkdir()
        sentinel = protected / "sentinel.json"
        sentinel.symlink_to(self.labels)
        self.record["bindings"] = [
            {"path": str(sentinel), "sha256": self.record["labels"]["sha256"]}
        ]
        self.pin()
        self.no_open(self.run_read)

    def test_bad_labels_spend_before_open(self):
        self.labels.write_text("corrupt label fixture\n")
        with self.assertRaises(ValueError):
            self.run_read()
        self.assertTrue(self.journal.exists())
        self.assertIn("KADID-TERMINAL-SPENT:", self.ledger.read_text())
        self.assertEqual(json.loads(self.output.read_text())["confirmation"], "ERROR")

    def test_complete_pass_and_no_second_read(self):
        result = self.run_read()
        self.assertEqual(result["confirmation"], "PASS")
        self.assertEqual(result["paired_bootstrap"]["resamples"], 10000)
        self.assertEqual(result["paired_bootstrap"]["delta_b_se"], 0)
        for m in owner.MODELS:
            self.assertEqual(result["metrics"][m]["srocc_signed"], 1)
            self.assertEqual(result["diagnostics"][m]["within_reference_srocc"], 1)
            self.assertEqual(len(result["diagnostics"][m]["per_distortion_type"]), 5)
            self.assertEqual(
                result["diagnostics"][m]["scatter"]["schema"], "scatter-v2"
            )
        self.output.unlink()  # Losing an output must not allow a second read.
        self.no_open(self.run_read)

    def test_negative_signed_result_spent(self):
        self.write_population(reverse=True)
        self.pin()
        result = self.run_read()
        self.assertEqual(result["confirmation"], "FAIL")
        self.assertEqual(result["metrics"]["production"]["srocc_signed"], -1)
        self.assertFalse(any(result["gates"].values()))
        self.assertIn("**FAIL**", self.ledger.read_text())

    def test_reference_paired_se_against_independent_oracle(self):
        import numpy as np
        from scipy.stats import spearmanr

        pop = owner.rows(self.population)
        y = np.array([(i % 100) + (i // 100) / 100 for i in range(2000)])
        shift = np.array([(i // 100) % 4 for i in range(2000)])
        a = y + shift * 15
        b = y - shift * 7
        with self.predictions.open("w") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(("source_row_id", *owner.MODELS))
            writer.writerows((i, a[i], b[i], y[i]) for i in range(2000))
        self.pin()
        rng = np.random.default_rng(self.record["bootstrap_seed"])
        groups = [
            np.array([i for i, p in enumerate(pop) if p["ref_basename"] == r])
            for r in sorted({p["ref_basename"] for p in pop})
        ]
        deltas = []
        for _ in range(10000):
            idx = np.concatenate(
                [groups[g] for g in rng.integers(0, len(groups), len(groups))]
            )
            deltas.append(
                spearmanr(a[idx], y[idx]).statistic
                - spearmanr(b[idx], y[idx]).statistic
            )
        result = self.run_read()["paired_bootstrap"]
        self.assertGreater(result["delta_b_se"], 0)
        self.assertAlmostEqual(
            result["delta_b_se"], float(np.std(deltas, ddof=1)), places=12
        )
        self.assertAlmostEqual(
            result["delta_b"],
            spearmanr(a, y).statistic - spearmanr(b, y).statistic,
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
