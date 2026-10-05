"""Surviving-file regression through the Rust CLI; synthetic rows only.

Build zensim_mlp_train first. SHIPPATH_TRAINER can select a pinned binary for
negative controls. No checkpoint from the failed reuse may be restamped.
"""
import json
import hashlib
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


class CheckpointOwnershipTests(unittest.TestCase):
    def test_surviving_checkpoint_refuses_different_seed_and_admission_contract(self):
        repo = Path(__file__).resolve().parents[2]
        binary = Path(os.environ.get("SHIPPATH_TRAINER", repo / "target/debug/zensim_mlp_train"))
        self.assertTrue(binary.is_file(), f"build trainer before running regression: {binary}")
        with tempfile.TemporaryDirectory(prefix="shippath3-checkpoint-") as tmp:
            root = Path(tmp)
            keep = root / "keep.txt"
            keep.write_text("0\n1\n2\n3\n")
            rng = np.random.default_rng(903)
            fields = {"ref_basename": [f"ref{i // 6}" for i in range(48)],
                      "human_score": np.tile(np.arange(6, dtype=np.float32), 8)}
            for i in range(720):
                fields[f"f{i}"] = (rng.uniform(0, 1, 48).astype(np.float32) if i < 228 or i >= 372
                                   else np.full(48, np.nan, dtype=np.float32))
            admitted = root / "oracle.parquet"
            pq.write_table(pa.table(fields), admitted, compression="zstd")
            declaration = {"feature_set_id": "basic+peaks+v2@w1825/rev5_localwin#36c3f3af",
                           "formula_revision": 5, "decoder_era": "synthetic-ownership-regression"}
            Path(f"{admitted}.manifest.json").write_text(json.dumps(declaration))
            historical = root / "historical.parquet"
            pq.write_table(pa.table({k: v for k, v in fields.items()
                                     if not k.startswith("f") or int(k[1:]) < 372}), historical, compression="zstd")

            def command(dest, seed, every, strict=False):
                dest.mkdir()
                checkpoint_dir = dest / "ckpt"
                checkpoint_dir.mkdir()
                table = admitted if strict else historical
                argv = [str(binary), "--group", f"synthetic:{table}:1.0:1.0:withinref,both",
                        "--target-column", "human_score", "--target-scale", "1", "--hidden", "8",
                        "--epochs", "3", "--pairs-per-epoch", "128", "--init-seed", str(seed),
                        "--sample-seed", "101", "--pair-sampling", "uniform", "--max-features",
                        "720" if strict else "372", "--keep-features", str(keep), "--mse-weight", "1",
                        "--early-stop-patience", "0", "--val-policy", "mean", "--val-aggregate", "geomean3",
                        "--out-dtype", "f32", "--log-every", "1", "--no-auto-eval", "--out", str(dest / "best.bin"),
                        "--dump-checkpoints-every", str(every), "--dump-checkpoints-dir", str(checkpoint_dir)]
                if not strict:
                    argv += ["--historical-replay", "synthetic historical checkpoint ownership regression; never ship"]
                return argv, checkpoint_dir

            def run(argv):
                return subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      env={**os.environ, "RAYON_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})

            old_cmd, old_ckpt = command(root / "old", 1101, 1)
            old = run(old_cmd)
            self.assertEqual(old.returncode, 0, old.stdout)
            surviving = (old_ckpt / "ckpt_epoch001.bin").read_bytes()
            evidence = {"scope": "synthetic-only surviving-file CLI regression",
                        "trainer_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                        "surviving_checkpoint_sha256": hashlib.sha256(surviving).hexdigest(),
                        "original_init_seed": 1101, "new_init_seed": 1103, "reuse": []}
            # Establish the new admission contract is real through a fresh Rust
            # invocation; a refusal caused by invalid input would prove nothing.
            fresh_cmd, fresh_ckpt = command(root / "fresh-qualified", 1103, 2, strict=True)
            fresh = run(fresh_cmd)
            self.assertEqual(fresh.returncode, 0, fresh.stdout)
            self.assertIn('"qualified_provenance":true', fresh.stdout.replace(" ", ""))
            self.assertTrue((fresh_ckpt / "ckpt_epoch002.bin").is_file())
            evidence["fresh_table_admission"] = json.loads(next(
                line.removeprefix("[table-admission] ") for line in fresh.stdout.splitlines()
                if line.startswith("[table-admission] ")))
            for strict in (False, True):
                argv, ckpt = command(root / f"reuse-{strict}", 1103, 2, strict=strict)
                stale = ckpt / "ckpt_epoch001.bin"
                stale.write_bytes(surviving)
                result = run(argv)
                with self.subTest(strict=strict):
                    self.assertNotEqual(result.returncode, 0, result.stdout)
                    self.assertIn("checkpoint directory must be empty", result.stdout)
                    self.assertEqual(stale.read_bytes(), surviving)
                    self.assertEqual(list(ckpt.iterdir()), [stale])
                    self.assertFalse((ckpt.parent / "best.bin").exists())
                    self.assertNotIn("[table-admission]", result.stdout)
                    evidence["reuse"].append({"contract": "qualified Rev5" if strict else "historical replay",
                                               "returncode": result.returncode,
                                               "surviving_bytes_unchanged": True,
                                               "new_outputs": 0, "table_admission_reached": False})
            print(json.dumps(evidence, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
