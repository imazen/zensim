"""First-fit compatibility through the unchanged p2_mlp.train owner, synthetic only."""
import hashlib
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rev4_featpot"))
# linear_probe's import installs its historical global scratch directory.
# Preserve the caller itself while containing that import-only side effect.
original_mkdir = Path.mkdir


def import_mkdir(path, *args, **kwargs):
    if path == Path("/var/tmp/rev4-featpot/tmp"):
        return None
    return original_mkdir(path, *args, **kwargs)


with (patch.object(Path, "mkdir", import_mkdir), patch.dict(os.environ),
      patch.object(tempfile, "tempdir", tempfile.tempdir)):
    import p2_mlp as caller


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ExistingCallerTests(unittest.TestCase):
    def test_p2_first_fit_allows_inputs_but_refuses_surviving_models(self):
        repo = Path(__file__).resolve().parents[2]
        trainer = Path(os.environ.get("SHIPPATH_TRAINER", repo / "target/debug/zensim_mlp_train"))
        self.assertTrue(trainer.is_file(), "build the trainer before running this test")
        with tempfile.TemporaryDirectory(prefix="shippath4-caller-") as tmp:
            root = Path(tmp)
            fields = {"ref_basename": [f"ref{i // 6}" for i in range(48)],
                      "human_score": np.tile(np.arange(6, dtype=np.float32) / 5, 8)}
            for i in range(946):
                fields[f"f{i}"] = (np.arange(48, dtype=np.float32) / 48 * (i + 1) if i < 4
                                   else np.zeros(48, dtype=np.float32))

            def prepare(name):
                dest = root / name
                dest.mkdir()
                for filename in ("fit.parquet", "test.parquet"):
                    pq.write_table(pa.table(fields), dest / filename, compression="zstd")
                (dest / "keep.txt").write_text("inputs may coexist with checkpoints\n")
                (dest / "unrelated.bin").write_bytes(b"unrelated caller input; never read or stamped")
                return dest

            def train(dest, seed):
                with patch.multiple(caller, TRAINER=trainer, EPOCHS=3, PAIRS_PER_EPOCH=128, LOG_EVERY=1):
                    caller.train(dest / "fit.parquet", None, dest, 32, "p2", seed, 101,
                                 dest / "unused_slice.txt", dump=True)

            first = prepare("first")
            inputs = {p.name: sha(p) for p in first.iterdir()}
            train(first, 1101)
            self.assertTrue((first / "best.bin").is_file())
            self.assertEqual(sorted(p.name for p in first.glob("ckpt_epoch*.bin")),
                             [f"ckpt_epoch{i:03}.bin" for i in range(3)])
            self.assertIn("Stamped 3 checkpoint dump(s)", (first / "train.log").read_text())
            self.assertTrue(all(sha(first / name) == digest for name, digest in inputs.items()))
            for artifact in ("ckpt_epoch001.bin", "ckpt_epoch999.bin", "ckpt_epochevil.bin",
                             "best.bin", "last.bin", "selected.bin", "custom-chosen.bin"):
                with self.subTest(artifact=artifact):
                    dest = prepare(artifact.replace(".", "-"))
                    shutil.copyfile(first / "ckpt_epoch001.bin", dest / artifact)
                    before = {p.name: sha(p) for p in dest.iterdir()}
                    original_run = caller.run

                    def configured_run(argv, log):
                        if artifact == "custom-chosen.bin":
                            argv[argv.index("--out") + 1] = str(dest / artifact)
                        original_run(argv, log)

                    with (patch.object(caller, "run", side_effect=configured_run),
                          self.assertRaisesRegex(RuntimeError, "rc=2")):
                        train(dest, 1103)
                    self.assertTrue(all(sha(dest / name) == digest for name, digest in before.items()))
                    self.assertEqual({p.name for p in dest.iterdir()}, set(before) | {"train.log"})
                    log = (dest / "train.log").read_text()
                    self.assertIn("checkpoint artifacts", log)
                    self.assertNotIn("[table-admission]", log)


if __name__ == "__main__":
    unittest.main()
