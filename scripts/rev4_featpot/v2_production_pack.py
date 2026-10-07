"""Canonical deployment order: selected epoch -> densify -> quantize -> TRAIN dial."""
from pathlib import Path
from v2_common import FITBIN, dense_bake, sha
from v2_lodo_mlp import run, strict_training_groups


def pack_production(bake, dest, anchor):
    # Teacher role, revision, ordered keys and admission hashes precede its read.
    strict_training_groups([("TRAIN calibration", anchor, 0., 0., "withinref,both")])
    dense = dense_bake(bake, dest)
    packed = Path(dest) / "refit/production-f16.bin"
    run([str(FITBIN), "pack", "--in", str(dense), "--out", str(packed),
         "--dtype", "f16", "--zerobias-bulk", "0", "--protect-last", "--neg-tail",
         "--anchor", str(anchor), "--target-col", "human_score", "--verify", "none"],
        Path(dest) / "pack.log")
    return {"packed_model": str(packed), "packed_model_sha256": sha(packed),
            "dense_model_sha256": sha(dense), "TRAIN_calibration_table_sha256": sha(anchor),
            "deployment_order": "final epoch; canonical densify; f16 quantization; TRAIN calibration"}
