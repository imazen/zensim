"""Verify full E28 executor smokes through the existing harvest and bake owners."""
import argparse
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys


def verify(root, arm, tools, inspector):
    sys.path.insert(0, str(tools))
    harvest = importlib.import_module("harvest_fit_cells")
    job = json.loads((root / f"SMOKE_JOB_{arm}.json").read_text())
    name = job["cell"]["image_path"]
    receipt = harvest.verify_blob(root / f"smoke-{arm}.tar.gz", root / f"verified-{arm}", name, job["kind"])
    cell = harvest.cell_dir(root / f"verified-{arm}", name, job["kind"])
    r = json.loads((cell / "result.json").read_text())
    assert r["selected_epoch"] == 119 and r["epoch_rule"] == "last"
    assert r["epochs"] == 120 and r["pairs_per_epoch"] == 50000
    assert r["heldout"] == "kadid" and r["seed_index"] == 0
    assert r["kept_features"] == 420 and r["head"] == "N" and r["hidden"] == 128
    assert r["ssim2_recipe"]["arm"] == arm
    assert receipt["tier"]["effective"] == "v3"
    assert len(r["prediction"]) == r["rows"] == 7869
    expected = {"safesyn", "cid22", "coverage", "tid2013", "konfig", "cid22_a25", "aic3"} if arm == "s2o" else {"cid22", "tid2013", "konfig"}
    assert set(r["train_weights"]) == expected
    assert ("coverage_leg" in r) == (arm == "s2o")
    assert "hdr_leg" not in r
    bake = cell / "refit/last.bin"
    inspected = json.loads(subprocess.check_output([str(inspector), "inspect", str(bake)], text=True))
    metadata = {m["key"]: m for m in inspected["metadata"]}
    repro = json.loads(metadata["zentrain.repro"]["value_text"])
    pooled = repro["pooled_objective"]
    active = ["safesyn", "cid22", "tid2013"] if arm == "s2o" else ["cid22", "tid2013", "konfig"]
    assert pooled["legs"] == active
    assert pooled["rank_share"] == pooled["pearson_weight"] == 0.5
    assert pooled["pearson_batch_rows"] == 32
    timing = (root / f"TIME_{arm}.txt").read_text()
    rss_kib = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", timing)[1])
    peak = int((root / f"MEMORY_PEAK_{arm}.txt").read_text())
    events = dict(line.split() for line in (root / f"MEMORY_EVENTS_{arm}.txt").read_text().splitlines())
    assert int(events["oom_kill"]) == 0 and peak < 6 * 1024**3
    log = (cell / "train.log").read_text()
    coverage = json.loads(metadata["zentrain.sample_coverage"]["value_text"])
    live_digest = re.search(r"ZENSIM_SAMPLE_DIGEST ([0-9a-f]+)", log)[1]
    assert coverage["digest"] == live_digest
    assert coverage["pooled_objective"]["legs"] == active
    assert coverage["pooled_objective"]["rank_share"] == 0.5
    final = next(line for line in log.splitlines() if line.startswith("  epoch 119"))
    seconds = float(re.search(r"t=([0-9.]+)s", final)[1])
    record = dict(status="PASS", cell=str(cell), selected_epoch=119, epochs=120, pairs_per_epoch=50000,
                  prediction_rows=r["rows"], selected_bake_sha=receipt["selected_bake_sha"],
                  program_sha=receipt["program_sha"], data_sha=receipt["data_sha"], argv_sha=receipt["argv_sha"],
                  blob_sha256=harvest.digest(root / f"smoke-{arm}.tar.gz"), tier=receipt["tier"],
                  training_seconds=seconds, live_sampler_digest=live_digest, pooled_objective=pooled, train_weights=r["train_weights"],
                  rss_max_kib=rss_kib, memory_peak_bytes=peak, memory_cap_bytes=6*1024**3,
                  score=r["score"], qualified_provenance=repro["table_admission"]["qualified_provenance"],
                  konfig_deviation=r["ssim2_recipe"]["konfig_deviation"],
                  verification_owner="harvest_fit_cells.verify_blob + zenpredict inspect")
    (root / f"SMOKE_PASS_{arm}.json").write_text(json.dumps(record, indent=1)+"\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--arm", choices=("s2o", "s2m"), required=True)
    ap.add_argument("--tools", type=Path, required=True)
    ap.add_argument("--inspector", type=Path, required=True)
    a = ap.parse_args()
    verify(a.root, a.arm, a.tools, a.inspector)
