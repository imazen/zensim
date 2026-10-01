"""EFFAUDIT D8 proof: pre-rendered bootstrap selectors give the stored (old-path) bootstrap arrays bit for bit."""
import json, sys, time
sys.path.insert(0, "scripts/rev4_featpot"); sys.path.insert(0, "scripts")
import numpy as np
import v2_compare as c
from lib.zen_stats import panel_batch_indexed
bad = 0
for source in c.SOURCE_ORDER:
    for spec, head, i in (("r0", "N", 0), ("oracle_hi", "F", 3)):
        cdir = c.V2 / "cells" / f"{spec}__{head}" / f"without_{source}_s{i}"
        z = np.load(cdir / f"boot_B{c.BOOT_B}_seed{c.BOOT_SEED}.npz")
        cell = json.loads((cdir / "result.json").read_text())
        keys = c.keys_for(spec, source)
        pred = np.asarray(cell["prediction"], dtype=np.float64); y = keys.target.to_numpy(dtype=np.float64)
        t0 = time.time()
        rows = panel_batch_indexed({"p": pred, "y": y}, None, stats="srocc", timeout=7200,
                                   rendered_jobs=c.rendered_jobs(source, keys))
        by = {r["label"]: r["srocc"] for r in rows}
        boot = np.asarray([by[f"B{b}"] for b in range(c.BOOT_B)], dtype=np.float64)
        same = float(by["POINT"]) == float(z["point"]) and np.array_equal(boot, z["boot"])
        print(f"{source} {spec}__{head} s{i} identical={same} {time.time()-t0:.1f}s", flush=True)
        bad += not same
print("BAD", bad)
