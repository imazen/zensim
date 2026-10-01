"""Admission gate for the v9 fleet predictor (bake_dial_refit built from main b926258e + pr/signedfeat a659715e).
P1: on bakes that read only f0-f1824 (acceptance-run cells, R4 last.bin), v9 predictions are byte-identical to the v8
    predictor's (bin-v3, cafed5ca) on each cell's held-out table.
P2: on a canon screen_main bake (reads f1825-f1852), v9 predictions on the cell's three dev legs reproduce the trainer's
    own epoch-119 dev SROCC (stride-decimated to 4096 rows exactly as zensim-validate panel.rs) to the logged 4 decimals."""
import glob, hashlib, json, random, subprocess, sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
from scipy.stats import spearmanr
V8 = "/var/tmp/fitv2/bin-v3/bake_dial_refit"; V9 = "/mnt/data/fitv2-canon/v9-target/release/bake_dial_refit"
OUT = Path.home() / "tmp/featpot-audit/v9"; ENV = {"ZENSIM_MAX_TIER": "v3", "RAYON_NUM_THREADS": "4", "PATH": "/usr/bin:/bin"}
def predict(binary, bake, corpus, out):
    subprocess.run([binary, "predict", "--bake", str(bake), "--corpus", str(corpus), "--score-units", "--out", str(out)],
                   check=True, env=ENV, capture_output=True)
    return out.read_bytes()
report = {"P1": [], "P2": []}
cells = sorted(glob.glob("/var/tmp/rev4-featpot/v2/cells/*@h32__*/without_*/refit/last.bin"))
random.Random(20261001).shuffle(cells)
seen, pick = set(), []
for b in cells:  # one per spec x head, up to 12
    spec_head = b.split("/")[-4]
    if spec_head not in seen:
        seen.add(spec_head); pick.append(b)
for b in pick[:12]:
    cell = Path(b).parent.parent; res = json.loads((cell / "result.json").read_text())
    spec = cell.parent.name.split("__")[0].split("@")[0]; src = cell.name.split("_s")[0].removeprefix("without_")
    lists = json.loads(Path("/var/tmp/rev4-featpot/v2/wide/keep_lists.json").read_text())["specs"][spec]
    table = Path(f"/var/tmp/rev4-featpot/v2/wide/{lists['family']}/{lists['variant']}/{src}.parquet")
    a = predict(V8, b, table, OUT / "p1_v8.tsv"); z = predict(V9, b, table, OUT / "p1_v9.tsv")
    report["P1"].append({"bake": b, "table": str(table), "identical": a == z, "sha_v8": hashlib.sha256(a).hexdigest()[:16],
                         "sha_v9": hashlib.sha256(z).hexdigest()[:16]})
    print("P1", cell.parent.name, cell.name, "identical" if a == z else "DIFFER", flush=True)
cell = Path(sys.argv[1]); root = Path("/var/tmp/rev4-featpot/v2c/wide/main/real")
log = cell / "train.log"; line = [l for l in log.read_text().splitlines() if l.strip().startswith("epoch 119 ")][0]
held = cell.name.split("_s")[0].removeprefix("without_")
legs = {"safesyn_development": root / "safesyn_dev.parquet", "cid22_development": root / "cid22_dev.parquet",
        "human_development": root / f"human_without_{held}_dev.parquet"}
for name, table in legs.items():
    logged = float(line.split(f"{name}: srocc=")[1].split()[0])
    predict(V9, cell / "refit/last.bin", table, OUT / "p2.tsv")
    pred = np.loadtxt(OUT / "p2.tsv", delimiter="\t", skiprows=1, usecols=-1)
    y = pq.read_table(table, columns=["human_score"]).column(0).to_numpy()
    n = min(len(pred), len(y)); stride = -(-n // 4096) if n > 4096 else 1
    rho = spearmanr(pred[:n:stride], y[:n:stride])[0]
    ok = bool(round(abs(rho), 4) == logged); rho = float(rho)
    report["P2"].append({"group": name, "rows": n, "stride": stride, "srocc": rho, "logged": logged, "match": ok})
    print("P2", name, f"n={n} stride={stride} srocc={rho:+.6f} logged={logged} {'MATCH' if ok else 'MISMATCH'}", flush=True)
report["pass"] = all(r["identical"] for r in report["P1"]) and all(r["match"] for r in report["P2"])
(OUT / "v9_predictor_gate.json").write_text(json.dumps(report, indent=1) + "\n")
print("PASS" if report["pass"] else "FAIL")
