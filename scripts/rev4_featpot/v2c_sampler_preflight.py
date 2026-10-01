"""CANONTAB review 14: sampler preflight for the confirmatory fits on the canon root. Rebuilds v2_confirm_fit's train groups
(TEACHERS legs + human_all; weights via acceptance_weight; human weight 32) for the main family, real variant, and
runs subset_sim --require-disjoint-sampler-windows over the ten confirm sample seeds at 120 x 50,000.
The sampler does not read the group loss mode (rank/mse/both), so `:withinref` replays the fits' draws exactly.
  python v2c_sampler_preflight.py [OUT.json] [SUBSET_SIM]   (subset_sim built from this tree, release)"""
import json, subprocess, sys
from pathlib import Path
OUT = sys.argv[1] if len(sys.argv) > 1 else str(Path.home() / "tmp/featpot-audit/preflight/confirm_sampler_preflight.json")
SUBSET_SIM = sys.argv[2] if len(sys.argv) > 2 else "/mnt/data/fitv2-canon/preflight-target/release/subset_sim"
sys.argv = [sys.argv[0], "--root", "/var/tmp/rev4-featpot/v2c"]  # v2_common reads the root from argv
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v2_common as c  # noqa: E402
from v2_lodo_mlp import checked, refs_of  # noqa: E402
receipt = json.loads((c.V2 / "wide/main/real/receipt.json").read_text())
legs = receipt["legs"]
groups = []
for leg, (_, _, val_w) in c.TEACHERS.items():
    fit, dev = checked(legs[leg]["fit"]), checked(legs[leg]["dev"])
    w = c.acceptance_weight(c.NOMINAL_WEIGHT[leg], refs_of(fit))
    groups += [f"{leg}:{fit}:{w}:0:withinref", f"{leg}_development:{dev}:0:{val_w}:withinref"]
hfit, hdev = checked(legs["human_all"]["fit"]), checked(legs["human_all"]["dev"])
w = c.acceptance_weight(32.0, refs_of(hfit))
groups += [f"human:{hfit}:{w}:0:withinref", f"human_development:{hdev}:0:{c.HUMAN_VAL_WEIGHT}:withinref"]
seeds = [c.confirm_seeds(i)[1] for i in range(10)]
cmd = [SUBSET_SIM, *[a for g in groups for a in ("--group", g)],
       "--seeds", ",".join(map(str, seeds)), "--epochs", "120", "--pairs-per-epoch", "50000",
       "--require-disjoint-sampler-windows", "--out", OUT]
print(" ".join(cmd), flush=True)
sys.exit(subprocess.run(cmd).returncode)
