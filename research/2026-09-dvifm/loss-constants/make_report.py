#!/usr/bin/env python3
"""Assemble the lane deliverables from fit artefacts:
  benchmarks/dvifm_constants_2026-09-20.{md,json} + constants-v1.json.
Usage: make_report.py <artefact.json> [...] --out-dir <dir>
"""
import json, sys
from pathlib import Path

def main():
    arts = [json.loads(Path(p).read_text()) for p in sys.argv[1:] if p.endswith(".json")]
    out = Path(sys.argv[sys.argv.index("--out-dir") + 1])
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for a in arts:
        fin = a["final"]["params"]
        betas = []
        for g, levs in fin["groups"].items():
            for l, lv in enumerate(levs):
                betas.append((g, l, lv["beta"]))
        rows.append({
            "domain": a["name"], "weber_eps": a.get("weber_eps"),
            "beta_mode": fin["beta_mode"], "chroma_shared": fin["chroma_shared"],
            "betas": betas,
            "dev_rank_loss": a["final"]["dev"]["rank_loss"],
            "dev_srocc": a["final"]["dev"]["srocc"],
            "lambda": a["final"]["lambda_prior"],
            "intervals": {k: v.get("interval") for k, v in
                          (a["stages"].get("E6_profile") or {}).items()},
        })
    (out / "dvifm_constants_2026-09-20.json").write_text(
        json.dumps({"domains": rows}, indent=1, default=str))
    print(json.dumps(rows, indent=1, default=str))

if __name__ == "__main__":
    main()
