#!/usr/bin/env python3
"""Build BAKE_INDEX.json — the single manifest of every bake in a dir, tying each to its
sha256, provenance (.spec.json), and factors (.metrics.json). The cure for "which bake was
that / what were its numbers / how was it trained" amnesia: one grep-able, committed index.

Usage: python3 scripts/v_next/build_bake_index.py <dir>   (default: corr-lq)
Writes <dir>/BAKE_INDEX.json + <dir>/BAKE_INDEX.md. Re-run after emit_bake_metrics.py.

Every row is sourced from the persisted sidecars (no recompute) — if a bake has no
.metrics.json it appears flagged, never silently dropped.
"""
import sys, json, glob
from pathlib import Path

DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/mnt/v/output/zensim/corr-lq")


def cid(m):
    return next((c["srocc"] for c in m["eval"]["panel"] if c["display"] == "CID22"), None)


def main():
    rows = []
    for b in sorted(glob.glob(str(DIR / "*.bin"))):
        name = Path(b).name
        mp = Path(b + ".metrics.json")
        sp = Path(b + ".spec.json")
        row = {"bake": name, "has_metrics": mp.exists(), "has_spec": sp.exists()}
        if mp.exists():
            m = json.load(open(mp))
            e = m["eval"]
            row.update({
                "bake_sha256": m["bake_sha256"][:16], "n_inputs": m["n_inputs"],
                "cid22": cid(m), "dial_mono": e["dial"]["monotonicity"],
                "dial_p5": e["dial"]["p5"], "dial_p95": e["dial"]["p95"],
                "ood_max": e["ood_max_abs_raw"], "corruption": e["corruption_gate_q20"],
                # closed_loop: {additive, basic_input_only, diffmap_basic_fraction}.
                # `additive` is the exact-gradient property (was mislabeled as diffmap_frac=1.0
                # for basic-input MLPs before 2026-07-18). Tolerate old-schema sidecars.
                "additive": (e.get("closed_loop") or {}).get("additive"),
                "basic_input": (e.get("closed_loop") or {}).get("basic_input_only"),
                "diffmap_frac": (e.get("closed_loop") or {}).get("diffmap_basic_fraction",
                                 e.get("diffmap_basic_fraction")),
                "train_corpora": (m["provenance"] or {}).get("train_corpora"),
                "reconstructed_prov": bool((m["provenance"] or {}).get("reconstructed")),
                "eval_at": m["tool"]["timestamp"],
            })
        rows.append(row)
    index = {"schema": "zensim.bake_index.v1", "dir": str(DIR), "n_bakes": len(rows),
             "n_with_metrics": sum(r["has_metrics"] for r in rows),
             "n_with_spec": sum(r["has_spec"] for r in rows), "bakes": rows}
    (DIR / "BAKE_INDEX.json").write_text(json.dumps(index, indent=2))
    # human-readable
    md = ["# Bake index — " + str(DIR), "",
          f"{index['n_bakes']} bakes · {index['n_with_metrics']} with metrics · {index['n_with_spec']} with provenance", "",
          "| bake | sha | n_in | CID22 | dial-mono | OOD max | corr | additive | basic-in | prov | evaluated |",
          "|---|---|--|--|--|--|--|--|--|--|--|"]
    for r in rows:
        if r["has_metrics"]:
            prov = "recon" if r.get("reconstructed_prov") else ("✓" if r["has_spec"] else "—")
            add = {True: "✓", False: "✗", None: "?"}[r.get("additive")]
            bin_ = {True: "✓", False: "—", None: "?"}[r.get("basic_input")]
            md.append(f"| {r['bake']} | `{r['bake_sha256']}` | {r['n_inputs']} | "
                      f"{r['cid22']:.4f} | {r['dial_mono']:.3f} | {r['ood_max']:.0f} | "
                      f"{r['corruption']*100:.0f}% | {add} | {bin_} | {prov} | {r['eval_at'][:10]} |")
        else:
            md.append(f"| {r['bake']} | — | — | **NO metrics.json** | | | | | | {'✓' if r['has_spec'] else '—'} | |")
    (DIR / "BAKE_INDEX.md").write_text("\n".join(md))
    print(f"wrote {DIR}/BAKE_INDEX.json + .md  ({index['n_bakes']} bakes, "
          f"{index['n_with_metrics']} metrics, {index['n_with_spec']} spec)")


if __name__ == "__main__":
    main()
