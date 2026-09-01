#!/usr/bin/env python3
"""Turn a stored reference-metric dial-grid TSV into the per-cell score table
`bake_verdict --dial-peer-scores` reads.

WHY: a reference metric (ssim2, butteraugli, cvvdp, iwssim) has no bake, so
until `--dial-peer-scores` existed the DIAL panel could not be run on one and
the board's peer rows carried a hand-rolled, self-described "presentation-grade"
monotonicity with `tied_pct: null` and no ladder zones. "Is zensim's dial more
monotone than ssim2's?" is a first-order product question with no owner-computed
answer. This script does the KEY NORMALIZATION and nothing else — every
statistic is computed by `bake_verdict`, which is the owner.

Input  : reports/refmetrics/dialgrid_<metric>.tsv
         (ref_path, dist_path, codec, q, knob_tuple_json, <value col>)
Output : image_id \t codec \t q \t pred
         — byte-shape-identical to what `ZENSIM_DIAL_PRED_OUT` dumps, so the
         two round-trip (that identity is the mode's gate).

Two normalizations, both derived from the grid, not assumed:
  * codec: the TSV names the ENCODER (`zenjpeg`), the grid names the FAMILY
    (`jpeg`). Mapping is checked against the grid's own codec set.
  * q: for distance-parameterized JXL the TSV's `q` column is a constant
    placeholder and the real knob is `knob_tuple_json.distance`; the grid
    stores the monotone q-equivalent `100 - 4*distance`
    (docs/EVAL_PANEL_REQUIREMENT.md). Applied only where a distance is present.

Orientation: a DISTANCE metric (butteraugli) must be negated with `--negate`
so "higher = better quality" holds, exactly as `build_peer_fullevals.py` does
for its rank rows. Monotonicity is not orientation-free — an un-negated
distance would read as 100% inverted.

Refuses to write unless EVERY grid row is covered: a dial panel over a subset
of ladders is a different measurement, not a smaller one.
"""
import argparse, csv, json, os, sys

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True, help="refmetrics dialgrid TSV")
    ap.add_argument("--value-col", required=True, help="metric column in the TSV")
    ap.add_argument("--grid", required=True, help="dial-grid parquet to cover")
    ap.add_argument("--out", required=True)
    ap.add_argument("--negate", action="store_true",
                    help="metric is a DISTANCE (butteraugli): negate for quality orientation")
    a = ap.parse_args()

    import pyarrow.parquet as pq
    g = pq.read_table(a.grid, columns=["image_id", "codec", "q"]).to_pydict()
    grid_codecs = set(g["codec"])

    lut = {}
    with open(a.tsv, newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            img = os.path.basename(r["ref_path"]).rsplit(".", 1)[0]
            enc = r["codec"]
            fam = enc[3:] if enc.startswith("zen") and enc[3:] in grid_codecs else enc
            if fam not in grid_codecs:
                sys.exit(f"codec {enc!r} maps to {fam!r}, absent from the grid's {sorted(grid_codecs)}")
            knob = json.loads(r["knob_tuple_json"] or "{}")
            q = 100.0 - 4.0 * float(knob["distance"]) if "distance" in knob else float(r["q"])
            v = float(r[a.value_col])
            lut[(img, fam, round(q, 4))] = -v if a.negate else v

    out, missing = [], []
    for i in range(len(g["q"])):
        k = (g["image_id"][i], g["codec"][i], round(float(g["q"][i]), 4))
        if k in lut:
            out.append((k[0], k[1], g["q"][i], lut[k]))
        else:
            missing.append(k)
    if missing:
        sys.exit(f"{len(missing)} of {len(g['q'])} grid rows have no {a.value_col} score "
                 f"(first: {missing[:3]}) — refusing to write a partial cell table.")

    with open(a.out, "w") as fh:
        fh.write("image_id\tcodec\tq\tpred\n")
        for img, cod, q, v in out:
            fh.write(f"{img}\t{cod}\t{q}\t{v!r}\n")
    print(f"wrote {len(out)} cells -> {a.out} "
          f"(TSV carried {len(lut)}; grid coverage 100%)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
