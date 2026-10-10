"""Write the E33 result summary (JSON + markdown) from the measured artifacts, without recomputing anything.

Inputs: assessment-e33/e33_e21.json (+ decision.json), gates/GATES.json, E33_VERDICT.json and the launch packet.
Outputs: benchmarks/e33_result_summary_2026-10-10.{json,md}. Every number is copied from those files.
"""

import argparse
import json
from pathlib import Path

from v2_common import sha

E = Path("/mnt/v/output/zensim/e33-impl-2026-10-09")
SOURCES = ("kadid", "tid2013", "konfig", "cid22_a25")
NAMES = {"kadid": "KADID", "tid2013": "TID2013", "konfig": "KonFiG", "cid22_a25": "CID22-A(25)"}


def g(x, digits=6):
    return f"{x:.{digits}g}" if isinstance(x, float) else str(x)


def build(out_json, out_md):
    e21 = json.loads((E / "assessment-e33/e33_e21.json").read_text())
    decision = json.loads((E / "assessment-e33/decision.json").read_text())
    gates = json.loads((E / "gates/GATES.json").read_text())
    verdict = json.loads((E / "E33_VERDICT.json").read_text())
    packet = json.loads((E / "packet/PACKET.json").read_text())
    pins = {name: sha(E / name) for name in ("assessment-e33/e33_e21.json", "assessment-e33/decision.json",
                                              "gates/GATES.json", "E33_VERDICT.json", "packet/PACKET.json")}
    summary = dict(schema="e33-result-summary-v1", registration="benchmarks/e33_registration_2026-10-09.md",
                   verdict=verdict, owner_note=dict(date="2026-10-10", verbatim="i think it might matter, dont write off c",
                                                    effect="registered verdict unchanged; arm C kept under investigation"), e21=dict(decisions=e21["decisions"], c_vs_a=e21["c_vs_a"],
                                             observation_counts=decision["observation_counts"],
                                             report_only=e21["report_only"]),
                   gates=gates, packet=dict(program_sha=packet["program_sha"], image_id=packet["image_id"],
                                            data_sha=packet["data_sha"], jobsets=packet["jobsets"]),
                   artifact_sha256=pins)
    out_json.write_text(json.dumps(summary, indent=1, allow_nan=False) + "\n")

    d = e21["decisions"]
    lines = ["# E33 registered results — 2026-10-10", ""]
    adopt = verdict["adopt"]
    lines += [f"**Verdict: {verdict['rule']}. Adopt: {'production seed 0 (keep control)' if adopt == 'control' else 'Arm ' + adopt.upper()}.** "
              "\"Adopt\" means the next production candidate, entering full qualification; E33 qualifies nothing.", ""]
    lines += ["Owner, verbatim 2026-10-10 (conveyed by the coordinator): \"i think it might matter, dont write off c\". "
              "The registered verdict above stands as computed; Arm C stays under investigation (its steering failures "
              "are being diagnosed with the STEERFIX method; no threshold change).", ""]
    lines += ["| Arm | E21 as-good | Failed label-free/runtime gates | Eligible |", "|---|---|---|---|"]
    for arm in ("a", "c"):
        v = verdict["arms"][arm]
        lines.append(f"| {arm.upper()} | {'yes' if v['e21_as_good'] else 'no'} | "
                     f"{', '.join(v['failed_gates']) or 'none'} | {'yes' if v['eligible'] else 'no'} |")
    lines += ["", "## E21 human rank (section 9.1), each arm against E33's fresh 40-cell control", "",
              "Ten seed units; source-equal weights. Guards: mean ≥ −0.002, each source ≥ −0.005, W2 Δ > −2·SE.", "",
              "| Arm | Mean Δ | SE | W2 Δ | W2 SE | Mean / source / W2 guards | As-good |", "|---|---:|---:|---:|---:|---|---|"]
    for arm in ("a", "c"):
        x = d[arm]
        gd = x["guards"]
        lines.append(f"| {arm.upper()} | {g(x['signed']['delta'])} | {g(x['signed']['se'])} | {g(x['w2']['delta'])} | "
                     f"{g(x['w2']['se'])} | {'/'.join('PASS' if gd[k] else 'FAIL' for k in ('mean', 'each_source', 'w2'))} | "
                     f"{'yes' if x['as_good'] else 'no'} |")
    lines += ["", "| Arm | " + " | ".join(NAMES[s] for s in SOURCES) + " |", "|---|" + "---:|" * len(SOURCES)]
    for arm in ("a", "c"):
        lines.append(f"| {arm.upper()} | " + " | ".join(g(d[arm]["per_source"][s]) for s in SOURCES) + " |")
    c = e21["c_vs_a"]
    lines += ["", f"C vs A improvement test (9.4): mean {g(c['signed']['delta'])}, SE {g(c['signed']['se'])}, "
              f"t = {g(c['t'], 4)}, df = 9, one-sided p = {g(c['p_one_sided'], 4)}; C beats A: "
              f"{'yes' if c['c_beats_a'] else 'no'} (needs mean > +0.002 and p < 0.05).", ""]
    counts = decision["observation_counts"]
    lines += ["Populations per rotation: " + ", ".join(f"{NAMES[s]} {counts[s]:,}" for s in SOURCES)
              + f" (total {sum(counts.values()):,}); each fit excludes its assessed source.", ""]
    lines += ["## Label-free gates on full-data seed 0 (section 9.2)", "",
              "| Gate | Bar | A | C | Production seed 0 |", "|---|---|---|---|---|"]
    a, cc = gates["arms"]["a"], gates["arms"]["c"]

    def pf(entry, value):
        ok = entry.get("pass", entry.get("pass_"))
        return f"{'PASS' if ok else 'FAIL'} ({value})"
    lines += [
        f"| N1 near-identity | both one-pixel rungs ≥ 99.0 on 24 refs | {pf(a['N1'], 'min ' + g(a['N1']['one_pixel_score_range'][0], 5))} | {pf(cc['N1'], 'min ' + g(cc['N1']['one_pixel_score_range'][0], 5))} | fails (88.69–97.67) |",
        f"| N2 no gap | highest nonidentical ≥ 99.0 per ref | {pf(a['N2'], 'min ' + g(a['N2']['highest_nonidentical_range'][0], 5))} | {pf(cc['N2'], 'min ' + g(cc['N2']['highest_nonidentical_range'][0], 5))} | fails (max 97.73) |",
        f"| N3 ladders | ≥ 122/144 nonincreasing | {pf(a['N3'], str(a['N3']['monotone_ladders']) + '/144')} | {pf(cc['N3'], str(cc['N3']['monotone_ladders']) + '/144')} | 122/144 |",
        f"| C2 ties | ≤ 0.05 standard and ladder | {pf(a['C2'], g(a['C2']['standard'], 4) + ' / ' + g(a['C2']['ladder'], 4))} | {pf(cc['C2'], g(cc['C2']['standard'], 4) + ' / ' + g(cc['C2']['ladder'], 4))} | 0.0065 / 0.036 |",
        f"| C5 identity | 38 raw identities exactly 100.0 (620 source×tier rows) | {pf(a['C5'], str(a['C5']['rows']) + ' rows')} | {pf(cc['C5'], str(cc['C5']['rows']) + ' rows')} | fails 38/38 |",
        f"| G-STEER | ≥ 128/135 | {pf(a['G-STEER'], str(a['G-STEER']['passed']) + '/135')} | {pf(cc['G-STEER'], str(cc['G-STEER']['passed']) + '/135')} | 128/135 |",
        f"| Output stage K1–K5 | K1–K3 at pack; K4 0 raw ≤ x_floor on every E33 population (table below); K5 C1/C3/C4/C6/G-DIAL | {pf(a['output_stage'], 'K4 ' + str(a['output_stage']['K4_rows_at_or_below_floor']) + ' at floor')} | {pf(cc['output_stage'], 'K4 ' + str(cc['output_stage']['K4_rows_at_or_below_floor']) + ' at floor')} | n/a |",
    ]
    lines += ["", "K4 (section 8) per population, raw units (candidate with its output spline stripped, same owners): rows, "
              "rows with raw ≤ x_floor, rows with raw < x0, and min raw − x_floor.", "",
              "| Population | Rows | A at floor | A raw < x0 | A min raw − x_floor | C at floor | C raw < x0 | C min raw − x_floor |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    pa_, pc_ = a["output_stage"]["K4_populations"], cc["output_stage"]["K4_populations"]
    for name in pa_:
        x, y = pa_[name], pc_[name]
        lines.append(f"| {name} | {x['rows']} | {x['at_or_below_floor']} | "
                     f"{x['below_x0']} | {g(x['min_raw_minus_x_floor'], 5)} | {y['at_or_below_floor']} | {y['below_x0']} | "
                     f"{g(y['min_raw_minus_x_floor'], 5)} |")
    lines += ["", "The 620-row identity proof (C5) serves exactly 100.0 on every row, so none is near the floor. "
              "The first results record counted K4 only on the standard and ladder grids and NEARID (14,665 served rows) "
              "plus the calibration rows. The results review (E33RESULTS_REVIEW) checked the missing populations on "
              "served scores and found 0 at the floor (served minima A/C: negative-tail −103.94/−122.57, identity 100.0, "
              "G-STEER forwards −64.84/−78.55). This table counts every registered population in raw units."]
    for arm, x in (("A", a), ("C", cc)):
        if x["G-STEER"]["failing"]:
            lines.append("")
            lines.append(f"{arm} G-STEER failing cases: " + ", ".join(x["G-STEER"]["failing"]) + ".")
    rt = gates["runtime"]
    lines += ["", "## Runtime (section 9.3)", ""]
    if isinstance(rt, dict):
        lines += ["Guard: no cell slower (paired CI wholly above +2% of production). One thread; whole-call medians; "
                  "first-32-clean paired rounds; pointwise 95% CIs.", "",
                  "| Cell | A % change | A label | C % change | C label |", "|---|---:|---|---:|---|"]
        for cell, row in sorted(rt["cells"].items()):
            lines.append(f"| {cell} | {g(row['e33_a']['pct_change'], 4)} | {row['e33_a']['label']} | "
                         f"{g(row['e33_c']['pct_change'], 4)} | {row['e33_c']['label']} |")
        lines += ["", "Fresh-process peak RSS at 1024², v4x, one thread (KiB): " + ", ".join(
            f"{k} {v:,}" for k, v in rt["rss_kib_1024"].items()) + ".",
                  "Model bytes: production " + f"{gates['production']['model_bytes']:,}"
                  + "; " + "; ".join(f"{k.upper()} {v['model_bytes']:,}" for k, v in gates["candidates"].items()) + "."]
    else:
        lines.append(str(rt))
    hq = e21["report_only"]["high_quality_slice"]
    lines += ["", "## Report-only (section 9.5)", "",
              "High-quality slice (top 20% human quality per source), ten-seed mean signed SROCC, arm minus control:", "",
              "| Source | Slice rows | Control | A − control | C − control |", "|---|---:|---:|---:|---:|"]
    for s in SOURCES:
        x = hq[s]
        lines.append(f"| {NAMES[s]} | {x['slice_rows']} | {g(x['control_mean'], 4)} | {g(x['a_minus_control'], 4)} | "
                     f"{g(x['c_minus_control'], 4)} |")
    lines += ["", "Slices are small; descriptive only.", ""]
    lines += seeds_1_2()
    lines += ["",
              "## Artifact pins", ""] + [f"- `{k}`: `{v}`" for k, v in pins.items()]
    out_md.write_text("\n".join(lines) + "\n")


def seeds_1_2():
    """Report-only (9.2): seeds 1-2 full-data gates, never used to choose."""
    import struct
    from e33_gates import identity_rows
    hundred = struct.unpack("<q", struct.pack("<d", 100.0))[0]
    out = ["Seeds 1–2 full-data gates (report-only, never used to choose):", "",
           "| Seed | Arm | N1 min one-pixel | N2 min highest | N3 ladders | C2 std/ladder | C5 exact 100 | G-STEER |",
           "|---|---|---:|---:|---:|---|---|---:|"]
    for seed in (1, 2):
        gates = E / "gates"
        rows, cands = identity_rows(gates / f"identity-s{seed}.jsonl")
        for arm in ("a", "c"):
            n = json.loads((gates / f"nearid-s{seed}" / f"candidate-{arm}.GATES.json").read_text())
            c2 = [next(c["measured"] for c in json.loads((gates / f"verdict-s{seed}-{arm}" / grid / "gaddr.json")
                                                      .read_text())["checks"] if c["id"] == "C2")
                  for grid in ("standard", "ladder")]
            steer = json.loads((gates / f"steer-s{seed}-{arm}.json").read_text())["rows"]
            model_sha = json.loads((gates / f"steer-s{seed}-{arm}-packet.json").read_text())["cases"][0]["model_sha256"]
            i = [c["sha256"] for c in cands].index(model_sha)
            exact = all(r["candidate_score_bits"][i] == hundred for r in rows) and len(rows) == 620
            out.append(f"| {seed} | {arm.upper()} | {g(n['N1']['one_pixel_score_range'][0], 5)} | "
                       f"{g(n['N2']['highest_nonidentical_range'][0], 5)} | {n['N3']['monotone_ladders']}/144 | "
                       f"{g(c2[0], 4)}/{g(c2[1], 4)} | {'yes' if exact else 'NO'} | "
                       f"{sum(r['pass'] for r in steer)}/{len(steer)} |")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", type=Path, default=Path("benchmarks/e33_result_summary_2026-10-10.json"))
    p.add_argument("--md", type=Path, default=Path("benchmarks/e33_result_summary_2026-10-10.md"))
    a = p.parse_args()
    build(a.json, a.md)


if __name__ == "__main__":
    main()
