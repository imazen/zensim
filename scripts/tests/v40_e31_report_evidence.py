"""Verify registered E31 report closure and preserve every cell's pooled ranks.

This reads the packet's already authorized report, not any dataset payload.
No additional statistic or adoption rule is computed.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1]),
    str(Path(__file__).resolve().parents[1] / "rev4_featpot"),
]
from v40_panels import bound_bytes  # noqa: E402


FOLDS = ("kadid", "tid2013", "konfig", "cid22_a25")


def verify(report, exposure):
    if (
        report.get("schema") != "e31-v40-upiq-training-report-v1"
        or report.get("report_only") is not True
        or report.get("independent_test") is not False
        or report.get("shipping_adoption_authorized") is not False
        or exposure.get("mode") != "upiq"
        or exposure.get("label_read_authorized") is not True
        or not exposure.get("coordinator_message")
        or report.get("control_pins_sha256") != exposure.get("control_pins_sha256")
    ):
        raise ValueError("registered report-only contract required")
    cells = {f"{f}_s{s}" for f in FOLDS for s in range(10)}
    if set(report["panels"]) != {"fit", "development"}:
        raise ValueError("both registered populations required")
    summary = {}
    for split, count, refs in (("fit", 330, 26), ("development", 50, 4)):
        arms = report["panels"][split]
        if set(arms) != {"control", "uh4"} or any(
            set(rows) != cells for rows in arms.values()
        ):
            raise ValueError("complete forty-cell matched grid required")
        target = None
        summary[split] = {}
        for arm, rows in arms.items():
            summary[split][arm] = {}
            for cell, entry in rows.items():
                if (
                    entry.get("schema") != "e31-upiq-training-report-v1"
                    or entry.get("independent_test") is not False
                    or entry.get("shipping_adoption_authorized") is not False
                    or set(entry["panels"]) != {split}
                ):
                    raise ValueError("foreign report population")
                panel = entry["panels"][split]
                scatter = panel["scatter"]
                if any(len(scatter[k]) != count for k in ("target", "prediction")):
                    raise ValueError("raw scatter census differs")
                if not all(math.isfinite(v) for v in sum(scatter.values(), [])):
                    raise ValueError("nonfinite raw geometry")
                if target is None:
                    target = scatter["target"]
                if scatter["target"] != target:
                    raise ValueError("targets/order differ across models")
                if set(panel["per_study"]) != {"korshunov", "narwaria"}:
                    raise ValueError("registered studies missing")
                if len(panel["within_reference"]) != refs:
                    raise ValueError("registered reference census differs")
                if any(
                    sum(p["n"] for p in panel[group].values()) != count
                    for group in ("per_study", "within_reference")
                ):
                    raise ValueError("group coverage differs")
                all_panels = [
                    panel["pooled"],
                    *panel["per_study"].values(),
                    *panel["within_reference"].values(),
                ]
                if panel["pooled"]["n"] != count or any(
                    p["n_dropped"] or not math.isfinite(p["srocc_signed"])
                    for p in all_panels
                ):
                    raise ValueError("incomplete registered rank panel")
                summary[split][arm][cell] = panel["pooled"]["srocc_signed"]
    return dict(
        schema="v40-e31-hdr-report-evidence-v1",
        report_only=True,
        independent_test=False,
        shipping_adoption_authorized=False,
        populations={
            "fit": dict(rows=330, references=26),
            "development": dict(rows=50, references=4),
        },
        cells_per_arm=40,
        arms=["control", "uh4"],
        pooled_signed_srocc=summary,
        external_reports=dict(
            hdrvdc="BLOCKED: no retained compatible Rev5 inputs",
            avt="BLOCKED: no retained compatible Rev5 inputs",
        ),
    )


# set: (label rows, per-study groups, references, legs)
VIDEO = {
    "hdrvdc": (464, {"bright-near", "bright-far", "dim-near", "dim-far"}, 16, {"i", "ii", "iii"}),
    "avt": (195, {"av1", "hevc", "vvc"}, 5, {"single"}),
}


def verify_video(report, exposure):
    """The external HDR video reports (HDRVID, 2026-10-10): same closure checks."""
    if (
        report.get("schema") != "e31-v40-external-video-report-v1"
        or report.get("report_only") is not True
        or report.get("independent_test") is not False
        or report.get("shipping_adoption_authorized") is not False
        or exposure.get("mode") != "e31video"
        or exposure.get("label_read_authorized") is not True
        or not exposure.get("coordinator_message")
        or report.get("control_pins_sha256") != exposure.get("control_pins_sha256")
        or set(report["panels"]) != set(VIDEO)
    ):
        raise ValueError("registered external-video report-only contract required")
    cells = {f"{f}_s{s}" for f in FOLDS for s in range(10)}
    summary = {}
    for name, (count, studies, refs, legs) in VIDEO.items():
        arms = report["panels"][name]
        if set(arms) != {"control", "uh4"} or any(set(rows) != cells for rows in arms.values()):
            raise ValueError("complete forty-cell matched grid required")
        target = None
        summary[name] = {}
        for arm, rows in arms.items():
            summary[name][arm] = {}
            for cell, entry in rows.items():
                if set(entry) != legs:
                    raise ValueError("registered legs missing")
                summary[name][arm][cell] = {}
                for leg, panel in entry.items():
                    scatter = panel["scatter"]
                    if any(len(scatter[k]) != count for k in ("target", "prediction")):
                        raise ValueError("raw scatter census differs")
                    if not all(math.isfinite(v) for v in sum(scatter.values(), [])):
                        raise ValueError("nonfinite raw geometry")
                    target = scatter["target"] if target is None else target
                    if scatter["target"] != target:
                        raise ValueError("targets/order differ across models")
                    if set(panel["per_study"]) != studies or len(panel["within_reference"]) != refs:
                        raise ValueError("registered study/reference census differs")
                    if any(sum(p["n"] for p in panel[g].values()) != count for g in ("per_study", "within_reference")):
                        raise ValueError("group coverage differs")
                    panels = [panel["pooled"], *panel["per_study"].values(), *panel["within_reference"].values()]
                    if panel["pooled"]["n"] != count or any(
                        p["n_dropped"] or not math.isfinite(p["srocc_signed"]) for p in panels
                    ):
                        raise ValueError("incomplete registered rank panel")
                    summary[name][arm][cell][leg] = dict(
                        pooled=panel["pooled"]["srocc_signed"],
                        per_study={k: v["srocc_signed"] for k, v in panel["per_study"].items()},
                    )
    return dict(
        schema="v40-e31-video-report-evidence-v1",
        report_only=True,
        independent_test=False,
        shipping_adoption_authorized=False,
        populations={k: dict(rows=v[0], references=v[2], legs=sorted(v[3])) for k, v in VIDEO.items()},
        cells_per_arm=40,
        arms=["control", "uh4"],
        signed_srocc=summary,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--exposure", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("fresh evidence output required")
    rb, eb = bound_bytes(args.report), bound_bytes(args.exposure)
    report = json.loads(rb)
    exposure = json.loads(eb)
    digest = hashlib.sha256(eb).hexdigest()
    if report.get("exposure_freeze_sha256") != digest:
        raise ValueError("exposure binding differs")
    video = report.get("schema") == "e31-v40-external-video-report-v1"
    result = (verify_video if video else verify)(report, exposure)
    result.update(report_sha256=hashlib.sha256(rb).hexdigest(), exposure_sha256=digest)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        "Verified both populations, all 160 reports, raw target/order identity and complete study/reference ranks."
        + (" (HDR-VDC legs i/ii/iii, AVT)" if video else "")
    )


if __name__ == "__main__":
    main()
