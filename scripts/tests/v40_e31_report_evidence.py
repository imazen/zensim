"""Verify registered E31 report closure and preserve every cell's pooled ranks.

This reads the packet's already authorized report, not any dataset payload.
No additional statistic or adoption rule is computed.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--exposure", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("fresh evidence output required")
    rb, eb = args.report.read_bytes(), args.exposure.read_bytes()
    report = json.loads(rb)
    exposure = json.loads(eb)
    digest = hashlib.sha256(eb).hexdigest()
    if report.get("exposure_freeze_sha256") != digest:
        raise ValueError("exposure binding differs")
    result = verify(report, exposure)
    result.update(report_sha256=hashlib.sha256(rb).hexdigest(), exposure_sha256=digest)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        "Verified both populations, all 160 reports, raw target/order identity and complete study/reference ranks."
    )


if __name__ == "__main__":
    main()
