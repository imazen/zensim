#!/usr/bin/env python3
"""Write the E31 external-video exposure freeze before any label read.

  freeze.py --out OUT --packet PACKET --dest benchmarks/<name>.json

Pins every object `v40_panels.py --mode e31video` may open: per set the table
manifest (metadata), label-free keys and pooled feature table (payload), the
original label file (payload) and the packet predictor (tool). Label-file
hashes come from the dataset pointers recorded in zenpapers when the files
were acquired; this script does not open the label files. The report's
`retain()` re-verifies every byte against these pins at read time.
"""

import argparse
import hashlib
import json
from pathlib import Path

LABELS = {
    # zenpapers/datasets/HDR-VDC.pointer.md and AVT-VQDB-UHD-1-HDR.pointer.md
    "hdrvdc": ("/mnt/v/datasets/hdr-vdc/HDR_VDC_JOD_Scores.csv",
               "81484245b53b3d3726f1fefcf06905952a24d1418ef92278851b5d7ff1f6b917"),
    "avt": ("/mnt/v/datasets/avt-vqdb-uhd-1-hdr/subjective_scores/mos_ci.csv",
            "6b1e5bac20f183ffdfc14f2478492593324842fee0407f91368d7afe11cc1a53"),
}
OWNER = (
    "Owner, verbatim 2026-10-10: \"do HDR-VDC with rav1d-safe now, and non-av1 videos with ffmpeg 8.1\". "
    "Earlier owner approval of E31's registered HDR-side reports including the external HDR-VDC and AVT "
    "panels, verbatim 2026-10-09: \"approved, all of it\". Scope: report-only E31 uh4 vs frozen matched V40 "
    "control on HDR-VDC (464 distorted condition observations; legs i/ii/iii over configs A-E) and "
    "AVT-VQDB-UHD-1-HDR (195 encoded videos); no KADID TERMINAL, AIC-family, T0 or sealed data; no new "
    "fit, calibration, selection or statistic."
)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--packet", required=True, type=Path)
    p.add_argument("--dest", required=True, type=Path)
    args = p.parse_args()
    if args.dest.exists():
        raise SystemExit("fresh freeze path required")
    owner = Path(__file__).resolve().parents[2] / "rev4_featpot" / "v40_panels.py"
    objects, tables = {}, {}
    for name, (label_path, label_sha) in LABELS.items():
        base = args.out / "tables" / f"{name}.parquet"
        manifest = json.loads(Path(f"{base}.manifest.json").read_text())
        if manifest["table_sha256"] != sha(base) or manifest["keys_sha256"] != sha(base.with_suffix(".keys.parquet")):
            raise SystemExit(f"{name}: table/manifest pins disagree")
        objects[f"{name}-manifest"] = dict(kind="metadata", path=f"{base}.manifest.json",
                                           sha256=sha(f"{base}.manifest.json"))
        objects[f"{name}-keys"] = dict(kind="payload", path=str(base.with_suffix(".keys.parquet")),
                                       sha256=manifest["keys_sha256"])
        objects[f"{name}-table"] = dict(kind="payload", path=str(base), sha256=manifest["table_sha256"])
        objects[f"{name}-labels"] = dict(kind="payload", path=label_path, sha256=label_sha)
        tables[name] = dict(manifest=f"{name}-manifest", keys=f"{name}-keys", table=f"{name}-table",
                            labels=f"{name}-labels")
    objects["predictor"] = dict(kind="tool", path=str(args.packet / "bin/predict_features_with_bake"),
                                sha256=sha(args.packet / "bin/predict_features_with_bake"))
    record = dict(
        schema="v40-assessment-exposure-freeze-v1",
        mode="e31video",
        label_read_authorized=True,
        coordinator_message=OWNER,
        assessment_source_sha256=sha(owner),
        program_sha256=sha(args.packet / "program.tar.gz"),
        control_pins_sha256=sha(args.packet / "V40_CONTROL_PINS.json"),
        required_action="Freeze exact assessment declarations/payload/tool pins after complete fits; "
                        "no protected population is permitted.",
        objects=objects,
        tables=tables,
    )
    args.dest.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print(args.dest, sha(args.dest))


if __name__ == "__main__":
    main()
