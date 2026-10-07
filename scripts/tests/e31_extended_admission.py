"""Actual extended-binary refusals, with strace proving zero feature opens."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rev4_featpot"))
import e31_training as e31


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trainer", type=Path, required=True)
    ap.add_argument("--fit", type=Path, required=True)
    ap.add_argument("--dest", type=Path, required=True)
    a = ap.parse_args()
    a.dest.mkdir(parents=True, exist_ok=False)
    decision = a.dest / "synthetic-decision.json"
    decision.write_text(
        json.dumps(
            dict(
                schema="e31-upiq-label-disposition-v1",
                state="approved",
                decision_id="E31-legacy-HDR-label-producer-gap",
                allowed_use="registered-E31-research-training",
                manifest_sha256=e31.MANIFEST_SHA,
                legacy_label_sha256=e31.LABEL_SHA,
                accept_unresolved_producer=True,
                decided_by="synthetic-negative-test-fixture",
            )
        )
    )
    ordinary = a.dest / "unread-ordinary.parquet"
    Path(f"{ordinary}.manifest.json").write_text(
        json.dumps(
            dict(
                feature_set_id="basic+peaks+v2@w1825/rev5_localwin#36c3f3af",
                formula_revision=5,
                human_sources=["aic3"],
            )
        )
    )
    original = json.loads(Path(f"{a.fit}.manifest.json").read_text())
    dev = a.dest / "unread-development.parquet"
    original["split"] = "development"
    Path(f"{dev}.manifest.json").write_text(json.dumps(original))
    selected = e31.columns("by_v2fy")
    cases = {
        "no-disposition": (a.fit, "rank", selected, "malformed feature_set_id"),
        "different-420-subset": (
            a.fit,
            "rank",
            list(range(420)),
            "420-slot projection",
        ),
        "development": (dev, "rank", selected, "pinned UPIQ fit manifest"),
        "foreign-aic": (a.fit, "rank", selected, "registered Rev5/D1 population"),
        "withinref": (a.fit, "withinref,rank", selected, "pooled rank-only"),
        "mse": (a.fit, "both", selected, "pooled rank-only"),
    }
    report = {}
    for name, (native, mode, ids, message) in cases.items():
        cmd = [
            str(a.trainer),
            "--group",
            f"upiq380:{native}:4.34410740924913:0:{mode}",
            "--group",
            f"ordinary:{ordinary}:1:0:rank",
            "--upiq-label-disposition",
            str(decision),
            "--keep-features",
            ",".join(map(str, ids)),
            "--max-features",
            "1853",
            "--target-column",
            "human_score",
            "--target-scale",
            "1",
            "--epochs",
            "1",
            "--pairs-per-epoch",
            "1",
            "--no-auto-eval",
            "--out",
            str(a.dest / f"{name}.bin"),
        ]
        if name == "no-disposition":
            index = cmd.index("--upiq-label-disposition")
            del cmd[index : index + 2]
        trace = a.dest / f"{name}.trace"
        result = subprocess.run(
            ["strace", "-f", "-e", "trace=openat", "-o", str(trace), *cmd],
            text=True,
            capture_output=True,
        )
        (a.dest / f"{name}.log").write_text(result.stdout + result.stderr)
        protected = [str(a.fit), str(native), str(ordinary)]
        opens = [
            line
            for line in trace.read_text().splitlines()
            if any(f'"{p}"' in line for p in protected)
        ]
        if result.returncode != 2 or message not in result.stderr or opens:
            raise AssertionError(
                f"{name}: rc={result.returncode}, feature opens={opens}; see {a.dest}"
            )
        report[name] = dict(status="PASS", exit_code=2, feature_payload_opens=0)
    (a.dest / "ADMISSION.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
