"""Synthetic actual-binary V40 admission probes; syscall evidence stays on disk."""

# ruff: noqa: E402
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "scripts/rev4_featpot"), str(REPO / "scripts/tests")]
from test_e32_palette_training import PaletteTraining
from test_shippath2_admission import RecipeAdmissionTests
import e32_palette as palette
import e31_training as upiq
import v2_common as common
import v2_d1_prepare as prepare
import v2_human_role as roles
import v2_teacher as teacher


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--upiq-manifest", type=Path, required=True)
    a = ap.parse_args()
    a.dest.mkdir(parents=True, exist_ok=False)
    report = {}

    def run(name, groups, ids, width, flags=(), reason=None, extra_payloads=()):
        out = a.dest / name
        out.mkdir()
        argv = [
            str(a.binary),
            "--target-column",
            "human_score",
            "--target-scale",
            "1",
            "--keep-features",
            ",".join(map(str, ids)),
            "--max-features",
            str(width),
            "--epochs",
            "2",
            "--pairs-per-epoch",
            "128",
            "--hidden",
            "128",
            "--early-stop-patience",
            "0",
            "--no-auto-eval",
            "--nonneg-distance",
            "--out",
            str(out / "model.bin"),
            *flags,
        ]
        for group in groups:
            argv += ["--group", ":".join(map(str, group))]
        trace = out / "open.trace"
        p = subprocess.run(
            ["strace", "-qq", "-f", "-e", "trace=openat", "-o", str(trace), *argv],
            capture_output=True,
            text=True,
        )
        (out / "stdout.log").write_text(p.stdout)
        (out / "stderr.log").write_text(p.stderr)
        (out / "argv.json").write_text(json.dumps(argv, indent=2) + "\n")
        opens = [
            line
            for line in trace.read_text().splitlines()
            if (any(f'"{p}"' in line for p in [*(g[1] for g in groups), *extra_payloads])
                or (".parquet\"" in line and ".keys.parquet\"" not in line))
            and "O_PATH" not in line and "O_DIRECTORY" not in line and "= -1" not in line
        ]
        assert p.returncode == 2 and not opens and not (out / "model.bin").exists(), (
            name,
            p.returncode,
            opens,
            p.stderr,
        )
        if reason is not None:
            assert reason in p.stderr, (name, p.stderr)
        report[name] = dict(
            status="PASS", exit_code=2, feature_payload_opens=0, model_writes=0
        )

    fixture = PaletteTraining()
    fixture.setUp()
    try:
        original = copy.deepcopy(fixture.d)
        kp = teacher.key_path(fixture.path)
        original_keys = kp.read_bytes()
        for case in (
            "VAL-keys",
            "development-training",
            "ordered-key-pin",
            "missing-identity",
            "ordinal-missing-identity",
            "ordinal-duplicate-index",
            "row-count",
            "late-development",
        ):
            kp.write_bytes(original_keys)
            d = copy.deepcopy(original)
            keys = pq.read_table(kp)
            if case == "VAL-keys":
                keys = keys.append_column("role", pa.array(["val"] * len(keys)))
            elif case in ("development-training", "late-development"):
                d["research_palette"]["role"] = "TRAIN-oracle-development"
            elif case == "ordered-key-pin":
                d["row_keys_sha256"] = "0" * 64
            elif case == "missing-identity":
                keys = pa.table({"member_set": ["safesyn"]})
                d["rows"] = 1
            elif case.startswith("ordinal-"):
                d["data_role"] = "TRAIN ordinal KADIS source_id%10<8; no human labels"
                d["research_palette"].update(
                    role="TRAIN-ordinal",
                    member_sets=["coverage_pool"],
                    key_domain="coverage-selection-ordinal",
                )
                keys = pa.table({"__index_level_0__": [0]})
                if case == "ordinal-duplicate-index":
                    keys = pa.table(
                        {
                            "ladder": ["r"] * 2,
                            "source_filename": ["r"] * 2,
                            "type": ["16"] * 2,
                            "family": ["light"] * 2,
                            "severity_level": [1, 2],
                            "severity": [1.0, 2.0],
                            "sign": [1.0, 1.0],
                            "__index_level_0__": [0, 0],
                        }
                    )
                d["rows"] = len(keys)
            elif case == "row-count":
                d["rows"] = len(keys) + 1
            pq.write_table(keys, kp, compression="zstd")
            d["keys_sha256"] = common.sha(kp)
            if case != "ordered-key-pin":
                d["row_keys_sha256"] = teacher.row_keys_sha(keys)
            fixture.save(d)
            groups = [("synthetic", fixture.path, 1, 0, "withinref,both")]
            if case == "late-development":
                late = fixture.path.with_name("late.parquet")
                Path(f"{late}.manifest.json").write_text(json.dumps(d))
                fixture.save(original)
                groups.append(("late", late, 1, 0, "withinref,both"))
            run("palette-" + case, groups, palette.ARM_IDS, 1867)
        # A receipt-bound development leg passed with a training weight must
        # reach the weight gate, rather than failing an earlier changed pin.
        dev = fixture.path.with_name("safesyn_dev.parquet")
        dev.write_bytes(fixture.path.read_bytes())
        teacher.key_path(dev).write_bytes(original_keys)
        d = copy.deepcopy(original)
        d["research_palette"]["role"] = "TRAIN-oracle-development"
        d["keys_sha256"] = common.sha(teacher.key_path(dev))
        d["row_keys_sha256"] = teacher.row_keys_sha(
            pq.read_table(teacher.key_path(dev))
        )
        Path(f"{dev}.manifest.json").write_text(json.dumps(d))
        rp = dev.parent / "receipt.json"
        r = json.loads(rp.read_text())
        r["legs"]["safesyn"]["dev"].update(
            sha256=d["table_sha256"],
            rows=d.get("rows", 24),
            manifest_sha256=common.sha(Path(f"{dev}.manifest.json")),
        )
        rp.write_text(json.dumps(r))
        run(
            "palette-bound-development-weight",
            [("safesyn_development", dev, 1, 0, "withinref,both")],
            palette.ARM_IDS,
            1867,
            reason="fit/development role disagrees with training/validation weights",
        )
    finally:
        fixture.doCleanups()

    f = RecipeAdmissionTests()
    f.setUp()
    try:
        f.admit()
        approval = f.root / "decision.json"
        f.json(
            approval,
            dict(
                schema="shippath-human-role-decision-v1",
                decision_id="SHIPPATH-human-production-role",
                state="approved",
                decided_by="TEST",
                allowed_use="qualified-recipe-training",
                sources=list(roles.PRODUCTION_SOURCES),
                ledger_commit=roles.LEDGER_COMMIT,
                source_receipt_sha256=common.sha(f.wide / "receipt.json"),
                source_frozen_sha256=common.sha(f.source / "wide/frozen.json"),
            ),
        )
        root = f.root / "d1"
        prepare.prepare(
            f.out,
            f.source,
            f.bank,
            f.root / "stage",
            root,
            approval,
            f.root / "logical",
        )
        coverage, _ = teacher.coverage_leg(0x98, f.root, admitted_root=root)
        base = root / "wide/main/real"
        groups = [
            (name, base / f"{stem}.parquet", tw, vw, "withinref,rank")
            for name, stem, tw, vw in [
                ("safesyn", "safesyn_fit", 1, 0),
                ("safesyn_development", "safesyn_dev", 0, 1),
                ("cid22", "cid22_fit", 16, 0),
                ("cid22_development", "cid22_dev", 0, 1),
                ("human", "human_without_kadid_fit", 32, 0),
                ("human_development", "human_without_kadid_dev", 0, 1),
            ]
        ]
        groups.append(("coverage", coverage, 1, 0, "withinref,rank"))
        # Manifest inputs must not bypass the complete group/key preflight.
        manifest = f.root / "train.toml"
        human = groups[4][1]
        manifest.write_text(f'[inputs.human]\npath = "{human}"\nsha256 = "{common.sha(human)}"\n')
        changed = groups[:]
        g = changed[5]
        changed[5] = (g[0], g[1], 1, 0, g[4])
        run("manifest-development-training", changed, upiq.columns("by_v2fy"), 1853,
            ["--manifest", str(manifest)], reason="fit/development role disagrees with training/validation weights")
        hp = Path(f"{human}.manifest.json")
        original_human, original_receipt = hp.read_bytes(), (base / "receipt.json").read_bytes()
        d = json.loads(original_human)
        d["role"] = "val"
        hp.write_text(json.dumps(d))
        r = json.loads(original_receipt)
        for leg in r["legs"].values():
            for split in ("fit", "dev", "full"):
                if isinstance(leg.get(split), dict) and Path(leg[split].get("rel", "")).name == human.name:
                    leg[split]["manifest_sha256"] = common.sha(hp)
        (base / "receipt.json").write_text(json.dumps(r))
        run("manifest-VAL-human", groups, upiq.columns("by_v2fy"), 1853,
            ["--manifest", str(manifest)], reason="unapproved role or training/development weight")
        hp.write_bytes(original_human)
        (base / "receipt.json").write_bytes(original_receipt)
        # The late protected entry blocks EVERY input hash, even with drift allowed.
        for component in ("kadid_terminal", "holdout", "_sealed", "labels__synthetic"):
            protected = f.root / component / "sentinel.parquet"
            protected.parent.mkdir()
            protected.write_bytes(b"SYNTHETIC SENTINEL; never an assessment label")
            alias = f.root / f"alias-{component}"
            alias.symlink_to(protected.parent, target_is_directory=True)
            for route, path in (("direct", protected), ("symlink", alias / protected.name)):
                manifest.write_text(f'[inputs.a_ordinary]\npath = "{human}"\nsha256 = "{common.sha(human)}"\n[inputs.z_protected]\npath = "{path}"\nsha256 = "{"0" * 64}"\n')
                for drift in (False, True):
                    run(f"manifest-{component}-{route}-drift-{drift}", groups, upiq.columns("by_v2fy"), 1853,
                        ["--manifest", str(manifest), *(["--manifest-allow-sha-drift"] if drift else [])],
                        reason="protected training input ancestry", extra_payloads=[protected, path])
        native = f.root / "upiq.parquet"
        Path(f"{native}.manifest.json").write_bytes(a.upiq_manifest.read_bytes())
        pq.write_table(
            pa.table({"role": ["val"]}), teacher.key_path(native), compression="zstd"
        )
        decision = f.root / "upiq-disposition.json"
        decision.write_text(
            json.dumps(
                dict(
                    schema="e31-upiq-label-disposition-v1",
                    state="approved",
                    decision_id="E31-legacy-HDR-label-producer-gap",
                    allowed_use="registered-E31-research-training",
                    manifest_sha256=upiq.MANIFEST_SHA,
                    legacy_label_sha256=upiq.LABEL_SHA,
                    accept_unresolved_producer=True,
                    decided_by="synthetic-negative-only",
                )
            )
        )
        groups.append(("upiq380", native, 4.34410740924913, 0, "rank"))
        flags = ["--upiq-label-disposition", str(decision)]
        run(
            "upiq-native-VAL-keys-all-groups",
            groups,
            upiq.columns("by_v2fy"),
            1853,
            flags,
        )
        sp = Path(f"{groups[0][1]}.manifest.json")
        original = json.loads(sp.read_text())
        bad = {**original, "role": "val", "split": "terminal", "tier": "T0"}
        sp.write_text(json.dumps(bad))
        run(
            "upiq-ordinary-development-training",
            groups,
            upiq.columns("by_v2fy"),
            1853,
            flags,
        )
        sp.write_text(json.dumps(original))
        kp = teacher.key_path(groups[0][1])
        keys = pq.read_table(kp).append_column("role", pa.array(["val"] * 24))
        pq.write_table(keys, kp, compression="zstd")
        original.update(
            keys_sha256=common.sha(kp), row_keys_sha256=teacher.row_keys_sha(keys)
        )
        sp.write_text(json.dumps(original))
        rp = base / "receipt.json"
        r = json.loads(rp.read_text())
        r["legs"]["safesyn"]["fit"]["manifest_sha256"] = common.sha(sp)
        rp.write_text(json.dumps(r))
        run(
            "upiq-ordinary-VAL-keys",
            groups,
            upiq.columns("by_v2fy"),
            1853,
            flags,
            reason="observation identity",
        )
        # These failures precede even the late native key rejection.
        for name, index, tw, vw in [
            ("human-development-training", 5, 1, 0),
            ("human-fit-development", 4, 0, 1),
            ("teacher-development-training", 1, 1, 0),
        ]:
            changed = groups[:]
            g = changed[index]
            changed[index] = (g[0], g[1], tw, vw, g[4])
            sp.write_text(
                json.dumps(
                    {
                        k: v
                        for k, v in original.items()
                        if k not in ["role", "split", "tier"]
                    }
                )
            )
            r["legs"]["safesyn"]["fit"]["manifest_sha256"] = common.sha(sp)
            rp.write_text(json.dumps(r))
            run(
                name,
                changed,
                upiq.columns("by_v2fy"),
                1853,
                flags,
                reason="fit/development role disagrees with training/validation weights",
            )
    finally:
        f.doCleanups()
    (a.dest / "RESULT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
