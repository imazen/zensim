"""Explicit, frozen post-fit panels. No default label population or discovery.

The coordinator supplies the exposure record after complete registered fits.
External and UPIQ panels are report-only. E29 delegates its rule to its owner.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lib.assessment_identity import safe_path
from lib.zen_stats import panel_batch
from v2_common import sha, refuse_immutable_output, dense_bake
from v2_human_role import PRODUCTION_SOURCES
from v40_score import complete, upiq_report
from external_sets import CONTENT_OVERLAP, GROUPS
from e32_palette import ARM_IDS, FEATURE_SET_ID
from v2_teacher import row_keys_sha

BASE_IDS = ARM_IDS[:420]


def bound_bytes(path):
    """No-follow every ancestor; retain the exact bytes hashed and parsed."""
    path = safe_path(Path(path).absolute())
    if any("terminal" in p.lower() or "aic" in p.lower() for p in path.parts):
        raise PermissionError("unregistered terminal/AIC assessment input")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            new = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = new
        file = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
        with os.fdopen(file, "rb") as stream:
            return stream.read()
    finally:
        os.close(fd)


def exposure(path, bundle, mode, pins):
    record = json.loads(bound_bytes(path))
    if (
        record.get("schema") != "v40-assessment-exposure-freeze-v1"
        or record.get("mode") != mode
        or record.get("label_read_authorized") is not True
        or not record.get("coordinator_message")
        or record.get("assessment_source_sha256") != sha(Path(__file__))
        or record.get("program_sha256") != sha(bundle / "program.tar.gz")
        or record.get("control_pins_sha256") != sha(pins)
        or not record.get("objects")
    ):
        raise PermissionError("explicit exact assessment exposure freeze required")
    # Check all names before opening even the first declaration.
    for alias, obj in record["objects"].items():
        if (
            Path(alias).name != alias
            or not alias
            or obj.get("kind") not in ("metadata", "payload", "tool")
            or len(obj.get("sha256", "")) != 64
        ):
            raise ValueError("unsafe/incomplete assessment object")
        safe_path(obj["path"])
        if any(
            "terminal" in p.lower() or "aic" in p.lower()
            for p in Path(obj["path"]).parts
        ):
            raise PermissionError("unregistered terminal/AIC assessment input")
    return record


def retain(record, kinds):
    blobs = {}
    for alias, obj in record["objects"].items():
        if obj["kind"] in kinds:
            data = bound_bytes(obj["path"])
            if hashlib.sha256(data).hexdigest() != obj["sha256"]:
                raise ValueError("frozen assessment input changed: " + alias)
            blobs[alias] = data
    return blobs


def table_object_kinds(record, entries, extra=()):
    expected = {}
    for entry in entries:
        for field, kind in (
            ("manifest", "metadata"),
            ("keys", "payload"),
            ("table", "payload"),
        ):
            alias = entry[field]
            if alias in expected and expected[alias] != kind:
                raise ValueError("ambiguous assessment object class")
            expected[alias] = kind
    expected.update(extra)
    if set(expected) != set(record["objects"]) or any(
        record["objects"][alias]["kind"] != kind for alias, kind in expected.items()
    ):
        raise ValueError("exact assessment object inventory/class required")


def external_metrics(pred, keys, name):
    pred = np.asarray(pred)
    if len(pred) != len(keys) or not np.isfinite(pred).all():
        raise ValueError("INCOMPLETE: external prediction identity")
    keys = keys.reset_index(drop=True)
    jobs = [("all", pred, keys.human_score.to_numpy())]
    jobs += [
        (f"{GROUPS[name][0]}:{g}", pred[ix], keys.human_score.to_numpy()[ix])
        for g, ix in keys.groupby(GROUPS[name][0]).indices.items()
    ]
    if name == "live":
        disjoint = ~keys.ref_path.map(lambda p: Path(p).name).isin(
            CONTENT_OVERLAP["live"]
        )
        if keys.loc[disjoint, "ref_path"].nunique() != 10:
            raise ValueError("LIVE fixed ten-reference disjoint panel changed")
        jobs.append(
            ("disjoint:all", pred[disjoint], keys.human_score.to_numpy()[disjoint])
        )
    if name == "mciqa":
        jobs += [
            ("dim:" + dim, pred, keys[dim].to_numpy())
            for dim in ("gn_z", "cs_z", "scm_z")
        ]
    panels = panel_batch(jobs, stats="srocc")
    if any(p["n_dropped"] or not np.isfinite(p["srocc_signed"]) for p in panels):
        raise ValueError("INCOMPLETE: undefined external panel")
    return {job[0]: panel["srocc_signed"] for job, panel in zip(jobs, panels)}


def external_summary(panels):
    """Four fold means inside each seed; no 40-cell pseudo-replication."""
    if set(panels) != {"control", "palette"}:
        raise ValueError("exact registered external arm/control required")
    expected = {f"{f}_s{s}" for f in PRODUCTION_SOURCES for s in range(10)}
    if any(set(rows) != expected for rows in panels.values()):
        raise ValueError("INCOMPLETE: external forty-cell grid")
    metrics = set(next(iter(panels["control"].values())))
    if not metrics or any(
        set(v) != metrics for rows in panels.values() for v in rows.values()
    ):
        raise ValueError("INCOMPLETE: inconsistent external metrics")
    result = {}
    for metric in sorted(metrics):
        a, b = [
            np.array(
                [
                    [panels[arm][f"{f}_s{s}"][metric] for f in PRODUCTION_SOURCES]
                    for s in range(10)
                ]
            )
            for arm in ("palette", "control")
        ]
        if not np.isfinite([a, b]).all():
            raise ValueError("INCOMPLETE: nonfinite external panel")
        d = (a - b).mean(axis=1)
        result[metric] = dict(
            mean=float(a.mean()),
            control_mean=float(b.mean()),
            delta=float(d.mean()),
            se=float(d.std(ddof=1) / np.sqrt(10)),
            seed_deltas=d.tolist(),
            seed_units=10,
            report_only=True,
        )
    return result


def external(record, cells, out):
    entries = record.get("tables", {})
    if set(entries) != {"nits", "live", "mciqa"} or any(
        set(e) != {"control", "palette"} for e in entries.values()
    ):
        raise ValueError("all three registered external populations required")
    table_object_kinds(
        record,
        [entry for arms in entries.values() for entry in arms.values()],
        (("predictor", "tool"),),
    )
    metadata = retain(record, {"metadata"})
    manifests = {}
    for name, arms in entries.items():
        for arm, entry in arms.items():
            m = json.loads(metadata[entry["manifest"]])
            ids = ARM_IDS if arm == "palette" else BASE_IDS
            if (
                m.get("data_role") != "assessment-only"
                or m.get("set") != name
                or str(m.get("formula_revision")) not in ("5", "Rev5")
                or not m.get("decoder_era")
                or m.get("requested_ids") != ids
                or m.get("rows") != {"nits": 405, "live": 779, "mciqa": 2000}[name]
                or (arm == "palette" and m.get("feature_set_id") != FEATURE_SET_ID)
                or m.get("table_sha256") != record["objects"][entry["table"]]["sha256"]
                or m.get("keys_sha256") != record["objects"][entry["keys"]]["sha256"]
            ):
                raise ValueError("unqualified external feature/key declaration")
            manifests[name, arm] = m
    blobs = retain(record, {"payload", "tool"})
    tool = out / "predict_features_with_bake"
    tool.write_bytes(blobs["predictor"])
    tool.chmod(0o500)

    report = {}
    for name, arms in entries.items():
        panels, original = {}, None
        for arm, entry in arms.items():
            keys = pq.read_table(pa.BufferReader(blobs[entry["keys"]]))
            if row_keys_sha(keys) != manifests[name, arm].get("row_keys_sha256"):
                raise ValueError("external ordered keys differ")
            frame = keys.to_pandas().reset_index(drop=True)
            if len(frame) != manifests[name, arm]["rows"]:
                raise ValueError("external row census differs")
            common = ["pair_key", "ref_path", "dist_path", "human_score"]
            if original is not None and not original[common].equals(frame[common]):
                raise ValueError("external arm/control labels/order differ")
            original = frame
            table = pq.read_table(pa.BufferReader(blobs[entry["table"]]))
            if table["pair_key"].to_pylist() != frame.pair_key.tolist():
                raise ValueError("external feature/key pair order differs")
            width = 1867 if arm == "palette" else 1825
            matrix = np.full((len(frame), width), np.nan, dtype="<f8")
            ids = ARM_IDS if arm == "palette" else BASE_IDS
            for id in ids:
                column = f"palette_f{id}" if id >= 1825 else f"f{id}"
                # Same one-time cast as training; numeric auxiliaries cannot
                # satisfy a named palette input.
                matrix[:, id] = (
                    table[column].to_numpy().astype(np.float32).astype(np.float64)
                )
            if not np.isfinite(matrix[:, ids]).all():
                raise ValueError("external primary features unmeasured/nonfinite")
            wire = out / f"{name}-{arm}.f64.wire"
            wire.write_bytes(struct.pack("<II", width, len(frame)) + matrix.tobytes())
            panels[arm] = {}
            for (fold, seed), cell in cells[arm].items():
                dest = out / name / arm / f"{fold}_s{seed}"
                dest.mkdir(parents=True)
                dense = dense_bake(cell / "refit/last.bin", dest, research_palette_cached=arm == "palette")
                pred = np.array(
                    [
                        float(v)
                        for v in subprocess.check_output(
                            [
                                str(tool),
                                "--bake",
                                str(dense),
                                "--features-file",
                                str(wire),
                                "--f64-wire",
                                "--production",
                                *(["--research-palette-cached"] if arm == "palette" else []),
                            ],
                            text=True,
                        ).split()
                    ]
                )
                (dest / "pred.json").write_text(json.dumps(pred.tolist()) + "\n")
                panels[arm][f"{fold}_s{seed}"] = external_metrics(pred, frame, name)
        report[name] = dict(summary=external_summary(panels), panels=panels)
    return dict(
        schema="e32-v40-external-report-v1",
        panels=report,
        report_only=True,
        shipping_adoption_authorized=False,
    )


def upiq(record, cells, out):
    entries = record.get("tables", {})
    if set(entries) != {"fit", "development"}:
        raise ValueError("UPIQ TRAIN fit and development reports required")
    table_object_kinds(record, entries.values(), (("predictor", "tool"),))
    fixed = {
        "fit": (
            "2da346bb17e08a4a63aae9ed89b159b36edd331199a9dd1c42b4a05b53ac939e",
            "7f09debedc591e7dd3494846ada9fe0c93b01779918b8358e2cf8053f5f1a6c4",
            330,
        ),
        "development": (
            "51f8adb58707993e3f12bda52c246cb28b30a07ccd68fb7ebd6c864a5f34538f",
            "bb56912e9b8d45542b1efdd8f7df506efcee939990e06cb909a816ecd42a0243",
            50,
        ),
    }
    metadata = retain(record, {"metadata"})
    for split, entry in entries.items():
        m = json.loads(metadata[entry["manifest"]])
        if (
            record["objects"][entry["manifest"]]["sha256"] != fixed[split][0]
            or record["objects"][entry["table"]]["sha256"] != fixed[split][1]
            or m.get("source") != "UPIQ-380"
            or m.get("role") != "train"
            or m.get("split") != split
            or m.get("rows") != fixed[split][2]
            or m.get("requested_ids") != BASE_IDS
            or m.get("keys_sha256") != record["objects"][entry["keys"]]["sha256"]
        ):
            raise ValueError("UPIQ report refuses foreign/non-TRAIN population")
    blobs = retain(record, {"payload", "tool"})
    tool = out / "predict_features_with_bake"
    tool.write_bytes(blobs["predictor"])
    tool.chmod(0o500)
    reports = {}
    for split, entry in entries.items():
        keys = (
            pq.read_table(pa.BufferReader(blobs[entry["keys"]]))
            .to_pandas()
            .reset_index(drop=True)
        )
        table = pq.read_table(pa.BufferReader(blobs[entry["table"]]))
        if (
            len(keys) != fixed[split][2]
            or set(keys.role) != {"train"}
            or set(keys.split) != {split}
            or table["pair_key"].to_pylist() != keys.pair_key.tolist()
        ):
            raise ValueError("UPIQ report keys/order changed")
        matrix = np.column_stack(
            [table[f"f{i}"].to_numpy() for i in range(1825)]
        ).astype("<f8")
        if not np.isfinite(matrix[:, BASE_IDS]).all():
            raise ValueError("UPIQ requested features unmeasured")
        wire = out / f"{split}.f64.wire"
        wire.write_bytes(struct.pack("<II", 1825, len(keys)) + matrix.tobytes())
        reports[split] = {}
        for arm, grid in cells.items():
            reports[split][arm] = {}
            for (fold, seed), cell in grid.items():
                dest = out / split / arm / f"{fold}_s{seed}"
                dest.mkdir(parents=True)
                dense = dense_bake(cell / "refit/last.bin", dest, research_palette_cached=arm == "palette")
                pred = np.array(
                    [
                        float(v)
                        for v in subprocess.check_output(
                            [
                                str(tool),
                                "--bake",
                                str(dense),
                                "--features-file",
                                str(wire),
                                "--f64-wire",
                                "--production",
                            ],
                            text=True,
                        ).split()
                    ]
                )
                reports[split][arm][f"{fold}_s{seed}"] = upiq_report(
                    pred, table["human_JOD"].to_numpy(), keys
                )
    return dict(
        schema="e31-v40-upiq-training-report-v1",
        panels=reports,
        report_only=True,
        independent_test=False,
        shipping_adoption_authorized=False,
    )


def run(args):
    study = {"external": "e32", "upiq": "e31", "hdr": "e29"}[args.mode]
    cells = complete(args.bundle, study, args.results, args.control, args.tools)
    frozen = json.loads(bound_bytes(args.control_pins))
    if frozen.get("control_choice") != "fresh-matched-v40" or frozen.get(
        "program_sha"
    ) != sha(args.bundle / "program.tar.gz"):
        raise ValueError("fresh V40 control freeze required")
    for (fold, seed), cell in cells["control"].items():
        if frozen["cells"].get(f"{fold}_s{seed}") != dict(
            result_sha256=sha(cell / "result.json"),
            bake_sha256=sha(cell / "refit/last.bin"),
        ):
            raise ValueError("frozen control changed")
    record = exposure(args.exposure, args.bundle, args.mode, args.control_pins)
    refuse_immutable_output(
        args.out,
        (
            args.results,
            args.control,
            args.bundle,
            *(Path(o["path"]).parent for o in record["objects"].values()),
        ),
    )
    if args.out.exists():
        raise ValueError("fresh assessment output required")
    args.out.mkdir(parents=True)
    if args.mode == "hdr":
        expected = {
            "_MANIFEST.json": "metadata",
            "native-proof.json": "metadata",
            "sdr.json": "metadata",
            "keys.parquet": "payload",
            "features.parquet": "payload",
            "predictor": "tool",
        }
        if set(record["objects"]) != set(expected) or any(
            record["objects"][alias]["kind"] != kind for alias, kind in expected.items()
        ):
            raise ValueError("exact E29 registered HDR assessment inputs required")
        metadata = retain(record, {"metadata"})
        manifest = json.loads(metadata["_MANIFEST.json"])
        proof = json.loads(metadata["native-proof.json"])
        from rev5_bank import HDR_TEACHER_SHA

        if (
            manifest.get("role") != "val"
            or manifest.get("rows") != 3900
            or manifest.get("formula_revision") != "Rev5"
            or manifest.get("requested_ids") != BASE_IDS
            or manifest.get("teacher_sha256") != HDR_TEACHER_SHA["val"]
            or manifest.get("keys_sha256")
            != record["objects"]["keys.parquet"]["sha256"]
            or manifest.get("features_parquet_sha256")
            != record["objects"]["features.parquet"]["sha256"]
            or proof.get("cached_executable_sha256")
            != record["objects"]["predictor"]["sha256"]
            or proof.get("val_bank_manifest_sha256")
            != record["objects"]["_MANIFEST.json"]["sha256"]
        ):
            raise ValueError("E29 registered VAL/native proof metadata differs")
        blobs = {**metadata, **retain(record, {"payload", "tool"})}
        bank = args.out / "retained-native-bank"
        bank.mkdir()
        for alias, data in blobs.items():
            (bank / alias).write_bytes(data)
            (bank / alias).chmod(0o400)
        (bank / "predictor").chmod(0o500)
        os.environ["ZL_PREDICT"] = str(bank / "predictor")
        from scripts.hdr.hdr_route_panel import _e26_panel

        _e26_panel(
            args.results,
            args.control,
            bank,
            args.out,
            bank / "native-proof.json",
            study="e29",
            e29_control_pins=args.control_pins.with_name("E29_CONTROL_PINS.json"),
            e29_sdr_decision=bank / "sdr.json",
        )
        # Registered descriptive Borda panel; it never enters the adoption rule.
        from e29_consensus import consensus

        keys = pq.read_table(pa.BufferReader(blobs["keys.parquet"])).to_pandas()
        target = consensus(keys.hdrvdp3_q_jod.to_numpy(), keys.cvvdp_jod.to_numpy())
        borda = {}
        for arm, grid in cells.items():
            borda[arm] = {}
            for fold, seed in grid:
                pred = np.load(
                    args.out / f"{arm}_{fold}_s{seed}.npy", allow_pickle=False
                )
                panel = panel_batch([("borda", pred, target)], stats="srocc")[0]
                if panel["n_dropped"] or not np.isfinite(panel["srocc_signed"]):
                    raise ValueError("INCOMPLETE: undefined Borda VAL panel")
                borda[arm][f"{fold}_s{seed}"] = panel["srocc_signed"]
        (args.out / "borda_report.json").write_text(
            json.dumps(
                dict(
                    report_only=True,
                    panels=borda,
                    exposure_freeze_sha256=sha(args.exposure),
                ),
                indent=2,
            )
            + "\n"
        )
        return
    report = (
        external(record, cells, args.out)
        if args.mode == "external"
        else upiq(record, cells, args.out)
    )
    report.update(
        exposure_freeze_sha256=sha(args.exposure),
        control_pins_sha256=sha(args.control_pins),
    )
    (args.out / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("external", "upiq", "hdr"), required=True)
    for name in (
        "bundle",
        "results",
        "control",
        "tools",
        "control-pins",
        "exposure",
        "out",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
