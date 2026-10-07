#!/usr/bin/env python3
"""D2 read orchestration over frozen Rust-surface predictions; never a scorer.

The receipt format and preparation commands are in release_gate_map_2026-10-07.md.
Authorization is checked before ANY label hashing. Exposure is reserved durably
before the first label open; errors also spend the design line. No default input
or label discovery. Original stimuli (including identity/collapsed bank rows)
must all appear in the pinned population and prediction table.
"""

import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
import stat
from pathlib import Path
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from contextlib import contextmanager

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import zen_stats
from lib.assessment_identity import safe_path
import v2c_labels

DESIGN = "by_v2fy-rev5-d1-20261007"
GATES = (
    "Table provenance",
    "Rust surface",
    "G-RANK",
    "G-DIAL",
    "G-ADDR",
    "codec floors",
    "negative tails/identity",
    "G-STEER",
    "G-RD",
    "G-TARGET",
    "integrity",
    "Rev5 correctness",
    "runtime/memory",
    "input/serving",
    "HDR scope",
)
MODELS = ("production", "profile_b", "research_by_v2fy")
REPO = Path(__file__).resolve().parents[2]
REGISTRATION = REPO / "benchmarks/kadid_terminal_registration_2026-10-07.md"
CONTRACT = REPO / "benchmarks/shippath_qualified_fit_contract_2026-10-07.json"
JOURNAL = Path.home() / "tmp/zensim-paper/rev4/KADID_TERMINAL_SPENT.json"
CODE = {
    p.name: p
    for p in (
        Path(__file__).resolve(),
        Path(v2c_labels.__file__).resolve(),
        Path(zen_stats.__file__).resolve(),
    )
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


# Metadata belongs to prepared artifacts, source or the exposure journal, never
# to original corpus stores. Roots are owned here, not supplied by the receipt.
METADATA_ROOTS = (
    Path(__file__).resolve().parents[2],
    Path.home() / "tmp/zensim-paper/rev4",
    Path("/mnt/v/output/zensim"),
    Path("/var/tmp/rev4-featpot"),
)
CORPUS_ROOTS = (
    Path("/mnt/v/dataset"),
    Path("/mnt/v/datasets"),
    Path("/mnt/v/input"),
    Path("/mnt/v/collections"),
    Path("/var/tmp/datasets"),
    v2c_labels.BANK,
)
# DATA_SPLITS T0/terminal bank populations and JPEG-AIC holdout family. Bank
# inputs are never needed: the label-free population is prepared outside it.
PROTECTED_POPULATIONS = {
    "bank",
    "kadid_terminal",
    "cid22_b",
    "cid22val",
    "cid22_validation_set",
    "aic3",
    "aic4",
    "aic2026",
    "sdr25",
    "jpeg-ai-sdr25",
    "csiq",
    "nits",
    "nits-iqa",
    "mciqa",
    "mciqa-2k",
    "upiq",
    "konjnd_jpeg_terminal",
}


class TerminalReadError(ValueError):
    """Only a fixed category can cross the post-read or CLI boundary."""

    def __init__(self, category):
        self.category = (
            category
            if category
            in {
                "preflight-refused",
                "exposure-refused",
                "label-adapter-error",
                "assessment-error",
                "execution-error",
            }
            else "execution-error"
        )
        super().__init__(self.category)


def metadata_path(path):
    p = safe_path(path)
    require(p.is_absolute(), "absolute metadata paths required")
    resolved = p.resolve()
    for candidate in (p, resolved):
        parts = {v.lower() for v in candidate.parts}
        require(
            not parts.intersection(PROTECTED_POPULATIONS)
            and not any("terminal" in v.lower() for v in candidate.parent.parts)
            and not any(
                v == "t0" or v.startswith("t0_") or v.startswith("t0-") for v in parts
            )
            and not any(
                candidate.is_relative_to(root.resolve()) for root in CORPUS_ROOTS
            ),
            "terminal/T0 metadata ancestry refused",
        )
        require(
            any(
                candidate.is_relative_to(root.resolve())
                for root in (*METADATA_ROOTS, REPO)
            ),
            "metadata outside trusted preparation roots",
        )
    # Return precisely the resolved pathname whose ancestry was admitted.
    # All later payload I/O walks this spelling without following any symlink.
    return resolved


def metadata_open(path, mode="rb", **kwargs):
    """Bind every component from / using directory fds and O_NOFOLLOW (Linux).

    Legitimate prepared aliases resolve during admission. Retargeting an alias
    afterwards cannot change this walk; replacing a resolved ancestor/leaf with
    a symlink refuses. Once a directory is open its handle survives renames.
    No fallback to a pathname open on platforms lacking the required flags.
    """
    resolved = metadata_path(path)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    modes = {
        "rb": os.O_RDONLY,
        "r+": os.O_RDWR,
        "x": os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        "w": os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        "a": os.O_WRONLY | os.O_APPEND,
    }
    flags = modes[mode] | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
    parent = os.open("/", directory_flags)
    fd = None
    try:
        for component in resolved.parts[1:-1]:
            next_parent = os.open(component, directory_flags, dir_fd=parent)
            os.close(parent)
            parent = next_parent
        fd = os.open(resolved.name, flags, 0o600, dir_fd=parent)
        require(stat.S_ISREG(os.fstat(fd).st_mode), "regular metadata file required")
        bound = os.fdopen(fd, mode, **kwargs)
        fd = None  # bound owns it now, including exceptional close paths.
        return bound
    finally:
        os.close(parent)
        if fd is not None:
            os.close(fd)


def metadata_bytes(path):
    with metadata_open(path) as bound:
        return bound.read()


def metadata_sha(path):
    return hashlib.sha256(metadata_bytes(path)).hexdigest()


def checked_bytes(spec):
    data = metadata_bytes(spec["path"])
    require(hashlib.sha256(data).hexdigest() == spec["sha256"], "changed input")
    return data


@contextmanager
def checked_open(spec):
    # The external model inspector gets the same fd that we hashed, not the
    # mutable model/inspector pathname. Its child inherits only these fds.
    with metadata_open(spec["path"]) as bound:
        require(
            hashlib.sha256(bound.read()).hexdigest() == spec["sha256"], "changed input"
        )
        bound.seek(0)
        yield bound


def validate_label_adapter(spec):
    require(isinstance(spec, dict), "original label adapter required")
    require(
        isinstance(spec.get("path"), str)
        and "\0" not in spec["path"]
        and Path(spec["path"]).is_absolute(),
        "explicit original label path required",
    )
    digest = spec.get("sha256")
    require(
        isinstance(digest, str)
        and len(digest) == 64
        and all(c in "0123456789abcdef" for c in digest),
        "64-hex label SHA-256 required",
    )
    require(
        spec.get("via_pairs") is None and spec.get("select") is None,
        "dedicated terminal-only original label manifest required",
    )
    cols = [spec.get(k) for k in ("ref_col", "dist_col", "label_col")]
    require(
        spec.get("format") in ("tsv", "csv", "json")
        and all(isinstance(c, str) and c.strip() for c in cols)
        and len(set(cols)) == 3,
        "original label adapter required",
    )
    if spec["format"] == "json":
        require(
            isinstance(spec.get("rows_key"), str) and bool(spec["rows_key"].strip()),
            "JSON label rows_key required",
        )
    usecols = spec.get("usecols")
    require(
        usecols is None
        or (
            isinstance(usecols, list)
            and all(isinstance(c, str) and c.strip() for c in usecols)
            and len(set(usecols)) == len(usecols)
            and set(cols).issubset(usecols)
        ),
        "label usecols must include every adapter column",
    )


def checked(spec):
    path = metadata_path(spec["path"])
    require(path.is_absolute(), "absolute paths required")
    checked_bytes(spec)
    return path


def rows(path):
    data = checked_bytes(path) if isinstance(path, dict) else metadata_bytes(path)
    return list(csv.DictReader(io.StringIO(data.decode(), newline=""), delimiter="\t"))


def committed_bytes(commit, pin):
    # Added jj workspaces have no .git. Resolve the shared store without
    # snapshotting working-copy files, including any exposure payloads.
    git_dir = REPO / ".git"
    if not git_dir.exists():
        git_dir = Path(
            subprocess.run(
                ["jj", "--ignore-working-copy", "git", "root"],
                cwd=REPO,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    return subprocess.run(
        ["git", "--git-dir", str(git_dir), "show", f"{commit}:{pin}"],
        cwd=REPO,
        capture_output=True,
        check=True,
    ).stdout


def preflight(receipt_path, authorization_path, ledger, journal, output):
    # Reject even metadata aliases into protected ancestry before opening them.
    receipt_path, authorization_path = (
        metadata_path(receipt_path),
        metadata_path(authorization_path),
    )
    # One read: retain the bytes matched against the committed object.
    receipt_live = metadata_bytes(receipt_path)
    receipt_sha = hashlib.sha256(receipt_live).hexdigest()
    auth = json.loads(metadata_bytes(authorization_path))
    require(
        auth.get("schema") == "kadid-terminal-authorization-v1"
        and auth.get("authorize_once") is True
        and auth.get("design_line") == DESIGN
        and auth.get("receipt_sha256") == receipt_sha
        and bool(auth.get("coordinator_message", "").strip()),
        "coordinator authorization required",
    )
    require(
        metadata_path(auth["ledger"]).resolve() == metadata_path(ledger).resolve()
        and metadata_path(auth["journal"]).resolve()
        == metadata_path(journal).resolve(),
        "exposure destinations not authorized",
    )
    # The model/scorer pin must already exist in a local committed tree.
    commit, pin = auth["pre_read_commit"], auth["receipt_repo_path"]
    require(
        len(commit) == 40 and all(c in "0123456789abcdef" for c in commit),
        "exact pre-read commit required",
    )
    require(
        not Path(pin).is_absolute() and ".." not in Path(pin).parts,
        "invalid committed pin path",
    )
    metadata_path(REPO / pin)
    committed = committed_bytes(commit, pin)
    require(
        hashlib.sha256(committed).hexdigest() == receipt_sha,
        "pre-read commit does not bind receipt",
    )
    receipt = json.loads(committed)
    validate_label_adapter(receipt.get("labels"))
    require(
        receipt.get("schema") == "kadid-terminal-final-model-v1"
        and receipt.get("design_line") == DESIGN
        and receipt.get("population_rows") == 2000
        and receipt.get("bootstrap_resamples") == 10000
        and type(receipt.get("bootstrap_seed")) is int
        and receipt["bootstrap_seed"] >= 0
        and receipt.get("orientation") == "quality",
        "registered population/statistics required",
    )
    require(
        receipt["registration_sha256"] == metadata_sha(REGISTRATION),
        "changed D2 registration",
    )
    require(
        receipt.get("code_sha256") == {k: metadata_sha(p) for k, p in CODE.items()},
        "unbound read/label/statistic owners",
    )
    require(
        receipt.get("human_sources") == ["kadid", "tid2013", "konfig", "cid22_a25"]
        and receipt.get("fit_jobset") == "fitv2d1-20261007"
        and receipt.get("selected_epoch") == 119,
        "D1 production fit required",
    )
    # Preflight the WHOLE non-label input set before hashing any member. Labels
    # are the sole explicit exception, and are deferred until after reservation.
    specs = [
        receipt["population"],
        receipt["predictions"],
        receipt["prediction_receipt"],
        receipt["panel"],
        receipt["scorer"],
        receipt["inspector"],
        receipt["qualification"],
        *receipt["models"].values(),
        *receipt["bindings"],
    ]
    labels = Path(receipt["labels"]["path"]).resolve()
    for spec in specs:
        path = metadata_path(spec["path"])
        require(path.resolve() != labels, "label alias in metadata/input inventory")
    ledger, journal, output = map(Path, (ledger, journal, output))
    for path in (ledger, journal, output):
        metadata_path(path)
        require(path.resolve() != labels, "label alias in output")
    require(
        ledger.is_file() and not journal.exists() and not output.exists(),
        "spent read or missing ledger/existing output",
    )
    for spec in specs:
        checked(spec)
    require(set(receipt["models"]) == set(MODELS), "exact comparators required")
    require(
        receipt["composition"].get("primary") == receipt["models"]["production"],
        "composition primary binding missing",
    )
    require(
        hashlib.sha256(checked_bytes(receipt["models"]["profile_b"])).hexdigest()
        == metadata_sha(
            REPO
            / "zensim/weights/b_sdr_linear_cid80_inclwinsor_dense_dial_byid_2026-09-06.bin"
        ),
        "profile B must be the shipped codec_target bake",
    )
    qualification = json.loads(checked_bytes(receipt["qualification"]))
    require(
        qualification.get("composition") == receipt["composition"]
        and qualification.get("model_sha256")
        == receipt["models"]["production"]["sha256"]
        and all(
            qualification.get("gates", {}).get(g, {}).get("state") == "pass"
            for g in GATES
        ),
        "every pre-terminal release gate must pass for exact composition",
    )
    report = qualification.get("reports", {}).get("E30", {})
    require(report.get("state") == "completed", "completed bound E30 report required")
    # E30 reports the removal cost; D1 is fixed regardless of its comparison.
    evidence = [qualification["gates"][g]["artifact"] for g in GATES] + [
        report["artifact"]
    ]
    for spec in evidence:
        require(
            metadata_path(spec["path"]).resolve() != labels,
            "label alias in gate evidence",
        )
    for spec in evidence:
        checked(spec)
    e30 = json.loads(checked_bytes(report["artifact"]))
    sources = ("kadid", "tid2013", "konfig", "cid22_a25")
    require(
        e30.get("schema") == "rev4-featpot-e30-four-source-v1"
        and e30.get("status") == "complete"
        and e30.get("missing_cells") == 0
        and e30.get("seeds") == list(range(10))
        and set(e30.get("rows", {})) == {"nA3"}
        and set(e30["rows"]["nA3"].get("per_source", {})) == set(sources)
        and set(e30.get("control_stats", {}))
        == {f"{source}_s{i}" for source in sources for i in range(10)}
        and e30.get("scope")
        == "Report removal cost only; D1 fixed; no adoption rule; no AIC fold"
        and "rule" not in e30
        and "adopt" not in e30["rows"]["nA3"],
        "complete registered report-only E30 artifact required",
    )
    prediction_receipt = json.loads(checked_bytes(receipt["prediction_receipt"]))
    require(
        prediction_receipt
        == {
            "schema": "kadid-terminal-surface-predictions-v1",
            "labels_read": False,
            "surface": "zensim::BakeScorer / Zensim::codec_target",
            "population": receipt["population"],
            "predictions": receipt["predictions"],
            "scorer": receipt["scorer"],
            "models": receipt["models"],
            "composition": receipt["composition"],
            "bindings": receipt["bindings"],
        },
        "prediction owner receipt does not bind final surface/bytes/population",
    )
    with (
        checked_open(receipt["inspector"]) as inspector,
        checked_open(receipt["models"]["production"]) as model,
    ):
        metadata = json.loads(
            subprocess.run(
                [
                    f"/proc/self/fd/{inspector.fileno()}",
                    f"/proc/self/fd/{model.fileno()}",
                ],
                pass_fds=(inspector.fileno(), model.fileno()),
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )
    require(
        metadata.get("qualified_provenance") is True
        and metadata.get("formula_revision") == 5
        and str(metadata.get("checkpoint_epoch")) == "119"
        and metadata.get("admitted_tables") == 7
        and metadata.get("pair_sampling") == "uniform",
        "final qualified Rev5 epoch119 required",
    )
    contract = json.loads(metadata_bytes(CONTRACT))
    seed = receipt["seed_index"]
    require(
        type(seed) is int and seed in (0, 1, 2), "registered full-data seed required"
    )
    repro = metadata["repro"]
    require(
        repro["epochs"] == repro["requested_epochs"] == contract["epochs"]
        and repro["pairs_per_epoch"] == contract["pairs_per_epoch"]
        and repro["init_seed"] == contract["init_seeds"][seed]
        and repro["sample_seed"] == contract["sample_seeds"][seed]
        and repro["keep_features_n"] == 420,
        "decoded model budget/seeds mismatch",
    )
    require(
        {v["name"]: v["sha256"] for v in repro["inputs"]}
        == {v["name"]: v["table_sha256"] for v in contract["routes"]["production"]},
        "decoded model does not bind registered D1 input tables",
    )
    population = rows(receipt["population"])
    predictions = rows(receipt["predictions"])
    require(
        len(population) == len(predictions) == 2000,
        "all 2000 original stimuli required",
    )
    key_columns = (
        "source_row_id",
        "pair_key",
        "ref_basename",
        "distortion_type",
        "ref_path",
        "dist_path",
        "ref_pixels_sha256",
        "dist_pixels_sha256",
    )
    require(
        all(set(r) == set(key_columns) for r in population),
        "label-free exact population schema required",
    )
    require(
        all(set(r) == {"source_row_id", *MODELS} for r in predictions),
        "label-free prediction schema required",
    )
    require(
        len({r["source_row_id"] for r in population}) == 2000
        and [r["source_row_id"] for r in population]
        == [r["source_row_id"] for r in predictions],
        "row identity/order mismatch",
    )
    require(
        len({(r["ref_path"], r["dist_path"]) for r in population}) == 2000,
        "original stimulus path pairs must be unique; do not discard collapsed keys",
    )
    require(
        all(
            len(r[c]) == 64 and all(v in "0123456789abcdef" for v in r[c])
            for r in population
            for c in ("ref_pixels_sha256", "dist_pixels_sha256")
        ),
        "pixel byte pins required",
    )
    require(
        all(np.isfinite(float(r[m])) for r in predictions for m in MODELS),
        "nonfinite prediction",
    )
    return receipt, receipt_sha, population, predictions


def reserve(ledger, journal, receipt_sha):
    token = f"KADID-TERMINAL-SPENT:{DESIGN}"
    # O_EXCL is shared across workspaces/outputs; flock also serializes ledger
    # append. A crash or failed statistic cannot permit a second look.
    with metadata_open(ledger, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        require(token not in f.read(), "terminal design line already spent")
        record = {
            "design_line": DESIGN,
            "receipt_sha256": receipt_sha,
            "state": "spent-before-label-open",
            "time_utc": datetime.now(timezone.utc).isoformat(),
        }
        with metadata_open(journal, "x") as j:
            json.dump(record, j, indent=2)
            j.flush()
            os.fsync(j.fileno())
        f.seek(0, os.SEEK_END)
        f.write(
            f"\n## KADID TERMINAL exposure — {record['time_utc']}\n\n"
            f"{token}\nPre-read receipt SHA-256 `{receipt_sha}`. Reserved once before label opening; "
            "this design line is spent even on failure. No adaptive reuse.\n"
        )
        f.flush()
        os.fsync(f.fileno())


def assess(receipt, population, predictions):
    # First label open occurs here, AFTER authorization, all gates and exposure.
    try:
        label_rows = v2c_labels.load_label_rows(receipt["labels"])
    except Exception:
        raise TerminalReadError("label-adapter-error") from None
    lookup = {(r.ref_path, r.dist_path): r.label for r in label_rows.itertuples()}
    require(
        len(label_rows) == len(lookup) == 2000
        and set(lookup) == {(r["ref_path"], r["dist_path"]) for r in population},
        "terminal label coverage mismatch",
    )
    y = [lookup[(r["ref_path"], r["dist_path"])] for r in population]
    x = {m: [float(r[m]) for r in predictions] for m in MODELS}
    os.environ["ZEN_PANEL_BIN"] = receipt["panel"]["path"]
    groups = {
        r: [i for i, p in enumerate(population) if p["ref_basename"] == r]
        for r in sorted({p["ref_basename"] for p in population})
    }
    require(len(groups) > 1, "reference bootstrap requires multiple references")
    full = zen_stats.panel_batch([(m, x[m], y) for m in MODELS])
    for row in full:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv") as wire:
            writer = csv.writer(wire, delimiter="\t")
            writer.writerow(("predicted", "target"))
            writer.writerows(zip(x[row["label"]], y))
            wire.flush()
            signed = json.loads(
                subprocess.run(
                    [
                        receipt["panel"]["path"],
                        "--input",
                        wire.name,
                        "--json",
                        "--signed-quality",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout
            )
        require(
            signed.get("schema") == "panel-signed-quality-v1"
            and signed.get("orientation") == "quality",
            "signed quality statistic owner required",
        )
        row.update(
            {
                k: signed[k]
                for k in ("srocc_signed", "krocc", "plcc", "plcc_raw", "plcc_form")
            }
        )
        row["srocc"] = row["srocc_signed"]
    diagnostics = {}
    for m in MODELS:
        within = zen_stats.panel_batch(
            [
                (r, [x[m][i] for i in idx], [y[i] for i in idx])
                for r, idx in groups.items()
            ],
            stats="srocc",
        )
        types = sorted({r["distortion_type"] for r in population})
        by_type = zen_stats.panel_batch(
            [
                (
                    t,
                    [
                        x[m][i]
                        for i, p in enumerate(population)
                        if p["distortion_type"] == t
                    ],
                    [
                        y[i]
                        for i, p in enumerate(population)
                        if p["distortion_type"] == t
                    ],
                )
                for t in types
            ],
            stats="srocc",
        )
        for row in (*within, *by_type):
            row["srocc"] = row["srocc_signed"]
        # Reduce the already-returned signed canonical per-reference results.
        # Match the owner min_len=3 and retain undefined/short groups explicitly.
        eligible = [
            r["srocc_signed"]
            for r in within
            if r["n"] >= 3 and np.isfinite(r["srocc_signed"])
        ]
        per_group = {
            "orientation": "quality",
            "min_len": 3,
            "n_groups": len(eligible),
            "n_excluded": len(within) - len(eligible),
            "mean": float(np.mean(eligible)) if eligible else float("nan"),
        }
        diagnostics[m] = {
            "within_reference": within,
            "within_reference_summary": per_group,
            "within_reference_srocc": per_group["mean"],
            "per_distortion_type": by_type,
            "scatter": zen_stats.scatter(x[m], y),
        }
    rng = np.random.default_rng(receipt["bootstrap_seed"])
    clusters = list(groups.values())
    draws = []
    # Bound wire memory: one batch of 100 paired reference draws at a time.
    for start in range(0, 10000, 100):
        jobs = []
        for b in range(start, start + 100):
            idx = [
                i
                for g in rng.integers(0, len(clusters), len(clusters))
                for i in clusters[g]
            ]
            jobs.extend(
                (f"{b}:{m}", m, "target", idx) for m in ("production", "profile_b")
            )
        values = zen_stats.panel_batch_indexed({**x, "target": y}, jobs, stats="srocc")
        draws.extend(
            values[i]["srocc_signed"] - values[i + 1]["srocc_signed"]
            for i in range(0, len(values), 2)
        )
    require(
        np.isfinite(draws).all(), "undefined bootstrap statistic; read remains spent"
    )
    metrics = {r["label"]: r for r in full}
    require(
        all(
            r["n"] == 2000 and r["n_dropped"] == 0 and np.isfinite(r["srocc_signed"])
            for r in full
        ),
        "invalid full panel",
    )
    delta_b = (
        metrics["production"]["srocc_signed"] - metrics["profile_b"]["srocc_signed"]
    )
    delta_research = (
        metrics["production"]["srocc_signed"]
        - metrics["research_by_v2fy"]["srocc_signed"]
    )
    se = float(np.std(draws, ddof=1))
    gates = {
        "profile_b": delta_b >= -2 * se,
        "research_by_v2fy": delta_research >= -0.005,
    }
    return {
        "metrics": metrics,
        "diagnostics": diagnostics,
        "paired_bootstrap": {
            "unit": "reference",
            "resamples": 10000,
            "seed": receipt["bootstrap_seed"],
            "delta_b": delta_b,
            "delta_b_se": se,
            "delta_research": delta_research,
        },
        "gates": gates,
        "confirmation": "PASS" if all(gates.values()) else "FAIL",
    }


def execute(receipt, authorization, ledger, journal, output):
    try:
        r, receipt_sha, pop, pred = preflight(
            receipt, authorization, ledger, journal, output
        )
    except Exception:
        raise TerminalReadError("preflight-refused") from None
    try:
        reserve(ledger, journal, receipt_sha)
    except Exception:
        raise TerminalReadError("exposure-refused") from None
    try:
        result = dict(
            assess(r, pop, pred),
            schema="kadid-terminal-result-v1",
            receipt_sha256=receipt_sha,
            labels_read=True,
            design_line_spent=True,
        )

        # Undefined auxiliary groups stay explicit as null; never turn them
        # into a numeric correlation or drop the stimulus population.
        def json_values(value):
            if isinstance(value, dict):
                return {k: json_values(v) for k, v in value.items()}
            if isinstance(value, list):
                return [json_values(v) for v in value]
            return (
                None if isinstance(value, float) and not np.isfinite(value) else value
            )

        result = json_values(result)
    except Exception as error:
        result = {
            "schema": "kadid-terminal-result-v1",
            "receipt_sha256": receipt_sha,
            "design_line_spent": True,
            "confirmation": "ERROR",
            "error": error.category
            if isinstance(error, TerminalReadError)
            else "assessment-error",
        }
        with metadata_open(output, "w") as f:
            f.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
        raise TerminalReadError(result["error"]) from None
    with metadata_open(output, "w") as f:
        f.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with metadata_open(ledger, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(
            f"\nD2 result `{Path(output)}` SHA-256 `{metadata_sha(output)}`: **{result['confirmation']}**. "
            "Labels read once; no retuning permitted.\n"
        )
        f.flush()
        os.fsync(f.fileno())
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--ledger", type=Path, default=REPO / "docs/DATA_SPLITS.md")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    # Caller-controlled scratch never falls back to /tmp.
    scratch = Path.home() / "tmp/kadid-terminal-read"
    scratch.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(scratch)
    try:
        result = execute(
            args.receipt, args.authorization, args.ledger, JOURNAL, args.output
        )
    except Exception as error:
        category = (
            error.category
            if isinstance(error, TerminalReadError)
            else "execution-error"
        )
        print(f"KADID TERMINAL REFUSED: {category}", file=sys.stderr)
        return 2
    print(result["confirmation"])
    return 0 if result["confirmation"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
