#!/usr/bin/env python3
"""Re-grade EVERY board cell on the 2026-09-05 FLOOR-DENSE LADDER instrument, and
graft the reading onto the board as its own `dial_ladder` block.

Pre-registration: `docs/PLAN_BOARD_LADDER_RULER_2026-09-06.md` (pushed before any
re-grade ran).  Read §3 there for the width + era rules this file implements.

WHY A SECOND BLOCK RATHER THAN A REPLACED ONE.  `dial.addressability` is cut on the
board's OWN dial grid and is same-grid gated by `promote_fulleval.py --graft-gaddr`;
a ladder reading is a different instrument by construction, so merging the two into
one column would produce silently wrong cross-cell comparisons (gate doc §17.7
measured the refusal).  So the ladder reading lands in `dial_ladder`, carries its
own instrument stamp, and `dial` is never opened for writing.

WHAT THIS FILE DOES NOT DO.  It computes no statistic.  `bake_verdict` owns the
measurement, `bake_block_profile` owns the width/block fingerprint, the registry
owns every bar and reference table, and `promote_fulleval.py --graft-gaddr-ladder`
owns the board write.  This file only decides WHICH instrument a cell is entitled
to, reconstructs the cell's own invocation, and records a reason whenever the
answer is NOT MEASURED.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REGISTRY = REPO / "benchmarks/dial_addressability_floor_2026-09-04.json"
LADDER = Path("/mnt/v/output/zensim/ladder-2026-09-05/instruments")

# The two ladder instruments, by CALLER width.  Both hold the SAME 9,593 distinct
# settings; the width only decides which bakes can be scored.  Registered (grid row
# AND per-codec floor rows, under both `distinct` and the operative `resolvable`)
# in benchmarks/dial_addressability_floor_2026-09-04.json.
LADDER_GRID = {372: LADDER / "dial_grid_372col_ladder.parquet",
               944: LADDER / "dial_grid_944col_ladder.parquet"}

# The 944 feature-set question, MEASURED (plan §3.2): the ladder-944 grid populates
# f156..371 (905 slots, slot-set sha8 b6811ae0) and `bake_verdict`'s DEFAULT 944 grid
# does NOT (689 slots, sha8 026c0aba).  A bake trained where those columns are always
# zero has UNTRAINED weights there, so grading it on the ladder would multiply live
# data by noise.  This sha is the one 944 grid that is already pools-era.
POOLS_944_SHA16 = "694e16c4520a5d41"

# Every negative-tail / identity probe a board cell could have been graded with,
# keyed by the sha16 `bake_verdict` prints in its "probe <sha16> not in the registry"
# note.  Same table as scripts/gaddr_board_regrade.py -- kept in sync deliberately
# rather than imported, because that file's copy is the 2026-09-04 replay's and this
# one is the ladder run's; a shared mutable default would couple two records.
D0904 = Path("/mnt/v/output/zensim/dialgate-2026-09-04")
PROBES = {
    "42f93e61c6e5f562": D0904 / "probes944/negtail_probe_944_2026-08-01era.parquet",
    "b73ce10655cb1c16": D0904 / "probes944/negtail_probe_944_era2r4_foldapp2.parquet",
    "5609d19fa10aef81": D0904 / "negtail_probe_372_2026-09-04.parquet",
}
IDENTITY_372 = D0904 / "identity_probe_372_2026-09-04.parquet"

# ── PROBE RECOVERY, VERIFIED RATHER THAN GUESSED ────────────────────────────
# A cell's C3/C4 (negative tail) and C5/C6 (identity) rows need the SAME probe the
# board cell was graded with, or the ladder read silently loses coverage the board
# had -- which would confound "the ruler moved" with "fewer rows were measurable".
# Not every board cell has an as-run log, so the probe is RECONSTRUCTED and then
# PROVEN: `measured.negtail` / `measured.identity` are functions of (bake, probe)
# ONLY -- they never touch the dial grid -- so the correct probe is the one whose
# read reproduces the board's block BIT-FOR-BIT. MEASURED: W10L9PH_s4007_packed's
# negtail block is identical between the board's canonical-grid read and this
# lane's ladder read (frac_below_zero 0.0265, min -69.61934236224508, p1
# -22.014636705459125, p5 7.100433565953122). A probe that does not reproduce the
# expected block is REJECTED and the rows stay NOT MEASURED.
NEGTAIL_CANDIDATES = {
    372: [D0904 / "negtail_probe_372_2026-09-04.parquet",
          Path("/mnt/v/output/zensim/dpeaks372-2026-09-05/instruments/negtail_probe_372_postC_2026-09-05.parquet"),
          Path("/mnt/v/output/zensim/dpeaks372-2026-09-05/instruments/negtail_probe_372_preC_2026-09-05.parquet")],
    944: [D0904 / "probes944/negtail_probe_944_2026-08-01era.parquet",
          D0904 / "probes944/negtail_probe_944_era2r4_foldapp2.parquet",
          Path("/mnt/v/output/zensim/dfree-2026-09-05/probes/negtail_probe_944pools_2026-09-05.parquet")],
}
IDENTITY_CANDIDATES = {
    372: [D0904 / "identity_probe_372_2026-09-04.parquet",
          Path("/mnt/v/output/zensim/dpeaks372-2026-09-05/instruments/identity_probe_372_postC_2026-09-05.parquet")],
    944: [Path("/mnt/v/output/zensim/dfree-2026-09-05/probes/identity_probe_944pools_2026-09-05.parquet")],
}
# Probe-scoped (not grid-scoped) peer score tables, so they carry to any instrument.
PEER_NEGTAIL = {"peer_ssim2": D0904 / "repin/negtail_peer_ssim2.tsv"}
PEER_IDENTITY = {"peer_ssim2": D0904 / "repin/identity_peer_ssim2.tsv"}
# The per-cell reference tables a PEER cell is scored FROM, on the ladder instrument.
PEER_DIAL_CELLS = {
    "peer_ssim2": LADDER / "dialcells_ssim2_ladder.tsv",
    "peer_butteraugli": LADDER / "dialcells_butteraugli_pnorm3_ladder.tsv",
}

# `bake_verdict` requires `--bake` even in peer mode; this is the carrier the gate
# doc's own peer reproduction uses. It supplies the rank panel only -- every dial
# number in a peer read comes from `--dial-peer-scores`.
PEER_CARRIER = REPO / "zensim/weights/d_sdr_add156_id100_negrich_dial_2026-09-05.bin"

MEMBER_ROOTS = [
    "/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28",
    "/mnt/v/output/zensim/wlin7b-2026-08-30/arms",
    "/mnt/v/output/zensim/wave-r4-2026-09-01/bakes",
    "/mnt/v/output/zensim/best-of-all-2026-09-06/bakes",
]


def registry() -> dict:
    return json.loads(REGISTRY.read_text())


def ladder_shas() -> dict:
    """{sha256: width} for the REGISTERED ladder grids -- read from the registry,
    never transcribed.  A grid whose registered path is not one of LADDER_GRID's is
    not a ladder grid as far as this run is concerned."""
    want = {str(p.resolve()): w for w, p in LADDER_GRID.items()}
    out = {}
    for row in registry().get("grids", []):
        p = row.get("path")
        if p and str(Path(p)) in want:
            out[row["dial_grid_sha256"]] = want[str(Path(p))]
    return out


def registry_tables(grid_sha: str) -> tuple[str | None, str | None]:
    """(mentor per-cell truth TSV, butteraugli reference-truth TSV:variant) for a
    grid, resolved BY SHA out of the registry.  A lookup, not a guess -- a mistyped
    sha yields None and the cell reads NOT MEASURED on the rows that need it."""
    r = registry()
    truth = None
    for row in r.get("grid_floor_representability", []):
        if row.get("dial_grid_sha256") == grid_sha and row.get("path"):
            # the floor rows name the GRID; the mentor cell table is the ladder's own
            pass
    # the mentor per-cell table travels with the instrument
    if grid_sha in ladder_shas():
        truth = str(LADDER / "dialcells_ssim2_ladder.tsv")
    ref = None
    for row in (r.get("inversion_truth") or {}).get("reference_tables", []):
        if row.get("grid_sha16") and grid_sha.startswith(row["grid_sha16"]):
            if row.get("variant") == "pnorm3" and Path(row["table"]).is_file():
                ref = f"{row['table']}:pnorm3"
                break
    return truth, ref


_BP_CACHE: dict[str, dict] = {}


def block_profile(bbp: str, bake: Path) -> dict | None:
    """`bake_block_profile --json` -- the OWNER of caller width + block usage.
    Cached by bake sha so a bake shared by many cells is fingerprinted once."""
    try:
        sha = hashlib.sha256(bake.read_bytes()).hexdigest()
    except OSError:
        return None
    if sha in _BP_CACHE:
        return _BP_CACHE[sha]
    r = subprocess.run([bbp, "--bake", str(bake), "--json"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    try:
        d = json.loads(r.stdout)
    except json.JSONDecodeError:
        return None
    d["_sha256"] = sha
    _BP_CACHE[sha] = d
    return d


def parse_asrun(path: Path) -> dict:
    """Recover the inputs a 2026-09-04 as-run log records.  Only the fields this run
    REPRODUCES are taken -- the grid is deliberately NOT taken, because replacing it
    is the one thing that changes."""
    t = path.read_text(errors="replace")
    hdr = re.search(r"^bake_verdict — (.*)$", t, re.M)
    if not hdr:
        return {}
    out: dict = {}
    for key, pat in (("corpora", r"corpora=(\S+)"),):
        m = re.search(pat, hdr.group(1))
        if m:
            out[key] = m.group(1)
    m = re.search(r"grid \| `([0-9a-f]+)`", t) or re.search(
        r"corpus \*\*dial grid[^|]*\| `([0-9a-f]+)`", t)
    if m:
        out["orig_grid_sha16"] = m.group(1)
    ens = re.search(r"^ENSEMBLE: (\d+) members[^\n]*\n((?:  \S+\n)+)", t, re.M)
    if ens:
        bk = re.search(r"bake=(\S+)", hdr.group(1))
        base = Path(bk.group(1)).parent if bk else Path(".")
        roots = [base] + [Path(p) for p in MEMBER_ROOTS]
        members = []
        for m2 in ens.group(2).splitlines():
            m2 = m2.strip()
            if not m2:
                continue
            hit = next((r / m2 for r in roots if (r / m2).is_file()), None)
            members.append(str(hit) if hit else m2)
        out["members"] = members
    m = re.search(r"\*\*A7\*\*: probe ([0-9a-f]+) not in", t)
    if m and m.group(1) in PROBES:
        out["probe"] = str(PROBES[m.group(1)])
    elif "no --negtail-probe supplied" not in t:
        out["probe"] = str(PROBES["5609d19fa10aef81"])
    if "no --identity-probe supplied" not in t:
        out["identity"] = str(IDENTITY_372)
    if "--cross-regime set" in t:
        out["cross_regime"] = True
    if path.name.startswith("HYA_w084."):
        out["weights"] = "0.84,0.16"
    return out


def _root_serves(root: Path, width: int) -> bool:
    """Does this features root carry a CID22 table for this caller width?  Checked
    rather than assumed -- see the comment at the call site."""
    names = ("cid22_features_372col_2026-05-15.parquet",) if width == 372 else (
        "ext_cid22val.parquet",)
    return any((root / n).is_file() for n in names)


def plan_cell(f: Path, bbp: str, asrun: Path | None) -> dict:
    """Decide what this cell is entitled to.  Returns a record with either
    `cmd` (grade it) or `reason` (NOT MEASURED)."""
    d = json.loads(f.read_text())
    name = f.name[: -len(".fulleval.json")]
    rec: dict = {"name": name, "fulleval": str(f)}
    model = d.get("model") or {}
    addr = (d.get("dial") or {}).get("addressability") or {}
    rec["orig_grid_sha16"] = (addr.get("grid_sha256") or "")[:16] or None

    log = asrun / f"{name}.active.log" if asrun else None
    inv = parse_asrun(log) if log and log.is_file() else {}
    rec["invocation_source"] = "as-run log" if inv else "fulleval fields"
    if inv.get("orig_grid_sha16") and not rec["orig_grid_sha16"]:
        rec["orig_grid_sha16"] = inv["orig_grid_sha16"]

    # ── PEER cells (a reference metric, no bake) ────────────────────────────────
    if model.get("kind") == "reference-metric" or not d.get("bake"):
        cells = PEER_DIAL_CELLS.get(name)
        if cells is None or not Path(cells).is_file():
            rec["reason"] = (f"peer cell `{name}` has no per-cell reference table on the "
                             f"ladder instrument (only ssim2 and butteraugli are measured "
                             f"there); NOT MEASURED rather than scored from another grid")
            return rec
        # A peer cell has no bake, so its width is whatever the grid is; the 372
        # ladder is the registered instrument the mentor's own bars were cut on.
        # `bake_verdict` still REQUIRES `--bake` in peer mode -- the carrier bake
        # supplies the rank panel while `--dial-peer-scores` supplies every dial
        # number, so the G-ADDR block describes the PEER (`scorer.kind == "peer"`),
        # not the carrier. Same carrier the gate doc's own peer reproduction uses
        # (S14.9 / S17.8), named here rather than "any 372 bake" so the invocation
        # is reproducible.
        if not PEER_CARRIER.is_file():
            rec["reason"] = (f"peer cell needs a carrier bake for `bake_verdict --bake` and "
                             f"{PEER_CARRIER} is not on disk")
            return rec
        grid, width = LADDER_GRID[372], 372
        cmd = ["--bake", str(PEER_CARRIER),
               "--dial-peer-scores", f"{name}={cells}", "--dial-grid", str(grid),
               "--corpora", "cid22"]
        if name in PEER_NEGTAIL and PEER_NEGTAIL[name].is_file():
            cmd += ["--negtail-probe", str(PROBES["5609d19fa10aef81"]),
                    "--negtail-peer-scores", f"{name}={PEER_NEGTAIL[name]}"]
        if name in PEER_IDENTITY and PEER_IDENTITY[name].is_file():
            cmd += ["--identity-probe", str(IDENTITY_372),
                    "--identity-peer-scores", f"{name}={PEER_IDENTITY[name]}"]
        rec.update(kind="peer", width=width, grid_sha16=None, cmd=cmd, era="n/a (peer)")
        return rec

    bake = Path(d["bake"])
    if not bake.is_file():
        rec["reason"] = f"bake not on disk: {bake}"
        return rec
    bp = block_profile(bbp, bake)
    if bp is None:
        rec["reason"] = f"bake_block_profile could not fingerprint {bake}"
        return rec
    if bp["_sha256"] != d.get("bake_sha256"):
        rec["reason"] = (f"bake_sha256 mismatch — the fulleval records "
                         f"{str(d.get('bake_sha256'))[:16]} and the file on disk is "
                         f"{bp['_sha256'][:16]}; refusing to grade a different artifact")
        return rec
    width = bp.get("caller_input_width") or bp.get("n_inputs")
    rec["caller_width"] = width
    rec["uses_f156_371"] = bp.get("uses_f156_371")
    if width not in LADDER_GRID:
        rec["reason"] = (f"no ladder instrument at caller width {width} — the 2026-09-05 "
                         f"ladder was built at 372 and 944 only")
        return rec

    # ── the 944 ERA gate (plan §3.2), MEASURED ─────────────────────────────────
    era = "372 ladder (single-era instrument)"
    if width == 944:
        if rec["orig_grid_sha16"] == POOLS_944_SHA16:
            era = "pools-era (its own instrument is already f156-371-live)"
        elif bp.get("uses_f156_371") is False:
            era = "immune (bake's f156-371 weights are exactly zero)"
        else:
            rec["reason"] = (
                "era mismatch: bake READS f156-371, which its own 944 instrument zeroes "
                "(slot-set 026c0aba, 689 populated) and the ladder-944 populates "
                "(b6811ae0, 905 populated) — grading it here would multiply live columns "
                "by weights that never received a gradient")
            rec["era"] = "MISMATCH"
            return rec
    rec["era"] = era

    grid = LADDER_GRID[width]
    cmd = ["--bake", str(bake), "--dial-grid", str(grid)]
    # The features root feeds the RANK panel only -- G-ADDR reads the dial grid and
    # the probes, nothing else -- so a root that cannot serve this width's corpus
    # files is dropped rather than made fatal, and the drop is RECORDED. (Measured:
    # five 372-width cells record a 944 root's `foldapp2_views/`, which has no
    # `cid22_features_372col_*`; `bake_verdict` then aborts with MISSING corpus and
    # the cell would read NOT MEASURED for a reason that has nothing to do with the
    # ruler.)
    root = ((d.get("features_root") or {}).get("path"))
    if root and Path(root).is_dir() and _root_serves(Path(root), width):
        cmd += ["--features-root", root]
    elif root:
        rec["features_root_dropped"] = root
    cmd += ["--corpora", inv.get("corpora", "cid22")]
    if width == 944:
        cmd += ["--regime", "944"]
    if inv.get("members"):
        cmd += ["--ensemble", ",".join(inv["members"])]
        if inv.get("weights"):
            cmd += ["--ensemble-weights", inv["weights"]]
    elif model.get("kind") == "ensemble":
        rec["ensemble_members_unrecoverable"] = True
    if inv.get("cross_regime"):
        cmd += ["--cross-regime"]
    if inv.get("probe"):
        cmd += ["--negtail-probe", inv["probe"]]
        rec["negtail"] = inv["probe"]
    if inv.get("identity"):
        cmd += ["--identity-probe", inv["identity"]]
        rec["identity"] = inv["identity"]
    meas = addr.get("measured") or {}
    rec["expect_negtail"] = meas.get("negtail")
    rec["expect_identity"] = meas.get("identity")
    rec["probe_candidates"] = [str(x) for x in NEGTAIL_CANDIDATES.get(width, []) if x.is_file()]
    rec["identity_candidates"] = [str(x) for x in IDENTITY_CANDIDATES.get(width, []) if x.is_file()]
    rec.update(kind="bake", width=width, cmd=cmd)
    return rec


# `measured.identity` mixes PROBE reads with GRID properties: `n_above_identity` and
# `n_grid_cells_total` count dial-grid cells (4,817 on the POOLS grid, 9,593 on the
# ladder) and so CANNOT match across instruments. MEASURED on Ffree@dfreelane, whose
# board block records `n_grid_cells_total: 4817`. The probe-scoped fields
# (`dial_min`/`dial_median`/`dial_max`/`n`) are pure functions of (bake, probe) and
# are what the recovery proof compares.
GRID_DEPENDENT_PROBE_FIELDS = {"n_above_identity", "n_grid_cells_total"}


def _probe_block_eq(got, want) -> bool:
    """Do two probe blocks agree on every field that is a function of (bake, probe)?
    Compared over the keys they SHARE, minus the grid-dependent counts -- a board
    block written by an older tool may simply carry fewer fields."""
    if want is None:
        return True
    if not isinstance(got, dict) or not isinstance(want, dict):
        return got == want
    keys = (set(got) & set(want)) - GRID_DEPENDENT_PROBE_FIELDS
    if not keys:
        return False
    return all(got[k] == want[k] for k in keys)


def _probe_args(cmd: list[str], negtail: str | None, identity: str | None) -> list[str]:
    """`cmd` with any --negtail-probe/--identity-probe replaced by these."""
    out, skip = [], 0
    for i, a in enumerate(cmd):
        if skip:
            skip -= 1
            continue
        if a in ("--negtail-probe", "--identity-probe"):
            skip = 1
            continue
        out.append(a)
    if negtail:
        out += ["--negtail-probe", negtail]
    if identity:
        out += ["--identity-probe", identity]
    return out


def run_cell(bv: str, rec: dict, out: Path, value_pins: str, tail_pins: str,
             floor_rule: str | None, floor_margin: float | None) -> dict:
    """Grade the cell, then PROVE the probe coverage matches the board's.

    If the board cell carries a `measured.negtail` / `measured.identity` block and
    this read does not reproduce it bit-for-bit, the probe was not recovered -- so
    every width-compatible candidate is tried and the one that REPRODUCES the board
    block is accepted. None matching leaves the rows NOT MEASURED with a reason;
    a wrong probe is never accepted just because it produced numbers."""
    first = _run_once(bv, rec, out, value_pins, tail_pins, floor_rule, floor_margin,
                      rec.get("negtail"), rec.get("identity"))
    exp_n, exp_i = rec.get("expect_negtail"), rec.get("expect_identity")
    if "reason" in first:
        return first
    def _ident_ok(m: dict) -> bool:
        if exp_i is None:
            return True
        if m.get("identity") is not None:
            return _probe_block_eq(m.get("identity"), exp_i)
        return bool(m.get("identity_rows_measured"))

    got_n = (first.get("_measured") or {}).get("negtail")
    if _probe_block_eq(got_n, exp_n) and _ident_ok(first.get("_measured") or {}):
        first.pop("_measured", None)
        return first
    ncands = [None] + list(rec.get("probe_candidates") or [])
    icands = [None] + list(rec.get("identity_candidates") or [])
    for n in ncands:
        for i in icands:
            r = _run_once(bv, rec, out, value_pins, tail_pins, floor_rule, floor_margin, n, i)
            if "reason" in r:
                continue
            m = r.get("_measured") or {}
            if _probe_block_eq(m.get("negtail"), exp_n) and _ident_ok(m):
                r["probe_recovered"] = {"negtail": n, "identity": i,
                                        "how": "reproduces the board block bit-for-bit"}
                r.pop("_measured", None)
                return r
    first["probe_recovery_failed"] = (
        "the board cell carries a probe block this run could not reproduce with any "
        "width-compatible probe on disk; C3/C4 (and/or C5/C6) stay NOT MEASURED rather "
        "than being filled from a probe that is not the one the board used")
    first.pop("_measured", None)
    return first


def _run_once(bv: str, rec: dict, out: Path, value_pins: str, tail_pins: str,
              floor_rule: str | None, floor_margin: float | None,
              negtail: str | None = None, identity: str | None = None) -> dict:
    import copy as _copy
    rec = _copy.deepcopy(rec)
    if negtail or identity:
        rec["cmd"] = _probe_args(rec["cmd"], negtail, identity)
    name = rec["name"]
    grid_sha = None
    gj = out / "gaddr" / f"{name}.json"
    truth, ref = registry_tables("")  # placeholder; resolved below by grid path
    # resolve the tables from the grid actually in the command
    gpath = rec["cmd"][rec["cmd"].index("--dial-grid") + 1]
    gsha = hashlib.sha256(Path(gpath).read_bytes()).hexdigest()
    grid_sha = gsha
    truth, ref = registry_tables(gsha)
    cmd = [bv] + rec["cmd"] + ["--gaddr-tail-pins", tail_pins,
                               "--gaddr-value-pins", value_pins,
                               "--gaddr-json", str(gj)]
    if truth:
        cmd += ["--gaddr-grid-truth", truth]
    if ref:
        cmd += ["--reference-truth", ref]
    if floor_rule:
        cmd += ["--floor-rule", floor_rule]
    if floor_margin is not None:
        cmd += ["--floor-margin", str(floor_margin)]
    with open(out / "logs" / f"{name}.log", "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    rec["grid_sha256"] = grid_sha
    rec["argv"] = cmd
    if rc != 0 or not gj.is_file():
        tail = (out / "logs" / f"{name}.log").read_text(errors="replace").strip().splitlines()
        why = next((l for l in reversed(tail) if "REFUS" in l or "error" in l.lower()), "")
        rec["reason"] = f"bake_verdict rc={rc}: {why[:300]}"
        rec.pop("cmd", None)
        return rec
    g = json.loads(gj.read_text())
    rec["gaddr_json"] = str(gj)
    rec["headline"] = g.get("headline")
    rec["contract"] = g.get("contract")
    rec["regression"] = g.get("regression")
    rec["shippable"] = bool(g.get("shippable"))
    rec["cfail"] = [c["id"] for c in g["checks"]
                    if c.get("tier") == "contract" and c.get("state") == "fail"]
    rec["cnm"] = [c["id"] for c in g["checks"]
                  if c.get("tier") == "contract" and c.get("state") == "not_measured"]
    rec["floors"] = [[c.get("codec"), c.get("represented_frac"),
                      c.get("represented_frac_reference"), c.get("state")]
                     for c in ((g.get("measured") or {}).get("codec_floor") or [])]
    a7 = next((c for c in g["checks"] if c.get("id") == "A7r"), None)
    rec["a7r"] = a7.get("state") if a7 else None
    rec["_measured"] = {"negtail": (g.get("measured") or {}).get("negtail"),
                        "identity": (g.get("measured") or {}).get("identity"),
                        # MEASURED on Ffree@dfreelane: this build's --gaddr-json
                        # emits C5/C6 as measured CHECKS but serialises no
                        # `measured.identity` dict, while the board block (written
                        # by another lane's build) carries one. So "the identity
                        # probe was accepted" is read from the CHECK STATES, which
                        # every build emits, and the dict is only compared when
                        # this build actually produced one.
                        "identity_rows_measured": all(
                            next((c.get("state") for c in g["checks"]
                                  if c.get("id") == cid), "not_measured") != "not_measured"
                            for cid in ("C5", "C6"))}
    rec.pop("cmd", None)
    return rec


def grade(a) -> int:
    board, out = Path(a.board), Path(a.out)
    (out / "gaddr").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    cells = sorted(board.glob("*.fulleval.json"))
    if not cells:
        print(f"no board cells under {board}", file=sys.stderr)
        return 2
    asrun = Path(a.asrun) if a.asrun else None
    plans = [plan_cell(f, a.bbp, asrun) for f in cells]
    todo = [p for p in plans if "cmd" in p]
    print(f"{len(cells)} board cells: {len(todo)} gradeable, "
          f"{len(plans) - len(todo)} NOT MEASURED before running", flush=True)
    done = [0]

    def work(p):
        r = run_cell(a.bv, p, out, a.value_pins, a.tail_pins, a.floor_rule, a.floor_margin)
        done[0] += 1
        if done[0] % 25 == 0:
            print(f"  [{done[0]}/{len(todo)}]", flush=True)
        return r

    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        graded = list(ex.map(work, todo))
    by = {r["name"]: r for r in graded}
    recs = [by.get(p["name"], p) for p in plans]
    (out / "cells.json").write_text(json.dumps(recs, indent=1))
    hdr = ["name", "status", "width", "era", "grid_sha16", "contract", "regression",
           "a7r", "shippable", "cfail", "cnm", "invocation_source", "reason"]
    with open(out / "ladder_board_summary.tsv", "w") as fh:
        fh.write("\t".join(hdr) + "\n")
        for r in recs:
            st = "NOT_MEASURED" if "reason" in r else "GRADED"
            fh.write("\t".join(str(x) for x in [
                r["name"], st, r.get("caller_width", r.get("width", "")), r.get("era", ""),
                (r.get("grid_sha256") or "")[:16], r.get("contract", ""),
                r.get("regression", ""), r.get("a7r", ""), r.get("shippable", ""),
                ",".join(r.get("cfail", [])), ",".join(r.get("cnm", [])),
                r.get("invocation_source", ""), r.get("reason", "")]) + "\n")
    ng = sum(1 for r in recs if "reason" not in r)
    print(f"\nGRADED {ng} / {len(recs)};  NOT MEASURED {len(recs) - ng}")
    print(f"summary: {out/'ladder_board_summary.tsv'}")
    return 0


def graft(a) -> int:
    out, board = Path(a.out), Path(a.board)
    recs = json.loads((out / "cells.json").read_text())
    ok = ref = 0
    for r in recs:
        if "gaddr_json" not in r:
            continue
        target = board / f"{r['name']}.fulleval.json"
        if not target.is_file():
            continue
        p = subprocess.run(
            [sys.executable, str(REPO / "scripts/promote_fulleval.py"),
             "--graft-into", str(target), "--graft-gaddr-ladder", r["gaddr_json"]],
            capture_output=True, text=True)
        if p.returncode != 0:
            print(f"  REFUSED {r['name']}: {(p.stdout + p.stderr).strip()[:200]}")
            ref += 1
        else:
            ok += 1
    print(f"grafted {ok}, refused {ref}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grade")
    g.add_argument("--bv", default=str(REPO / "target/release/bake_verdict"))
    g.add_argument("--bbp", default=str(REPO / "target/release/bake_block_profile"))
    g.add_argument("--board", default="/mnt/v/output/zensim/reports/fulleval")
    g.add_argument("--out", required=True)
    g.add_argument("--asrun", default="/mnt/v/output/zensim/gaddr-board-2026-09-04/logs")
    g.add_argument("--jobs", type=int, default=8)
    g.add_argument("--value-pins", default="report", choices=["report", "hard"])
    g.add_argument("--tail-pins", default="product", choices=["product", "retired"])
    g.add_argument("--floor-rule", default=None,
                   choices=["distinct", "resolvable", "spaced"],
                   help="omit (default) to grade under the OPERATIVE rule the registry's "
                        "active pin set names. `distinct` reproduces the pre-ruling window.")
    g.add_argument("--floor-margin", type=float, default=None)
    g.set_defaults(fn=grade)
    p = sub.add_parser("graft")
    p.add_argument("--out", required=True)
    p.add_argument("--board", default="/mnt/v/output/zensim/reports/fulleval")
    p.set_defaults(fn=graft)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
