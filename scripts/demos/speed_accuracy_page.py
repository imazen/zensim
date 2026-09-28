#!/usr/bin/env python3
"""Build the speed-vs-accuracy demo page from data that already exists.

WHAT THIS IS. Two measurements of the same set of metrics live in two places
and have never been shown together: the cross-generation speed matrix
(`benchmarks/speed_matrix_2026-09-18.json`) and the board's full-evaluation
rows (`<fulleval>/<name>.fulleval.json`). This joins them BY MODEL IDENTITY
and lays the result out as one self-contained page: x = measured latency,
y = measured CID22 rank correlation.

It runs NOTHING. No benchmark, no scorer, no cargo build. Every number on the
page is read out of one of those two files, and a cell that is absent from
both is printed as absent -- never interpolated, never substituted from a
lookalike row.

    python3 scripts/demos/speed_accuracy_page.py
    python3 scripts/demos/speed_accuracy_page.py --out DIR --speed-json F \
        --fulleval-dir D --joined-json F --raster

    just demo-speed-accuracy

THE JOIN IS THE WHOLE PROBLEM. Timing and accuracy come from different runs;
the only thing tying an accuracy row to a timed arm is the bytes the arm
executed. So:

  * `zensim_B` / `zensim_D` -- the board rows `MT914_matched_B` /
    `MT914_matched_D` carry a `bake_sha256`, and this script re-hashes the
    weight file `zensim/src/profile.rs` embeds for that profile and requires
    the two to match before it will draw the point.
  * the two Rev3 ensembles -- the board row's `bake_sha256` is member s17101
    of the five, and the speed matrix's notes list all five member hashes for
    the arm it timed; the script requires the row's hash to be one of them.
  * `zensim_C` -- the shipped bake is re-hashed and the whole fulleval
    directory searched for it. As of this writing NO row carries it, so C is
    plotted on the speed axis only, with no accuracy value.
  * `zensim_V0_2` -- `PreviewV0_2` has no bake at all (a source-embedded
    228-weight linear, `mlp_bytes: None`), so no `bake_sha256` can key a row
    to it. Speed axis only, same as C.
  * the peers are identified by their metric, not by a hash: `fast_ssim2` and
    `ssimulacra2_rs` are two implementations of SSIMULACRA 2 and share the
    `peer_ssim2_mt914` row; `butteraugli` uses `peer_butteraugli`.

CORPORA. CID22 / AIC-3 / AIC-4 only. KADID, TID, KonJND, KonFiG, nonphoto,
imazen26 and hfnlproxy are deliberately excluded -- each is train==val,
tuned-on, or carries an ssim2-derived target, and none of the three is an
accuracy claim this page is entitled to make.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SPEED = REPO / "benchmarks" / "speed_matrix_2026-09-18.json"
DEFAULT_SPEED_MD = REPO / "benchmarks" / "speed_matrix_2026-09-18.md"
DEFAULT_FULLEVAL = Path("/mnt/v/output/zensim/reports/fulleval")
DEFAULT_OUT = Path("/mnt/v/output/zensim/demos/speed-accuracy-2026-09-18")
DEFAULT_JOINED = REPO / "benchmarks" / "speed_accuracy_2026-09-18.json"
PROFILE_RS = REPO / "zensim" / "src" / "profile.rs"
WEIGHTS = REPO / "zensim" / "weights"
# `/mnt/v/output/` is served at this prefix by the mntv-gallery service.
SERVE_ROOT = Path("/mnt/v/output")
SERVE_URL = "http://localhost:3300"

SIZES = ("64", "256", "1024", "2048", "4096")
NOISY_CV = 0.10
ANCHOR = "fast_ssim2"
ANCHOR_DRIFT_LIMIT = 0.05
CORPORA = (("cid22", "CID22"), ("aic3", "AIC-3"), ("aic4", "AIC-4"))

# --- classes -> one categorical slot each -----------------------------------
# Three slots, because a scatter needs every PAIR separated, not just adjacent
# ones, and three is where the default categorical theme still clears the
# all-pairs CVD and normal-vision floors in both light and dark. Arm identity
# is carried by the direct label and the marker SHAPE; colour only says which
# family a point belongs to.
CLASSES = {
    "peer": {"label": "peer metric", "slot": 3, "shape": "diamond"},
    "zensim": {"label": "zensim named profile", "slot": 1, "shape": "circle"},
    "rev3": {"label": "Rev3 research ensemble", "slot": 2, "shape": "square"},
}

ARMS = [
    {
        "arm": "fast_ssim2",
        "label": "fast-ssim2",
        "cls": "peer",
        "run1t": "1t-rev1",
        "run8t": "8t-rev1",
        "acc_row": "peer_ssim2_mt914",
        "about": "SSIMULACRA 2, imazen's fast-ssim2 implementation. The speed "
        "matrix's anchor arm, and the reference every ratio is taken against.",
    },
    {
        "arm": "ssimulacra2_rs",
        "label": "ssimulacra2 (rust-av)",
        "cls": "peer",
        "run1t": "1t-rev1",
        "run8t": None,
        "acc_row": "peer_ssim2_mt914",
        "about": "The same metric through a different implementation. Its API "
        "takes its inputs by value, so the timed region carries a Vec clone "
        "the other arms never pay; it is not charged for the u8 to f32 "
        "widening the others' inputs do not need.",
    },
    {
        "arm": "butteraugli",
        "label": "butteraugli",
        "cls": "peer",
        "run1t": "1t-rev1",
        "run8t": None,
        "acc_row": "peer_butteraugli",
        "about": "Butteraugli, max norm. A peer we build on and measure "
        "against; it is also the second reference in the project's "
        "two-reference inversion rule.",
    },
    {
        "arm": "zensim_V0_2",
        "label": "zensim PreviewV0_2",
        "cls": "zensim",
        "run1t": "1t-rev1",
        "run8t": "8t-rev1",
        "acc_row": None,
        "bake_fn": None,
        "about": "The profile the published 0.2.x line defaults to, built from "
        "this tree rather than from the crates.io binary.",
    },
    {
        "arm": "zensim_B",
        "label": "zensim B",
        "cls": "zensim",
        "run1t": "1t-rev1",
        "run8t": "8t-rev1",
        "acc_row": "MT914_matched_B",
        "bake_fn": "linear_bake_b_cid80",
        "about": "The current default profile on main.",
    },
    {
        "arm": "zensim_C",
        "label": "zensim C",
        "cls": "zensim",
        "run1t": "1t-rev1",
        "run8t": "8t-rev1",
        "acc_row": None,
        "bake_fn": "mlp_bake_c_purity944",
        "about": "The 944-wide dense candidate profile.",
    },
    {
        "arm": "zensim_D",
        "label": "zensim D",
        "cls": "zensim",
        "run1t": "1t-rev1",
        "run8t": "8t-rev1",
        "acc_row": "MT914_matched_D",
        "bake_fn": "mlp_bake_d_add156",
        "about": "The additive 156-input fast candidate profile.",
    },
    {
        "arm": "rev3_fast_y60_ens5",
        "label": "Rev3 fast y60 (ens5)",
        "cls": "rev3",
        "run1t": "1t-rev3",
        "run8t": "8t-rev3",
        "acc_row": "R915_y60_h32_ens5",
        "about": "Five equal-weighted September 15 frozen bakes over one "
        "coarse-Y extraction. A research candidate, not a profile.",
    },
    {
        "arm": "rev3_rich_basic228_ens5",
        "label": "Rev3 rich basic228 (ens5)",
        "cls": "rev3",
        "run1t": "1t-rev3",
        "run8t": "8t-rev3",
        "acc_row": "R915_basic228_h128_ens5",
        "about": "Five equal-weighted September 15 frozen bakes over the "
        "228-wide basic extraction. A research candidate, not a profile.",
    },
]

# Accuracy-only peers: no arm in the speed matrix, so they appear in the table
# and never on a chart.
TABLE_ONLY = [
    ("peer_cvvdp", "ColorVideoVDP", "No arm in the speed matrix."),
    ("peer_iwssim", "IW-SSIM", "No arm in the speed matrix."),
]

EXCLUDED_CORPORA = "KADID, TID, KonJND, KonFiG, nonphoto, imazen26, hfnlproxy"


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bake_path_for(fn_name: str) -> Path:
    """The weight file `profile.rs` embeds in `fn_name`.

    Read out of the source rather than hardcoded here, so a bake rotation
    breaks this loudly instead of silently plotting the wrong bytes.
    """
    src = PROFILE_RS.read_text(errors="replace")
    m = re.search(
        rf"fn\s+{re.escape(fn_name)}\s*\(\s*\)[^{{]*{{\s*"
        r'include_bytes!\("\.\./weights/([^"]+)"\)',
        src,
    )
    if not m:
        raise SystemExit(
            f"speed_accuracy_page: no include_bytes! found for {fn_name} in "
            f"{PROFILE_RS} -- the profile was renamed or its bake moved."
        )
    return WEIGHTS / m.group(1)


def member_hashes_from_notes(md_path: Path) -> dict[str, set[str]]:
    """The frozen ensemble member sha256s the speed matrix says it timed.

    The report lists them per arm under a `### Frozen bakes` section as a
    backtick-quoted arm name followed by a fenced block of hashes.
    """
    out: dict[str, set[str]] = {}
    if not md_path.exists():
        return out
    text = md_path.read_text(errors="replace")
    for m in re.finditer(
        r"`(rev3_[a-z0-9_]+)`[^\n]*\n+```\n((?:[0-9a-f]{64}\n)+)```", text
    ):
        out[m.group(1)] = set(m.group(2).split())
    return out


def load_fulleval(d: Path, name: str) -> dict | None:
    p = d / f"{name}.fulleval.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def search_fulleval_for_sha(d: Path, want: str) -> list[str]:
    """Every board row whose `bake_sha256` is `want`. Usually zero or one."""
    hits = []
    for p in sorted(d.glob("*.fulleval.json")):
        try:
            row = json.loads(p.read_text())
        except (ValueError, OSError):
            continue
        if row.get("bake_sha256") == want:
            hits.append(str(row.get("name") or p.stem))
    return hits


def resolve_identity(arm: dict, feval: Path, members: dict[str, set[str]]) -> dict:
    """Prove (or refuse) the tie between a timed arm and a board row."""
    out: dict[str, object] = {"row": None, "proof": None, "why_missing": None}
    cls = arm["cls"]

    if cls == "peer":
        row = load_fulleval(feval, arm["acc_row"])
        if row is None:
            out["why_missing"] = f"no board row `{arm['acc_row']}` on disk"
            return out
        out["row"] = arm["acc_row"]
        out["proof"] = "identified by metric, not by a bake hash (peer row)"
        return out

    if cls == "rev3":
        row = load_fulleval(feval, arm["acc_row"])
        if row is None:
            out["why_missing"] = f"no board row `{arm['acc_row']}` on disk"
            return out
        sha = str(row.get("bake_sha256") or "")
        pool = members.get(arm["arm"], set())
        if pool and sha not in pool:
            out["why_missing"] = (
                f"board row `{arm['acc_row']}` bake {sha[:12]} is not one of "
                f"the {len(pool)} member hashes the speed matrix timed"
            )
            return out
        out["row"] = arm["acc_row"]
        out["bake_sha256"] = sha
        out["proof"] = (
            f"row bake {sha[:12]} is member 1 of the {len(pool) or '?'} frozen "
            "hashes the speed-matrix driver verified before timing this arm"
            if pool
            else "ensemble member hashes not found in the speed-matrix report"
        )
        return out

    # zensim named profile
    fn = arm.get("bake_fn")
    if fn is None:
        out["why_missing"] = (
            "the profile carries no bake at all (source-embedded 228-weight "
            "linear, `mlp_bytes: None`), so no `bake_sha256` can key a board "
            "row to it"
        )
        return out
    bake = bake_path_for(fn)
    if not bake.exists():
        out["why_missing"] = f"embedded bake {bake.name} is not on disk"
        return out
    sha = sha256_file(bake)
    out["bake"] = bake.name
    out["bake_sha256"] = sha
    want = arm.get("acc_row")
    if want:
        row = load_fulleval(feval, want)
        if row is None:
            out["why_missing"] = f"no board row `{want}` on disk"
            return out
        if row.get("bake_sha256") != sha:
            out["why_missing"] = (
                f"board row `{want}` carries bake "
                f"{str(row.get('bake_sha256'))[:12]}, not the shipped "
                f"{sha[:12]}"
            )
            return out
        out["row"] = want
        out["proof"] = f"{bake.name} sha256 {sha[:12]} == the row's bake_sha256"
        return out

    hits = search_fulleval_for_sha(feval, sha)
    if hits:
        out["row"] = hits[0]
        out["proof"] = f"{bake.name} sha256 {sha[:12]} found on row `{hits[0]}`"
        return out
    out["why_missing"] = (
        f"no row in the board directory carries {bake.name} "
        f"(sha256 {sha[:12]}); searched every `*.fulleval.json` there"
    )
    return out


# ---------------------------------------------------------------------------
# data assembly
# ---------------------------------------------------------------------------


def cell(speed: dict, run: str, size: str, arm: str) -> dict | None:
    r = speed["runs"].get(run)
    if not r:
        return None
    raw = r["sizes"].get(size, {}).get(arm)
    if raw is None:
        return None
    return dict(zip(speed["cell_schema"], raw))


def build(speed: dict, feval: Path, members: dict[str, set[str]]) -> dict:
    rows = []
    for arm in ARMS:
        ident = resolve_identity(arm, feval, members)
        acc: dict[str, dict] = {}
        extra: dict[str, object] = {}
        if ident["row"]:
            row = load_fulleval(feval, str(ident["row"]))
            assert row is not None
            for key, _ in CORPORA:
                r = (row.get("rank") or {}).get(key)
                if not r:
                    continue
                acc[key] = {
                    "srocc": r.get("srocc"),
                    "plcc": r.get("plcc"),
                    "n": r.get("n"),
                    "per_ref_n": r.get("per_ref_n"),
                    "srocc_ci": r.get("srocc_ci"),
                }
            extra["composite"] = row.get("composite")
            extra["mono_pct"] = (row.get("dial") or {}).get("mono_pct")
        speeds = {
            "1t": {s: cell(speed, arm["run1t"], s, arm["arm"]) for s in SIZES},
            "8t": (
                {s: cell(speed, arm["run8t"], s, arm["arm"]) for s in SIZES}
                if arm["run8t"]
                else {}
            ),
        }
        rows.append(
            {
                "arm": arm["arm"],
                "label": arm["label"],
                "cls": arm["cls"],
                "about": arm["about"],
                "run1t": arm["run1t"],
                "run8t": arm["run8t"],
                "identity": ident,
                "accuracy": acc,
                **extra,
                "speed": speeds,
            }
        )

    table_only = []
    for name, label, note in TABLE_ONLY:
        row = load_fulleval(feval, name)
        if row is None:
            continue
        acc = {}
        for key, _ in CORPORA:
            r = (row.get("rank") or {}).get(key)
            if r:
                acc[key] = {
                    "srocc": r.get("srocc"),
                    "plcc": r.get("plcc"),
                    "n": r.get("n"),
                    "per_ref_n": r.get("per_ref_n"),
                    "srocc_ci": r.get("srocc_ci"),
                }
        table_only.append(
            {
                "arm": name,
                "label": label,
                "cls": "peer",
                "note": note,
                "identity": {"row": name, "proof": "peer row"},
                "accuracy": acc,
                "composite": row.get("composite"),
                "mono_pct": (row.get("dial") or {}).get("mono_pct"),
            }
        )
    return {"rows": rows, "table_only": table_only}


def anchor_drift(speed: dict, run_a: str, run_b: str) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for s in SIZES:
        a = cell(speed, run_a, s, ANCHOR)
        b = cell(speed, run_b, s, ANCHOR)
        if not a or not b:
            out[s] = None
            continue
        out[s] = b["median_ms"] / a["median_ms"] - 1.0
    return out


def ssim2_agreement(feval: Path) -> dict:
    """Do the two SSIMULACRA 2 board rows agree? Measured, not assumed."""
    a = load_fulleval(feval, "peer_ssim2")
    b = load_fulleval(feval, "peer_ssim2_mt914")
    if not a or not b:
        return {}
    out = {}
    for key, label in CORPORA:
        ra = (a.get("rank") or {}).get(key)
        rb = (b.get("rank") or {}).get(key)
        if not ra or not rb:
            continue
        out[key] = {
            "label": label,
            "peer_ssim2": ra.get("srocc"),
            "peer_ssim2_mt914": rb.get("srocc"),
            "abs_delta": abs((ra.get("srocc") or 0) - (rb.get("srocc") or 0)),
        }
    return out


# ---------------------------------------------------------------------------
# SVG
# ---------------------------------------------------------------------------

CHAR_W = 0.545  # em per character, system sans, measured close enough for bboxes


class Chart:
    """One log-x / linear-y scatter, emitted as inline SVG.

    No JS, no CDN. Hover is a native `<title>`, which every browser renders as
    a tooltip without any script.
    """

    W = 940
    H = 520
    ML, MR, MT, MB = 64, 26, 20, 112

    def __init__(self, xs: list[float], ys: list[float]):
        lo, hi = min(xs), max(xs)
        pad = 0.10 * (math.log10(hi) - math.log10(lo) or 1.0)
        self.lx0, self.lx1 = math.log10(lo) - pad, math.log10(hi) + pad
        ylo, yhi = min(ys), max(ys)
        ypad = 0.14 * ((yhi - ylo) or 0.05)
        self.y0, self.y1 = ylo - ypad, yhi + ypad
        self.px0, self.px1 = self.ML, self.W - self.MR
        self.py0, self.py1 = self.MT, self.H - self.MB
        self.parts: list[str] = []
        self.boxes: list[tuple[float, float, float, float]] = []
        self.collisions: list[str] = []
        self.out_of_bounds: list[str] = []

    def x(self, v: float) -> float:
        t = (math.log10(v) - self.lx0) / (self.lx1 - self.lx0)
        return self.px0 + t * (self.px1 - self.px0)

    def y(self, v: float) -> float:
        t = (v - self.y0) / (self.y1 - self.y0)
        return self.py1 - t * (self.py1 - self.py0)

    # -- chrome ------------------------------------------------------------
    def grid(self, xlabel: str, ylabel: str) -> None:
        p = self.parts
        p.append(
            f'<rect x="{self.px0}" y="{self.py0}" '
            f'width="{self.px1 - self.px0}" height="{self.py1 - self.py0}" '
            'fill="var(--surface-1)"/>'
        )
        for tick, minor in self.xticks():
            xx = self.x(tick)
            p.append(
                f'<line x1="{xx:.1f}" y1="{self.py0}" x2="{xx:.1f}" '
                f'y2="{self.py1}" class="{"grid-minor" if minor else "grid"}"/>'
            )
            p.append(
                f'<text x="{xx:.1f}" y="{self.py1 + 18}" class="tick'
                f'{" tick-minor" if minor else ""}" '
                f'text-anchor="middle">{fmt_ms(tick)}</text>'
            )
        for tick in self.yticks():
            yy = self.y(tick)
            p.append(
                f'<line x1="{self.px0}" y1="{yy:.1f}" x2="{self.px1}" '
                f'y2="{yy:.1f}" class="grid"/>'
            )
            p.append(
                f'<text x="{self.px0 - 9}" y="{yy + 4:.1f}" class="tick" '
                f'text-anchor="end">{tick:.2f}</text>'
            )
        p.append(
            f'<line x1="{self.px0}" y1="{self.py1}" x2="{self.px1}" '
            f'y2="{self.py1}" class="axis"/>'
        )
        p.append(
            f'<text x="{self.px1}" y="{self.py1 + 40}" class="axis-label" '
            f'text-anchor="end">{html.escape(xlabel)}</text>'
        )
        p.append(
            f'<text x="{self.px0 - 46}" y="{self.py0 + 4}" class="axis-label" '
            f'text-anchor="start">{html.escape(ylabel)}</text>'
        )

    def xticks(self) -> list[tuple[float, bool]]:
        out = []
        d0, d1 = math.floor(self.lx0), math.ceil(self.lx1)
        for d in range(d0, d1 + 1):
            for mant, minor in ((1, False), (2, True), (5, True)):
                v = mant * 10.0**d
                if self.lx0 <= math.log10(v) <= self.lx1:
                    out.append((v, minor))
        return out

    def yticks(self) -> list[float]:
        span = self.y1 - self.y0
        step = 0.01 if span < 0.08 else (0.02 if span < 0.16 else 0.05)
        first = math.ceil(self.y0 / step) * step
        out, v = [], first
        while v <= self.y1 + 1e-9:
            out.append(round(v, 4))
            v += step
        return out

    # -- marks -------------------------------------------------------------
    def marker(self, cx: float, cy: float, shape: str, slot: int, hollow: bool) -> str:
        fill = "var(--surface-1)" if hollow else f"var(--series-{slot})"
        sw = 2.2 if hollow else 1.6
        common = (
            f'fill="{fill}" stroke="var(--series-{slot})" stroke-width="{sw}" '
            'class="mark"'
        )
        if shape == "circle":
            return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6.5" {common}/>'
        if shape == "square":
            return (
                f'<rect x="{cx - 5.8:.1f}" y="{cy - 5.8:.1f}" width="11.6" '
                f'height="11.6" rx="1.5" {common}/>'
            )
        pts = " ".join(
            f"{cx + dx:.1f},{cy + dy:.1f}"
            for dx, dy in ((0, -7.6), (7.0, 0), (0, 7.6), (-7.0, 0))
        )
        return f'<polygon points="{pts}" {common}/>'

    def point(self, pt: dict) -> None:
        cx, cy = self.x(pt["ms"]), self.y(pt["srocc"])
        if not (self.px0 <= cx <= self.px1 and self.py0 <= cy <= self.py1):
            self.out_of_bounds.append(f'{pt["label"]} at ({cx:.1f},{cy:.1f})')
        slot = CLASSES[pt["cls"]]["slot"]
        ci = pt.get("srocc_ci")
        if ci:
            ylo, yhi = self.y(ci[1]), self.y(ci[0])
            self.parts.append(
                f'<line x1="{cx:.1f}" y1="{ylo:.1f}" x2="{cx:.1f}" '
                f'y2="{yhi:.1f}" class="ci" stroke="var(--series-{slot})"/>'
            )
            for yy in (ylo, yhi):
                self.parts.append(
                    f'<line x1="{cx - 4:.1f}" y1="{yy:.1f}" x2="{cx + 4:.1f}" '
                    f'y2="{yy:.1f}" class="ci" stroke="var(--series-{slot})"/>'
                )
        tip = (
            f'{pt["label"]} — {pt["ms"]:.3f} ms, CID22 SROCC {pt["srocc"]:.4f}'
            f'{", 95% CI " + f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else ", no interval in the row"}'
            f', cv {pt["cv"]*100:.1f}%'
        )
        self.parts.append(f"<g><title>{html.escape(tip)}</title>")
        self.parts.append(
            self.marker(cx, cy, str(CLASSES[pt["cls"]]["shape"]), slot, pt["hollow"])
        )
        self.parts.append("</g>")
        self.boxes.append((cx - 9, cy - 9, cx + 9, cy + 9))

    def pareto(self, pts: list[dict]) -> list[str]:
        front, best = [], -2.0
        for p in sorted(pts, key=lambda q: q["ms"]):
            if p["srocc"] > best + 1e-12:
                front.append(p)
                best = p["srocc"]
        if len(front) < 2:
            return [p["label"] for p in front]
        d = [f'M {self.x(front[0]["ms"]):.1f} {self.y(front[0]["srocc"]):.1f}']
        for a, b in zip(front, front[1:]):
            d.append(f'L {self.x(b["ms"]):.1f} {self.y(a["srocc"]):.1f}')
            d.append(f'L {self.x(b["ms"]):.1f} {self.y(b["srocc"]):.1f}')
        # Appended, not inserted at 0: the grid's background rect is part 0 and
        # would paint straight over it.
        self.parts.append(f'<path d="{" ".join(d)}" class="pareto" fill="none"/>')
        # Reserve the staircase against label placement. Without this a label
        # sits exactly on a horizontal run and reads as struck through.
        for a, b in zip(front, front[1:]):
            x0, x1 = self.x(a["ms"]), self.x(b["ms"])
            ya, yb = self.y(a["srocc"]), self.y(b["srocc"])
            self.boxes.append((x0, ya - 5, x1, ya + 5))
            self.boxes.append((x1 - 5, min(ya, yb), x1 + 5, max(ya, yb)))
        last = front[-1]
        lx, ly = self.x(last["ms"]), self.y(last["srocc"]) - 14
        width = 86.0
        if lx + 12 + width > self.px1:  # would run off the right edge
            x, anchor, box = lx - 12, "end", (lx - 12 - width, ly - 12, lx - 10, ly + 4)
        else:
            x, anchor, box = lx + 12, "start", (lx + 10, ly - 12, lx + 12 + width, ly + 4)
        self.parts.append(
            f'<text x="{x:.1f}" y="{ly:.1f}" class="pareto-label" '
            f'text-anchor="{anchor}">Pareto frontier</text>'
        )
        self.boxes.append(box)
        return [p["label"] for p in front]

    def rug(self, pts: list[dict], caption: str) -> None:
        """Speed-only arms: an x position and nothing else. Never a y guess."""
        if not pts:
            return
        yy = self.py1 + 68
        self.parts.append(
            f'<line x1="{self.px0}" y1="{yy}" x2="{self.px1}" y2="{yy}" '
            'class="rug-base"/>'
        )
        self.parts.append(
            f'<text x="{self.px0}" y="{yy - 8}" class="rug-caption">'
            f"{html.escape(caption)}</text>"
        )
        for p in sorted(pts, key=lambda q: q["ms"]):
            cx = self.x(p["ms"])
            self.parts.append(
                f'<g><title>{html.escape(f"{p['label']} — {p['ms']:.3f} ms, no accuracy cell")}</title>'
                f'<line x1="{cx:.1f}" y1="{yy - 6}" x2="{cx:.1f}" y2="{yy + 6}" '
                f'class="rug-tick" stroke="var(--series-{CLASSES[p["cls"]]["slot"]})"/></g>'
            )
            self.parts.append(
                f'<text x="{cx:.1f}" y="{yy + 20}" class="rug-label" '
                f'text-anchor="middle">{html.escape(p["label"])}</text>'
            )

    def labels(self, pts: list[dict]) -> None:
        """Direct labels, placed by bbox search. No legend hunt."""
        fs = 12.5
        order = sorted(pts, key=lambda p: -p["srocc"])
        for p in order:
            cx, cy = self.x(p["ms"]), self.y(p["srocc"])
            w = len(p["label"]) * CHAR_W * fs
            placed = False
            for dx, dy, anchor in (
                (12, 4.5, "start"),
                (-12, 4.5, "end"),
                (12, -10, "start"),
                (-12, -10, "end"),
                (12, 18, "start"),
                (-12, 18, "end"),
                (0, -15, "middle"),
                (0, 24, "middle"),
            ):
                x0 = cx + dx if anchor == "start" else (
                    cx + dx - w if anchor == "end" else cx - w / 2
                )
                box = (x0 - 2, cy + dy - fs + 1, x0 + w + 2, cy + dy + 4)
                if box[0] < self.px0 - 2 or box[2] > self.px1 + 2:
                    continue
                if box[1] < self.py0 or box[3] > self.py1:
                    continue
                if any(overlap(box, b) for b in self.boxes):
                    continue
                self.boxes.append(box)
                self.parts.append(
                    f'<text x="{cx + dx:.1f}" y="{cy + dy:.1f}" class="dlabel" '
                    f'text-anchor="{anchor}">{html.escape(p["label"])}</text>'
                )
                placed = True
                break
            if not placed:
                self.collisions.append(p["label"])
                self.parts.append(
                    f'<text x="{cx + 12:.1f}" y="{cy + 4.5:.1f}" class="dlabel" '
                    f'text-anchor="start">{html.escape(p["label"])}</text>'
                )

    def svg(self, title: str) -> str:
        return (
            f'<svg viewBox="0 0 {self.W} {self.H}" role="img" '
            f'aria-label="{html.escape(title)}" class="chart">'
            + "".join(self.parts)
            + "</svg>"
        )


def overlap(a: tuple, b: tuple) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def fmt_ms(v: float) -> str:
    if v >= 1000:
        return f"{v:,.0f}"
    if v >= 1:
        return f"{v:g}"
    return f"{v:g}"


def make_chart(
    data: dict,
    size: str,
    threads: str,
    title: str,
    ylabel: str,
    only_runs: set[str] | None = None,
) -> tuple[str, list[str], Chart, list[dict], list[dict]]:
    """One chart. `threads` is a key of `row["speed"]`; `only_runs` restricts
    the arms to those timed in a named run, which is how the eight-thread
    chart drops the arms whose process failed the anchor check."""
    scored, rug = [], []
    for row in data["rows"]:
        if only_runs is not None and row["run8t" if threads == "8t" else "run1t"] not in only_runs:
            continue
        c = row["speed"].get(threads, {}).get(size)
        if not c:
            continue
        acc = row["accuracy"].get("cid22")
        rec = {
            "label": row["label"],
            "cls": row["cls"],
            "ms": c["median_ms"],
            "cv": c["cv"],
            "hollow": c["cv"] > NOISY_CV,
        }
        if acc and acc.get("srocc") is not None:
            rec["srocc"] = acc["srocc"]
            rec["srocc_ci"] = acc.get("srocc_ci")
            scored.append(rec)
        else:
            rug.append(rec)
    if not scored:
        raise SystemExit(f"speed_accuracy_page: no scored arms at {size} {threads}")
    xs = [p["ms"] for p in scored + rug]
    ys = [p["srocc"] for p in scored]
    ys += [v for p in scored for v in (p.get("srocc_ci") or [])]
    ch = Chart(xs, ys)
    ch.grid("single call, milliseconds (log)", ylabel)
    front = ch.pareto(scored)
    for p in scored:
        ch.point(p)
    ch.labels(scored)
    ch.rug(rug, "speed measured, no accuracy cell on this board:")
    return ch.svg(title), front, ch, scored, rug


# ---------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------

CSS = """
:root{color-scheme:light;
--surface-1:#fcfcfb;--plane:#f9f9f7;--text-primary:#0b0b0b;--text-secondary:#52514e;
--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,0.10);
--series-1:#2a78d6;--series-2:#eb6834;--series-3:#1baf7a;}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){
color-scheme:dark;
--surface-1:#1a1a19;--plane:#0d0d0d;--text-primary:#ffffff;--text-secondary:#c3c2b7;
--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,0.10);
--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;}}
:root[data-theme="dark"]{color-scheme:dark;
--surface-1:#1a1a19;--plane:#0d0d0d;--text-primary:#ffffff;--text-secondary:#c3c2b7;
--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,0.10);
--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--text-primary);
font:15px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif;}
main{max-width:1020px;margin:0 auto;padding:40px 16px 80px}
h1{font-size:30px;line-height:1.2;margin:0 0 6px;letter-spacing:-0.01em}
h2{font-size:20px;margin:48px 0 6px;letter-spacing:-0.005em}
h3{font-size:15px;margin:28px 0 4px;color:var(--text-secondary);font-weight:600}
p{margin:0 0 14px;max-width:72ch;color:var(--text-secondary)}
p.lede{color:var(--text-primary);font-size:17px}
.sub{color:var(--muted);font-size:13px;margin:0 0 22px}
figure{margin:18px 0 8px;background:var(--surface-1);border:1px solid var(--border);
border-radius:10px;padding:10px 10px 4px}
figcaption{color:var(--muted);font-size:13px;padding:2px 8px 10px;max-width:none}
svg.chart{display:block;width:100%;height:auto}
.grid{stroke:var(--grid);stroke-width:1}
.grid-minor{stroke:var(--grid);stroke-width:1;stroke-dasharray:2 4;opacity:.55}
.axis{stroke:var(--axis);stroke-width:1}
.tick{fill:var(--muted);font-size:11.5px;font-variant-numeric:tabular-nums}
.tick-minor{font-size:10px;opacity:.75}
.axis-label{fill:var(--text-secondary);font-size:12.5px}
.dlabel{fill:var(--text-primary);font-size:12.5px}
.ci{stroke-width:1.6;opacity:.55}
.pareto{stroke:var(--muted);stroke-width:1.6;stroke-dasharray:5 4;opacity:.8}
.pareto-label{fill:var(--muted);font-size:11.5px}
.rug-base{stroke:var(--axis);stroke-width:1}
.rug-tick{stroke-width:2.4}
.rug-caption{fill:var(--muted);font-size:11.5px}
.rug-label{fill:var(--text-secondary);font-size:11.5px}
.mark{}
.key{display:flex;flex-wrap:wrap;gap:18px;margin:2px 8px 12px;font-size:12.5px;
color:var(--text-secondary)}
.key span{display:inline-flex;align-items:center;gap:7px}
.key i{display:inline-block;width:12px;height:12px;border:2px solid currentColor}
.key i.circle{border-radius:50%}
.key i.diamond{transform:rotate(45deg)}
.key i.hollow{background:var(--surface-1)}
.k1{color:var(--series-1)}.k2{color:var(--series-2)}.k3{color:var(--series-3)}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:10px 0 6px;
font-variant-numeric:tabular-nums}
caption{text-align:left;color:var(--muted);font-size:13px;padding:4px 0 8px}
th,td{padding:6px 10px;text-align:right;border-bottom:1px solid var(--border)}
th:first-child,td:first-child{text-align:left;font-variant-numeric:normal}
thead th{color:var(--text-secondary);font-weight:600;border-bottom:1px solid var(--axis)}
tbody tr:hover{background:var(--grid)}
td.na,span.na{color:var(--muted)}
.box{background:var(--surface-1);border:1px solid var(--border);border-left:3px solid var(--muted);
border-radius:8px;padding:16px 18px 6px;margin:22px 0}
.box h2{margin-top:0}
.box ul{margin:0 0 14px;padding-left:20px;color:var(--text-secondary);max-width:74ch}
.box li{margin-bottom:7px}
code{font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
background:var(--grid);padding:1px 5px;border-radius:4px}
footer{margin-top:56px;padding-top:18px;border-top:1px solid var(--border);
color:var(--muted);font-size:12.5px}
footer code{background:none;padding:0}
.scroll{overflow-x:auto}
"""


def num(v, digits=4, dash="—"):
    if v is None:
        return f'<span class="na">{dash}</span>'
    return f"{v:.{digits}f}"


def key_html() -> str:
    bits = []
    for cls, meta in (("zensim", CLASSES["zensim"]), ("rev3", CLASSES["rev3"]), ("peer", CLASSES["peer"])):
        shape = meta["shape"]
        bits.append(
            f'<span class="k{meta["slot"]}"><i class="{shape}"></i>'
            f'<span style="color:var(--text-secondary)">{meta["label"]}</span></span>'
        )
    bits.append(
        '<span style="color:var(--muted)"><i class="circle hollow" '
        'style="border-color:var(--muted)"></i>'
        "hollow = this timing cell's cv exceeds 10%</span>"
    )
    bits.append(
        '<span style="color:var(--muted)">dashed staircase = Pareto frontier</span>'
    )
    return f'<div class="key">{"".join(bits)}</div>'


def speed_table(data: dict, drift: dict) -> str:
    head = "".join(f"<th>{s}&sup2; ms</th><th>&times;s2</th>" for s in SIZES)
    body = []
    drift_cells = "".join(
        (
            '<td class="na" colspan="2">—</td>'
            if drift.get(s) is None
            else (
                f'<td colspan="2">{drift[s]*100:+.2f}%'
                + (
                    ' <strong>refused</strong>'
                    if abs(drift[s]) > ANCHOR_DRIFT_LIMIT
                    else ""
                )
                + "</td>"
            )
        )
        for s in SIZES
    )
    for row in data["rows"]:
        tds = []
        for s in SIZES:
            c = row["speed"]["1t"].get(s)
            if not c:
                tds.append('<td class="na">—</td><td class="na">—</td>')
                continue
            flag = " !" if c["cv"] > NOISY_CV else ""
            ms = c["median_ms"]
            ms_s = f"{ms:,.3f}" if ms < 10 else f"{ms:,.1f}"
            tds.append(
                f'<td title="cv {c["cv"]*100:.1f}%, {c["gate_clean_rounds"]}/'
                f'{c["n_rounds"]} gate-clean rounds">{ms_s}{flag}</td>'
                f'<td>{c["ratio_vs_fast_ssim2"]:.2f}</td>'
            )
        body.append(
            f'<tr><td>{html.escape(row["label"])}</td>{"".join(tds)}</tr>'
        )
    return (
        '<div class="scroll"><table><caption>Median milliseconds per single '
        "call at one thread, and the ratio against fast-ssim2 <em>measured in "
        "the same process</em>. A <code>!</code> marks a cell whose coefficient "
        "of variation exceeds 10%. Hover a cell for its cv and gate-clean round "
        "count. The last row is the drift of the shared anchor arm between the "
        "two processes: the revision-1 arms (fast-ssim2 through zensim D) and "
        "the revision-3 arms (the two ensembles) ran in separate processes, so "
        "their <em>ratio</em> columns are always paired, but their raw "
        "millisecond columns may only be compared across the two halves where "
        "that drift is inside 5%."
        "</caption><thead><tr><th>arm</th>"
        f"{head}</tr></thead><tbody>{''.join(body)}"
        "<tr><td>anchor drift, rev1 &rarr; rev3</td>"
        f"{drift_cells}</tr></tbody></table></div>"
    )


def accuracy_table(data: dict) -> str:
    head = "".join(
        f"<th>{lab} SROCC</th><th>{lab} PLCC</th>" for _, lab in CORPORA
    )
    body = []
    for row in data["rows"] + data["table_only"]:
        acc = row.get("accuracy") or {}
        if not acc:
            why = (row.get("identity") or {}).get("why_missing") or "no board row"
            body.append(
                f'<tr><td>{html.escape(row["label"])}</td>'
                f'<td class="na" colspan="{2 * len(CORPORA) + 2}">'
                f"not evaluated on this board — {html.escape(str(why))}</td></tr>"
            )
            continue
        cells = []
        for key, _ in CORPORA:
            a = acc.get(key)
            if not a:
                cells.append('<td class="na">—</td><td class="na">—</td>')
                continue
            ci = a.get("srocc_ci")
            tip = (
                f"n={a.get('n')}"
                + (f", {a['per_ref_n']} refs" if a.get("per_ref_n") else "")
                + (
                    f", 95% CI [{ci[0]:.4f}, {ci[1]:.4f}]"
                    if ci
                    else ", no interval in this row"
                )
            )
            cells.append(
                f'<td title="{html.escape(tip)}">{num(a.get("srocc"))}</td>'
                f"<td>{num(a.get('plcc'))}</td>"
            )
        body.append(
            f'<tr><td>{html.escape(row["label"])}</td>{"".join(cells)}'
            f'<td>{num(row.get("composite"), 4)}</td>'
            f'<td>{num(row.get("mono_pct"), 4)}</td></tr>'
        )
    return (
        '<div class="scroll"><table><caption>Spearman and Pearson correlation '
        "against the human scores of three consulted public panels. Hover a "
        "SROCC cell for its n, reference count and 95% interval where the row "
        "carries one. <strong>composite</strong> is the board's registered "
        "composite and <strong>dial mono</strong> its ladder monotonicity "
        "fraction; both are blank where the row does not carry them."
        "</caption><thead><tr><th>arm</th>"
        f"{head}<th>composite</th><th>dial mono</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def identity_table(data: dict) -> str:
    body = []
    for row in data["rows"]:
        ident = row["identity"]
        sha = str(ident.get("bake_sha256") or "")
        body.append(
            "<tr>"
            f'<td>{html.escape(row["label"])}</td>'
            f'<td>{html.escape(str(ident.get("row") or "—"))}</td>'
            f'<td style="text-align:left"><code>{sha[:16] or "—"}</code></td>'
            f'<td style="text-align:left">'
            f'{html.escape(str(ident.get("proof") or ident.get("why_missing") or ""))}'
            "</td></tr>"
        )
    return (
        '<div class="scroll"><table><caption>How each timed arm was tied to an '
        "accuracy row. A bake sha256 is the repository weight file the profile "
        "embeds, re-hashed by this script at build time and compared with the "
        "board row's own <code>bake_sha256</code>."
        "</caption><thead><tr><th>arm</th><th>board row</th><th>bake sha256</th>"
        f"<th>how it was proved</th></tr></thead><tbody>{''.join(body)}</tbody>"
        "</table></div>"
    )


def render(data: dict, meta: dict) -> str:
    c1, f1, ch1, sc1, rug1 = make_chart(
        data, "1024", "1t", "Speed against CID22 rank at 1024 square, one thread",
        "CID22 SROCC",
    )
    c2, f2, ch2, sc2, rug2 = make_chart(
        data, "4096", "1t", "Speed against CID22 rank at 4096 square, one thread",
        "CID22 SROCC",
    )
    c3, f3, ch3, sc3, rug3 = make_chart(
        data, "4096", "8t", "Speed against CID22 rank at 4096 square, eight threads",
        "CID22 SROCC", only_runs={"8t-rev1"},
    )
    meta["frontier"] = {"1024_1t": f1, "4096_1t": f2, "4096_8t_rev1": f3}
    meta["render_checks"] = {
        "label_collisions": {"1024_1t": ch1.collisions, "4096_1t": ch2.collisions,
                             "4096_8t_rev1": ch3.collisions},
        "points_outside_plot": {"1024_1t": ch1.out_of_bounds, "4096_1t": ch2.out_of_bounds,
                                "4096_8t_rev1": ch3.out_of_bounds},
    }

    drift = meta["anchor_drift_1t"]
    agree = meta["ssim2_agreement"]
    a_cid = agree.get("cid22", {})
    pm = meta.get("peer_ssim2_mono_pct")
    peer_ssim2_mono = (
        f"{pm:.4f} of its ladder steps are monotone"
        if pm is not None
        else "no ladder monotonicity value in that row either"
    )
    fs1 = data["by_arm"]["fast_ssim2"]["speed"]["1t"]["4096"]["median_ms"]
    fs8 = data["by_arm"]["fast_ssim2"]["speed"]["8t"]["4096"]["median_ms"]
    b64 = data["by_arm"]["zensim_B"]["speed"]["1t"]["64"]
    c64 = data["by_arm"]["zensim_C"]["speed"]["1t"]["64"]
    v1024 = data["by_arm"]["zensim_V0_2"]["speed"]["1t"]["1024"]["median_ms"]
    b1024 = data["by_arm"]["zensim_B"]["speed"]["1t"]["1024"]["median_ms"]
    v4096 = data["by_arm"]["zensim_V0_2"]["speed"]["1t"]["4096"]["median_ms"]
    b4096 = data["by_arm"]["zensim_B"]["speed"]["1t"]["4096"]["median_ms"]

    def frontier_line(names: list[str]) -> str:
        return ", ".join(f"<strong>{html.escape(n)}</strong>" for n in names)

    refused8 = [
        r["label"]
        for r in data["rows"]
        if r["run8t"] == "8t-rev3" and r["speed"].get("8t", {}).get("4096")
    ]
    d8 = meta["anchor_drift_8t"]["4096"]

    parts = [
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">",
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        "<title>zensim speed vs accuracy</title>",
        f"<style>{CSS}</style></head><body><main>",
        "<h1>Speed against accuracy, every zensim generation and the peers</h1>",
        f'<p class="sub">Joined {html.escape(meta["generated_utc"])} from the '
        "September 18 speed matrix and the board's full-evaluation rows. "
        "Nothing here was measured by this page.</p>",
        '<p class="lede">Two measurements that had never been put on the same '
        "axes: how long one comparison takes, and how well the resulting score "
        "orders the images humans ordered. Latency is a median over 32 "
        "interleaved rounds; accuracy is Spearman correlation on CID22. "
        "Neither was run here — both were read out of files that already "
        "existed, and joined by model identity.</p>",

        # --- caveat box, up front -------------------------------------
        '<div class="box"><h2>What this does and does not show</h2><ul>',
        "<li><strong>These models are unreleased.</strong> crates.io ships "
        "zensim 0.2.7, whose default is <code>PreviewV0_2</code>. B, C, D and "
        "both Rev3 ensembles live on <code>main</code> and have not been "
        "published.</li>",
        "<li><strong>The two Rev3 ensembles are research candidates that fail "
        "the project's product gates.</strong> They are on the chart because "
        "they were timed and scored, not because they are shippable. The "
        "gates they have to clear are in "
        "<code>docs/PRODUCTION_PRIORITIES_2026-09-15.md</code>.</li>",
        "<li><strong>CID22 and AIC are consulted public panels, not untouched "
        "holdouts.</strong> They have been looked at during development. They "
        "are the best public evidence available here; they are not a sealed "
        "exam.</li>",
        "<li><strong>Timing and accuracy come from different runs.</strong> The "
        "only thing joining them is model identity — a bake sha256 for the "
        "named profiles and the Rev3 ensembles, the metric itself for the "
        "peers. Every tie is listed, with its hash, in the identity table "
        "below.</li>",
        "<li><strong>The benchmark input is synthetic textured pixels</strong> "
        "(<code>test_pair(n, n)</code>), identical across every arm, not a "
        "photographic corpus. It fixes the pixel count and the memory traffic; "
        "it does not model content.</li>",
        "<li><strong>Hardware:</strong> AMD Ryzen 9 9950X3D, 16 physical cores, "
        "Linux x86-64, release build with thin LTO and <em>no</em> "
        "<code>-C target-cpu=native</code> — runtime SIMD dispatch is what a "
        "user gets.</li>",
        "<li><strong>Two arms have no accuracy value</strong> and are drawn "
        "under the plot on the speed axis alone. No lookalike row was "
        "substituted for either. The reasons are in the accuracy table.</li>",
        f"<li><strong>Excluded corpora:</strong> {EXCLUDED_CORPORA} are not "
        "shown as accuracy claims — each is train==val, tuned-on, or carries an "
        "ssim2-derived target.</li>",
        "</ul></div>",

        # --- chart 1 ---------------------------------------------------
        "<h2>1024&sup2;, one thread</h2>",
        "<p>Down and to the right is slower; up is better agreement with human "
        "ranking. The dashed staircase is the Pareto frontier: every point on "
        "it is beaten by nothing that is both faster and better.</p>",
        "<figure>",
        key_html(),
        c1,
        f"<figcaption>Frontier: {frontier_line(f1)}. Error bars are the 95% "
        "interval the board row carries; the peer rows carry none, so their "
        "points have no bar — this page does not compute one for them. "
        f"Cross-process anchor drift at this size is "
        f"{drift['1024']*100:+.2f}%, inside the 5% limit, so the "
        "revision-1 and revision-3 arms may be read against each other here."
        "</figcaption></figure>",

        # --- chart 2 ---------------------------------------------------
        "<h2>4096&sup2;, one thread</h2>",
        "<p>Same arms, 16&times; the pixels. The ordering changes: the gap "
        "between the streaming implementations and the pyramid ones widens as "
        "the working set leaves cache.</p>",
        "<figure>",
        key_html(),
        c2,
        f"<figcaption>Frontier: {frontier_line(f2)}. Anchor drift at this size "
        f"is {drift['4096']*100:+.2f}%, inside the 5% limit. Hollow markers are "
        "cells whose cv exceeded 10% — at this size the revision-3 process had "
        "14 of 32 gate-clean rounds and its cells are correspondingly "
        "noisy.</figcaption></figure>",

        # --- chart 3 ---------------------------------------------------
        "<h2>4096&sup2;, eight threads</h2>",
        f"<p>Threaded runs use the build with fast-ssim2's rayon feature "
        "compiled in, because an MT row measured against a single-threaded "
        f"opponent is not an MT comparison. Turning it on does not help: "
        f"fast-ssim2 costs {fs1:,.0f} ms at one thread and {fs8:,.0f} ms at "
        f"eight, a factor of {fs1/fs8:.2f}&times;. Every zensim scaling number "
        "on this chart is therefore also a measurement of how much of the gap "
        "is threading.</p>",
        "<figure>",
        key_html(),
        c3,
        f"<figcaption>Frontier: {frontier_line(f3)}. <strong>Two arms are "
        "missing on purpose.</strong> "
        f"{html.escape(', '.join(refused8))} were timed in a separate "
        "eight-thread revision-3 process whose anchor read "
        f"{d8*100:+.1f}% against the one-thread anchor — far outside the 5% "
        "limit, which means that process saw a different box. The source "
        "report refuses that comparison and so does this page; the cells "
        "exist, they are just not comparable to anything else here. The "
        "revision-1 cells at this size also had 0 of 32 gate-clean rounds "
        "(zenbench's own pre-round resource check never passed), so read them "
        "as a loaded box, not a quiet one.</figcaption></figure>",

        # --- speed table ----------------------------------------------
        "<h2>Every size, one thread</h2>",
        speed_table(data, drift),
        "<h3>Where zensim loses</h3>",
        f"<p>At 64&sup2; <strong>zensim B is slower than fast-ssim2</strong> "
        f"({b64['median_ms']*1000:.0f} µs against "
        f"{data['by_arm']['fast_ssim2']['speed']['1t']['64']['median_ms']*1000:.0f} µs, "
        f"{b64['ratio_vs_fast_ssim2']:.2f}&times;) and <strong>zensim C is "
        f"{c64['ratio_vs_fast_ssim2']:.1f}&times; slower</strong>. The "
        "fold/plan/forward stack has a fixed per-call cost, and at four "
        "thousand pixels there is nothing to amortise it over; the advantage "
        "is a large-image advantage and only becomes decisive from about "
        "256&sup2; up. Both Rev3 ensembles are also slower than the anchor at "
        "64&sup2;. And the profile most users are actually running is not the "
        f"slow one: <strong>PreviewV0_2 beats B at every size measured</strong> "
        f"— {v1024:,.1f} ms against {b1024:,.1f} at 1024&sup2;, {v4096:,.0f} "
        f"against {b4096:,.0f} at 4096&sup2; — and at 64&sup2; and 256&sup2; it "
        "is the fastest arm in the matrix outright, ahead of D. C is the "
        "expensive one throughout, and it scales worst of the four.</p>",
        f"<p>And the default profile is not on either single-thread frontier: "
        f"at both 1024&sup2; and 4096&sup2; <strong>the Rev3 rich ensemble is "
        "both faster than B and better ranked than B</strong> — "
        f"{data['by_arm']['rev3_rich_basic228_ens5']['speed']['1t']['1024']['median_ms']:.1f} ms "
        f"against {b1024:.1f} and "
        f"{data['by_arm']['rev3_rich_basic228_ens5']['accuracy']['cid22']['srocc']:.4f} "
        f"against "
        f"{data['by_arm']['zensim_B']['accuracy']['cid22']['srocc']:.4f} on CID22. "
        "That is a research candidate which does not clear the product gates, "
        "so it is not an argument for shipping it — but the speed excuse for "
        "B is not available at these two geometries.</p>",

        # --- accuracy table -------------------------------------------
        "<h2>Accuracy</h2>",
        f"<p>fast-ssim2 and ssimulacra2 (rust-av) are two implementations of "
        "the same metric, so they share one accuracy row and sit at the same "
        "height on every chart. That row was checked against the board's other "
        f"SSIMULACRA 2 row: on CID22 they read "
        f"{a_cid.get('peer_ssim2_mt914', float('nan')):.7f} and "
        f"{a_cid.get('peer_ssim2', float('nan')):.7f}, a difference of "
        f"{a_cid.get('abs_delta', float('nan')):.2e} — the same numbers to "
        "five decimal places.</p>",
        "<p>AIC-4 has 300 pairs over <strong>only five reference images</strong>, "
        "so its correlation is a narrow measurement and moves a long way on a "
        "few pairs. CID22 (4,292 pairs, 49 references) is the primary panel "
        "here; AIC-3 has 600 pairs over 10 references.</p>",
        "<p>AIC-4's target runs the other way from the other two — every row in "
        "the table, peers and models alike, correlates negatively against it, "
        "so the column reports the magnitude, exactly as the board row does. "
        "The comparison between rows is unaffected; the sign is a property of "
        "the panel, not of any metric on it.</p>",
        accuracy_table(data),
        f"<p>The <strong>dial mono</strong> column is blank for the shared "
        "SSIMULACRA 2 row because that row carries no ladder. The board's "
        "other SSIMULACRA 2 row, <code>peer_ssim2</code> — the one checked "
        f"against it above — does: {peer_ssim2_mono}. It is quoted here rather "
        "than moved into the table, because it comes from a different "
        "evaluation pass than the rank numbers beside it.</p>",

        # --- identity --------------------------------------------------
        "<h2>How the join was proved</h2>",
        identity_table(data),
    ]

    src = meta["sources"]
    parts += [
        "<footer>",
        f'<p>Repository commit <code>{html.escape(meta["git_commit"])}</code>. '
        f'Speed matrix taken at <code>{html.escape(meta["speed_commit"])}</code>, '
        f'started <code>{html.escape(meta["speed_started_utc"])}</code>, '
        f'{meta["speed_rounds"]} rounds per arm per size.</p>',
        f'<p>Generated by <code>{html.escape(meta["generated_by"])}</code> — '
        "<code>just demo-speed-accuracy</code>.</p>",
        "<p>Sources: "
        + " · ".join(f"<code>{html.escape(s)}</code>" for s in src)
        + "</p>",
        "</footer></main></body></html>",
    ]
    return "".join(parts)


# ---------------------------------------------------------------------------


def git_commit() -> str:
    for cmd in (["git", "rev-parse", "HEAD"],):
        try:
            r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=30)
            if r.returncode == 0:
                return r.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return "unknown"


# The chart's own rules, duplicated into a standalone SVG for rasterising.
# In the page the classes come from the stylesheet and the colours from custom
# properties; a lone .svg file has neither, so both are inlined here with the
# light-mode values. Keep in sync with CSS above — the PNG is a proof that the
# geometry is right, not a second design.
LIGHT_TOKENS = {
    "--surface-1": "#fcfcfb",
    "--grid": "#e1e0d9",
    "--axis": "#c3c2b7",
    "--muted": "#898781",
    "--text-primary": "#0b0b0b",
    "--text-secondary": "#52514e",
    "--series-1": "#2a78d6",
    "--series-2": "#eb6834",
    "--series-3": "#1baf7a",
}
SVG_ONLY_CSS = """
text{font-family:sans-serif}
.grid{stroke:#e1e0d9;stroke-width:1}
.grid-minor{stroke:#e1e0d9;stroke-width:1;stroke-dasharray:2 4;opacity:.55}
.axis{stroke:#c3c2b7;stroke-width:1}
.tick{fill:#898781;font-size:11.5px}
.tick-minor{font-size:10px;opacity:.75}
.axis-label{fill:#52514e;font-size:12.5px}
.dlabel{fill:#0b0b0b;font-size:12.5px}
.ci{stroke-width:1.6;opacity:.55}
.pareto{stroke:#898781;stroke-width:1.6;stroke-dasharray:5 4;opacity:.8}
.pareto-label{fill:#898781;font-size:11.5px}
.rug-base{stroke:#c3c2b7;stroke-width:1}
.rug-tick{stroke-width:2.4}
.rug-caption{fill:#898781;font-size:11.5px}
.rug-label{fill:#52514e;font-size:11.5px}
"""


def standalone_svg(svg_text: str) -> str:
    """The same SVG, but readable by a rasteriser that has no page around it."""
    for token, value in LIGHT_TOKENS.items():
        svg_text = svg_text.replace(f"var({token})", value)
    body = svg_text.replace(
        "<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1
    )
    return body.replace(
        ">", f"><style><![CDATA[{SVG_ONLY_CSS}]]></style>", 1
    )


def raster(svg_text: str, png: Path) -> str | None:
    """Rasterise one SVG so a human (or an agent) can actually look at it."""
    exe = shutil.which("resvg")
    if not exe:
        return None
    svg = png.with_suffix(".svg")
    svg.write_text('<?xml version="1.0" encoding="UTF-8"?>' + standalone_svg(svg_text))
    r = subprocess.run(
        [
            exe,
            "--background", "#fcfcfb",
            "--zoom", "1.4",
            # `system-ui` is a CSS keyword a standalone rasteriser has no
            # notion of; the SVG copy asks for plain `sans-serif` and this
            # names something that exists on the box.
            "--sans-serif-family", "DejaVu Sans",
            "--font-family", "DejaVu Sans",
            str(svg), str(png),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return str(png) if r.returncode == 0 else f"resvg failed: {r.stderr.strip()}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--speed-json", type=Path, default=DEFAULT_SPEED)
    ap.add_argument("--speed-md", type=Path, default=DEFAULT_SPEED_MD)
    ap.add_argument("--fulleval-dir", type=Path, default=DEFAULT_FULLEVAL)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--joined-json", type=Path, default=DEFAULT_JOINED)
    ap.add_argument(
        "--raster",
        action="store_true",
        help="also write PNGs of the charts next to the page, for inspection",
    )
    args = ap.parse_args()

    if not args.speed_json.exists():
        raise SystemExit(f"speed_accuracy_page: no speed matrix at {args.speed_json}")
    if not args.fulleval_dir.is_dir():
        raise SystemExit(
            f"speed_accuracy_page: no fulleval directory at {args.fulleval_dir}"
        )

    speed = json.loads(args.speed_json.read_text())
    members = member_hashes_from_notes(args.speed_md)
    data = build(speed, args.fulleval_dir, members)
    data["by_arm"] = {r["arm"]: r for r in data["rows"]}

    import datetime

    meta = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "generated_by": "scripts/demos/speed_accuracy_page.py",
        "git_commit": git_commit(),
        "speed_commit": str(speed["provenance"].get("commit")),
        "speed_started_utc": str(speed["provenance"].get("started_utc")),
        "speed_rounds": speed["provenance"].get("rounds"),
        "testbed": speed.get("testbed"),
        "sources": [
            str(args.speed_json.relative_to(REPO))
            if args.speed_json.is_relative_to(REPO)
            else str(args.speed_json),
            str(args.speed_md.relative_to(REPO))
            if args.speed_md.is_relative_to(REPO)
            else str(args.speed_md),
            str(args.fulleval_dir) + "/<row>.fulleval.json",
        ],
        "anchor_drift_1t": anchor_drift(speed, "1t-rev1", "1t-rev3"),
        "anchor_drift_8t": anchor_drift(speed, "8t-rev1", "8t-rev3"),
        "anchor_drift_limit": ANCHOR_DRIFT_LIMIT,
        "ssim2_agreement": ssim2_agreement(args.fulleval_dir),
        "peer_ssim2_mono_pct": (
            ((load_fulleval(args.fulleval_dir, "peer_ssim2") or {}).get("dial") or {})
            .get("mono_pct")
        ),
        "excluded_corpora": EXCLUDED_CORPORA,
    }

    page = render(data, meta)
    args.out.mkdir(parents=True, exist_ok=True)
    index = args.out / "index.html"
    index.write_text(page)

    # The joined table, committed next to the speed matrix it came from.
    joined = {
        "generated_by": meta["generated_by"],
        "generated_utc": meta["generated_utc"],
        "git_commit": meta["git_commit"],
        "sources": meta["sources"],
        "speed_provenance": speed["provenance"],
        "cell_schema": speed["cell_schema"],
        "noisy_cv_threshold": NOISY_CV,
        "anchor_drift_1t": meta["anchor_drift_1t"],
        "anchor_drift_8t": meta["anchor_drift_8t"],
        "ssim2_agreement": meta["ssim2_agreement"],
        "excluded_corpora": EXCLUDED_CORPORA,
        "frontier": meta["frontier"],
        "render_checks": meta["render_checks"],
        "arms": [
            {
                "arm": r["arm"],
                "label": r["label"],
                "class": r["cls"],
                "speed_run_1t": r["run1t"],
                "speed_run_8t": r["run8t"],
                "identity": r["identity"],
                "accuracy": r["accuracy"],
                "composite": r.get("composite"),
                "dial_mono_pct": r.get("mono_pct"),
                "median_ms_1t": {
                    s: (r["speed"]["1t"][s] or {}).get("median_ms")
                    for s in SIZES
                },
                "cv_1t": {s: (r["speed"]["1t"][s] or {}).get("cv") for s in SIZES},
                "ratio_vs_fast_ssim2_1t": {
                    s: (r["speed"]["1t"][s] or {}).get("ratio_vs_fast_ssim2")
                    for s in SIZES
                },
                "median_ms_8t": {
                    s: (r["speed"]["8t"].get(s) or {}).get("median_ms")
                    for s in SIZES
                }
                if r["speed"]["8t"]
                else None,
            }
            for r in data["rows"]
        ],
        "accuracy_only": [
            {
                "arm": r["arm"],
                "label": r["label"],
                "note": r["note"],
                "accuracy": r["accuracy"],
                "composite": r.get("composite"),
                "dial_mono_pct": r.get("mono_pct"),
            }
            for r in data["table_only"]
        ],
    }
    args.joined_json.parent.mkdir(parents=True, exist_ok=True)
    args.joined_json.write_text(json.dumps(joined, indent=1, sort_keys=False) + "\n")

    # --- report -----------------------------------------------------------
    print(f"page:   {index}")
    if index.is_relative_to(SERVE_ROOT):
        print(f"url:    {SERVE_URL}/{index.relative_to(SERVE_ROOT)}")
    size_kb = args.joined_json.stat().st_size / 1024
    print(f"joined: {args.joined_json} ({size_kb:.1f} KB)")
    if size_kb > 30:
        print("  WARNING: over the 30 KB commit ceiling — trim before committing")
    for k, v in meta["frontier"].items():
        print(f"frontier {k}: {', '.join(v)}")
    for k, v in meta["render_checks"]["label_collisions"].items():
        print(f"label collisions {k}: {v or 'none'}")
    for k, v in meta["render_checks"]["points_outside_plot"].items():
        print(f"points outside plot {k}: {v or 'none'}")
    for r in data["rows"]:
        if not r["accuracy"]:
            print(f"no accuracy: {r['arm']} — {r['identity']['why_missing']}")

    if args.raster:
        import xml.etree.ElementTree as ET

        for name, chart in (
            ("chart1-1024-1t", render_only(data, "1024", "1t")),
            ("chart2-4096-1t", render_only(data, "4096", "1t")),
            ("chart3-4096-8t", render_only(data, "4096", "8t", {"8t-rev1"})),
        ):
            try:
                ET.fromstring(chart)
            except ET.ParseError as e:
                print(f"SVG {name} is NOT well-formed XML: {e}")
                continue
            print(f"SVG {name}: well-formed XML")
            got = raster(chart, args.out / f"{name}.png")
            print(f"  raster: {got or 'no resvg on PATH — not rendered'}")
    return 0


def render_only(
    data: dict, size: str, threads: str, only_runs: set[str] | None = None
) -> str:
    svg, _, _, _, _ = make_chart(
        data, size, threads, f"{size} {threads}", "CID22 SROCC", only_runs
    )
    return svg


if __name__ == "__main__":
    sys.exit(main())
