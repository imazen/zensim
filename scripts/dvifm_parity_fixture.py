#!/usr/bin/env python3
"""Regenerate zensim/tests/fixtures/dvifm_parity_2026-09-19.txt.

Runs the numpy DVIFM reference (../zenpapers/scripts/dvifm_block_visibility.py,
imported by path so zenpapers stays read-only) on the same closed-form input
planes the Rust parity test in zensim/src/dvifm.rs regenerates, and writes the
expected Laplacian pyramid planes plus the 30-feature vectors.

Fixture format (all numbers %.9e, row-major, one row per line):

    case <name> <h> <w>
    plane <level> <h> <w>
    <h rows of w values>
    features <band>          # band in {laplacian, local}
    <5 rows of 6 values>     # per level: F1 then F2_0..F2_4
    end

Empty levels (no complete 5x5 block) emit an all-zero feature row — the Rust
kernel's contract; numpy would produce NaN.

Usage: python3 scripts/dvifm_parity_fixture.py [path-to-reference.py]
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REF = ROOT.parent / "zenpapers" / "scripts" / "dvifm_block_visibility.py"
OUT = ROOT / "zensim" / "tests" / "fixtures" / "dvifm_parity_2026-09-19.txt"

CENTERS = np.log(np.array([1e-3, 1e-2, 5e-2, 0.2, 0.8]))  # DvifmLevelParams::SEED.f2_centers
THETA = (0.01, 0.65, 4.0)  # SEED c0, beta, sharp


def load_reference(path: Path):
    spec = importlib.util.spec_from_file_location("dvifm_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- closed-form inputs: keep the expressions identical to the Rust test ---

def case_a(h: int = 37, w: int = 31):
    r, c = np.mgrid[0:h, 0:w].astype(np.float64)
    ref = 0.5 + 0.28 * np.sin(0.21 * r + 0.13 * c) * np.cos(0.17 * r - 0.11 * c) \
        + 0.07 * np.sin(0.53 * (r + c))
    dist = ref + 0.03 * np.sin(1.3 * r - 0.7 * c) * np.cos(0.31 * r + 0.9 * c)
    return ref, dist


def case_b(h: int = 45, w: int = 37):
    r, c = np.mgrid[0:h, 0:w].astype(np.float64)
    ref = 0.45 + 0.25 * np.sin(0.35 * r) * np.cos(0.28 * c) + 0.12 * (c >= 18)
    dist = ref + 0.02 * np.cos(0.9 * r - 0.4 * c)
    return ref, dist


def case_c(h: int = 13, w: int = 9):
    r, c = np.mgrid[0:h, 0:w].astype(np.float64)
    ref = 0.5 + 0.2 * np.sin(0.9 * r) * np.cos(1.1 * c)
    dist = ref + 0.03 * np.sin(3.0 * r + 2.0 * c)
    return ref, dist


def case_d(h: int = 7, w: int = 5):
    r, c = np.mgrid[0:h, 0:w].astype(np.float64)
    ref = 0.5 + 0.3 * np.sin(0.8 * r + 0.5 * c)
    dist = ref + 0.04 * np.cos(1.7 * r - 0.6 * c)
    return ref, dist


def features(mod, ref: np.ndarray, dist: np.ndarray, band: str) -> np.ndarray:
    """[F1, F2_0..4] per level — mirrors pool_block/level_out in dvifm.rs."""
    sts = mod.block_stats_pyramid(ref, dist, band=band)  # offset=0, s_c=1, n=5
    rows = np.zeros((len(sts), 6), dtype=np.float64)
    for l, st in enumerate(sts):
        if st["hardmax"].size == 0:
            continue  # Rust contract: exact zeros for a block-less level
        rows[l, 0] = mod.f1_parametric(st, g=1.0, theta=THETA, p=1.0,
                                       edge=True, error="hardmax")
        rows[l, 1:] = mod.f2_binned(st, CENTERS, p=1.0, eps=1e-6, error="hardmax")
    return rows


def emit_plane(f, plane: np.ndarray, level: int) -> None:
    h, w = plane.shape
    f.write(f"plane {level} {h} {w}\n")
    for row in plane:
        f.write(" ".join(f"{v:.9e}" for v in row) + "\n")


def emit_features(f, rows: np.ndarray, band: str) -> None:
    f.write(f"features {band}\n")
    for row in rows:
        f.write(" ".join(f"{v:.9e}" for v in row) + "\n")


def main() -> None:
    ref_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_REF
    mod = load_reference(ref_path)
    with OUT.open("w") as f:
        f.write("# DVIFM parity fixture. Generator: scripts/dvifm_parity_fixture.py\n")
        f.write("# Reference: ../zenpapers/scripts/dvifm_block_visibility.py\n")
        f.write("# Inputs are closed-form; dvifm.rs tests regenerate them.\n")
        for name, gen, bands in (
            ("a", case_a, ("laplacian", "local")),
            ("b", case_b, ("laplacian",)),
            ("c", case_c, ("laplacian",)),
            ("d", case_d, ("laplacian",)),
        ):
            ref, dist = gen()
            f.write(f"case {name} {ref.shape[0]} {ref.shape[1]}\n")
            if name == "a":
                for level, plane in enumerate(mod.laplacian_pyramid(ref)):
                    emit_plane(f, plane, level)
            for band in bands:
                emit_features(f, features(mod, ref, dist, band), band)
            f.write("end\n")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
