#!/usr/bin/env python3
"""Independent PU21 golden-value generator for the cross-crate parity drift-guard.

Computes the PU21 encoded value V from the *published gfxdisp/pu21* coefficients
and formula in float64 — deliberately NOT calling our Rust implementation, so the
output is an independent reference. The emitted table is pasted into both
`zensim/src/pu21.rs::tests::reference_parity_gfxdisp_goldens` and
`zenmetrics` `crates/zenmetrics-api/src/hdr.rs` tests. Both copies asserting the
same numbers means neither can drift from the other or from gfxdisp.

  V = max( p7 * ( ((p1 + p2*Y^p4)/(1 + p3*Y^p4))^p5 - p6 ), 0 ),  Y in [0.005, 10000].

Coefficients: gfxdisp/pu21 `pu21_encoder.m` (BSD-3-Clause, (c) Rafal Mantiuk),
updated 2020-02-06. Run: `python3 scripts/pu21_golden.py`.
"""

COEF = {
    "Banding": [1.070275272, 0.4088273932, 0.153224308, 0.2520326168,
                1.063512885, 1.14115047, 521.4527484],
    "BandingGlare": [0.353487901, 0.3734658629, 8.277049286e-05, 0.9062562627,
                     0.09150303166, 0.9099517204, 596.3148142],
    "Peaks": [1.043882782, 0.6459495343, 0.3194584211, 0.374025247,
              1.114783422, 1.095360363, 384.9217577],
    "PeaksGlare": [816.885024, 1479.463946, 0.001253215609, 0.9329636822,
                   0.06746643971, 1.573435413, 419.6006374],
}

YS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]


def encode(y, p):
    y = min(max(y, 0.005), 10000.0)
    yp = y ** p[3]
    inner = (p[0] + p[1] * yp) / (1.0 + p[2] * yp)
    return max(p[6] * (inner ** p[4] - p[5]), 0.0)


if __name__ == "__main__":
    for name, p in COEF.items():
        vals = ", ".join(f"{encode(y, p):.4f}" for y in YS)
        print(f"        Pu21Variant::{name} => [{vals}],")
    print("    // Y points (cd/m²):", YS)
