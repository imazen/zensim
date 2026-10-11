"""Which interval a stored ``rank.<corpus>.srocc_ci`` is, so readers never mix the two.

``bake_verdict --full-json`` writes ``srocc_ci``, the 95% bootstrap interval of a corpus'
pooled SROCC, in two kinds:

* ``pair``: a pair (row) bootstrap. Every verdict written before 2026-10-11 carries this
  kind and has no ``srocc_ci_method`` block. Since then, a corpus without reference ids
  is resampled by rows too, recorded as ``srocc_ci_method.design == "rows"``.
* ``ref``: a reference-clustered bootstrap (``zenstats::resample::spearman_interval`` with
  ``Design::OneWay(reference)``), recorded as ``design == "one-way: reference"``.

They are not the same quantity: whole references are resampled in one and individual
pairs in the other. Measured for profile C, the reference interval is 3.95x as wide on
CID22, 1.96x on KADID SELECT and 1.72x on TID SELECT
(``benchmarks/stats_migration_2026-10-11.md``). A reader labels every interval with its
kind and never compares, sorts or pools widths across kinds.

This module only classifies stored records; it computes no statistic.
"""

from __future__ import annotations

REF = "ref"
PAIR = "pair"
KINDS = (REF, PAIR)
LABEL = {
    REF: "reference-clustered bootstrap",
    PAIR: "pair (row) bootstrap",
}
_DESIGNS = {"one-way: reference": REF, "rows": PAIR}


def kind(block) -> str | None:
    """The kind of ``block["srocc_ci"]``: ``"ref"``, ``"pair"``, or None when the block
    carries no usable interval. An unknown recorded design raises ``ValueError`` rather
    than being guessed."""
    if not isinstance(block, dict) or interval_bounds(block) is None:
        return None
    method = block.get("srocc_ci_method")
    if method is None:
        return PAIR
    design = method.get("design") if isinstance(method, dict) else None
    if design not in _DESIGNS:
        raise ValueError(f"unknown srocc_ci_method design {design!r}")
    return _DESIGNS[design]


def interval_bounds(block):
    """``(lo, hi)`` of a stored ``srocc_ci``, or None when absent or not two finite numbers."""
    ci = block.get("srocc_ci") if isinstance(block, dict) else None
    if not isinstance(ci, (list, tuple)) or len(ci) != 2:
        return None
    lo, hi = ci
    if not all(isinstance(v, (int, float)) and v == v and abs(v) != float("inf") for v in (lo, hi)):
        return None
    return float(lo), float(hi)


def interval(block):
    """``(lo, hi, kind)`` of a stored ``srocc_ci``, or None."""
    b = interval_bounds(block)
    return None if b is None else (b[0], b[1], kind(block))


def common_kind(kinds) -> str | None:
    """The single kind shared by ``kinds`` (Nones ignored), or None when they mix or
    there are none. A caller that would pool or compare widths uses this as its gate."""
    found = {k for k in kinds if k is not None}
    return found.pop() if len(found) == 1 else None
