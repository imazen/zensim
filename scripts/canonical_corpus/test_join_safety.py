#!/usr/bin/env python3
"""Self-tests for join_safety.py — the structural guards against the
2026-05-25 kadid/tid metric corruption.

Run: python3 scripts/canonical_corpus/test_join_safety.py
Exits 0 if all pass, nonzero (with traceback) on the first failure.
No pytest dependency — plain asserts so it runs in minimal CI.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parent))
from join_safety import (  # noqa: E402
    JoinSafetyError,
    assert_metric_not_constant_per_ref,
    assert_no_leaked_metric_columns,
    attach_metric_positional,
    safe_metric_join,
)


def _expect_raise(fn, *, contains: str = ""):
    try:
        fn()
    except JoinSafetyError as e:
        if contains and contains not in str(e):
            raise AssertionError(f"raised but message lacked {contains!r}: {e}")
        return
    raise AssertionError("expected JoinSafetyError, none raised")


def test_ref_only_join_raises():
    """Fix 1: target with ONLY ref_basename must raise, never collapse."""
    target = pd.DataFrame({"ref_basename": ["a", "a", "b"], "human_score": [10, 20, 30]})
    metric = pd.DataFrame({
        "ref_basename": ["a", "a", "b"],
        "codec": ["x", "y", "x"],
        "q": [50, 60, 50],
        "knob_tuple_json": ["{}", "{}", "{}"],
        "ssim2_gpu": [80.0, 70.0, 90.0],
    })
    _expect_raise(
        lambda: safe_metric_join(
            target, metric,
            ["ref_basename", "codec", "q", "knob_tuple_json"], "ssim2_gpu",
        ),
        contains="ref_basename",
    )


def test_full_key_join_succeeds():
    """A full-key join with unique metric rows works and preserves per-pair signal."""
    keys = ["ref_basename", "codec", "q", "knob_tuple_json"]
    target = pd.DataFrame({
        "ref_basename": ["a", "a", "b"],
        "codec": ["x", "y", "x"],
        "q": [50, 60, 50],
        "knob_tuple_json": ["{}", "{}", "{}"],
        "human_score": [10, 20, 30],
    })
    metric = pd.DataFrame({
        "ref_basename": ["a", "a", "b"],
        "codec": ["x", "y", "x"],
        "q": [50, 60, 50],
        "knob_tuple_json": ["{}", "{}", "{}"],
        "ssim2_gpu": [80.0, 70.0, 90.0],
    })
    out = safe_metric_join(target, metric, keys, "ssim2_gpu")
    assert list(out["ssim2_gpu"]) == [80.0, 70.0, 90.0], out["ssim2_gpu"].tolist()


def test_duplicate_metric_rows_raise():
    """A metric source with duplicate per-key rows must raise (no silent mean)."""
    keys = ["ref_basename", "codec", "q", "knob_tuple_json"]
    target = pd.DataFrame({
        "ref_basename": ["a"], "codec": ["x"], "q": [50], "knob_tuple_json": ["{}"],
    })
    metric = pd.DataFrame({
        "ref_basename": ["a", "a"], "codec": ["x", "x"], "q": [50, 50],
        "knob_tuple_json": ["{}", "{}"], "ssim2_gpu": [80.0, 81.0],
    })
    _expect_raise(lambda: safe_metric_join(target, metric, keys, "ssim2_gpu"),
                  contains="NOT unique")


def test_positional_attach_ok_and_length_guard():
    target = pd.DataFrame({"ref_basename": ["a", "b", "c"], "human_score": [1, 2, 3]})
    out = attach_metric_positional(target, [0.1, 0.2, 0.3], "ssim2_gpu")
    assert list(out["ssim2_gpu"]) == [0.1, 0.2, 0.3]
    _expect_raise(lambda: attach_metric_positional(target, [0.1, 0.2], "ssim2_gpu"),
                  contains="positional alignment")


def test_constant_per_ref_raises():
    """Fix 1 (Mode B): ssim2 constant within every ref group must raise."""
    refs = [f"r{i}" for i in range(6) for _ in range(8)]
    # one constant value per ref group (the broadcast signature)
    vals = np.repeat(np.arange(6, dtype=float) * 10, 8)
    _expect_raise(
        lambda: assert_metric_not_constant_per_ref("test", refs, vals, "ssim2_gpu"),
        contains="constant within every",
    )


def test_constant_per_ref_per_pair_sidecar_ok():
    """A per-pair sidecar (each ref unique, group size 1) must NOT raise."""
    refs = [f"r{i}" for i in range(20)]
    vals = np.random.default_rng(0).uniform(0, 100, 20)
    assert_metric_not_constant_per_ref("sidecar", refs, vals, "ssim2_gpu")  # no raise


def test_varied_per_ref_ok():
    refs = [f"r{i}" for i in range(6) for _ in range(8)]
    vals = np.random.default_rng(1).uniform(0, 100, 48)
    assert_metric_not_constant_per_ref("varied", refs, vals, "ssim2_gpu")  # no raise


def _tbl(cols: dict):
    arrays, names = [], []
    for k, v in cols.items():
        names.append(k)
        arrays.append(pa.array(v))
    return pa.Table.from_arrays(arrays, names=names)


def test_human_copy_rejected():
    """Fix 2: a metric column bit-identical to human_score must raise."""
    hs = list(np.random.default_rng(2).uniform(0, 100, 200))
    tbl = _tbl({"human_score": hs, "iwssim": list(hs)})
    _expect_raise(
        lambda: assert_no_leaked_metric_columns("kadid", tbl.schema.names, tbl),
        contains="bit-identical copy",
    )


def test_mock_column_rejected():
    """Fix 2: any *_mock* column is forbidden in a corpus."""
    hs = list(np.random.default_rng(3).uniform(0, 100, 200))
    tbl = _tbl({"human_score": hs, "iwssim_MOCK_VAL_ONLY": list(hs)})
    _expect_raise(
        lambda: assert_no_leaked_metric_columns("kadid", tbl.schema.names, tbl),
        contains="MOCK column",
    )


def test_clean_metrics_pass():
    """Real metrics correlated-but-not-identical to human_score pass."""
    rng = np.random.default_rng(4)
    hs = rng.uniform(0, 100, 200)
    tbl = _tbl({
        "human_score": list(hs),
        "iwssim": list(np.clip(hs / 100 + rng.normal(0, 0.05, 200), 0, 1)),
        "ssim2_gpu": list(hs + rng.normal(0, 5, 200)),
    })
    assert_no_leaked_metric_columns("clean", tbl.schema.names, tbl)  # no raise


def test_linear_rescale_not_flagged():
    """safesyn legitimacy: human_score = ssim2_gpu / 100 (corr=1.0, but NOT
    bit-identical) must pass — perfect correlation alone is not a leak."""
    rng = np.random.default_rng(5)
    s2 = rng.uniform(-700, 99, 200)
    hs = s2 / 100.0  # exact linear rescale → corr 1.0, identical% 0
    tbl = _tbl({"human_score": list(hs), "ssim2_gpu": list(s2)})
    assert_no_leaked_metric_columns("safesyn", tbl.schema.names, tbl)  # no raise


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\nALL {len(tests)} join_safety tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
