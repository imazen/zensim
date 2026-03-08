#!/usr/bin/env python3
"""Systematic optimizer for zensim encoder loop parameters.

Uses Nelder-Mead simplex optimization to find optimal parameter values
by driving the encoder and measuring quality (SSIM2) vs size tradeoffs.

Unlike grid sweep, this converges efficiently in ~100-200 evaluations
for the 5-7 continuous parameters.

Usage:
    # Quick optimization (3 images, 2 distances, ~30 min)
    python3 tools/zensim-optimize.py --quick

    # Full optimization (8 images, 3 distances, ~2-3 hours)
    python3 tools/zensim-optimize.py

    # Resume from previous run
    python3 tools/zensim-optimize.py --resume results/optimize_20260308.json

    # Custom image set
    python3 tools/zensim-optimize.py --images img1.png img2.png --dists 1.0 2.0

Environment:
    CJXL_RS     - path to cjxl-rs binary
    DJXL        - path to djxl binary
    SS2         - path to ssimulacra2 binary
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import minimize


# ── Parameter space ──────────────────────────────────────────────────────

@dataclass
class ParamSpace:
    """Defines the optimization space with bounds and transforms."""

    # (name, env_var, low, high, default, log_scale)
    PARAMS = [
        ("masking",    "ZENSIM_MASKING",    1.0,  20.0,  8.0,  False),
        ("norm",       "ZENSIM_NORM",       1.0,  8.0,   2.0,  False),
        ("spatial_w",  "ZENSIM_SPATIAL_W",  0.1,  1.0,   0.6,  False),
        ("ratio_max",  "ZENSIM_RATIO_MAX",  1.5,  10.0,  3.0,  True),
        ("alpha",      "ZENSIM_ALPHA",      0.05, 0.50,  0.25, False),
        ("factor_max", "ZENSIM_FACTOR_MAX", 1.02, 1.50,  1.15, False),
    ]

    # Fixed params (not optimized)
    FIXED = {
        "ZENSIM_SQRT": "0",
        "ZENSIM_HF": "1",
        "ZENSIM_EDGE_MSE": "1",
    }

    @classmethod
    def names(cls):
        return [p[0] for p in cls.PARAMS]

    @classmethod
    def defaults(cls):
        return np.array([p[4] for p in cls.PARAMS])

    @classmethod
    def bounds(cls):
        return [(p[2], p[3]) for p in cls.PARAMS]

    @classmethod
    def to_normalized(cls, values):
        """Map parameter values to [0, 1] for optimizer."""
        norm = np.zeros(len(cls.PARAMS))
        for i, (_, _, lo, hi, _, log_scale) in enumerate(cls.PARAMS):
            if log_scale:
                norm[i] = (np.log(values[i]) - np.log(lo)) / (np.log(hi) - np.log(lo))
            else:
                norm[i] = (values[i] - lo) / (hi - lo)
        return norm

    @classmethod
    def from_normalized(cls, norm):
        """Map [0, 1] back to parameter values."""
        values = np.zeros(len(cls.PARAMS))
        for i, (_, _, lo, hi, _, log_scale) in enumerate(cls.PARAMS):
            n = np.clip(norm[i], 0, 1)
            if log_scale:
                values[i] = np.exp(np.log(lo) + n * (np.log(hi) - np.log(lo)))
            else:
                values[i] = lo + n * (hi - lo)
        return values

    @classmethod
    def to_env(cls, values):
        """Convert parameter values to env var dict."""
        env = dict(cls.FIXED)
        for i, (_, env_var, _, _, _, _) in enumerate(cls.PARAMS):
            env[env_var] = f"{values[i]:.4f}"
        return env

    @classmethod
    def format(cls, values):
        """Human-readable parameter string."""
        parts = []
        for i, (name, _, _, _, _, _) in enumerate(cls.PARAMS):
            parts.append(f"{name}={values[i]:.3f}")
        return " ".join(parts)


# ── Encoder evaluation ──────────────────────────────────────────────────

@dataclass
class EncodeResult:
    image: str
    distance: float
    mode: str
    size: int
    ssim2: float
    elapsed: float


@dataclass
class EvalResult:
    params: list
    delta_ss2: float  # avg SSIM2 gain vs e7 baseline
    delta_sz_pct: float  # avg size change % vs e7 baseline
    score: float  # composite objective
    n_encodes: int
    elapsed: float
    details: list = field(default_factory=list)


class Evaluator:
    """Drives encoder + decoder + ssimulacra2 to evaluate parameter sets."""

    def __init__(self, images, distances, cjxl, djxl, ss2, outdir, iters=4,
                 size_threshold=0.0):
        self.images = images
        self.distances = distances
        self.cjxl = cjxl
        self.djxl = djxl
        self.ss2 = ss2
        self.outdir = Path(outdir)
        self.iters = iters
        self.size_threshold = size_threshold
        self.outdir.mkdir(parents=True, exist_ok=True)

        # Prepare stripped references
        self.refs = {}
        for img in images:
            name = Path(img).stem[:12]
            ref = self.outdir / f"{name}_ref.png"
            if not ref.exists():
                subprocess.run(
                    ["convert", img, "-strip", str(ref)],
                    capture_output=True
                )
                if not ref.exists():
                    import shutil
                    shutil.copy2(img, ref)
            self.refs[img] = ref

        # Cache baselines
        self._baselines = {}
        self.eval_count = 0

    def get_baseline(self, img, dist):
        """Get or compute e7 baseline for an image/distance pair."""
        key = (img, dist)
        if key not in self._baselines:
            result = self._encode_one(img, dist, "e7", "baseline", env_extra=None)
            self._baselines[key] = result
        return self._baselines[key]

    def _encode_one(self, img, dist, mode, tag, env_extra=None):
        """Encode one image and measure quality."""
        name = Path(img).stem[:12]
        jxl = self.outdir / f"{tag}_{name}_d{dist}.jxl"
        dec = self.outdir / f"{tag}_{name}_d{dist}_dec.png"
        dec_s = self.outdir / f"{tag}_{name}_d{dist}_dec_s.png"

        env = dict(os.environ)
        if env_extra:
            env.update(env_extra)
        # Remove any leftover ZENSIM vars when running baseline
        if env_extra is None:
            for var in ["ZENSIM_MASKING", "ZENSIM_SQRT", "ZENSIM_HF",
                        "ZENSIM_EDGE_MSE", "ZENSIM_NORM", "ZENSIM_SPATIAL_W",
                        "ZENSIM_RATIO_MAX", "ZENSIM_ALPHA", "ZENSIM_FACTOR_MAX"]:
                env.pop(var, None)

        t0 = time.time()

        # Encode
        cmd = [self.cjxl, img, str(jxl), "-d", str(dist), "-e", "7"]
        if mode == "e7-zen":
            cmd.extend(["--zensim-iters", str(self.iters)])
        elif mode == "e8-zen":
            cmd = [self.cjxl, img, str(jxl), "-d", str(dist), "-e", "8",
                   "--zensim-iters", str(self.iters)]

        subprocess.run(cmd, capture_output=True, env=env)

        if not jxl.exists():
            return EncodeResult(name, dist, mode, 0, 0.0, time.time() - t0)

        size = jxl.stat().st_size

        # Decode
        subprocess.run([self.djxl, str(jxl), str(dec)], capture_output=True)
        if not dec.exists():
            jxl.unlink(missing_ok=True)
            return EncodeResult(name, dist, mode, size, 0.0, time.time() - t0)

        # Strip ICC for comparison
        subprocess.run(["convert", str(dec), "-strip", str(dec_s)], capture_output=True)
        if not dec_s.exists():
            dec_s = dec

        # Measure SSIM2
        ref = self.refs[img]
        result = subprocess.run(
            [self.ss2, str(ref), str(dec_s)],
            capture_output=True, text=True
        )
        ssim2 = float(result.stdout.strip().split('\n')[-1]) if result.returncode == 0 else 0.0

        # Cleanup
        jxl.unlink(missing_ok=True)
        dec.unlink(missing_ok=True)
        dec_s.unlink(missing_ok=True)

        elapsed = time.time() - t0
        return EncodeResult(name, dist, mode, size, ssim2, elapsed)

    def evaluate(self, param_values, size_penalty=0.5):
        """Evaluate a parameter set. Returns negative score (for minimization)."""
        t0 = time.time()
        self.eval_count += 1
        tag = f"opt_{self.eval_count:04d}"

        env_extra = ParamSpace.to_env(param_values)

        total_ds2 = 0.0
        total_dsz = 0.0
        n = 0
        details = []

        for img in self.images:
            for dist in self.distances:
                # Get baseline
                base = self.get_baseline(img, dist)
                if base.size == 0:
                    continue

                # Encode with current params
                result = self._encode_one(img, dist, "e7-zen", tag, env_extra)
                if result.size == 0:
                    # Encoding failed — return bad score
                    return EvalResult(
                        params=param_values.tolist(),
                        delta_ss2=0.0, delta_sz_pct=100.0,
                        score=-100.0, n_encodes=n,
                        elapsed=time.time() - t0
                    )

                ds2 = result.ssim2 - base.ssim2
                dsz = (result.size - base.size) / base.size * 100
                total_ds2 += ds2
                total_dsz += dsz
                n += 1
                details.append({
                    "image": result.image, "dist": dist,
                    "base_ss2": base.ssim2, "base_sz": base.size,
                    "zen_ss2": result.ssim2, "zen_sz": result.size,
                    "ds2": ds2, "dsz_pct": dsz
                })

        if n == 0:
            return EvalResult(
                params=param_values.tolist(),
                delta_ss2=0.0, delta_sz_pct=0.0,
                score=-100.0, n_encodes=0,
                elapsed=time.time() - t0
            )

        avg_ds2 = total_ds2 / n
        avg_dsz = total_dsz / n

        # Score: quality gain, penalize positive size inflation
        # Only penalize size inflation above threshold (e.g., 1.5% is acceptable)
        excess_size = max(0, avg_dsz - getattr(self, 'size_threshold', 0.0))
        score = avg_ds2 - size_penalty * excess_size

        return EvalResult(
            params=param_values.tolist(),
            delta_ss2=avg_ds2, delta_sz_pct=avg_dsz,
            score=score, n_encodes=n,
            elapsed=time.time() - t0,
            details=details
        )


# ── Optimization runner ─────────────────────────────────────────────────

class OptimizerState:
    """Tracks optimization progress for logging and resumption."""

    def __init__(self, logfile):
        self.logfile = Path(logfile)
        self.history = []
        self.best_score = -999.0
        self.best_params = None
        self.best_result = None
        self.start_time = time.time()

    def record(self, result: EvalResult):
        entry = {
            "eval": len(self.history) + 1,
            "params": {n: v for n, v in zip(ParamSpace.names(), result.params)},
            "delta_ss2": result.delta_ss2,
            "delta_sz_pct": result.delta_sz_pct,
            "score": result.score,
            "elapsed": result.elapsed,
            "wall_time": time.time() - self.start_time,
        }
        self.history.append(entry)

        is_best = result.score > self.best_score
        if is_best:
            self.best_score = result.score
            self.best_params = result.params
            self.best_result = entry

        # Print progress
        marker = " *** BEST ***" if is_best else ""
        print(f"  [{len(self.history):3d}] score={result.score:+.4f}  "
              f"ΔSS2={result.delta_ss2:+.3f}  ΔSz={result.delta_sz_pct:+.2f}%  "
              f"({result.elapsed:.1f}s){marker}")

        # Append to log
        with open(self.logfile, 'w') as f:
            json.dump({
                "best": self.best_result,
                "history": self.history,
                "param_names": ParamSpace.names(),
            }, f, indent=2)

    def load(self):
        if self.logfile.exists():
            with open(self.logfile) as f:
                data = json.load(f)
            self.history = data.get("history", [])
            self.best_result = data.get("best")
            if self.best_result:
                self.best_score = self.best_result["score"]
                self.best_params = list(self.best_result["params"].values())
            print(f"Resumed from {len(self.history)} evaluations, "
                  f"best score={self.best_score:.4f}")
            return True
        return False


def run_optimization(evaluator, state, method="nelder-mead",
                     size_penalty=0.5, maxiter=200):
    """Run Nelder-Mead optimization in normalized parameter space."""

    def objective(x_norm):
        """Objective function for scipy minimize (operates in [0,1] space)."""
        params = ParamSpace.from_normalized(x_norm)
        result = evaluator.evaluate(params, size_penalty=size_penalty)
        state.record(result)
        return -result.score  # minimize negative score = maximize score

    # Start from defaults in normalized space
    x0 = ParamSpace.to_normalized(ParamSpace.defaults())

    # If resuming and we have a best, start from there
    if state.best_params is not None:
        x0 = ParamSpace.to_normalized(np.array(state.best_params))

    print(f"\nStarting {method} optimization ({len(ParamSpace.PARAMS)} params)")
    print(f"Initial: {ParamSpace.format(ParamSpace.from_normalized(x0))}")
    print(f"Max iterations: {maxiter}")
    print()

    if method == "nelder-mead":
        result = minimize(
            objective, x0,
            method='Nelder-Mead',
            options={
                'maxiter': maxiter,
                'maxfev': maxiter,
                'xatol': 0.02,   # 2% of normalized range
                'fatol': 0.005,  # 0.005 score tolerance
                'adaptive': True,
                'initial_simplex': _initial_simplex(x0),
            }
        )
    elif method == "powell":
        result = minimize(
            objective, x0,
            method='Powell',
            options={
                'maxiter': maxiter,
                'maxfev': maxiter,
                'ftol': 0.005,
            },
            bounds=[(0, 1)] * len(x0),
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    best_params = ParamSpace.from_normalized(result.x)
    print(f"\nOptimization complete after {state.eval_count} evaluations")
    print(f"Best score: {-result.fun:.4f}")
    print(f"Best params: {ParamSpace.format(best_params)}")

    return best_params


def _initial_simplex(x0):
    """Create initial simplex with ~15% perturbation per dimension."""
    n = len(x0)
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    for i in range(n):
        simplex[i + 1] = x0.copy()
        # Perturb by ±15% of normalized range
        delta = 0.15
        if x0[i] + delta > 1.0:
            simplex[i + 1, i] = x0[i] - delta
        else:
            simplex[i + 1, i] = x0[i] + delta
    return simplex


def run_coordinate_descent(evaluator, state, size_penalty=0.5,
                           rounds=3, steps_per_dim=7):
    """Coordinate descent: optimize one parameter at a time.

    More robust than Nelder-Mead for noisy objectives. Each dimension
    gets a 1D search with golden section bisection.
    """
    current = ParamSpace.defaults().copy()
    if state.best_params is not None:
        current = np.array(state.best_params)

    print(f"\nStarting coordinate descent ({len(ParamSpace.PARAMS)} params, "
          f"{rounds} rounds, {steps_per_dim} steps/dim)")
    print(f"Initial: {ParamSpace.format(current)}")
    print()

    for round_idx in range(rounds):
        print(f"\n── Round {round_idx + 1}/{rounds} ──")
        improved = False

        for dim_idx, (name, _, lo, hi, _, log_scale) in enumerate(ParamSpace.PARAMS):
            print(f"\n  Optimizing {name} [{lo:.2f}, {hi:.2f}]"
                  f" (current={current[dim_idx]:.3f})")

            # Generate test points across the range
            if log_scale:
                test_vals = np.exp(np.linspace(
                    np.log(lo), np.log(hi), steps_per_dim))
            else:
                test_vals = np.linspace(lo, hi, steps_per_dim)

            # Always include current value
            test_vals = np.unique(np.append(test_vals, current[dim_idx]))
            test_vals.sort()

            best_dim_score = -999.0
            best_dim_val = current[dim_idx]

            for val in test_vals:
                trial = current.copy()
                trial[dim_idx] = val
                result = evaluator.evaluate(trial, size_penalty=size_penalty)
                state.record(result)

                if result.score > best_dim_score:
                    best_dim_score = result.score
                    best_dim_val = val

            if best_dim_val != current[dim_idx]:
                print(f"  → {name}: {current[dim_idx]:.3f} → {best_dim_val:.3f} "
                      f"(score {best_dim_score:+.4f})")
                current[dim_idx] = best_dim_val
                improved = True
            else:
                print(f"  → {name}: unchanged at {current[dim_idx]:.3f}")

        if not improved:
            print(f"\nNo improvement in round {round_idx + 1}, stopping early")
            break

    print(f"\nCoordinate descent complete: {ParamSpace.format(current)}")
    return current


# ── Main ─────────────────────────────────────────────────────────────────

def find_images(quick=False):
    """Find test images from standard locations."""
    clic_dir = Path(os.environ.get(
        "CLIC_DIR", os.path.expanduser("~/work/codec-corpus/clic2025-1024")))
    cid22_dir = Path(os.environ.get(
        "CID22_DIR", "/mnt/v/dataset/cid22/CID22/original"))

    clic = sorted(clic_dir.glob("*.png"))
    cid22 = sorted(cid22_dir.glob("*.png"))

    if quick:
        return [str(p) for p in clic[:2] + cid22[:2]]
    else:
        return [str(p) for p in clic[:5] + cid22[:5]]


def main():
    parser = argparse.ArgumentParser(description="Optimize zensim loop parameters")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 4 images, 2 distances")
    parser.add_argument("--images", nargs="+", help="Custom image paths")
    parser.add_argument("--dists", nargs="+", type=float,
                        default=[1.0, 2.0], help="Distances to test")
    parser.add_argument("--method", choices=["nelder-mead", "powell", "coord"],
                        default="coord",
                        help="Optimization method (default: coord)")
    parser.add_argument("--maxiter", type=int, default=200,
                        help="Max evaluations (default: 200)")
    parser.add_argument("--rounds", type=int, default=3,
                        help="Coordinate descent rounds (default: 3)")
    parser.add_argument("--steps", type=int, default=7,
                        help="Steps per dimension in coord descent (default: 7)")
    parser.add_argument("--size-penalty", type=float, default=0.5,
                        help="Weight for size inflation penalty (default: 0.5)")
    parser.add_argument("--size-threshold", type=float, default=0.0,
                        help="Size inflation %% allowed before penalty kicks in (default: 0)")
    parser.add_argument("--resume", help="Resume from previous log file")
    parser.add_argument("--outdir", default="/tmp/zensim_optimize",
                        help="Working directory")
    parser.add_argument("--log", help="Output log file (JSON)")
    args = parser.parse_args()

    # Find tools
    cjxl = os.environ.get("CJXL_RS", os.path.expanduser(
        "~/work/zen/jxl-encoder-rs/target/release/cjxl-rs"))
    djxl = os.environ.get("DJXL", os.path.expanduser(
        "~/work/jxl-efforts/libjxl/build/tools/djxl"))
    ss2 = os.environ.get("SS2", os.path.expanduser(
        "~/work/jxl-efforts/libjxl/build/tools/ssimulacra2"))

    for tool, path in [("cjxl-rs", cjxl), ("djxl", djxl), ("ssimulacra2", ss2)]:
        if not Path(path).exists():
            print(f"Error: {tool} not found at {path}")
            sys.exit(1)

    # Find images
    images = args.images or find_images(quick=args.quick)
    if not images:
        print("Error: no images found")
        sys.exit(1)

    n_evals = len(images) * len(args.dists)
    print(f"Images: {len(images)}, Distances: {len(args.dists)}, "
          f"Encodes per evaluation: {n_evals}")
    for img in images:
        print(f"  {Path(img).name}")

    # Set up evaluator and state
    logfile = args.log or f"/tmp/zensim_optimize_{time.strftime('%Y%m%d_%H%M%S')}.json"
    evaluator = Evaluator(images, args.dists, cjxl, djxl, ss2, args.outdir,
                          size_threshold=args.size_threshold)
    state = OptimizerState(logfile)

    if args.resume:
        state.logfile = Path(args.resume)
        state.load()

    # Compute baselines first
    print("\nComputing baselines...")
    for img in images:
        for dist in args.dists:
            base = evaluator.get_baseline(img, dist)
            name = Path(img).stem[:12]
            print(f"  e7 {name} d={dist}: size={base.size} ss2={base.ssim2:.2f}")

    # Run optimization
    if args.method == "coord":
        best = run_coordinate_descent(
            evaluator, state,
            size_penalty=args.size_penalty,
            rounds=args.rounds,
            steps_per_dim=args.steps,
        )
    else:
        best = run_optimization(
            evaluator, state,
            method=args.method,
            size_penalty=args.size_penalty,
            maxiter=args.maxiter,
        )

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Evaluations: {len(state.history)}")
    print(f"Best score:  {state.best_score:+.4f}")
    print(f"Best params: {ParamSpace.format(np.array(state.best_params))}")
    print(f"\nEnvironment variables for best config:")
    env = ParamSpace.to_env(np.array(state.best_params))
    for k, v in sorted(env.items()):
        print(f"  export {k}={v}")
    print(f"\nResults saved to: {logfile}")


if __name__ == "__main__":
    main()
