"""Run final invented-ingress and build-mode checks, with zero optimizer steps."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--scripts", type=Path, default=Path(__file__).parent)
    a = p.parse_args()
    r = a.root
    here = a.scripts
    subprocess.run(
        [
            sys.executable,
            str(here / "e36_admission_fixtures.py"),
            "--binary",
            str(r / "bin/e36-fit-final/zensim_mlp_train"),
            "--layout",
            str(r / "FEATURE_LAYOUT_BINDING.json"),
            "--out",
            str(r / "ingress-fixtures-final"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(here / "e36_nontraining_modes.py"),
            "--root",
            str(r),
            "--private",
            str(r / "bin/e36-fit-final/zensim_mlp_train"),
            "--normal",
            str(r / "bin/e36-normal-final"),
            "--raw",
            str(r / "bin/e36-raw-score-final"),
            "--fixtures",
            str(r / "ingress-fixtures-final"),
            "--out",
            str(r / "mode-admission-final"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
