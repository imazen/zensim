#!/usr/bin/env python3
"""Compatibility entry point for the summer gauntlet's stored-verdict renderer.

The July NPZ/Matplotlib scoring mode was retired on 2026-09-07. Its source,
private instrument and training recipes remain at commit 45e1ec9a. Current
usage and options belong to gauntlet.py; this entry point forwards them intact.
"""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("gauntlet.py")), run_name="__main__")
