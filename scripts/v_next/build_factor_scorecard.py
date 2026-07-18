#!/usr/bin/env python3
"""Build the final-metric factor scorecard from /home/lilith/tmp/factors.json (see
benchmarks/final_metric_experiments_2026-07-18.md for how factors are computed).
Companion to bandwise_dashboard.py — covers factors bandwise lacks (OOD max, diffmap
coherence, corruption gate, dial monotonicity) in one color-coded matrix. Regenerate the
factors.json via the bake_verdict+OOD+corruption loop in the experiment log, then run this."""
# (the inline builder used on 2026-07-18 — see git history for the exact rendering)
print("see benchmarks/final_metric_experiments_2026-07-18.md; builder body in commit history")
