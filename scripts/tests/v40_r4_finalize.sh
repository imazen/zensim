#!/usr/bin/env bash
# Local package rehearsal only; caller holds heavy.lock and runs run-heavy.
set -euo pipefail
bundle=$1
logs=$2
metrics=$3
producer=$4
source_commit=$5
metrics_commit=$6
image=$7
mirror=$8
control=$9

just v40-r4-prepare "$bundle" "$producer" "$source_commit" "$metrics_commit" "$image" > "$logs/program-image-final.log" 2>&1
just v40-r3-executors "$bundle" "$image" 1 > "$logs/executors.log" 2>&1
just --justfile "$metrics/justfile" --working-directory "$metrics" v40-harvest-refusals "$bundle" 1 > "$logs/harvest-refusals.log" 2>&1
just v40-scoring-rehearsal "$bundle" "$image" "$logs/scoring-rehearsal" 77 > "$logs/scoring-rehearsal.log" 2>&1
just v40-r3-parity "$bundle" "$control" > "$logs/parity.log" 2>&1
just v40-cached-projection "$bundle" "$bundle/cached-projection-1" 1 > "$logs/cached-projection.log" 2>&1
just v40-freeze "$bundle" "$source_commit" "$metrics_commit" > "$logs/freeze.log" 2>&1
just v40-bundle-check "$bundle" > "$logs/bundle-check.log" 2>&1
mkdir "$logs/gate-evidence" "$logs/gate-scratch"
just v40-image-authorization "$bundle" "$image" "$logs/gate-evidence" "$logs/gate-scratch" > "$logs/image-authorization.log" 2>&1
just v40-evidence-archive "$bundle" "$logs" "$mirror" > "$logs/archive.log" 2>&1
just v40-own-cargo-cleanup "$bundle" "$mirror" > "$logs/target-cleanup.log" 2>&1
