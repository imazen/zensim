#!/usr/bin/env bash
# EXP-CROSS-CODEC-V9 entry point — thin shim (issue #41 Tier-1 item 3).
#
# The recipe lives once in run_cross_codec_seed.sh; this experiment's knobs and
# its full rationale header live in cross_codec_variants/v9.conf. This name is
# kept because benchmarks/*.md (and run_v9_full_pipeline.sh) quote it verbatim.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CC_ENTRY="$0" exec "${HERE}/run_cross_codec_seed.sh" v9 "$@"
