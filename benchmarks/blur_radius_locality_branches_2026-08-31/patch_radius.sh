#!/bin/bash
# Apply a scale-0 blur radius R to a zensim worktree, coherently across every
# site that hard-codes 5. REVERSIBLE: run with R=5 to restore.
#   patch_radius.sh <worktree> <R>
set -eu
WT="$1"; R="$2"
S="$WT/zensim/src"
sed -i -E "s/^pub\(crate\) const BLUR_RADIUS: usize = [0-9]+;/pub(crate) const BLUR_RADIUS: usize = $R;/" "$S/feature_v2.rs"
sed -i -E "s/^const V1_BAND_OVERLAP: usize = [0-9]+;/const V1_BAND_OVERLAP: usize = $R;/"              "$S/feature_v2.rs"
sed -i -E "s/^( *)blur_radius: [0-9]+,/\1blur_radius: $R,/"                                            "$S/metric.rs" "$S/profile.rs"
sed -i -E "s/config\.blur_radius == [0-9]+/config.blur_radius == $R/"                                  "$S/fold_engine.rs"
echo "== radius=$R applied; sites now:"
grep -n "blur_radius: [0-9]\+,\|blur_radius == [0-9]\+\|const BLUR_RADIUS\|const V1_BAND_OVERLAP" "$S"/*.rs
