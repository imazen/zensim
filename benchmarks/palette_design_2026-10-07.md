# PALETTE research design, registered before code (2026-10-07)

Caller: research::extract and extract_features_372col, `--palette-only`.
New explicit family `palette`, IDs f1825..f1866 (N=2..8, six signals
per N, N-major). New ComputeToken::Palette appended to preserve all token
bits; no new public function or type. Serving refuses these IDs. This is
an authorized research-only addition, not an adoption or bake.

Owner search: zensim feature_defs/research have no dominant-colour extractor.
zenanalyze/src/palette.rs counts occupied RGB bins and grayscale pixels; it
returns neither colours nor populations and is crate-private. Extending that
count-only contract would not supply perceptual centroids without changing
its semantics. Put the pair-specific kernel in zensim's research owner;
reuse ImageSource row/stride access and existing linear-sRGB conversion.
No sibling implementation or shipped dependency is modified.

Use OKLab, from [Ottosson's derivation](https://bottosson.github.io/posts/oklab/):
L is lightness and a/b supply polar chroma and hue. XYB's opponent channels
are optimized for compression, and do not give this L/chroma/hue interpretation.
Legacy RGB8 sRGB/BT.709 D65 only; other source contracts refuse explicitly.
Sample at the centres of a grid of at most 32 by 32 cells, without spatial
averaging (averaging invents colours). Sort samples lexicographically,
collapse exact duplicates with integer population weights. Deterministic
weighted median cut: split the box with greatest population times maximum
coordinate range, on its widest axis at the weighted median, never splitting
an equal colour. At most eight boxes; sort by population then centre. These
are dominant clusters representing all sampled colours, not merely the N
most frequent exact code values. Fewer than N unique colours repeat the
largest centre at zero weight. Bound: 1024 samples, seven splits per image;
all seven N share the sampling and incremental partition. Small islands
may be missed by this sampling; there is no spatial misalignment feature.

Optimal one-to-one weighted assignment uses subset dynamic programming
(minimize sum of mean population times Euclidean OKLab centre distance,
canonical tie-breaking). N<=8 bounds it at 256 masks. Six signals per N:

1. Weighted mean centre shift: assignment objective; nonnegative.
2. Signed mean lightness change (distorted minus reference).
3. Signed mean chroma change: desaturation negative, oversaturation positive.
4. Signed hue rotation in turns, weighted by shared chroma (min of both
   chromas), zero for achromatic centres; wrapped angle avoids hue seam.
5. Palette population redistribution EMD: after assignment, transport the
   distorted weights onto reference centres with Euclidean ground distance.
   This isolates population movement from centre movement. Exact residual
   min-cost flow, not a greedy approximation; reverse edges allow rerouting.
6. Largest matched centre shift with nonzero mean population.

[Rubner, Tomasi and Guibas](https://xenon.stanford.edu/~rubner/papers/rubnerIjcv00.pdf)
use weighted signatures and transport to compare colour distributions; that
supports a palette signal but does not establish IQA accuracy. zenpapers'
manifest contains palette-transfer/recolouring papers, no verified palette
IQA derivation. We do not copy that metadata into a claim of quality.

Kernel uses scalar f64 with libm roots/atan2, no tier-dependent reductions.
Identity and palette permutation yield positive exact zeros. Existing plans
and arithmetic remain untouched: research adds a separately planned block,
then gathers into the existing layout. Rev5 scope for the existing walk is
unchanged; palette has its own `palette_v1` semantics at every revision.
Prove old values with tier/corpus gates and fresh base-versus-candidate vectors.
No label payload read, fitted model, source constant or serving change.

Extraction: thin keyed sidecars for authorized v2c5 TRAIN/design roles only
(KADID TRAIN+SELECT, TID2013, KonFiG TRAIN+VAL, CID22-A25, KonJND BPG and
oracle teacher legs), plus NITS/LIVE/MCIQA features only. AIC-3 and every
protected/terminal set are refused. Pin producer, binary, keys and pixels;
no reinterpretation of existing instrument auxiliary slots as palette IDs.
Registration E32 remains a draft outside git; no fit is authorized here.
