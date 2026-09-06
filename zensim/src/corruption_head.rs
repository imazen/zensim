//! **The corruption head** — a gradient-boosted tree ensemble that answers
//! "is this a structural CORRUPTION rather than an honest encode?", and the
//! one owner of the deploy composition that folds its answer into a dial
//! score.
//!
//! Design + pre-registration: `docs/PLAN_CORRHEAD_SERVING_2026-09-06.md`.
//! Record: `benchmarks/corruption_head_serving_2026-09-06.md`.
//! The modelling result this serves:
//! `benchmarks/corruption_head_theories_2026-09-06.md` — on identical
//! features, split and calibration, `HistGradientBoostingClassifier` reads
//! **98.90 % detection / 1.23 % honest FP / 2.38 % near-lossless FP** against
//! the shipped logistic head's 86.01 / 11.37 / 50.00, and needs no dial guard.
//!
//! ## Why this is not a ZNPR bake
//!
//! [`zenpredict`] is frozen at the `zenanalyze-api` contract level, its
//! `Model` is layers-and-activations, and **every** consumer that holds one
//! dispatches through `Predictor::predict` — so a tree smuggled into a
//! `metadata[]` entry behind a plausible identity layer would be silently
//! mis-scored by anything that did not know to look for it. That is the same
//! shape as the `--regime 944` defect in `CLAUDE.md`'s Known Bugs. A distinct
//! `b"ZCTH"` magic makes head-vs-dial confusion a refusal at byte 0 instead.
//!
//! What ZNPR got right is copied rather than re-invented: a magic, a `u16`
//! format version, a `u64` schema hash over the canonical shape, a section
//! table of `(offset, len)`, and a declared-feature-id list so a head obeys
//! the same dense contract as [`crate::declared_feature_ids`].
//!
//! ## Wire format, `ZCTH` v1 (little-endian throughout)
//!
//! ```text
//! Header, 120 bytes
//!    0..4    magic                b"ZCTH"
//!    4..6    format_version  u16  = 1
//!    6..8    flags           u16  bit0 has_isotonic, bit1 has_scaler
//!    8..16   schema_hash     u64  FNV-1a over the canonical shape descriptor
//!   16..20   caller_input_width u32
//!   20..24   n_declared      u32
//!   24..28   n_trees         u32
//!   28..32   n_nodes         u32
//!   32..40   baseline        f64
//!   40..48   deadband_t      f64  fires when P > t
//!   48..52   clip            f32  standardisation clip, +-clip
//!   52..56   reserved        u32  = 0
//!   56..64   sec_declared_ids  Section  u16  * n_declared
//!   64..72   sec_scaler_mean   Section  f64  * n_declared
//!   72..80   sec_scaler_scale  Section  f64  * n_declared
//!   80..88   sec_tree_offsets  Section  u32  * (n_trees + 1)
//!   88..96   sec_nodes         Section  Node * n_nodes
//!   96..104  sec_iso_x         Section  f64  * n_knots
//!  104..112  sec_iso_y         Section  f64  * n_knots
//!  112..120  sec_meta          Section  utf8 JSON provenance
//!
//! Node, 32 bytes
//!    0..8    threshold     f64
//!    8..12   left          u32   node index WITHIN this tree's range
//!   12..16   right         u32
//!   16..20   feature_pos   u32   index into sec_declared_ids
//!   20..24   node_flags    u32   bit0 is_leaf, bit1 missing_go_to_left
//!   24..32   value         f64   leaf value (0.0 for an internal node)
//! ```
//!
//! `sec_meta` is provenance only — an evaluator that ignores it still scores
//! correctly.
//!
//! ## The evaluation contract
//!
//! ```text
//! z_j   = clamp((x[declared_ids[j]] - mean[j]) / scale[j], -clip, +clip)
//! raw   = baseline + SUM_over_trees walk(tree, z)
//! p_raw = 1 / (1 + exp(-raw))
//! p     = interp_linear(clamp(p_raw, iso_x[0], iso_x[n-1]); iso_x, iso_y)
//! score = 100 * (1 - p)
//! ```
//!
//! Both the tree walk and the interpolation reproduce their Python
//! counterparts' *arithmetic*, not merely their intent — see
//! [`interp_linear`] for why the bracket choice and the evaluation order are
//! load-bearing rather than stylistic.

/// The magic every `ZCTH` file starts with.
pub const MAGIC: [u8; 4] = *b"ZCTH";
/// The only format version this build reads or writes.
pub const FORMAT_VERSION: u16 = 1;
/// Header length in bytes; the section table ends here.
const HEADER_LEN: usize = 120;
/// One node's serialized width.
const NODE_LEN: usize = 32;

const FLAG_HAS_ISOTONIC: u16 = 1 << 0;
const FLAG_HAS_SCALER: u16 = 1 << 1;
const NODE_FLAG_LEAF: u32 = 1 << 0;
const NODE_FLAG_MISSING_LEFT: u32 = 1 << 1;

/// Why a `ZCTH` file cannot be loaded, or cannot be applied.
///
/// Every variant names the actionable detail. A refusal that says only
/// "mismatch" is the failure mode `feature_plan::PlanError` exists to avoid,
/// and the same rule applies here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CorruptionHeadError {
    /// The first four bytes are not `ZCTH`. Most often a ZNPR dial bake
    /// handed to the head loader by mistake — which is exactly the confusion
    /// the separate magic exists to make impossible.
    BadMagic { got: [u8; 4] },
    /// A format version this build does not read.
    UnsupportedVersion { got: u16, supported: u16 },
    /// The stored schema hash disagrees with the one recomputed from the
    /// file's own shape — the file was edited, truncated or mis-assembled.
    SchemaHashMismatch { stored: u64, computed: u64 },
    /// A section addresses bytes outside the file.
    SectionOutOfRange {
        name: &'static str,
        offset: u32,
        len: u32,
        file_len: usize,
    },
    /// A section's length is not a whole number of its element stride, or the
    /// element count disagrees with the header.
    SectionShape {
        name: &'static str,
        len: u32,
        stride: usize,
        expected_elems: usize,
    },
    /// The file is shorter than its own header.
    Truncated { expected: usize, got: usize },
    /// The caller supplied a feature row of the wrong width. Never scored as
    /// a prefix — the head declares its ids against a specific layout and a
    /// short row would silently gather the wrong slots.
    FeatureLenMismatch { expected: usize, got: usize },
    /// A tree's node graph is not a well-formed binary tree rooted at its
    /// first node: a child index out of range, a cycle, or a non-leaf whose
    /// children are unreachable.
    MalformedTree { tree: u32, detail: &'static str },
    /// A declared feature id lies outside the caller's declared width.
    DeclaredIdOutOfRange { pos: usize, id: u16, width: usize },
    /// The head reads slots the profile's extraction plan does not populate.
    /// Attaching a head must NEVER widen the walk, so this is a refusal, not
    /// a silent upgrade.
    NotServable {
        profile: &'static str,
        detail: String,
    },
}

impl core::fmt::Display for CorruptionHeadError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BadMagic { got } => write!(
                f,
                "not a ZCTH corruption head: magic {got:?} (a ZNPR dial bake starts with `ZNPR`)"
            ),
            Self::UnsupportedVersion { got, supported } => {
                write!(f, "ZCTH format version {got}; this build reads {supported}")
            }
            Self::SchemaHashMismatch { stored, computed } => write!(
                f,
                "ZCTH schema hash mismatch: file says {stored:#018x}, its own shape hashes to \
                 {computed:#018x}"
            ),
            Self::SectionOutOfRange {
                name,
                offset,
                len,
                file_len,
            } => write!(
                f,
                "ZCTH section `{name}` at {offset}+{len} lies outside the {file_len}-byte file"
            ),
            Self::SectionShape {
                name,
                len,
                stride,
                expected_elems,
            } => write!(
                f,
                "ZCTH section `{name}`: {len} bytes is not {expected_elems} x {stride}"
            ),
            Self::Truncated { expected, got } => {
                write!(f, "ZCTH truncated: need {expected} header bytes, got {got}")
            }
            Self::FeatureLenMismatch { expected, got } => write!(
                f,
                "corruption head expects {expected} caller features, got {got}"
            ),
            Self::MalformedTree { tree, detail } => {
                write!(f, "ZCTH tree {tree}: {detail}")
            }
            Self::DeclaredIdOutOfRange { pos, id, width } => write!(
                f,
                "ZCTH declared id[{pos}] = f{id} is outside the declared caller width {width}"
            ),
            Self::NotServable { profile, detail } => write!(
                f,
                "corruption head is not servable by profile {profile}: {detail}"
            ),
        }
    }
}

impl core::error::Error for CorruptionHeadError {}

/// A `(offset, len)` pair into the file, read from the header's table.
#[derive(Clone, Copy, Debug)]
struct Section {
    offset: u32,
    len: u32,
}

impl Section {
    fn slice<'a>(
        &self,
        bytes: &'a [u8],
        name: &'static str,
    ) -> Result<&'a [u8], CorruptionHeadError> {
        let end = (self.offset as usize)
            .checked_add(self.len as usize)
            .ok_or(CorruptionHeadError::SectionOutOfRange {
                name,
                offset: self.offset,
                len: self.len,
                file_len: bytes.len(),
            })?;
        bytes
            .get(self.offset as usize..end)
            .ok_or(CorruptionHeadError::SectionOutOfRange {
                name,
                offset: self.offset,
                len: self.len,
                file_len: bytes.len(),
            })
    }
}

/// One decision node. `left`/`right` are indices **within the owning tree's
/// node range**, which is what keeps a tree relocatable and makes the
/// bounds check local.
#[derive(Clone, Copy, Debug)]
struct Node {
    threshold: f64,
    left: u32,
    right: u32,
    feature_pos: u32,
    flags: u32,
    value: f64,
}

impl Node {
    #[inline]
    fn is_leaf(&self) -> bool {
        self.flags & NODE_FLAG_LEAF != 0
    }
    #[inline]
    fn missing_go_to_left(&self) -> bool {
        self.flags & NODE_FLAG_MISSING_LEFT != 0
    }
}

/// FNV-1a 64 — the schema hash. Chosen because it is four lines in both
/// Python and Rust with no dependency on either side, so the exporter and
/// the reader cannot drift on the hash they are supposed to agree about.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// The canonical shape descriptor the schema hash is taken over.
///
/// Deliberately covers SHAPE and the read set, not the fitted numbers: the
/// hash answers "is this the head I think it is, structurally?", while a
/// changed threshold is a different model that must get a different file, not
/// a corrupted one. Both sides build this from the same field order.
fn schema_descriptor(
    caller_input_width: u32,
    n_declared: u32,
    n_trees: u32,
    n_nodes: u32,
    clip: f32,
    declared_ids: &[u16],
    n_knots: u32,
) -> Vec<u8> {
    let mut d = Vec::with_capacity(26 + declared_ids.len() * 2);
    d.extend_from_slice(&MAGIC);
    d.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    d.extend_from_slice(&caller_input_width.to_le_bytes());
    d.extend_from_slice(&n_declared.to_le_bytes());
    d.extend_from_slice(&n_trees.to_le_bytes());
    d.extend_from_slice(&n_nodes.to_le_bytes());
    d.extend_from_slice(&clip.to_le_bytes());
    for id in declared_ids {
        d.extend_from_slice(&id.to_le_bytes());
    }
    d.extend_from_slice(&n_knots.to_le_bytes());
    d
}

/// **THE deploy composition**, in the dial's SCORE units — one owner, used by
/// the runtime companion and by `bake_verdict` for BOTH head kinds.
///
/// A flagged row is forced to `min(perceptual, 0)` so it can no longer
/// out-rank its own honest anchor; an unflagged row passes through untouched.
/// This is, verbatim, the rule `bake_verdict` computed inline before this
/// function existed (`if h < thr { d.min(0.0) } else { d }`) and the rule the
/// theory lane measured in Python (`np.where(p > T, np.minimum(dial, 0), dial)`,
/// with `T` and the score threshold related by `thr = 100 * (1 - T)`), so
/// adopting it moves no number. `scripts/verify_corrhead_composition.sh` gates
/// that on a full `--full-json`.
///
/// The head is **not** a second ranker. It cannot raise a score, only floor
/// one — which is what makes attaching it safe for a dial that is already
/// calibrated.
#[inline]
#[must_use]
pub fn gate_score(perceptual: f64, head_score: f64, deadband_score: f64) -> f64 {
    if head_score < deadband_score {
        perceptual.min(0.0)
    } else {
        perceptual
    }
}

/// A loaded corruption head.
#[derive(Clone, Debug)]
pub struct CorruptionHead {
    caller_input_width: usize,
    declared_ids: Vec<u16>,
    mean: Vec<f64>,
    scale: Vec<f64>,
    clip: f64,
    baseline: f64,
    deadband_t: f64,
    /// `tree_offsets[t]..tree_offsets[t + 1]` is tree `t`'s node range.
    tree_offsets: Vec<u32>,
    nodes: Vec<Node>,
    iso_x: Vec<f64>,
    iso_y: Vec<f64>,
    schema_hash: u64,
    meta: String,
}

fn rd_u16(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}
fn rd_u32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}
fn rd_u64(b: &[u8], at: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[at..at + 8]);
    u64::from_le_bytes(a)
}
fn rd_f32(b: &[u8], at: usize) -> f32 {
    f32::from_bits(rd_u32(b, at))
}
fn rd_f64(b: &[u8], at: usize) -> f64 {
    f64::from_bits(rd_u64(b, at))
}
fn rd_section(b: &[u8], at: usize) -> Section {
    Section {
        offset: rd_u32(b, at),
        len: rd_u32(b, at + 4),
    }
}

fn read_f64_vec(
    bytes: &[u8],
    sec: Section,
    name: &'static str,
    n: usize,
) -> Result<Vec<f64>, CorruptionHeadError> {
    let raw = sec.slice(bytes, name)?;
    if raw.len() != n * 8 {
        return Err(CorruptionHeadError::SectionShape {
            name,
            len: sec.len,
            stride: 8,
            expected_elems: n,
        });
    }
    Ok((0..n).map(|i| rd_f64(raw, i * 8)).collect())
}

impl CorruptionHead {
    /// Parse a `ZCTH` v1 file.
    ///
    /// Validates the magic, the version, the schema hash, every section's
    /// range and stride, every declared id against the caller width, and every
    /// tree's node graph — so a `CorruptionHead` that exists is one that can
    /// be evaluated without a single further bounds question in the hot loop.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CorruptionHeadError> {
        if bytes.len() < HEADER_LEN {
            return Err(CorruptionHeadError::Truncated {
                expected: HEADER_LEN,
                got: bytes.len(),
            });
        }
        let magic = [bytes[0], bytes[1], bytes[2], bytes[3]];
        if magic != MAGIC {
            return Err(CorruptionHeadError::BadMagic { got: magic });
        }
        let version = rd_u16(bytes, 4);
        if version != FORMAT_VERSION {
            return Err(CorruptionHeadError::UnsupportedVersion {
                got: version,
                supported: FORMAT_VERSION,
            });
        }
        let flags = rd_u16(bytes, 6);
        let stored_hash = rd_u64(bytes, 8);
        let caller_input_width = rd_u32(bytes, 16) as usize;
        let n_declared = rd_u32(bytes, 20) as usize;
        let n_trees = rd_u32(bytes, 24) as usize;
        let n_nodes = rd_u32(bytes, 28) as usize;
        let baseline = rd_f64(bytes, 32);
        let deadband_t = rd_f64(bytes, 40);
        let clip = rd_f32(bytes, 48);

        let sec_ids = rd_section(bytes, 56);
        let sec_mean = rd_section(bytes, 64);
        let sec_scale = rd_section(bytes, 72);
        let sec_toff = rd_section(bytes, 80);
        let sec_nodes = rd_section(bytes, 88);
        let sec_iso_x = rd_section(bytes, 96);
        let sec_iso_y = rd_section(bytes, 104);
        let sec_meta = rd_section(bytes, 112);

        // declared ids
        let raw_ids = sec_ids.slice(bytes, "declared_ids")?;
        if raw_ids.len() != n_declared * 2 {
            return Err(CorruptionHeadError::SectionShape {
                name: "declared_ids",
                len: sec_ids.len,
                stride: 2,
                expected_elems: n_declared,
            });
        }
        let declared_ids: Vec<u16> = (0..n_declared).map(|i| rd_u16(raw_ids, i * 2)).collect();
        for (pos, &id) in declared_ids.iter().enumerate() {
            if usize::from(id) >= caller_input_width {
                return Err(CorruptionHeadError::DeclaredIdOutOfRange {
                    pos,
                    id,
                    width: caller_input_width,
                });
            }
        }

        // scaler. `has_scaler` is honoured so a future tree-only head (trees
        // are scale-invariant when fitted on raw values) needs no dummy
        // identity vectors; today's exporter always writes it.
        let (mean, scale) = if flags & FLAG_HAS_SCALER != 0 {
            (
                read_f64_vec(bytes, sec_mean, "scaler_mean", n_declared)?,
                read_f64_vec(bytes, sec_scale, "scaler_scale", n_declared)?,
            )
        } else {
            (vec![0.0; n_declared], vec![1.0; n_declared])
        };

        // tree offsets
        let raw_toff = sec_toff.slice(bytes, "tree_offsets")?;
        if raw_toff.len() != (n_trees + 1) * 4 {
            return Err(CorruptionHeadError::SectionShape {
                name: "tree_offsets",
                len: sec_toff.len,
                stride: 4,
                expected_elems: n_trees + 1,
            });
        }
        let tree_offsets: Vec<u32> = (0..=n_trees).map(|i| rd_u32(raw_toff, i * 4)).collect();

        // nodes
        let raw_nodes = sec_nodes.slice(bytes, "nodes")?;
        if raw_nodes.len() != n_nodes * NODE_LEN {
            return Err(CorruptionHeadError::SectionShape {
                name: "nodes",
                len: sec_nodes.len,
                stride: NODE_LEN,
                expected_elems: n_nodes,
            });
        }
        let nodes: Vec<Node> = (0..n_nodes)
            .map(|i| {
                let b = i * NODE_LEN;
                Node {
                    threshold: rd_f64(raw_nodes, b),
                    left: rd_u32(raw_nodes, b + 8),
                    right: rd_u32(raw_nodes, b + 12),
                    feature_pos: rd_u32(raw_nodes, b + 16),
                    flags: rd_u32(raw_nodes, b + 20),
                    value: rd_f64(raw_nodes, b + 24),
                }
            })
            .collect();

        // isotonic
        let (iso_x, iso_y) = if flags & FLAG_HAS_ISOTONIC != 0 {
            let nx = sec_iso_x.len as usize / 8;
            let xs = read_f64_vec(bytes, sec_iso_x, "iso_x", nx)?;
            let ys = read_f64_vec(bytes, sec_iso_y, "iso_y", nx)?;
            (xs, ys)
        } else {
            (Vec::new(), Vec::new())
        };

        let meta = core::str::from_utf8(sec_meta.slice(bytes, "meta")?)
            .unwrap_or("")
            .to_string();

        let computed = fnv1a64(&schema_descriptor(
            caller_input_width as u32,
            n_declared as u32,
            n_trees as u32,
            n_nodes as u32,
            clip,
            &declared_ids,
            iso_x.len() as u32,
        ));
        if computed != stored_hash {
            return Err(CorruptionHeadError::SchemaHashMismatch {
                stored: stored_hash,
                computed,
            });
        }

        let head = Self {
            caller_input_width,
            declared_ids,
            mean,
            scale,
            clip: f64::from(clip),
            baseline,
            deadband_t,
            tree_offsets,
            nodes,
            iso_x,
            iso_y,
            schema_hash: stored_hash,
            meta,
        };
        head.validate_trees()?;
        Ok(head)
    }

    /// Every tree is a well-formed binary tree over its own node range, every
    /// internal node's `feature_pos` is in range, and no walk can cycle.
    ///
    /// Checked ONCE at load so [`Self::probability`] needs no bounds test in
    /// its inner loop, and so a malformed file fails at `from_bytes` rather
    /// than as a wrong number a thousand rows later.
    fn validate_trees(&self) -> Result<(), CorruptionHeadError> {
        let n_declared = self.declared_ids.len();
        for t in 0..self.tree_offsets.len().saturating_sub(1) {
            let lo = self.tree_offsets[t] as usize;
            let hi = self.tree_offsets[t + 1] as usize;
            let tno = t as u32;
            if lo > hi || hi > self.nodes.len() {
                return Err(CorruptionHeadError::MalformedTree {
                    tree: tno,
                    detail: "node range out of bounds",
                });
            }
            if lo == hi {
                return Err(CorruptionHeadError::MalformedTree {
                    tree: tno,
                    detail: "empty tree",
                });
            }
            let span = hi - lo;
            for node in &self.nodes[lo..hi] {
                if node.is_leaf() {
                    continue;
                }
                if node.left as usize >= span || node.right as usize >= span {
                    return Err(CorruptionHeadError::MalformedTree {
                        tree: tno,
                        detail: "child index outside the tree",
                    });
                }
                if node.feature_pos as usize >= n_declared {
                    return Err(CorruptionHeadError::MalformedTree {
                        tree: tno,
                        detail: "feature_pos outside the declared id list",
                    });
                }
            }
            // Acyclicity, cheaply and exactly: a walk from the root can visit
            // at most `span` nodes, so a walk that has not reached a leaf by
            // then contains a cycle. Checked structurally instead — every
            // child index must be strictly greater than its parent's, which
            // is how sklearn emits its node array and is a stronger, O(n)
            // property than "no cycle on the paths we happened to take".
            for (i, node) in self.nodes[lo..hi].iter().enumerate() {
                if node.is_leaf() {
                    continue;
                }
                if (node.left as usize) <= i || (node.right as usize) <= i {
                    return Err(CorruptionHeadError::MalformedTree {
                        tree: tno,
                        detail: "child index does not increase (cycle)",
                    });
                }
            }
        }
        Ok(())
    }

    /// The number of features a caller must supply, in the head's declared
    /// layout. Mirrors `zenpredict::Model::caller_input_width`.
    #[must_use]
    pub fn caller_input_width(&self) -> usize {
        self.caller_input_width
    }

    /// The feature ids the trees actually read, in the order the trees index
    /// them. The dense contract's consumer half, for a head.
    #[must_use]
    pub fn declared_feature_ids(&self) -> &[u16] {
        &self.declared_ids
    }

    /// Number of trees in the ensemble.
    #[must_use]
    pub fn n_trees(&self) -> usize {
        self.tree_offsets.len().saturating_sub(1)
    }

    /// Total node count across every tree — the size that matters for the
    /// forward cost.
    #[must_use]
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// The schema hash the file carries, already verified against its shape.
    #[must_use]
    pub fn schema_hash(&self) -> u64 {
        self.schema_hash
    }

    /// The provenance JSON blob, verbatim. Never load-bearing for scoring.
    #[must_use]
    pub fn metadata_json(&self) -> &str {
        &self.meta
    }

    /// The baked deadband in probability units: the head fires when
    /// [`Self::probability`] is strictly greater than this.
    #[must_use]
    pub fn deadband(&self) -> f64 {
        self.deadband_t
    }

    /// The same deadband in the dial's score units, `100 * (1 - deadband())`.
    ///
    /// The two are related by the same expression [`Self::score`] applies to a
    /// probability, so `score(row) < deadband_score()` and
    /// `probability(row) > deadband()` agree on every row rather than
    /// disagreeing by an ulp near the boundary.
    #[must_use]
    pub fn deadband_score(&self) -> f64 {
        100.0 * (1.0 - self.deadband_t)
    }

    /// `P(corrupt)` for one caller-width feature row of `f64`.
    pub fn probability_f64(&self, features: &[f64]) -> Result<f64, CorruptionHeadError> {
        if features.len() != self.caller_input_width {
            return Err(CorruptionHeadError::FeatureLenMismatch {
                expected: self.caller_input_width,
                got: features.len(),
            });
        }
        Ok(self.probability_unchecked(|j| features[usize::from(self.declared_ids[j])]))
    }

    /// `P(corrupt)` for one caller-width feature row of `f32`.
    ///
    /// The `f32` is widened to `f64` before standardisation, which is what the
    /// exporter's own parity harness feeds sklearn, so the two agree on the
    /// same bits rather than on a rounding convention.
    pub fn probability(&self, features: &[f32]) -> Result<f64, CorruptionHeadError> {
        if features.len() != self.caller_input_width {
            return Err(CorruptionHeadError::FeatureLenMismatch {
                expected: self.caller_input_width,
                got: features.len(),
            });
        }
        Ok(self.probability_unchecked(|j| f64::from(features[usize::from(self.declared_ids[j])])))
    }

    /// The head's score in the dial's units, `100 * (1 - P)`.
    pub fn score(&self, features: &[f32]) -> Result<f64, CorruptionHeadError> {
        self.probability(features).map(|p| 100.0 * (1.0 - p))
    }

    /// [`Self::score`] over an `f64` row.
    pub fn score_f64(&self, features: &[f64]) -> Result<f64, CorruptionHeadError> {
        self.probability_f64(features).map(|p| 100.0 * (1.0 - p))
    }

    /// The raw ensemble output before the logistic and the calibration —
    /// `baseline + sum of tree outputs`, i.e. sklearn's `decision_function`.
    ///
    /// Exposed because it is the quantity the parity gate can compare at
    /// **0 ulp**: the tree walk is exact arithmetic, so any difference here is
    /// a real defect, while a difference after `exp` is a libm question.
    pub fn decision_function(&self, features: &[f64]) -> Result<f64, CorruptionHeadError> {
        if features.len() != self.caller_input_width {
            return Err(CorruptionHeadError::FeatureLenMismatch {
                expected: self.caller_input_width,
                got: features.len(),
            });
        }
        Ok(self.raw(|j| features[usize::from(self.declared_ids[j])]))
    }

    fn probability_unchecked(&self, get: impl Fn(usize) -> f64) -> f64 {
        let raw = self.raw(get);
        let p_raw = 1.0 / (1.0 + (-raw).exp());
        if self.iso_x.len() < 2 {
            // A degenerate calibration (0 or 1 knot) is sklearn's
            // "single y, constant prediction" branch; with no knots at all
            // the head is uncalibrated and reports the logistic directly.
            return self.iso_y.first().copied().unwrap_or(p_raw);
        }
        interp_linear(p_raw, &self.iso_x, &self.iso_y)
    }

    /// `baseline + sum of tree outputs`, accumulated **in tree order** —
    /// which is the order `_predict_iterations` adds them in, and f64
    /// addition is not associative, so the order is part of the contract.
    fn raw(&self, get: impl Fn(usize) -> f64) -> f64 {
        // Standardise once into a small scratch rather than per tree: a
        // 228-wide row is 1.8 KiB and every tree re-reads a handful of its
        // lanes, so recomputing per node would dominate the walk.
        let n = self.declared_ids.len();
        let mut z = Vec::with_capacity(n);
        for j in 0..n {
            let v = (get(j) - self.mean[j]) / self.scale[j];
            z.push(v.clamp(-self.clip, self.clip));
        }
        let mut raw = self.baseline;
        for t in 0..self.n_trees() {
            let lo = self.tree_offsets[t] as usize;
            let hi = self.tree_offsets[t + 1] as usize;
            let tree = &self.nodes[lo..hi];
            let mut node = &tree[0];
            loop {
                if node.is_leaf() {
                    raw += node.value;
                    break;
                }
                let v = z[node.feature_pos as usize];
                // sklearn `_predictor.pyx`: NaN is routed by the node's own
                // flag; otherwise `<=` goes left. Reproduced, not paraphrased.
                let go_left = if v.is_nan() {
                    node.missing_go_to_left()
                } else {
                    v <= node.threshold
                };
                node = if go_left {
                    &tree[node.left as usize]
                } else {
                    &tree[node.right as usize]
                };
            }
        }
        raw
    }
}

/// sklearn's isotonic evaluation, reproduced arithmetic-for-arithmetic:
/// clip the query to the knot range, then **`np.interp`**.
///
/// Getting here took one wrong turn worth recording, because the wrong answer
/// looked right. `IsotonicRegression._build_f` constructs a
/// `scipy.interpolate.interp1d(kind="linear")`, so the obvious reading is
/// scipy's `_call_linear` — leftmost `searchsorted` bracket, convex-combination
/// evaluation. **That is not the code that runs.** `interp1d.__init__` routes
/// plain `linear` to `_call_linear_np`, which is a one-line call to
/// `np.interp`, and the two disagree:
///
/// * `np.interp` brackets on the **rightmost** `j` with `xs[j] <= t`, so a
///   query exactly ON knot `j` evaluates as `slope * 0 + ys[j]` and returns
///   `ys[j]` **exactly**. The leftmost bracket returns
///   `slope * (xs[j] - xs[j-1]) + ys[j-1]`, which is `ys[j]` in real arithmetic
///   and not always in f64.
/// * `np.interp` uses the **slope** form; `_call_linear` uses the convex
///   combination. MEASURED on a real 90-knot isotonic fit over 25,092 queries
///   (knots, midpoints, uniform draws, both out-of-range ends): the slope form
///   with the rightmost bracket is **bit-identical** to `iso.predict`, and the
///   convex form with the leftmost bracket differs by up to 1.11e-16.
///
/// The NaN retry and the `j == n - 1` short-circuit are `np.interp`'s own, kept
/// because they are the only thing standing between a duplicated end knot and a
/// `0/0`.
fn interp_linear(q: f64, xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len();
    // sklearn clips the QUERY to `[X_min_, X_max_]` before interpolating
    // (`IsotonicRegression._transform`), so out-of-range inputs return the
    // endpoint value rather than extrapolating.
    let t = q.clamp(xs[0], xs[n - 1]);
    // The largest `j` with `xs[j] <= t`. `t >= xs[0]` after the clamp, so the
    // partition point is at least 1 and the subtraction cannot underflow.
    let j = xs.partition_point(|&x| x <= t) - 1;
    if j >= n - 1 {
        return ys[n - 1];
    }
    let slope = (ys[j + 1] - ys[j]) / (xs[j + 1] - xs[j]);
    let r = slope * (t - xs[j]) + ys[j];
    if r.is_nan() {
        let r2 = slope * (t - xs[j + 1]) + ys[j + 1];
        if r2.is_nan() && ys[j] == ys[j + 1] {
            return ys[j];
        }
        return r2;
    }
    r
}

// ── Servability: attaching a head must never widen the walk ──────────────
#[cfg(feature = "feature-regime-v2")]
impl CorruptionHead {
    /// Refuse unless every declared id is a slot this profile's extraction
    /// plan already populates.
    ///
    /// This is the **"make sure everything can be served"** contract applied
    /// to the head: a head is attachable exactly when the walk that produced
    /// the score it is gating already computed everything the head reads. The
    /// `f0..f227` (basic + peaks) slice the theory lane selected is free at
    /// `ZensimProfile::D`'s `V1PoolsMode::Peaks` walk, which is why that slice
    /// was chosen; a head that reached into `f228..f371` would force
    /// `V1PoolsMode::Full` and silently make D as expensive as B, so it is
    /// refused rather than served.
    pub fn check_servable_by(
        &self,
        profile: crate::profile::ZensimProfile,
    ) -> Result<(), CorruptionHeadError> {
        let params = profile.params();
        let config = crate::metric::config_from_params(params, false);
        let plan = crate::fold_engine::score_plan(params, &config, true).ok_or_else(|| {
            CorruptionHeadError::NotServable {
                profile: profile.name(),
                detail: "the profile has no derivable extraction plan".to_string(),
            }
        })?;
        let want = crate::feature_set_id::SlotSet::from_slots(
            self.declared_ids.iter().map(|&id| usize::from(id)),
        );
        if plan.covers(&want) {
            Ok(())
        } else {
            Err(CorruptionHeadError::NotServable {
                profile: profile.name(),
                detail: format!("head reads {want}; the profile's plan emits {}", plan.emit),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A hand-assembled ZCTH file, so the reader is tested against bytes
    /// rather than against the exporter that produced them.
    struct Builder {
        caller_input_width: u32,
        declared_ids: Vec<u16>,
        mean: Vec<f64>,
        scale: Vec<f64>,
        clip: f32,
        baseline: f64,
        deadband_t: f64,
        tree_offsets: Vec<u32>,
        nodes: Vec<Node>,
        iso_x: Vec<f64>,
        iso_y: Vec<f64>,
        meta: &'static str,
    }

    impl Builder {
        fn single_stump(feature_id: u16, threshold: f64, lo: f64, hi: f64) -> Self {
            Self {
                caller_input_width: 8,
                declared_ids: vec![feature_id],
                mean: vec![0.0],
                scale: vec![1.0],
                clip: 8.0,
                baseline: 0.0,
                deadband_t: 0.9,
                tree_offsets: vec![0, 3],
                nodes: vec![
                    Node {
                        threshold,
                        left: 1,
                        right: 2,
                        feature_pos: 0,
                        flags: 0,
                        value: 0.0,
                    },
                    Node {
                        threshold: 0.0,
                        left: 0,
                        right: 0,
                        feature_pos: 0,
                        flags: NODE_FLAG_LEAF,
                        value: lo,
                    },
                    Node {
                        threshold: 0.0,
                        left: 0,
                        right: 0,
                        feature_pos: 0,
                        flags: NODE_FLAG_LEAF,
                        value: hi,
                    },
                ],
                iso_x: Vec::new(),
                iso_y: Vec::new(),
                meta: "{}",
            }
        }

        fn build(&self) -> Vec<u8> {
            let mut body: Vec<u8> = Vec::new();
            let mut push = |body: &mut Vec<u8>, data: &[u8]| -> Section {
                let off = HEADER_LEN + body.len();
                body.extend_from_slice(data);
                Section {
                    offset: off as u32,
                    len: data.len() as u32,
                }
            };
            let ids: Vec<u8> = self
                .declared_ids
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            let sec_ids = push(&mut body, &ids);
            let f64s = |v: &[f64]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
            let sec_mean = push(&mut body, &f64s(&self.mean));
            let sec_scale = push(&mut body, &f64s(&self.scale));
            let toff: Vec<u8> = self
                .tree_offsets
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            let sec_toff = push(&mut body, &toff);
            let mut nb: Vec<u8> = Vec::new();
            for n in &self.nodes {
                nb.extend_from_slice(&n.threshold.to_le_bytes());
                nb.extend_from_slice(&n.left.to_le_bytes());
                nb.extend_from_slice(&n.right.to_le_bytes());
                nb.extend_from_slice(&n.feature_pos.to_le_bytes());
                nb.extend_from_slice(&n.flags.to_le_bytes());
                nb.extend_from_slice(&n.value.to_le_bytes());
            }
            let sec_nodes = push(&mut body, &nb);
            let sec_ix = push(&mut body, &f64s(&self.iso_x));
            let sec_iy = push(&mut body, &f64s(&self.iso_y));
            let sec_meta = push(&mut body, self.meta.as_bytes());

            let mut flags = FLAG_HAS_SCALER;
            if !self.iso_x.is_empty() {
                flags |= FLAG_HAS_ISOTONIC;
            }
            let hash = fnv1a64(&schema_descriptor(
                self.caller_input_width,
                self.declared_ids.len() as u32,
                (self.tree_offsets.len() - 1) as u32,
                self.nodes.len() as u32,
                self.clip,
                &self.declared_ids,
                self.iso_x.len() as u32,
            ));
            let mut h = vec![0u8; HEADER_LEN];
            h[0..4].copy_from_slice(&MAGIC);
            h[4..6].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
            h[6..8].copy_from_slice(&flags.to_le_bytes());
            h[8..16].copy_from_slice(&hash.to_le_bytes());
            h[16..20].copy_from_slice(&self.caller_input_width.to_le_bytes());
            h[20..24].copy_from_slice(&(self.declared_ids.len() as u32).to_le_bytes());
            h[24..28].copy_from_slice(&((self.tree_offsets.len() - 1) as u32).to_le_bytes());
            h[28..32].copy_from_slice(&(self.nodes.len() as u32).to_le_bytes());
            h[32..40].copy_from_slice(&self.baseline.to_le_bytes());
            h[40..48].copy_from_slice(&self.deadband_t.to_le_bytes());
            h[48..52].copy_from_slice(&self.clip.to_le_bytes());
            for (i, s) in [
                sec_ids, sec_mean, sec_scale, sec_toff, sec_nodes, sec_ix, sec_iy, sec_meta,
            ]
            .iter()
            .enumerate()
            {
                let at = 56 + i * 8;
                h[at..at + 4].copy_from_slice(&s.offset.to_le_bytes());
                h[at + 4..at + 8].copy_from_slice(&s.len.to_le_bytes());
            }
            h.extend_from_slice(&body);
            h
        }
    }

    #[test]
    fn round_trip_a_stump() {
        let b = Builder::single_stump(3, 0.5, -2.0, 7.0);
        let head = CorruptionHead::from_bytes(&b.build()).expect("parse");
        assert_eq!(head.caller_input_width(), 8);
        assert_eq!(head.declared_feature_ids(), &[3]);
        assert_eq!(head.n_trees(), 1);
        // f3 = 0.25 <= 0.5 -> left leaf -2.0; sigmoid(-2) = 0.11920292202211755
        let row = [0.0f64, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(head.decision_function(&row).unwrap(), -2.0);
        let p = head.probability_f64(&row).unwrap();
        assert!((p - 1.0 / (1.0 + 2.0f64.exp())).abs() < 1e-18, "{p}");
        // f3 = 0.75 > 0.5 -> right leaf 7.0
        let row2 = [0.0f64, 0.0, 0.0, 0.75, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(head.decision_function(&row2).unwrap(), 7.0);
    }

    #[test]
    fn a_znpr_bake_is_refused_at_byte_zero() {
        let mut bytes = vec![0u8; 256];
        bytes[0..4].copy_from_slice(b"ZNPR");
        match CorruptionHead::from_bytes(&bytes) {
            Err(CorruptionHeadError::BadMagic { got }) => assert_eq!(&got, b"ZNPR"),
            other => panic!("expected BadMagic, got {other:?}"),
        }
    }

    #[test]
    fn a_flipped_shape_field_fails_the_schema_hash() {
        let b = Builder::single_stump(3, 0.5, -2.0, 7.0);
        let mut bytes = b.build();
        // Claim a wider caller layout without re-hashing: exactly the edit a
        // hand-patched file would make, and the one that would otherwise
        // gather the wrong slot.
        bytes[16..20].copy_from_slice(&16u32.to_le_bytes());
        assert!(matches!(
            CorruptionHead::from_bytes(&bytes),
            Err(CorruptionHeadError::SchemaHashMismatch { .. })
        ));
    }

    #[test]
    fn a_short_feature_row_is_refused_never_scored_as_a_prefix() {
        let head = CorruptionHead::from_bytes(&Builder::single_stump(3, 0.5, -2.0, 7.0).build())
            .expect("parse");
        let short = [0.0f64; 4];
        assert!(matches!(
            head.probability_f64(&short),
            Err(CorruptionHeadError::FeatureLenMismatch {
                expected: 8,
                got: 4
            })
        ));
    }

    #[test]
    fn a_cyclic_tree_is_refused_at_load() {
        let mut b = Builder::single_stump(3, 0.5, -2.0, 7.0);
        // Point the root's left child at itself.
        b.nodes[0].left = 0;
        let bytes = b.build();
        assert!(matches!(
            CorruptionHead::from_bytes(&bytes),
            Err(CorruptionHeadError::MalformedTree { .. })
        ));
    }

    #[test]
    fn a_declared_id_past_the_caller_width_is_refused() {
        let mut b = Builder::single_stump(3, 0.5, -2.0, 7.0);
        b.declared_ids = vec![9]; // width is 8
        let bytes = b.build();
        assert!(matches!(
            CorruptionHead::from_bytes(&bytes),
            Err(CorruptionHeadError::DeclaredIdOutOfRange { .. })
        ));
    }

    #[test]
    fn nan_is_routed_by_the_nodes_own_flag() {
        let mut b = Builder::single_stump(3, 0.5, -2.0, 7.0);
        b.nodes[0].flags = NODE_FLAG_MISSING_LEFT;
        let head = CorruptionHead::from_bytes(&b.build()).expect("parse");
        let mut row = [0.0f64; 8];
        row[3] = f64::NAN;
        assert_eq!(head.decision_function(&row).unwrap(), -2.0);

        let mut b2 = Builder::single_stump(3, 0.5, -2.0, 7.0);
        b2.nodes[0].flags = 0;
        let head2 = CorruptionHead::from_bytes(&b2.build()).expect("parse");
        assert_eq!(head2.decision_function(&row).unwrap(), 7.0);
    }

    #[test]
    fn the_clip_is_applied_before_the_split_test() {
        let mut b = Builder::single_stump(3, 7.5, -2.0, 7.0);
        b.clip = 4.0;
        let head = CorruptionHead::from_bytes(&b.build()).expect("parse");
        let mut row = [0.0f64; 8];
        // Raw 100 would be > 7.5 (right); clipped to 4.0 it is <= 7.5 (left).
        row[3] = 100.0;
        assert_eq!(head.decision_function(&row).unwrap(), -2.0);
    }

    /// The bracket rule is the one place a "reasonable" implementation
    /// silently disagrees with what sklearn actually runs, so it gets its own
    /// test. Every expected value below was read off `iso.predict` /
    /// `np.interp` on this exact fixture, not derived by hand.
    #[test]
    fn interp_matches_np_interps_rightmost_bracket() {
        let xs = [0.0, 0.25, 0.25, 0.75, 1.0];
        let ys = [0.0, 0.10, 0.60, 0.60, 1.0];
        // On a duplicated knot the RIGHTMOST bracket wins: (2, 3) -> 0.60.
        // scipy's `_call_linear` would say 0.10 here; `np.interp`, which is
        // what `interp1d(kind="linear")` dispatches to, says 0.60 — and 0.60
        // is what `scipy.interpolate.interp1d(...)(0.25)` returns.
        assert_eq!(interp_linear(0.25, &xs, &ys), 0.60);
        // Below/above the range clips to the endpoint values.
        assert_eq!(interp_linear(-5.0, &xs, &ys), 0.0);
        assert_eq!(interp_linear(5.0, &xs, &ys), 1.0);
        // Interior of the flat run stays on the plateau.
        assert_eq!(interp_linear(0.5, &xs, &ys), 0.60);
        assert_eq!(interp_linear(0.3, &xs, &ys), 0.60);
    }

    /// The property the rightmost bracket buys, stated on its own: a query
    /// exactly on a knot returns that knot's value with ZERO rounding, because
    /// the evaluation degenerates to `slope * 0 + ys[j]`. Isotonic fits are
    /// made of plateau edges, so this is the common case, not a corner.
    #[test]
    fn an_on_knot_query_returns_the_knot_value_exactly() {
        let xs = [0.0, 0.1, 0.37, 0.61, 0.9, 1.0];
        let ys = [0.02, 0.31, 0.315, 0.7, 0.7, 0.99];
        for (i, &x) in xs.iter().enumerate() {
            assert_eq!(
                interp_linear(x, &xs, &ys).to_bits(),
                ys[i].to_bits(),
                "knot {i}"
            );
        }
    }

    #[test]
    fn gate_score_floors_only_a_flagged_row_and_never_raises() {
        // Not flagged (head score above the deadband): passthrough.
        assert_eq!(gate_score(83.0, 55.0, 10.0), 83.0);
        // Flagged, positive dial: floored to 0.
        assert_eq!(gate_score(83.0, 2.0, 10.0), 0.0);
        // Flagged, already-negative dial: left alone (min, not clamp).
        assert_eq!(gate_score(-40.0, 2.0, 10.0), -40.0);
        // Exactly at the deadband is NOT flagged (strict `<`), matching the
        // Python rule's strict `p > T`.
        assert_eq!(gate_score(83.0, 10.0, 10.0), 83.0);
    }

    #[test]
    fn deadband_score_and_deadband_agree_on_the_boundary() {
        let head = CorruptionHead::from_bytes(&Builder::single_stump(3, 0.5, -2.0, 7.0).build())
            .expect("parse");
        assert_eq!(head.deadband(), 0.9);
        assert_eq!(head.deadband_score(), 100.0 * (1.0 - 0.9));
    }
}
