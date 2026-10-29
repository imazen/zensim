//! Feature-set identifiers: compute families, extractor era, and a hash of the slot IDs.
//!
//! The optional `@w<N>` layout width helps reconstruct a sparse set, but does not
//! participate in identity. Families render in registry order; slot hashes use
//! sorted, deduplicated IDs.

use core::fmt;

/// One registered COMPUTE token: a slot family whose presence changes which
/// slots are POPULATED (not merely their values).
///
/// Sub-toggles that reshape a family from the inside (`gradient`,
/// `blockiness`, `transducer_bank`, `append2_dst_activity`) are deliberately
/// absent: they move the populated set, so they move the hash, so they cannot
/// collide — and keeping them out of the short form is what keeps a 944 name
/// readable. See `docs/FEATURE_SET_IDS.md` §2.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum ComputeToken {
    /// v1 basic fold, `f0..155`.
    Basic,
    /// The ten `fused944native` carrier slots (`V1PoolsMode::Carriers`).
    Carriers,
    /// v1 soft-peak pool, `f156..227` (`V1PoolsMode::Peaks`).
    Peaks,
    /// v1 masked pool, `f228..299`.
    Masked,
    /// v1 IW pool, `f300..371`.
    Iw,
    /// v2-348 dense block, `f372..719`.
    V2,
}

impl ComputeToken {
    /// Every token, in the canonical registry order the short form renders in.
    pub const ALL: &'static [ComputeToken] = &[
        ComputeToken::Basic,
        ComputeToken::Carriers,
        ComputeToken::Peaks,
        ComputeToken::Masked,
        ComputeToken::Iw,
        ComputeToken::V2,
    ];

    /// The token's wire spelling (charset `[a-z0-9_]+`, per the
    /// `zenanalyze-api` name rule).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            ComputeToken::Basic => "basic",
            ComputeToken::Carriers => "carriers",
            ComputeToken::Peaks => "peaks",
            ComputeToken::Masked => "masked",
            ComputeToken::Iw => "iw",
            ComputeToken::V2 => "v2",
        }
    }

    /// Parse one token. Unknown spellings are `None` — the vocabulary is
    /// CLOSED, so a typo is an error rather than a new set.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|t| t.as_str() == s)
    }

    const fn bit(self) -> u32 {
        1u32 << (self as u32)
    }
}

impl fmt::Display for ComputeToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A set of [`ComputeToken`]s. Renders in registry order regardless of
/// insertion order; the empty set renders `none`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash, PartialOrd, Ord)]
pub struct ComputeParts(u32);

impl ComputeParts {
    /// No families at all.
    pub const EMPTY: Self = ComputeParts(0);

    /// This set plus `t`.
    #[must_use]
    pub const fn with(self, t: ComputeToken) -> Self {
        ComputeParts(self.0 | t.bit())
    }

    /// Is `t` in the set?
    #[must_use]
    pub const fn contains(self, t: ComputeToken) -> bool {
        self.0 & t.bit() != 0
    }

    /// Is the set empty?
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The tokens, in registry order.
    pub fn iter(self) -> impl Iterator<Item = ComputeToken> {
        ComputeToken::ALL
            .iter()
            .copied()
            .filter(move |t| self.contains(*t))
    }

    /// Parse a `+`-joined token list (order-tolerant), or `none`.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        if s == "none" {
            return Some(Self::EMPTY);
        }
        if s.is_empty() {
            return None;
        }
        let mut out = Self::EMPTY;
        for part in s.split('+') {
            out = out.with(ComputeToken::parse(part)?);
        }
        Some(out)
    }
}

impl fmt::Display for ComputeParts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return f.write_str("none");
        }
        let mut first = true;
        for t in ComputeToken::ALL.iter().copied() {
            if !self.contains(t) {
                continue;
            }
            if !first {
                f.write_str("+")?;
            }
            first = false;
            f.write_str(t.as_str())?;
        }
        Ok(())
    }
}

/// A canonical set of slot indices: sorted, de-duplicated, stored as merged
/// half-open ranges and rendered compactly (`"0-155,720-923"`, a lone slot as
/// `"42"`, the empty set as `""`).
///
/// This is the machine half of a feature-set id — the thing the `#hash8`
/// hashes and the thing [`SlotSet::covers`] compares. A PRODUCER's slot set is
/// what it populates; a CONSUMER's (a bake's) is what it reads.
#[derive(Debug, Clone, PartialEq, Eq, Default, Hash)]
pub struct SlotSet {
    /// Sorted, disjoint, merged, half-open `[start, end)`.
    ranges: Vec<(usize, usize)>,
}

impl SlotSet {
    /// Canonicalise an arbitrary slot iterator (any order, duplicates fine).
    #[must_use]
    pub fn from_slots(slots: impl IntoIterator<Item = usize>) -> Self {
        let mut v: Vec<usize> = slots.into_iter().collect();
        v.sort_unstable();
        v.dedup();
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for s in v {
            match ranges.last_mut() {
                Some(last) if last.1 == s => last.1 = s + 1,
                _ => ranges.push((s, s + 1)),
            }
        }
        Self { ranges }
    }

    /// Canonicalise a half-open range list (any order, overlaps fine).
    #[must_use]
    pub fn from_ranges(ranges: impl IntoIterator<Item = (usize, usize)>) -> Self {
        let mut v: Vec<(usize, usize)> = ranges.into_iter().filter(|(a, b)| a < b).collect();
        v.sort_unstable();
        let mut out: Vec<(usize, usize)> = Vec::new();
        for (a, b) in v {
            match out.last_mut() {
                Some(last) if a <= last.1 => last.1 = last.1.max(b),
                _ => out.push((a, b)),
            }
        }
        Self { ranges: out }
    }

    /// Number of slots in the set.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ranges.iter().map(|(a, b)| b - a).sum()
    }

    /// Is the set empty?
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// Is `slot` in the set?
    #[must_use]
    pub fn contains(&self, slot: usize) -> bool {
        self.ranges.iter().any(|(a, b)| slot >= *a && slot < *b)
    }

    /// Every slot, ascending.
    pub fn iter_slots(&self) -> impl Iterator<Item = usize> + '_ {
        self.ranges.iter().flat_map(|(a, b)| *a..*b)
    }

    /// The merged half-open ranges.
    #[must_use]
    pub fn ranges(&self) -> &[(usize, usize)] {
        &self.ranges
    }

    /// Does this set contain every slot of `other`? **The compatibility test:**
    /// a read is sound only when the producer covers the consumer.
    #[must_use]
    pub fn covers(&self, other: &SlotSet) -> bool {
        other.iter_slots().all(|s| self.contains(s))
    }

    /// The slots of `other` this set does NOT contain — the actionable half of
    /// a failed [`covers`](Self::covers).
    #[must_use]
    pub fn missing_from(&self, other: &SlotSet) -> SlotSet {
        SlotSet::from_slots(other.iter_slots().filter(|s| !self.contains(*s)))
    }

    /// Intersection.
    #[must_use]
    pub fn intersect(&self, other: &SlotSet) -> SlotSet {
        SlotSet::from_slots(self.iter_slots().filter(|s| other.contains(*s)))
    }

    /// Union.
    #[must_use]
    pub fn union(&self, other: &SlotSet) -> SlotSet {
        SlotSet::from_ranges(
            self.ranges
                .iter()
                .copied()
                .chain(other.ranges.iter().copied()),
        )
    }

    /// Everything below `width` — the LAYOUT clip. A family that does not
    /// exist at this width cannot be populated at it.
    #[must_use]
    pub fn clipped_to(&self, width: usize) -> SlotSet {
        SlotSet::from_ranges(
            self.ranges
                .iter()
                .filter(|(a, _)| *a < width)
                .map(|(a, b)| (*a, (*b).min(width))),
        )
    }

    /// This set's content hash — [`slots_hash8`], the ONE owner.
    #[must_use]
    pub fn hash8(&self) -> u32 {
        slots_hash8(self.iter_slots())
    }
}

impl fmt::Display for SlotSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for (a, b) in &self.ranges {
            if !first {
                f.write_str(",")?;
            }
            first = false;
            if b - a == 1 {
                write!(f, "{a}")?;
            } else {
                write!(f, "{}-{}", a, b - 1)?;
            }
        }
        Ok(())
    }
}

/// **THE hash owner.** 32-bit content hash of a slot set.
///
/// FNV-1a/64 over the canonical rendering of the SORTED, DE-DUPLICATED slot
/// list (decimal, comma-separated), folded to 32 bits by the exact rule
/// `zenanalyze_api::NamedFeature::fold_hash` mandates:
/// `(h >> 32) ^ (h & 0xffff_ffff)`.
///
/// Set semantics: order and duplicates in the input cannot move the hash.
/// Ordering *within* the emitted vector is the LAYOUT's job, not this one's.
///
/// Every producer and consumer calls this. Nothing re-derives it — a silent
/// hash mismatch is worse than no hash (the same discipline `zenanalyze`
/// applies to `feature_qualified_names`, which ships a committed TSV rather
/// than letting Python recompute the fold).
#[must_use]
pub fn slots_hash8(slots: impl IntoIterator<Item = usize>) -> u32 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut v: Vec<usize> = slots.into_iter().collect();
    v.sort_unstable();
    v.dedup();
    let mut h = FNV_OFFSET;
    let mut first = true;
    for s in v {
        if !first {
            h ^= u64::from(b',');
            h = h.wrapping_mul(FNV_PRIME);
        }
        first = false;
        for byte in itoa_bytes(s) {
            h ^= u64::from(byte);
            h = h.wrapping_mul(FNV_PRIME);
        }
    }
    // The mandated 64→32 fold — identical to `NamedFeature::fold_hash`.
    ((h >> 32) ^ (h & 0xffff_ffff)) as u32
}

/// Decimal bytes of `n`, most-significant first, without allocating a String.
fn itoa_bytes(n: usize) -> impl Iterator<Item = u8> {
    let mut buf = [0u8; 20];
    let mut i = buf.len();
    let mut n = n;
    loop {
        i -= 1;
        buf[i] = b'0' + (n % 10) as u8;
        n /= 10;
        if n == 0 {
            break;
        }
    }
    (i..buf.len()).map(move |k| buf[k])
}

/// Is `s` a valid COMPUTE/ERA token spelling? Charset `[a-z0-9_]+`, exactly
/// `zenanalyze_api::NamedFeature::is_valid_name`.
#[must_use]
pub fn is_valid_token(s: &str) -> bool {
    !s.is_empty()
        && s.bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
}

/// A feature-set identifier: `<compute>/<era>#<hash8>`, optionally carrying a
/// legacy `@w<layout>` width.
///
/// **Equality is over compute + era + hash, and NOT over the layout.** Same
/// slots at a different era is a DIFFERENT id, which is the whole point (the
/// `ext944` / `era2r4` pair is bit-identical except `f720..923`, and a
/// published number cannot be corrected across the boundary, only
/// re-verdicted). Same slots at a different WIRE WIDTH is the SAME id, which
/// is what makes every `@w<N>` spelling an alias of its layout-free form
/// rather than a different set — see the module docs for the measurement that
/// forced this.
#[derive(Debug, Clone, Eq)]
pub struct FeatureSetId {
    compute: ComputeParts,
    /// The legacy wire width, when the id string carried one. A
    /// RECONSTRUCTION AID, never part of the identity: excluded from
    /// [`PartialEq`] and [`Hash`] by hand below, so an id parsed from
    /// `…@w372/…` and the same id parsed from `…/…` are interchangeable in
    /// every set, map and comparison.
    layout_width: Option<usize>,
    era: String,
    slots_hash: u32,
}

impl PartialEq for FeatureSetId {
    fn eq(&self, other: &Self) -> bool {
        self.compute == other.compute
            && self.era == other.era
            && self.slots_hash == other.slots_hash
    }
}

impl core::hash::Hash for FeatureSetId {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.compute.hash(state);
        self.era.hash(state);
        self.slots_hash.hash(state);
    }
}

impl FeatureSetId {
    /// Build the canonical, LAYOUT-FREE id from an already-computed slot hash.
    /// `None` when `era` is not a valid token.
    #[must_use]
    pub fn new(compute: ComputeParts, era: &str, slots_hash: u32) -> Option<Self> {
        is_valid_token(era).then(|| Self {
            compute,
            layout_width: None,
            era: era.to_string(),
            slots_hash,
        })
    }

    /// This id carrying `layout_width`. Changes the rendering, never the
    /// identity: the result compares and hashes EQUAL to `self`.
    #[must_use]
    pub fn with_layout(mut self, layout_width: usize) -> Self {
        self.layout_width = Some(layout_width);
        self
    }

    /// The COMPUTE token set.
    #[must_use]
    pub fn compute(&self) -> ComputeParts {
        self.compute
    }
    /// The slot-set content hash.
    #[must_use]
    pub fn slots_hash(&self) -> u32 {
        self.slots_hash
    }

    /// Parse either form. Strict, exactly like
    /// `zenanalyze_api::NamedFeature::parse`: 8 LOWERCASE hex digits, a closed
    /// token vocabulary, `[a-z0-9_]+` eras — and, when the legacy `@w<N>`
    /// component is present, a mandatory `w` on the width.
    ///
    /// **Both spellings of the same set parse to EQUAL ids**, which is what
    /// makes every `@w<N>` string ever written an alias rather than a
    /// different set.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        // The `@w<N>` component is optional. Split on the FIRST `/` after the
        // compute part: everything before it is `<compute>` or
        // `<compute>@w<N>`.
        let (head, rest) = s.split_once('/')?;
        let (compute_s, layout_width) = match head.split_once('@') {
            Some((c, layout_s)) => (c, Some(layout_s.strip_prefix('w')?.parse::<usize>().ok()?)),
            None => (head, None),
        };
        let (era, hash_s) = rest.split_once('#')?;
        let compute = ComputeParts::parse(compute_s)?;
        if hash_s.len() != 8
            || !hash_s
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return None;
        }
        let slots_hash = u32::from_str_radix(hash_s, 16).ok()?;
        let id = Self::new(compute, era, slots_hash)?;
        Some(match layout_width {
            Some(w) => id.with_layout(w),
            None => id,
        })
    }
}

impl fmt::Display for FeatureSetId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.layout_width {
            Some(w) => write!(
                f,
                "{}@w{}/{}#{:08x}",
                self.compute, w, self.era, self.slots_hash
            ),
            None => write!(f, "{}/{}#{:08x}", self.compute, self.era, self.slots_hash),
        }
    }
}
