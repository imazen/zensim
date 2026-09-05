//! **Feature-set identity for bakes, features roots and grids** — the
//! consumer of `zensim::feature_set_id`, and the check that closes the
//! `--regime 944` silent-mis-scoring bug at the root.
//!
//! Design + the alias table: `docs/FEATURE_SET_IDS.md`. Registry:
//! `benchmarks/feature_sets_registry.json`, embedded here with `include_str!`
//! so a binary is never one missing file away from silently having no
//! registry.
//!
//! # Producer vs consumer
//!
//! Both are a [`FeatureSetRef`]; they differ only in what the slot set means.
//!
//! * a **PRODUCER** (an extractor, a features root, a dial grid) — its slots
//!   are what it POPULATES;
//! * a **CONSUMER** (a bake) — its slots are what it READS
//!   ([`crate::block_profile::used_caller_lines`]).
//!
//! A read is sound iff the widths agree, `consumer.slots ⊆ producer.slots`,
//! and the eras agree. The middle one is the bug: shipped **B** reads 49 lines
//! in `f156..371`, the `ext944` root populates none of them, and the resulting
//! CID22 **0.3862** (true: **0.8764**) came back with no error and no warning.
//!
//! # Why the registry is embedded, not read from disk
//!
//! The check is a REFUSAL surface. A registry that can be absent is a check
//! that can be skipped by deleting a file; embedding it makes "no registry" a
//! compile error instead of a silent pass. It is also what lets
//! `every_registered_set_agrees_with_the_hash_owner` hold the committed JSON
//! to `zensim::feature_set_id::slots_hash8` — nothing here re-derives the hash.

use std::collections::BTreeMap;
use std::path::Path;

use zenpredict::Model;
use zensim::feature_set_id::{ComputeParts, ComputeToken, FeatureSetId, SlotSet, ERA_UNKNOWN};

/// The committed registry, embedded at build time.
const REGISTRY_JSON: &str = include_str!("../../benchmarks/feature_sets_registry.json");

/// A registered feature set. `pinned` is `Some` for `kind: "set"` (slots and
/// hash pinned) and `None` for `kind: "class"` (compute + layout + era only —
/// a MODEL class whose exact read-set differs per bake, so the name is the
/// class and the hash is the instance).
#[derive(Debug, Clone)]
pub struct RegisteredSet {
    /// The registry key — the canonical id string, or the class form for a
    /// `class` entry.
    pub key: String,
    /// Compute + layout + era, always present.
    pub id: ClassId,
    /// `|slots|` as recorded, for a pinned set.
    pub n_slots: Option<usize>,
    /// `"producer"` (populates) or `"consumer"` (reads).
    pub role: String,
    /// The legacy count/name this set has been called.
    pub legacy_name: String,
    /// The doc/record that establishes the set exists.
    pub evidence: String,
    pinned: Option<FeatureSetRef>,
}

impl RegisteredSet {
    /// The fully-pinned reference, for a `kind: "set"` entry.
    #[must_use]
    pub fn as_ref(&self) -> Option<&FeatureSetRef> {
        self.pinned.as_ref()
    }
}

/// Compute + layout + era, without a slot hash — what a `class` entry pins and
/// what the canonical id string renders minus its `#hash8`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassId {
    /// The COMPUTE token set.
    pub compute: ComputeParts,
    /// The emitted vector width.
    pub layout: usize,
    /// The registered extractor-era token.
    pub era: String,
}

impl ClassId {
    /// `"<compute>@w<layout>/<era>"` — the id without its hash.
    #[must_use]
    pub fn to_string_class(&self) -> String {
        format!("{}@w{}/{}", self.compute, self.layout, self.era)
    }
}

/// A resolved feature-set reference: the id, the slot set it stands for, and
/// HOW we learned it.
#[derive(Debug, Clone)]
pub struct FeatureSetRef {
    /// The full id (compute + layout + era + slot hash).
    pub id: FeatureSetId,
    /// The slots this reference stands for — populated (producer) or read
    /// (consumer).
    pub slots: SlotSet,
    /// Human description of where the id came from, printed beside every use.
    pub source: String,
    /// `true` when the id was INFERRED (from the registry's root/regime tables
    /// or an alias) rather than read from a stored `feature_set_id`. An
    /// inferred id is evidence about the artifact's NAME, never about its
    /// BYTES — every consumer must badge it.
    pub inferred: bool,
}

/// What kind of disagreement a [`check`] found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MismatchKind {
    /// The consumer's declared width differs from the producer's layout. The
    /// runtime already fails loud on this (`FeatureLenMismatch`); reported
    /// here so a dry check sees it too.
    LayoutDiffers,
    /// **The bug.** The consumer reads slots the producer does not populate.
    SlotsNotPopulated,
    /// Same slots, different extractor era — the values differ, and the shift
    /// is model-specific, so the number cannot be corrected, only re-verdicted.
    EraDiffers,
    /// One side's era is `unknown`. Never silently a match.
    EraUnknown,
    /// The consumer's COMPUTE tokens are not a subset of the producer's. A
    /// weaker, more legible restatement of `SlotsNotPopulated`, reported
    /// alongside it.
    ComputeNotCovered,
}

/// One disagreement, with the actionable detail.
#[derive(Debug, Clone)]
pub struct Mismatch {
    /// Which kind.
    pub kind: MismatchKind,
    /// One line naming exactly what differs.
    pub detail: String,
}

impl std::fmt::Display for Mismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.detail)
    }
}

/// Compare a CONSUMER (a bake) against a PRODUCER (a features root / grid).
///
/// Returns every disagreement, most-severe first. An empty result means the
/// read is sound on all three axes of `docs/FEATURE_SET_IDS.md` §4.
#[must_use]
pub fn check(consumer: &FeatureSetRef, producer: &FeatureSetRef) -> Vec<Mismatch> {
    let mut out = Vec::new();
    if consumer.id.layout_width() != producer.id.layout_width() {
        out.push(Mismatch {
            kind: MismatchKind::LayoutDiffers,
            detail: format!(
                "bake declares w{} but the table is w{}",
                consumer.id.layout_width(),
                producer.id.layout_width()
            ),
        });
    }
    if !producer.slots.covers(&consumer.slots) {
        let missing = producer.slots.missing_from(&consumer.slots);
        out.push(Mismatch {
            kind: MismatchKind::SlotsNotPopulated,
            detail: format!(
                "the bake READS {} slot(s) the table does not POPULATE: {missing}",
                missing.len()
            ),
        });
        let uncovered: Vec<&str> = ComputeToken::ALL
            .iter()
            .filter(|t| consumer.id.compute().contains(**t) && !producer.id.compute().contains(**t))
            .map(|t| t.as_str())
            .collect();
        if !uncovered.is_empty() {
            out.push(Mismatch {
                kind: MismatchKind::ComputeNotCovered,
                detail: format!(
                    "compute families read but not produced: {}",
                    uncovered.join("+")
                ),
            });
        }
    }
    if consumer.id.era() == ERA_UNKNOWN || producer.id.era() == ERA_UNKNOWN {
        out.push(Mismatch {
            kind: MismatchKind::EraUnknown,
            detail: format!(
                "era not established (bake {}, table {}) — 'we do not know which extractor made this' \
                 has the same consequence for a published number as 'a different one'",
                consumer.id.era(),
                producer.id.era()
            ),
        });
    } else if consumer.id.era() != producer.id.era() {
        out.push(Mismatch {
            kind: MismatchKind::EraDiffers,
            detail: format!(
                "bake trained on era {}, table is era {} — the shift is model-specific, so the \
                 number cannot be corrected across the boundary, only re-verdicted",
                consumer.id.era(),
                producer.id.era()
            ),
        });
    }
    out
}

/// Derive a bake's CONSUMER reference from its bytes.
///
/// Slots = the structurally-used caller lines
/// ([`crate::block_profile::used_caller_lines`], the SAME definition
/// `bake_block_profile` tabulates). COMPUTE = the families those reads fall
/// in. LAYOUT = `Model::caller_input_width()` — never `n_inputs()`, which is
/// the pruned internal width and is a third, different number.
pub fn bake_feature_set_ref(model: &Model, era: &str) -> Result<FeatureSetRef, String> {
    let used = crate::block_profile::used_caller_lines(model)?;
    let slots = SlotSet::from_slots(used);
    let width = model.caller_input_width();
    let compute = compute_parts_for_slots(&slots);
    let id = FeatureSetId::from_slots(compute, width, era, &slots)
        .ok_or_else(|| format!("invalid era token {era:?} (charset [a-z0-9_]+)"))?;
    Ok(FeatureSetRef {
        id,
        slots,
        source: "derived from bake bytes (structurally-used caller lines)".to_string(),
        inferred: false,
    })
}

/// The era a bake declares it was TRAINED on, from its embedded
/// `zentrain.feature_set_id` metadata, or [`ERA_UNKNOWN`].
///
/// The stored value is the PRODUCER id of the tables the bake trained on — the
/// bake's own consumer id is always derivable from its bytes, so it is not
/// stored.
#[must_use]
pub fn bake_declared_training_set(model: &Model) -> Option<FeatureSetId> {
    let md = model.metadata();
    let entry = md.get("zentrain.feature_set_id")?;
    FeatureSetId::parse(core::str::from_utf8(entry.value).ok()?.trim())
}

/// Classify a slot set into COMPUTE tokens: a token is named when the set
/// touches its family at all.
///
/// Family granularity is deliberate — this is the NAME, and the exact read set
/// is the hash. The two scattered tranches (`moments` / `classc`) are named
/// only when their owning block is NOT also touched, mirroring
/// `ComputeSet::raw_moments` / `bounded_err`: when `append` runs it owns
/// `GLOBAL_*`, so a set covering the append block is `append`, not
/// `append+moments`.
#[must_use]
pub fn compute_parts_for_slots(slots: &SlotSet) -> ComputeParts {
    use ComputeToken as T;
    let reg = registry();
    let touches = |tok: T| -> bool {
        reg.token_slots(tok)
            .map(|s| !s.intersect(slots).is_empty())
            .unwrap_or(false)
    };
    let mut p = ComputeParts::EMPTY;
    for t in [T::Basic, T::Peaks, T::Masked, T::Iw, T::V2, T::Append, T::Append2, T::Csfw] {
        if touches(t) {
            p = p.with(t);
        }
    }
    // `carriers` is a SUBSET of the pool block: name it only when the pool
    // reads are exactly the ten carrier slots (otherwise `peaks`/`masked`/`iw`
    // already describe them).
    if let (Some(carriers), Some(pool)) = (reg.token_slots(T::Carriers), reg.pool_block()) {
        if !slots.intersect(pool).is_empty() && slots.intersect(pool) == *carriers {
            p = ComputeParts::EMPTY
                .with(T::Carriers)
                .with_all_except_pools(p);
        }
    }
    if !p.contains(T::Append)
        && let Some(m) = reg.token_slots(T::Moments)
        && !m.intersect(slots).is_empty()
    {
        p = p.with(T::Moments);
    }
    if !p.contains(T::V2)
        && let Some(c) = reg.token_slots(T::ClassC)
        && !c.intersect(slots).is_empty()
    {
        p = p.with(T::ClassC);
    }
    p
}

/// Small local extension so `compute_parts_for_slots` can swap the pool
/// tokens for `carriers` without reaching into `ComputeParts`' bits.
trait ComputePartsExt {
    fn with_all_except_pools(self, other: ComputeParts) -> ComputeParts;
}
impl ComputePartsExt for ComputeParts {
    fn with_all_except_pools(self, other: ComputeParts) -> ComputeParts {
        use ComputeToken as T;
        let mut out = self;
        for t in other.iter() {
            if !matches!(t, T::Peaks | T::Masked | T::Iw) {
                out = out.with(t);
            }
        }
        out
    }
}

/// Resolve a features root's PRODUCER reference.
///
/// Order, most authoritative first:
/// 1. the root's `_MANIFEST.json` `"feature_set_id"` key (asserted, not
///    inferred) — what every new artifact carries per the design's §6.1;
/// 2. the registry's `roots` table, keyed on the exact path (INFERRED);
/// 3. the root's `_MANIFEST.json` `"regime"` prose string, via the registry's
///    `regime_strings` table (INFERRED).
///
/// `None` when nothing resolves — which callers must report as "era not
/// established", never as a pass.
#[must_use]
pub fn root_feature_set_ref(root: &Path) -> Option<FeatureSetRef> {
    let reg = registry();
    let manifest = std::fs::read_to_string(root.join("_MANIFEST.json")).ok();
    // 1. an asserted, stored id.
    if let Some(txt) = manifest.as_deref()
        && let Some(v) = json_string_field(txt, "feature_set_id")
        && let Some(id) = FeatureSetId::parse(v.trim())
    {
        // The slot set comes from the registry when the id is registered; an
        // unregistered id still identifies itself, with an empty slot set that
        // makes every coverage check report rather than silently pass.
        let slots = reg
            .set(&id.to_string())
            .and_then(|s| s.as_ref().map(|r| r.slots.clone()))
            .unwrap_or_default();
        return Some(FeatureSetRef {
            id,
            slots,
            source: format!("{}/_MANIFEST.json feature_set_id", root.display()),
            inferred: false,
        });
    }
    // 2. the registry's root table.
    let key = root.to_string_lossy();
    let key = key.trim_end_matches('/');
    if let Some(set_id) = reg.roots.get(key)
        && let Some(set) = reg.set(set_id)
        && let Some(r) = set.as_ref()
    {
        let mut r = r.clone();
        r.source = format!("registry roots[{key}]");
        r.inferred = true;
        return Some(r);
    }
    // 3. the manifest's prose regime string.
    if let Some(txt) = manifest.as_deref()
        && let Some(regime) = json_string_field(txt, "regime")
        && let Some((compute, layout, slots)) = reg.regime(regime.trim())
    {
        let era = reg
            .era_for_root(key)
            .unwrap_or_else(|| ERA_UNKNOWN.to_string());
        let id = FeatureSetId::from_slots(compute, layout, &era, &slots)?;
        return Some(FeatureSetRef {
            id,
            slots,
            source: format!("{}/_MANIFEST.json regime {regime:?}", root.display()),
            inferred: true,
        });
    }
    None
}

/// Narrow scan for a top-level `"key": "value"` string — the same deliberate
/// non-serde shape `bake_verdict::root_declared_regime` uses, for the same
/// reason: these manifests have several historical shapes and the guard needs
/// one string.
fn json_string_field(txt: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let i = txt.find(&needle)?;
    let rest = &txt[i + needle.len()..];
    let c = rest.find(':')?;
    let after = &rest[c + 1..];
    let q1 = after.find('"')?;
    let after = &after[q1 + 1..];
    let q2 = after.find('"')?;
    Some(after[..q2].to_string())
}

/// The parsed registry.
#[derive(Debug)]
pub struct Registry {
    sets: BTreeMap<String, RegisteredSet>,
    /// features-root path → set key.
    pub roots: BTreeMap<String, String>,
    regimes: BTreeMap<String, (ComputeParts, usize, SlotSet)>,
    aliases: BTreeMap<String, Vec<String>>,
    token_slots: BTreeMap<&'static str, SlotSet>,
}

impl Registry {
    /// A registered set by its key (the canonical id string, or the class
    /// form for a `class` entry).
    #[must_use]
    pub fn set(&self, key: &str) -> Option<&RegisteredSet> {
        self.sets.get(key)
    }
    /// Every registered set, key-ordered.
    pub fn sets(&self) -> impl Iterator<Item = (&str, &RegisteredSet)> {
        self.sets.iter().map(|(k, v)| (k.as_str(), v))
    }
    /// Every alias, name-ordered.
    pub fn aliases(&self) -> impl Iterator<Item = (&str, &[String])> {
        self.aliases.iter().map(|(k, v)| (k.as_str(), v.as_slice()))
    }
    /// The ids a legacy name has meant. Length > 1 is AMBIGUOUS by
    /// construction, and a consumer must say so rather than pick one.
    #[must_use]
    pub fn aliases_for(&self, name: &str) -> &[String] {
        self.aliases.get(name).map(Vec::as_slice).unwrap_or(&[])
    }
    /// The canonical slot set of one COMPUTE token, at 4 scales.
    #[must_use]
    pub fn token_slots(&self, t: ComputeToken) -> Option<&SlotSet> {
        self.token_slots.get(t.as_str())
    }
    /// The whole v1 pool block (`peaks ∪ masked ∪ iw`).
    #[must_use]
    pub fn pool_block(&self) -> Option<&SlotSet> {
        self.token_slots.get("__pool_block")
    }
    /// Resolve a manifest `regime` prose string to `(compute, layout, slots)`.
    #[must_use]
    pub fn regime(&self, s: &str) -> Option<(ComputeParts, usize, SlotSet)> {
        self.regimes.get(s).cloned()
    }
    /// The era token registered for a features-root path, via the `roots`
    /// table's target set.
    #[must_use]
    pub fn era_for_root(&self, path: &str) -> Option<String> {
        let key = self.roots.get(path)?;
        Some(self.sets.get(key)?.id.era.clone())
    }
}

/// The embedded registry, parsed once.
#[must_use]
pub fn registry() -> &'static Registry {
    use std::sync::OnceLock;
    static REG: OnceLock<Registry> = OnceLock::new();
    REG.get_or_init(|| parse_registry(REGISTRY_JSON).expect("the COMMITTED registry must parse"))
}

fn parse_registry(txt: &str) -> Result<Registry, String> {
    let v: serde_json::Value =
        serde_json::from_str(txt).map_err(|e| format!("feature_sets_registry.json: {e}"))?;
    let mut token_slots = BTreeMap::new();
    if let Some(map) = v.get("compute_tokens").and_then(|x| x.as_object()) {
        for (k, e) in map {
            if let Some(s) = e.get("slots").and_then(|x| x.as_str())
                && let Some(set) = SlotSet::parse(s)
                && let Some(tok) = ComputeToken::parse(k)
            {
                token_slots.insert(tok.as_str(), set);
            }
        }
    }
    // The v1 pool block, composed from its three registered tokens — never a
    // second literal.
    let pool = [ComputeToken::Peaks, ComputeToken::Masked, ComputeToken::Iw]
        .iter()
        .filter_map(|t| token_slots.get(t.as_str()))
        .fold(SlotSet::default(), |acc, s| acc.union(s));
    token_slots.insert("__pool_block", pool);

    let mut sets = BTreeMap::new();
    if let Some(map) = v.get("sets").and_then(|x| x.as_object()) {
        for (key, e) in map {
            let compute = e
                .get("compute")
                .and_then(|x| x.as_str())
                .and_then(ComputeParts::parse)
                .ok_or_else(|| format!("{key}: bad compute"))?;
            let layout = e
                .get("layout")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| format!("{key}: bad layout"))? as usize;
            let era = e
                .get("era")
                .and_then(|x| x.as_str())
                .ok_or_else(|| format!("{key}: bad era"))?
                .to_string();
            let id = ClassId {
                compute,
                layout,
                era: era.clone(),
            };
            let pinned = match e.get("slots").and_then(|x| x.as_str()) {
                None => None,
                Some(s) => {
                    let slots =
                        SlotSet::parse(s).ok_or_else(|| format!("{key}: unparseable slots"))?;
                    let fid = FeatureSetId::from_slots(compute, layout, &era, &slots)
                        .ok_or_else(|| format!("{key}: bad era token"))?;
                    let stored = e.get("slots_hash8").and_then(|x| x.as_str());
                    if let Some(h) = stored
                        && h != format!("{:08x}", fid.slots_hash())
                    {
                        return Err(format!(
                            "{key}: stored slots_hash8 {h} != owner {:08x}",
                            fid.slots_hash()
                        ));
                    }
                    Some(FeatureSetRef {
                        id: fid,
                        slots,
                        source: format!("registry sets[{key}]"),
                        inferred: false,
                    })
                }
            };
            sets.insert(
                key.clone(),
                RegisteredSet {
                    key: key.clone(),
                    id,
                    n_slots: e
                        .get("n_slots")
                        .and_then(serde_json::Value::as_u64)
                        .map(|n| n as usize),
                    role: e
                        .get("role")
                        .and_then(|x| x.as_str())
                        .unwrap_or("producer")
                        .to_string(),
                    legacy_name: e
                        .get("legacy_name")
                        .and_then(|x| x.as_str())
                        .unwrap_or("")
                        .to_string(),
                    evidence: e
                        .get("evidence")
                        .and_then(|x| x.as_str())
                        .unwrap_or("")
                        .to_string(),
                    pinned,
                },
            );
        }
    }
    let mut roots = BTreeMap::new();
    if let Some(map) = v.get("roots").and_then(|x| x.as_object()) {
        for (k, e) in map {
            if let Some(s) = e.as_str() {
                roots.insert(k.clone(), s.to_string());
            }
        }
    }
    let mut regimes = BTreeMap::new();
    if let Some(map) = v.get("regime_strings").and_then(|x| x.as_object()) {
        for (k, e) in map {
            let (Some(c), Some(l), Some(s)) = (
                e.get("compute").and_then(|x| x.as_str()).and_then(ComputeParts::parse),
                e.get("layout").and_then(serde_json::Value::as_u64),
                e.get("slots").and_then(|x| x.as_str()).and_then(SlotSet::parse),
            ) else {
                continue;
            };
            regimes.insert(k.clone(), (c, l as usize, s));
        }
    }
    let mut aliases = BTreeMap::new();
    if let Some(map) = v.get("aliases").and_then(|x| x.as_object()) {
        for (k, e) in map {
            let ids: Vec<String> = e
                .get("ids")
                .and_then(|x| x.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect())
                .unwrap_or_default();
            aliases.insert(k.clone(), ids);
        }
    }
    Ok(Registry {
        sets,
        roots,
        regimes,
        aliases,
        token_slots,
    })
}
