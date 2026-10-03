//! REV4SERVE gate (2026-10-02, SWE-2): served Rev4 entries == the canonical
//! `research::extract` arithmetic, BIT FOR BIT, on every SIMD tier.
//!
//! Own test executable: `for_each_token_permutation` mutates process-wide
//! SIMD dispatch, so tier-forced probes may never share a process with the
//! parallel unit tests (`rev4_featbank_parity.rs` convention), and
//! `ZENSIM_FORMULA_REV` is a `OnceLock`, so each pinned revision's body owns
//! a child process via the `at_revision` re-exec protocol.
//!
//! * **G-SERVE** — every served feature-bearing entry at Rev4 emits the
//!   same bits as `research::extract(&Request::everything())` on the same
//!   pixels, under every dispatchable token permutation: the fold entry,
//!   the buffered and strip engines of `Zensim::compute`, the ref-cached
//!   compare, the declared-HDR fold entry and the PU-linear v1 entry.
//! * **G-MAPS** — Rev4 diffmap and attribution density are bit-identical
//!   across permutations (the maps have no research twin; tier-identity is
//!   the canonicality proof).
//! * **G-BAKE** — corpus-gated (`#[ignore]`): the real featpot Rev4 bake
//!   (`set:v2+basic@h32:H128__N/without_aic3_s0`) is loaded, served from
//!   pixels by `BakeScorer` — score AND feature row bit-identical to
//!   `score_features` over `research::extract(&Request::for_bake_bytes())` —
//!   and steered by `prepare_steering` on its held-out aic3 pixels.
//! * **G-REG** — `ZENSIM_REV4SERVE_CAPTURE=<tsv>` capture harness for the
//!   Rev1–Rev3 byte-identity gate: run under each pinned revision before
//!   and after this change set; the capture lines must be identical.

#![cfg(feature = "feature-regime-v2")]

mod common;

use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

/// Fewest dispatch permutations a host must run for the tier-parity checks to mean anything: x86_64 has at least
/// scalar plus two SIMD tiers, aarch64 scalar plus NEON, every other target (i686, wasm32, …) only its single tier.
/// Every permutation that runs is still checked bit for bit; this only guards against a vacuous pass.
fn min_tier_permutations() -> usize {
    if cfg!(target_arch = "x86_64") {
        3
    } else if cfg!(target_arch = "aarch64") {
        2
    } else {
        1
    }
}
use zensim::feature_v2::{V2NewFeatureToggles, V2Scratch};
use zensim::fold_engine::ScoringEngine;
use zensim::research::{self, Request};
use zensim::source::{AlphaMode, ImageSource, PixelFormat};
use zensim::{BakeScorer, RgbSlice, Zensim, ZensimProfile};

// ─── Revision pinning (same protocol as featcanon_rev4_contract.rs) ────

fn at_revision(rev: Option<&str>, test_path: &str, sentinel: &str) -> bool {
    if std::env::var("ZENSIM_FORMULA_REV").ok().as_deref() == rev
        && std::env::var("REV4SERVE_GATE_CHILD").is_ok()
    {
        return true;
    }
    let exe = std::env::current_exe().expect("test binary path");
    let mut cmd = std::process::Command::new(exe);
    cmd.args([
        test_path,
        "--exact",
        "--nocapture",
        "--test-threads=1",
        "--include-ignored",
    ])
    .env("REV4SERVE_GATE_CHILD", "1");
    match rev {
        Some(r) => cmd.env("ZENSIM_FORMULA_REV", r),
        None => cmd.env_remove("ZENSIM_FORMULA_REV"),
    };
    let out = cmd.output().expect("re-exec the test binary");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "{test_path} failed at ZENSIM_FORMULA_REV={rev:?}\n--- stdout ---\n{stdout}\n--- stderr ---\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        stdout.contains(sentinel),
        "{test_path} at ZENSIM_FORMULA_REV={rev:?} never reached its body (sentinel {sentinel:?} absent)\n{stdout}"
    );
    false
}

// ─── Shared fixtures ────────────────────────────────────────────────────

fn pair(w: usize, h: usize, seed: u32) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
    let r = common::generators::gen_value_noise(w, h, seed);
    let d = common::generators::distort_block_artifacts(&r, w, h);
    (r, d)
}

/// sRGB code → absolute linear luminance (cd/m², 203-nit SDR white) — the
/// same conversion `featcanon_rev4_contract.rs` uses for its declared-HDR
/// fixtures.
fn nits(c: u8) -> f32 {
    let v = c as f32 / 255.0;
    let lin = if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    };
    lin * 203.0
}

fn nits_rgb(px: &[[u8; 3]]) -> Vec<f32> {
    px.iter().flat_map(|p| p.map(nits)).collect()
}

/// A declared-HDR `LinearF32Rgba` source (absolute linear cd/m²) — the
/// `research::extract` HDR route input.
struct HdrLinear {
    bytes: Vec<u8>,
    width: usize,
    height: usize,
}

impl HdrLinear {
    fn new(px: &[[u8; 3]], width: usize, height: usize) -> Self {
        let bytes = px
            .iter()
            .flat_map(|p| {
                let [r, g, b] = p.map(nits);
                [r, g, b, 1.0f32]
            })
            .flat_map(f32::to_ne_bytes)
            .collect();
        Self {
            bytes,
            width,
            height,
        }
    }
}

impl ImageSource for HdrLinear {
    fn width(&self) -> usize {
        self.width
    }
    fn height(&self) -> usize {
        self.height
    }
    fn pixel_format(&self) -> PixelFormat {
        PixelFormat::LinearF32Rgba
    }
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Opaque
    }
    fn is_hdr(&self) -> bool {
        true
    }
    fn row_bytes(&self, y: usize) -> &[u8] {
        let row = self.width * 16;
        &self.bytes[y * row..(y + 1) * row]
    }
}

/// Full-width canonical extraction for the SDR pair (1825 slots).
fn extract_everything(src: &impl ImageSource, dst: &impl ImageSource) -> Vec<f64> {
    research::extract(&Request::everything(), src, dst)
        .expect("research::extract serves Rev4")
        .into_values()
}

/// Canonical extraction over the declared-HDR pair. `everything()`'s
/// restored-cut families assert SDR (pre-existing), so the HDR request is
/// the 944-wide identity — the same request the contract test uses.
fn extract_hdr(src: &impl ImageSource, dst: &impl ImageSource) -> Vec<f64> {
    research::extract(
        &Request::for_slots(
            zensim::feature_set_id::SlotSet::from_ranges([(0, 944)]),
            944,
        ),
        src,
        dst,
    )
    .expect("research::extract HDR serves Rev4")
    .into_values()
}

// ─── G-SERVE: every served entry == research::extract, every tier ──────

/// The slot set `toggles` populates under `NUM_SCALES = 4` at `width`.
fn populated(toggles: V2NewFeatureToggles, width: usize) -> zensim::feature_set_id::SlotSet {
    toggles.populated_slots(4, width)
}

/// The served feature-bearing entries for one pair. Each arm carries its
/// POPULATED slot set — positions outside it carry the walk's structural
/// fill, which is not part of the served contract.
fn served_feature_rows(
    src: &RgbSlice<'_>,
    dst: &RgbSlice<'_>,
) -> Vec<(String, zensim::feature_set_id::SlotSet, Vec<f64>)> {
    use zensim::feature_set_id::SlotSet;
    let mut out: Vec<(String, SlotSet, Vec<f64>)> = Vec::new();
    let z = Zensim::new(ZensimProfile::B).with_parallel(false);
    let mut scratch = V2Scratch::new();
    // The fold entries (research's own walk, through the served API): the
    // materialized append2 impl (944) and the streaming append impl (924).
    let app2 = V2NewFeatureToggles {
        append_block: true,
        append2_block: true,
        ..V2NewFeatureToggles::default()
    };
    out.push((
        "folded720_append2".into(),
        populated(app2, 944),
        z.compute_folded720_append2_features(src, dst)
            .expect("materialized fold entry serves Rev4")
            .features()
            .to_vec(),
    ));
    let app = V2NewFeatureToggles {
        append_block: true,
        ..V2NewFeatureToggles::default()
    };
    out.push((
        "folded720_append_streaming".into(),
        populated(app, 924),
        z.compute_folded720_append_features_streaming(
            src,
            dst,
            V2NewFeatureToggles::default(),
            &mut scratch,
        )
        .expect("fold entry serves Rev4")
        .features()
        .to_vec(),
    ));
    // The profile compute on BOTH engines — fold-backed (where backable)
    // and the forced-buffered strips walk. B's config (extended + IW) populates
    // all of v1's 372 slots.
    let v1_full = SlotSet::from_ranges([(0, 372)]);
    for (name, engine) in [
        ("fold", ScoringEngine::Fold),
        ("buffered", ScoringEngine::Buffered),
    ] {
        let res = Zensim::new(ZensimProfile::B)
            .with_parallel(false)
            .with_engine(engine)
            .compute(src, dst)
            .unwrap_or_else(|e| panic!("{name} engine compute serves Rev4: {e:?}"));
        out.push((
            format!("compute.{name}"),
            v1_full.clone(),
            res.features().to_vec(),
        ));
    }
    // Ref-cached compare (precompute + with_ref) — the batch scoring path.
    let pre = z
        .precompute_reference(src)
        .expect("precompute_reference serves Rev4");
    let res = z
        .compute_with_ref(&pre, dst)
        .expect("compute_with_ref serves Rev4");
    out.push((
        "compute_with_ref".into(),
        v1_full.clone(),
        res.features().to_vec(),
    ));
    // The 256-row strips entries merge per-strip ScaleAccumulators — an
    // epsilon-equivalent (not bit-identical) summation tree vs the fold's
    // tiling, so at Rev4 they refuse by name. (At small heights the f64
    // sums stay exact and coincidentally match; at 4MP they don't.)
    // Sub-64 in either dimension delegates to the buffered path before
    // the strips walk runs — the refusal only guards the strips walk.
    if dst.height() >= 64 && dst.width() >= 64 {
        let err = z
            .compute_with_ref_streaming_strips(&pre, dst, 256, 128)
            .expect_err("the strips walk must refuse at Rev4");
        assert!(
            format!("{err:?}").contains("strips walk merges"),
            "compute_with_ref_streaming_strips refused, but not the strips refusal: {err:?}"
        );
        let err = z
            .compute_streaming_strips(src, dst, 256, 128)
            .expect_err("the strips walk must refuse at Rev4");
        assert!(
            format!("{err:?}").contains("strips walk merges"),
            "compute_streaming_strips refused, but not the strips refusal: {err:?}"
        );
    }
    // The V2Bounded buffered walk is a DIFFERENT (non-canonical) engine —
    // at Rev4 it must refuse by name, not serve divergent bits.
    let err = z
        .compute_v2_features(src, dst)
        .expect_err("the V2Bounded walk must refuse at Rev4");
    assert!(
        format!("{err:?}").contains("V2Bounded"),
        "compute_v2_features refused, but not the V2Bounded refusal: {err:?}"
    );
    let err = z
        .compute_v2_features_with_toggles(src, dst, V2NewFeatureToggles::default())
        .expect_err("the V2Bounded walk must refuse at Rev4");
    assert!(
        format!("{err:?}").contains("V2Bounded"),
        "compute_v2_features_with_toggles refused, but not the V2Bounded refusal: {err:?}"
    );
    out
}

/// The served HDR entries (declared-HDR fold walk + PU-linear v1) for one
/// pair, against `extract_hdr`'s prefix.
fn served_hdr_rows(
    src: &[[u8; 3]],
    dst: &[[u8; 3]],
    w: usize,
    h: usize,
) -> Vec<(String, zensim::feature_set_id::SlotSet, Vec<f64>)> {
    use zensim::feature_set_id::SlotSet;
    let hs = HdrLinear::new(src, w, h);
    let hd = HdrLinear::new(dst, w, h);
    let z = Zensim::new(ZensimProfile::B).with_parallel(false);
    let mut scratch = V2Scratch::new();
    let mut out = Vec::new();
    let app2 = V2NewFeatureToggles {
        append_block: true,
        append2_block: true,
        ..V2NewFeatureToggles::default()
    };
    out.push((
        "folded720_append2_hdr".into(),
        populated(app2, 944),
        z.compute_folded720_append2_features_hdr(
            &hs,
            &hd,
            zensim::feature_v2::HdrEncoding::Linear,
            V2NewFeatureToggles::default(),
            &mut scratch,
        )
        .expect("HDR fold entry serves Rev4")
        .features()
        .to_vec(),
    ));
    // PU-linear v1 entry (interleaved nits) — B's HDR params are Full pools.
    let rn = nits_rgb(src);
    let dn = nits_rgb(dst);
    let res = z
        .compute_pu_linear(&rn, &dn, w, h, 3 * w, 3 * w)
        .expect("compute_pu_linear serves Rev4");
    out.push((
        "compute_pu_linear".into(),
        SlotSet::from_ranges([(0, 372)]),
        res.features().to_vec(),
    ));
    out
}

#[test]
fn rev4_served_vectors_bitmatch_research_extract_on_every_tier() {
    const SENTINEL: &str = "G_SERVE_OK";
    if !at_revision(
        Some("4"),
        "rev4_served_vectors_bitmatch_research_extract_on_every_tier",
        SENTINEL,
    ) {
        return;
    }
    let _lock = archmage::testing::lock_token_testing();
    // Odd widths, sub-64 (reflect-pad), non-multiple-of-8, multi-strip
    // heights (>128 rows), one >= 4 MP image.
    let geoms: [(usize, usize); 8] = [
        (64, 64),
        (97, 63),
        (200, 140),
        (127, 257),
        (40, 300),
        (33, 33),
        (2048, 2048), // 4.19 MP even — isolates odd-height last strip
        (2048, 2049), // 4.19 MP — the large-image arm
    ];
    for &(w, h) in &geoms {
        let (src, dst) = pair(w, h, 0x5EED ^ w as u32 ^ ((h as u32) << 16));
        let s = RgbSlice::new(&src, w, h);
        let d = RgbSlice::new(&dst, w, h);
        let want = extract_everything(&s, &d);
        let hdr_want = extract_hdr(&HdrLinear::new(&src, w, h), &HdrLinear::new(&dst, w, h));
        // The capture entry runs once per dispatch permutation; canon is
        // tier-independent, so `want` computed once is the right oracle for
        // every tier — the served side under each permutation must still
        // land on it exactly.
        let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            let mut bad = String::new();
            for (what, emit, got) in served_feature_rows(&s, &d) {
                for i in emit.iter_slots() {
                    let (i, g, wv) = (i, got[i], want[i]);
                    if g.to_bits() != wv.to_bits() && bad.len() < 4000 {
                        bad.push_str(&format!(
                            "\n  {what} slot {i}: {g:e} vs {wv:e} (bits {} vs {})",
                            g.to_bits(),
                            wv.to_bits()
                        ));
                    }
                }
            }
            for (what, emit, got) in served_hdr_rows(&src, &dst, w, h) {
                for i in emit.iter_slots() {
                    let (i, g, wv) = (i, got[i], hdr_want[i]);
                    if g.to_bits() != wv.to_bits() && bad.len() < 4000 {
                        bad.push_str(&format!(
                            "\n  {what} slot {i}: {g:e} vs {wv:e} (bits {} vs {})",
                            g.to_bits(),
                            wv.to_bits()
                        ));
                    }
                }
            }
            assert!(bad.is_empty(), "{w}x{h} tier {}:{bad}", perm.label);
        });
        eprintln!(
            "{w}x{h}: {} permutations served==research bit-for-bit",
            report.permutations_run
        );
        assert!(
            report.permutations_run >= min_tier_permutations(),
            "tier coverage too thin: {} permutations",
            report.permutations_run
        );
    }
    println!("{SENTINEL}");
}

// ─── G-MAPS: Rev4 diffmap / attribution density tier-identical ─────────

#[test]
fn rev4_maps_are_tier_identical() {
    const SENTINEL: &str = "G_MAPS_OK";
    if !at_revision(Some("4"), "rev4_maps_are_tier_identical", SENTINEL) {
        return;
    }
    let _lock = archmage::testing::lock_token_testing();
    // Per-geometry references — a tier's map is compared against the same
    // geometry's first-tier capture, never across geometries.
    type GeoMap = (usize, usize, Vec<u32>);
    let mut first: Vec<GeoMap> = Vec::new();
    #[cfg(feature = "custom-profiles")]
    let mut first_attr: Vec<GeoMap> = Vec::new();
    let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        for &(w, h) in &[(97usize, 63usize), (130, 130)] {
            let (src, dst) = pair(w, h, 0xABBA ^ w as u32);
            let s = RgbSlice::new(&src, w, h);
            let d = RgbSlice::new(&dst, w, h);
            let z = Zensim::new(ZensimProfile::B).with_parallel(false);
            // Diffmap: v2 tail weighting.
            let s_v2 = vec![0.5f64; 8 * 3 * zensim::feature_v2::FEATURES_PER_CHANNEL_V2_TOTAL];
            let map = z
                .compute_v2_diffmap(&s, &d, &s_v2)
                .expect("diffmap serves Rev4");
            let bits: Vec<u32> = map.iter().map(|v| v.to_bits()).collect();
            match first.iter_mut().find(|(gw, gh, _)| (*gw, *gh) == (w, h)) {
                None => first.push((w, h, bits)),
                Some((_, _, f)) => assert_eq!(
                    *f, bits,
                    "{w}x{h}: diffmap differs across tiers (tier {})",
                    perm.label
                ),
            }
            // Attribution density over the same pair (basic 156 weighting).
            #[cfg(feature = "custom-profiles")]
            {
                let pre = z.precompute_reference(&s).expect("precompute");
                let sv = vec![-0.5f64; 156];
                let (_, attr) = z
                    .compute_with_ref_score_and_attribution(&pre, &d, &sv)
                    .expect("attribution serves Rev4");
                let bits: Vec<u32> = attr.density().iter().map(|v| v.to_bits()).collect();
                match first_attr
                    .iter_mut()
                    .find(|(gw, gh, _)| (*gw, *gh) == (w, h))
                {
                    None => first_attr.push((w, h, bits)),
                    Some((_, _, f)) => assert_eq!(
                        *f, bits,
                        "{w}x{h}: attribution density differs across tiers (tier {})",
                        perm.label
                    ),
                }
            }
        }
    });
    eprintln!(
        "{} permutations, maps tier-identical",
        report.permutations_run
    );
    assert!(
        report.permutations_run >= min_tier_permutations(),
        "tier coverage too thin: {} permutations",
        report.permutations_run
    );
    println!("{SENTINEL}");
}

// ─── G-BAKE: the real featpot Rev4 bake served from pixels ─────────────

/// Default gate bake: the E9″-lane v2+basic POTENTIAL cell, aic3 held out.
const FEATPOT_BAKE: &str =
    "/var/tmp/rev4-featpot/v2c/cells/set:v2+basic@h32:H128__N/without_aic3_s0/refit/last.bin";
const AIC3_ORIGINALS: &str = "/mnt/v/dataset/aic3_ctc_epfl/original";
const AIC3_DECODED: &str = "/mnt/v/dataset/aic3_ctc_epfl/decoded";

fn decode_png_rgb8(path: &std::path::Path) -> (Vec<[u8; 3]>, usize, usize) {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("decode {path:?}: {e}"))
        .to_rgb8();
    let px = img
        .as_raw()
        .as_chunks::<3>()
        .0
        .iter()
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    (px, img.width() as usize, img.height() as usize)
}

/// Up to `n` held-out aic3 pairs (original vs its decoded variants).
type RgbPair = (Vec<[u8; 3]>, Vec<[u8; 3]>, usize, usize);

fn aic3_pairs(n: usize) -> Vec<RgbPair> {
    let mut out = Vec::new();
    let mut refs: Vec<_> = std::fs::read_dir(AIC3_ORIGINALS)
        .expect("aic3 originals")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "png"))
        .collect();
    refs.sort();
    'refs: for r in refs {
        let stem = r.file_stem().unwrap().to_string_lossy().to_string();
        let dir = std::path::Path::new(AIC3_DECODED).join(&stem);
        let Ok(read) = std::fs::read_dir(&dir) else {
            continue;
        };
        let mut dists: Vec<_> = read
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "png"))
            .collect();
        dists.sort();
        let (src, w, h) = decode_png_rgb8(&r);
        for dp in dists.iter().take(3) {
            let (dst, dw, dh) = decode_png_rgb8(dp);
            assert_eq!((w, h), (dw, dh), "dim mismatch {r:?} vs {dp:?}");
            out.push((src.clone(), dst, w, h));
            if out.len() >= n {
                break 'refs;
            }
        }
    }
    out
}

/// The Rev4 v2+basic featpot bake SERVED: pixel scoring matches the
/// feature-space forward (`score_features` over `research::extract` at the
/// bake's declared read set) BIT FOR BIT on >= 20 held-out aic3 pairs, and
/// `prepare_steering` serves it (scalar score = same features, map = the
/// attribution path). Runs across forced tiers for the map-identity arm.
#[test]
#[ignore = "corpus + bake gate: run `just rev4serve-gate` (requires /mnt/v aic3 + /var/tmp/rev4-featpot; REV4SERVE_BAKE overrides the bake path)"]
fn rev4_featpot_bake_served_and_steered() {
    const SENTINEL: &str = "G_BAKE_OK";
    if !at_revision(Some("4"), "rev4_featpot_bake_served_and_steered", SENTINEL) {
        return;
    }
    let _lock = archmage::testing::lock_token_testing();
    let bake_path = std::env::var("REV4SERVE_BAKE").unwrap_or_else(|_| FEATPOT_BAKE.to_string());
    let bake = std::fs::read(&bake_path).expect("read featpot bake");
    // The v2c cells predate the trainer's stamp: every input table in the
    // bake's `zentrain.repro` declares `formula_revision: 4`, but
    // `table_admission.formula_revision` resolved null ("feature-set
    // identity is unknown"), so no `zentrain.formula_revision` key was
    // written and the bake parses as pre-stamp Rev1. Stamping a copy is
    // exactly the declaration the stamp exists to carry — the bake's own
    // read set is fold-walk slots on Rev4 tables.
    let bake = zenpredict_bake::append_metadata_utf8(&bake, "zentrain.formula_revision", "4")
        .expect("stamp the gate bake's declared revision");
    let model = zenpredict::Model::from_bytes(&bake).expect("bake parses");
    assert_eq!(
        zensim::feature_v2::bake_formula_revision_public(&model).expect("revision"),
        zensim::feature_v2::FormulaRevision::Rev4,
        "the gate bake must declare Rev4"
    );
    let pairs = aic3_pairs(24);
    assert!(pairs.len() >= 20, "need >= 20 held-out pairs");
    let request = Request::for_bake_bytes(&bake).expect("the bake's read set is plannable");
    let mut scorer = BakeScorer::new(&model).expect("Rev4 bake loads in a Rev4 process");
    scorer = scorer.with_parallel(false);
    for (i, (src, dst, w, h)) in pairs.iter().enumerate() {
        let s = RgbSlice::new(src, *w, *h);
        let d = RgbSlice::new(dst, *w, *h);
        let served = scorer.compute(&s, &d, None).unwrap_or_else(|e| {
            panic!("pair {i}: BakeScorer::compute serves the Rev4 bake: {e:?}")
        });
        let ext = research::extract(&request, &s, &d)
            .unwrap_or_else(|e| panic!("pair {i}: research::extract at Rev4: {e}"));
        let feature_score = scorer
            .score_features(ext.values(), *w as u32, *h as u32, None)
            .unwrap_or_else(|e| panic!("pair {i}: score_features: {e:?}"));
        assert_eq!(
            served.score().to_bits(),
            feature_score.to_bits(),
            "pair {i} {w}x{h}: pixel score {} vs feature score {}",
            served.score(),
            feature_score
        );
        // The served row and the extraction row must agree on every slot
        // the bake READS. (The plan's pool-promotion policy legitimately
        // leaves live values in slots outside the read set where the
        // extraction emits structural zeros — not part of the contract.)
        for k in request.want().iter_slots() {
            assert_eq!(
                served.features()[k].to_bits(),
                ext.values()[k].to_bits(),
                "pair {i}: served feature slot {k} differs from research::extract \
                 ({:e} vs {:e})",
                served.features()[k],
                ext.values()[k]
            );
        }
    }
    eprintln!(
        "{} held-out pairs: pixel score == feature score, bit for bit",
        pairs.len()
    );

    // Steering: the session's scalar score must equal BakeScorer::compute's
    // (same features, same forward), and the density map is served.
    // (`prepare_steering`/`SteeringSession` are custom-profiles gated.)
    #[cfg(feature = "custom-profiles")]
    {
        let (src, dst, w, h) = &pairs[0];
        let s = RgbSlice::new(src, *w, *h);
        let d = RgbSlice::new(dst, *w, *h);
        let mut session = scorer
            .prepare_steering(&s, 1)
            .expect("prepare_steering serves the Rev4 v2+basic bake");
        let steered = session
            .compute(&d, None)
            .expect("steering compute serves Rev4");
        let direct = scorer.compute(&s, &d, None).expect("direct compute");
        assert_eq!(
            steered.result().score().to_bits(),
            direct.score().to_bits(),
            "steering scalar score must equal the pixel-path score bit for bit"
        );
        assert_eq!(
            steered.attribution().density().len(),
            w * h,
            "the steering map covers the full image"
        );
        assert!(
            steered
                .attribution()
                .density()
                .iter()
                .all(|v| v.is_finite()),
            "the steering map must be finite"
        );
    }
    println!("{SENTINEL}");
}

// ─── G-REG: Rev1–Rev3 byte-identity capture harness ────────────────────
//
// `ZENSIM_REV4SERVE_CAPTURE=<tsv>` turns the body into a capture run: every
// pair's served outputs are hashed/written. Run under each pinned revision
// on the base build (before) and this tree (after); the files must be
// identical. 120 deterministic pairs per revision.

fn fnv64(bits: &[u64]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in bits {
        h ^= b;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn hash_f64s(v: &[f64]) -> u64 {
    fnv64(&v.iter().map(|x| x.to_bits()).collect::<Vec<_>>())
}

fn hash_f32s(v: &[f32]) -> u64 {
    fnv64(&v.iter().map(|x| x.to_bits() as u64).collect::<Vec<_>>())
}

/// Run one capture entry, recording panics and errors as markers: byte
/// identity of behavior includes identical failures (e.g. Rev2's clamp
/// form can panic on out-of-range `d` — that must reproduce exactly).
fn entry(f: impl FnOnce() -> Result<String, zensim::ZensimError>) -> String {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(Ok(s)) => s,
        Ok(Err(_)) => "ERR".to_string(),
        Err(_) => "PANIC".to_string(),
    }
}

fn capture_lines(rev: &str) -> Vec<String> {
    let mut lines = vec![format!("# rev={rev} rev4serve-gate-v1")];
    for i in 0..120usize {
        // Deterministic, varied geometries: odd, sub-64, wide, tall.
        let w = 64 + (i * 37) % 160;
        let h = 48 + (i * 53) % 120;
        let (src, dst) = pair(w, h, 0xC001 ^ i as u32);
        let s = RgbSlice::new(&src, w, h);
        let d = RgbSlice::new(&dst, w, h);
        let z = Zensim::new(ZensimProfile::B).with_parallel(false);
        let l = entry(|| {
            let res = z.compute(&s, &d)?;
            Ok(format!(
                "score={:016x}\tfeats={:016x}",
                res.score().to_bits(),
                hash_f64s(res.features())
            ))
        });
        lines.push(format!("{i}\t{l}"));
        let l = entry(|| {
            let res = Zensim::new(ZensimProfile::B)
                .with_parallel(false)
                .with_engine(ScoringEngine::Fold)
                .compute(&s, &d)?;
            Ok(format!(
                "fold_score={:016x}\tfold_feats={:016x}",
                res.score().to_bits(),
                hash_f64s(res.features())
            ))
        });
        lines.push(format!("{i}\t{l}"));
        let l = entry(|| {
            let pre = z.precompute_reference(&s)?;
            let res = z.compute_with_ref(&pre, &d)?;
            Ok(format!("ref_score={:016x}", res.score().to_bits()))
        });
        lines.push(format!("{i}\t{l}"));
        let s_v2 = vec![0.5f64; 8 * 3 * zensim::feature_v2::FEATURES_PER_CHANNEL_V2_TOTAL];
        let l = entry(|| {
            let map = z.compute_v2_diffmap(&s, &d, &s_v2)?;
            Ok(format!("diffmap={:016x}", hash_f32s(&map)))
        });
        lines.push(format!("{i}\t{l}"));
        // HDR front end (canonical at every revision it serves).
        let rn = nits_rgb(&src);
        let dn = nits_rgb(&dst);
        let l = entry(|| {
            let res = z.compute_pu_linear(&rn, &dn, w, h, 3 * w, 3 * w)?;
            Ok(format!(
                "pu_score={:016x}\tpu_feats={:016x}",
                res.score().to_bits(),
                hash_f64s(res.features())
            ))
        });
        lines.push(format!("{i}\t{l}"));
    }
    lines
}

fn run_capture(rev: &str, test_path: &str) {
    let sentinel = format!("CAPTURE_{}_DONE", rev);
    if !at_revision(Some(rev), test_path, &sentinel) {
        return;
    }
    let lines = capture_lines(rev);
    if let Ok(path) = std::env::var("ZENSIM_REV4SERVE_CAPTURE") {
        std::fs::write(format!("{path}.rev{rev}"), lines.join("\n") + "\n").expect("write capture");
    }
    println!("{sentinel}");
}

#[test]
fn rev1_baseline_capture() {
    run_capture("1", "rev1_baseline_capture");
}

#[test]
fn rev2_baseline_capture() {
    run_capture("2", "rev2_baseline_capture");
}

#[test]
fn rev3_baseline_capture() {
    run_capture("3", "rev3_baseline_capture");
}
