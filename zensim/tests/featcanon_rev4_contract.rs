//! The Rev4 CONTRACT, as opposed to its arithmetic (`featcanon_tier_parity.rs`
//! owns the arithmetic): featcanon-fix D1 (2026-09-26) and the REV4SERVE
//! revision of D2 (2026-10-02).
//!
//! * **D1 — no silent mixing.** A Rev1 bake served in a `ZENSIM_FORMULA_REV=4`
//!   process used to be accepted and scored with Rev4 features (measured by
//!   the featcanon review: the D bake, 12/12 pairs). A Rev4 request in a
//!   non-Rev4 process used to be accepted and computed as Rev3. Both must now
//!   be errors.
//! * **D2 — Rev4 is served.** REV4SERVE made every leaf a Rev4 served
//!   computation reaches canonical (`det_math` mid-precision transcendentals,
//!   `pu21_encode_canon`, `pu_xyb_canon`, revision-aware HDR transfer decode,
//!   and the all-SSIM admission that keeps Rev4 off the tier-dispatched
//!   edge-only/MSE-only strips routes). A `ZENSIM_FORMULA_REV=4` process
//!   therefore serves every entry that resolves to Rev4 — `Zensim::*`,
//!   the HDR/PU entries, precompute, diffmap, and `research::extract` on
//!   declared-HDR pairs too — while a bake or request that names another
//!   revision still refuses at the Rev4 boundary (D1), and a Rev4 bake that
//!   needs a wide, unproven family refuses by name.
//!
//! Each assertion block also runs below Rev4 and must SUCCEED there, so a
//! refusal can never pass because the fixture itself is broken.
//!
//! `ZENSIM_FORMULA_REV` is a `OnceLock`, so each revision's assertions own a
//! child process ([`at_revision`], the `ssim_form::run_at_revision`
//! protocol): the parent re-executes this binary with the variable set (or
//! removed) and requires a sentinel on the child's stdout.

#![allow(deprecated)]

use zensim::feature_v2::{FormulaRevision, V2NewFeatureToggles};
use zensim::research::{self, Request};
use zensim::source::{AlphaMode, ImageSource, PixelFormat};
use zensim::{BakeScorer, RgbSlice, Zensim, ZensimProfile};

/// The shipped D bake (declares revision 1) — the review's direction-1 case.
const D_BAKE: &[u8] =
    include_bytes!("../weights/d_sdr_add156_id100_negrich_dial_byid_2026-09-06.bin");
const B_BAKE: &[u8] =
    include_bytes!("../weights/b_sdr_linear_cid80_inclwinsor_dense_dial_byid_2026-09-06.bin");
const C_BAKE: &[u8] = include_bytes!("../weights/c_sdr_purity944_byid_2026-09-07.bin");
const BHDR_BAKE: &[u8] =
    include_bytes!("../weights/bhdr_linear_shaped_cvvdpmix_byid_2026-09-06.bin");
const BAKES: &[(&str, &[u8])] = &[
    ("D", D_BAKE),
    ("B", B_BAKE),
    ("C", C_BAKE),
    ("BHdr", BHDR_BAKE),
];

/// What `ssim_form::refuse_rev4_mix` says — the D1 boundary every public
/// entry still enforces. (`ssim_form::REV4_MIX` is `pub(crate)`; this is a
/// substring of it.)
const MIX_REASON: &str = "cannot be mixed";

/// Run the calling test's body in a process whose `ZENSIM_FORMULA_REV` is
/// `rev` (`None` = unset). Returns `true` in that process; otherwise
/// re-executes this binary for exactly `test_path`, asserts success and the
/// sentinel, and returns `false`.
fn at_revision(rev: Option<&str>, test_path: &str, sentinel: &str) -> bool {
    if std::env::var("ZENSIM_FORMULA_REV").ok().as_deref() == rev
        && std::env::var("FEATCANON_CONTRACT_CHILD").is_ok()
    {
        return true;
    }
    let exe = std::env::current_exe().expect("test binary path");
    let mut cmd = std::process::Command::new(exe);
    cmd.args([test_path, "--exact", "--nocapture", "--test-threads=1"])
        .env("FEATCANON_CONTRACT_CHILD", "1");
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

/// Textured, distorted pair — real SSIM activity at every scale.
fn pair(w: usize, h: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
    let mut src = vec![[0u8; 3]; w * h];
    let mut dst = vec![[0u8; 3]; w * h];
    for y in 0..h {
        for x in 0..w {
            let g = ((x * 7 + y * 13) % 251) as u8;
            let t = (((x * 31) ^ (y * 17)) % 67) as u8;
            let s = g.wrapping_add(t);
            src[y * w + x] = [s, g.wrapping_add(t / 2), 255u8 - s];
            let dv = (s as i32 + ((x * y) % 29) as i32 - 14).clamp(0, 255) as u8;
            dst[y * w + x] = [dv, g.wrapping_add(t / 3), 255u8 - dv];
        }
    }
    (src, dst)
}

/// sRGB code → absolute linear luminance (cd/m², 203-nit SDR white).
fn nits(c: u8) -> f32 {
    let v = c as f32 / 255.0;
    let lin = if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    };
    lin * 203.0
}

/// Interleaved RGB nits for `compute_pu_linear`.
fn nits_rgb(px: &[[u8; 3]]) -> Vec<f32> {
    px.iter().flat_map(|p| p.map(nits)).collect()
}

/// A declared-HDR `LinearF32Rgba` source (absolute linear cd/m²) — the
/// `research::extract` HDR route.
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

/// The research request used on a declared-HDR pair: the 944-wide
/// layout. `Request::everything()` includes the restored-cut families
/// (`mapdev`/`z1max`), whose walk `assert!`s an SDR pair and panics on a
/// declared-HDR one at every revision (pre-existing, recorded in
/// FEATCANON_FIX_DONE.md), so it cannot serve as the below-Rev4 control.
fn hdr_request() -> Request {
    Request::for_slots(
        zensim::feature_set_id::SlotSet::from_ranges([(0, 944)]),
        944,
    )
}

fn profiles() -> Vec<(&'static str, ZensimProfile)> {
    vec![
        ("A", ZensimProfile::A),
        ("PreviewV0_1", ZensimProfile::PreviewV0_1),
        ("PreviewV0_2", ZensimProfile::PreviewV0_2),
        ("B", ZensimProfile::B),
        ("BHdr", ZensimProfile::BHdr),
        ("C", ZensimProfile::C),
        ("CHdr", ZensimProfile::CHdr),
        ("D", ZensimProfile::D),
    ]
}

fn score_bake(bytes: &[u8], s: &RgbSlice<'_>, d: &RgbSlice<'_>) -> Result<f64, String> {
    let model = zenpredict::Model::from_bytes(bytes).expect("bake parses");
    let mut scorer = BakeScorer::new(&model).map_err(|e| format!("new: {e:?}"))?;
    scorer
        .compute(s, d, None)
        .map(|r| r.score())
        .map_err(|e| format!("compute: {e:?}"))
}

fn assert_refused<T: std::fmt::Debug>(what: &str, r: Result<T, String>, reason: &str) {
    match r {
        Ok(v) => panic!("{what}: expected a refusal containing {reason:?}, got Ok({v:?})"),
        Err(e) => assert!(
            e.contains(reason),
            "{what}: refused, but not for the Rev4 contract: {e}"
        ),
    }
}

/// D1, direction 1 — the test that failed before featcanon-fix: the D bake
/// (declared Rev1) under `ZENSIM_FORMULA_REV=4` must ERROR, not score. Every
/// shipped bake likewise.
#[test]
fn rev1_bake_errors_in_a_rev4_process() {
    const SENTINEL: &str = "D1_DIR1_OK";
    if !at_revision(Some("4"), "rev1_bake_errors_in_a_rev4_process", SENTINEL) {
        return;
    }
    let (w, h) = (97, 63);
    let (src, dst) = pair(w, h);
    let (s, d) = (RgbSlice::new(&src, w, h), RgbSlice::new(&dst, w, h));
    let d_model = zenpredict::Model::from_bytes(D_BAKE).expect("D parses");
    assert_eq!(
        zensim::feature_v2::bake_formula_revision_public(&d_model).expect("known revision"),
        FormulaRevision::Rev1,
        "the fixture must be the Rev1-declared D bake"
    );
    for (name, bytes) in BAKES {
        assert_refused(
            &format!("bake {name} in a Rev4 process"),
            score_bake(bytes, &s, &d),
            MIX_REASON,
        );
    }
    println!("{SENTINEL}");
}

/// Non-vacuity of [`rev1_bake_errors_in_a_rev4_process`]: the same bakes on
/// the same pair score at the shipped revision.
#[test]
fn rev1_bake_scores_outside_rev4() {
    const SENTINEL: &str = "D1_DIR1_CONTROL_OK";
    if !at_revision(None, "rev1_bake_scores_outside_rev4", SENTINEL) {
        return;
    }
    let (w, h) = (97, 63);
    let (src, dst) = pair(w, h);
    let (s, d) = (RgbSlice::new(&src, w, h), RgbSlice::new(&dst, w, h));
    for (name, bytes) in BAKES.iter().filter(|(n, _)| *n != "BHdr") {
        let score = score_bake(bytes, &s, &d).unwrap_or_else(|e| panic!("bake {name}: {e}"));
        assert!(score.is_finite(), "bake {name}: score {score}");
    }
    println!("{SENTINEL}");
}

/// Toggles for a v1-only (basic/peak) request at `revision` — the per-request
/// route the review's direction-2 probe used.
fn v1_only(revision: FormulaRevision) -> V2NewFeatureToggles {
    V2NewFeatureToggles {
        v1_only: true,
        formula_revision: revision,
        ..V2NewFeatureToggles::default()
    }
}

/// D1, direction 2: a Rev4 request in a non-Rev4 process — unset (Rev1) and
/// pinned to 3 — must error. Before featcanon-fix it was accepted and
/// returned the Rev3 request's bits. The Rev3 request beside it must still
/// be served (non-vacuity).
fn rev4_request_refused_here() {
    let (w, h) = (97, 63);
    let (src, dst) = pair(w, h);
    let (s, d) = (RgbSlice::new(&src, w, h), RgbSlice::new(&dst, w, h));
    let z = Zensim::new(ZensimProfile::B);
    assert_refused(
        "v1-only Rev4 request",
        z.compute_v2_features_with_toggles(&s, &d, v1_only(FormulaRevision::Rev4))
            .map(|r| r.features().len())
            .map_err(|e| format!("{e:?}")),
        MIX_REASON,
    );
    let ok = z
        .compute_v2_features_with_toggles(&s, &d, v1_only(FormulaRevision::Rev3))
        .unwrap_or_else(|e| panic!("v1-only Rev3 request must be served: {e:?}"));
    assert!(!ok.features().is_empty());
}

#[test]
fn rev4_request_errors_in_an_unpinned_process() {
    const SENTINEL: &str = "D1_DIR2_UNSET_OK";
    if !at_revision(None, "rev4_request_errors_in_an_unpinned_process", SENTINEL) {
        return;
    }
    rev4_request_refused_here();
    println!("{SENTINEL}");
}

#[test]
fn rev4_request_errors_in_a_rev3_process() {
    const SENTINEL: &str = "D1_DIR2_REV3_OK";
    if !at_revision(Some("3"), "rev4_request_errors_in_a_rev3_process", SENTINEL) {
        return;
    }
    rev4_request_refused_here();
    println!("{SENTINEL}");
}

/// Every served entry, applied to one pair. `Ok` carries a short digest so a
/// served result is visible in the failure message.
fn served_calls(w: usize, h: usize) -> Vec<(String, Result<String, String>)> {
    let (src, dst) = pair(w, h);
    let (s, d) = (RgbSlice::new(&src, w, h), RgbSlice::new(&dst, w, h));
    let (rn, dn) = (nits_rgb(&src), nits_rgb(&dst));
    let hs = HdrLinear::new(&src, w, h);
    let hd = HdrLinear::new(&dst, w, h);
    let e = |e: zensim::ZensimError| format!("{e:?}");
    let mut out: Vec<(String, Result<String, String>)> = Vec::new();
    for (name, p) in profiles() {
        let z = Zensim::new(p);
        let hdr_profile = matches!(name, "BHdr" | "CHdr");
        if !hdr_profile {
            out.push((
                format!("{name}.compute"),
                z.compute(&s, &d)
                    .map(|r| format!("{:.6}", r.score()))
                    .map_err(e),
            ));
            out.push((
                format!("{name}.precompute_reference"),
                z.precompute_reference(&s)
                    .map(|r| format!("{}x{}", r.width(), r.height()))
                    .map_err(e),
            ));
        }
        // CHdr is not servable through the PU v1 entry at any revision (it reads
        // 944-walk ids: "the bake declares feature ids this feature vector does
        // not reach", measured on main at the shipped revision); its HDR route is
        // `compute_folded720_features_hdr` below.
        if matches!(name, "A" | "B" | "BHdr") {
            out.push((
                format!("{name}.compute_pu_linear"),
                z.compute_pu_linear(&rn, &dn, w, h, 3 * w, 3 * w)
                    .map(|r| format!("{:.6}", r.score()))
                    .map_err(e),
            ));
        }
    }
    let z = Zensim::new(ZensimProfile::B);
    // REV4SERVE: the V2Bounded buffered walk is not the canonical owner —
    // at Rev4 it refuses by name (the folded entries are the canonical
    // equivalents). Below Rev4 it serves unchanged, and the caller maps
    // that difference: this entry is checked against both contracts.
    out.push((
        "compute_v2_features".into(),
        z.compute_v2_features(&s, &d)
            .map(|r| r.features().len().to_string())
            .map_err(e),
    ));
    out.push((
        "compute_folded720_features".into(),
        z.compute_folded720_features(&s, &d)
            .map(|r| r.features().len().to_string())
            .map_err(e),
    ));
    let mut scratch = zensim::feature_v2::V2Scratch::new();
    out.push((
        "compute_folded720_features_streaming".into(),
        z.compute_folded720_features_streaming(
            &s,
            &d,
            V2NewFeatureToggles::default(),
            &mut scratch,
        )
        .map(|r| r.features().len().to_string())
        .map_err(e),
    ));
    out.push((
        "compute_folded720_features_hdr".into(),
        z.compute_folded720_features_hdr(
            &hs,
            &hd,
            zensim::feature_v2::HdrEncoding::Linear,
            V2NewFeatureToggles::default(),
            &mut scratch,
        )
        .map(|r| r.features().len().to_string())
        .map_err(e),
    ));
    for (name, bytes) in BAKES.iter().filter(|(n, _)| *n != "BHdr") {
        out.push((
            format!("BakeScorer({name}).compute"),
            score_bake(bytes, &s, &d).map(|v| format!("{v:.6}")),
        ));
    }
    out
}

/// D2 (REV4SERVE) — the served-path test: in a Rev4 process every served
/// entry whose request resolves to Rev4 COMPUTES — the profiles, the HDR/PU
/// entries, precompute, the folded/v2 feature walks including declared-HDR,
/// the diffmap, and `research::extract` on both SDR and HDR pairs. The one
/// refusal left is D1's: a bake that declares another revision (all four
/// shipped bakes declare Rev1) must not score Rev4 features under its own
/// label.
#[test]
fn served_paths_serve_rev4() {
    const SENTINEL: &str = "D2_SERVED_COMPUTES_OK";
    if !at_revision(Some("4"), "served_paths_serve_rev4", SENTINEL) {
        return;
    }
    for &(w, h) in &[(64usize, 64usize), (97, 63)] {
        let calls = served_calls(w, h);
        assert!(
            calls.len() >= 20,
            "the served-entry list shrank: {}",
            calls.len()
        );
        for (what, r) in calls {
            if what.starts_with("BakeScorer(") {
                // The shipped bakes declare Rev1 — the D1 boundary.
                assert_refused(&format!("{w}x{h} {what}"), r, MIX_REASON);
            } else if what == "compute_v2_features" {
                // The V2Bounded buffered walk is not the canonical Rev4
                // owner; it refuses by name (the folded entries cover the
                // same slots canonically).
                assert_refused(&format!("{w}x{h} {what}"), r, "V2Bounded");
            } else {
                r.unwrap_or_else(|e| panic!("{w}x{h} {what} must compute at Rev4: {e}"));
            }
        }
        let (src, dst) = pair(w, h);
        let (s, d) = (RgbSlice::new(&src, w, h), RgbSlice::new(&dst, w, h));
        let x = research::extract(&Request::everything(), &s, &d)
            .unwrap_or_else(|e| panic!("{w}x{h}: research::extract must compute Rev4: {e}"));
        assert_eq!(x.values().len(), research::full_width());
        // Palette belongs to the comprehensive research request only.
        // The explicit pre-palette scope keeps its original width and bits.
        let legacy = research::extract(
            &Request::for_slots(
                zensim::feature_set_id::SlotSet::from_ranges([(0, 1825)]),
                1825,
            ),
            &s,
            &d,
        )
        .expect("pre-palette research extraction at Rev4");
        assert_eq!(legacy.values().len(), 1825);
        for (id, (a, b)) in legacy.values().iter().zip(x.values()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "legacy f{id} changed");
        }
        // The canonical PU front end serves declared-HDR extraction at Rev4.
        let (hs, hd) = (HdrLinear::new(&src, w, h), HdrLinear::new(&dst, w, h));
        let x = research::extract(&hdr_request(), &hs, &hd)
            .unwrap_or_else(|e| panic!("{w}x{h}: research HDR must compute at Rev4: {e}"));
        assert!(!x.values().is_empty());
        // The diffmap serves at Rev4: `s_v2` is a per-scale/channel/local
        // weighting (any nonzero weights exercise the walk), sized for more
        // scales than these small images produce.
        let z = Zensim::new(ZensimProfile::B);
        let s_v2 = vec![0.5f64; 8 * 3 * zensim::feature_v2::FEATURES_PER_CHANNEL_V2_TOTAL];
        z.compute_v2_diffmap(&s, &d, &s_v2)
            .unwrap_or_else(|e| panic!("{w}x{h}: diffmap must compute at Rev4: {e:?}"));
    }
    println!("{SENTINEL}");
}

/// Non-vacuity of [`served_paths_serve_rev4`]: at the shipped revision the
/// same calls serve — including the bakes, whose Rev1 declaration matches
/// the unpinned process. (Shipped rather than Rev3 because wide bakes
/// rightly refuse a Rev3 process.)
#[test]
fn served_paths_serve_at_the_shipped_revision() {
    const SENTINEL: &str = "D2_SERVED_SHIPPED_OK";
    if !at_revision(None, "served_paths_serve_at_the_shipped_revision", SENTINEL) {
        return;
    }
    for &(w, h) in &[(64usize, 64usize), (97, 63)] {
        for (what, r) in served_calls(w, h) {
            if let Err(e) = r {
                panic!("{w}x{h} {what} must serve at Rev3: {e}");
            }
        }
        let (src, dst) = pair(w, h);
        let (hs, hd) = (HdrLinear::new(&src, w, h), HdrLinear::new(&dst, w, h));
        research::extract(&hdr_request(), &hs, &hd)
            .unwrap_or_else(|e| panic!("{w}x{h}: research HDR must compute below Rev4: {e}"));
    }
    println!("{SENTINEL}");
}

/// D1 at the research layer: a research extraction runs at the process
/// revision, so its reported arithmetic is Rev4 exactly when the process is.
#[test]
fn research_reports_the_revision_it_computed() {
    const SENTINEL: &str = "D1_RESEARCH_REV_OK";
    if !at_revision(
        Some("4"),
        "research_reports_the_revision_it_computed",
        SENTINEL,
    ) {
        return;
    }
    let (w, h) = (64, 64);
    let (src, dst) = pair(w, h);
    let x = research::extract(
        &Request::everything(),
        &RgbSlice::new(&src, w, h),
        &RgbSlice::new(&dst, w, h),
    )
    .expect("extract");
    let manifest = x.manifest_json();
    assert!(
        manifest.contains("\"formula_revision\": \"Rev4\""),
        "manifest does not name Rev4:\n{}",
        &manifest[..manifest.len().min(800)]
    );
    assert!(
        manifest.contains("\"tiercanon\""),
        "manifest lacks the tiercanon era"
    );
    println!("{SENTINEL}");
}

/// The era label a Rev4 producer stamps. The `feature_set_id` token charset
/// is `[a-z0-9_]`, so the label is `tiercanon_c3negfold` (a hyphen is refused).
const REV4_ERA_LABEL: &str = "tiercanon_c3negfold";

/// rev4canon D6: the `c3negfold` definition fix is visible to a consumer.
///
/// * the manifest names the era in `formula_revision_eras`, and the eight
///   tailhist `Bin` slots (`*_p95`/`*_p99`: 2 x 4 maps x 4 scales x 3 channels
///   = 96) carry it in their per-slot `proposed_revision` while `*_max` does
///   not (`revision` itself reads `tiercanon` on every slot: the arithmetic
///   era is every slot's era);
/// * the id's era field is CALLER-SUPPLIED (`Request::with_era_label`), not
///   derived from the revision's eras, so two labels give two ids and the
///   default label gives a third. A producer must pass [`REV4_ERA_LABEL`];
///   auto-deriving it is a follow-up, not done here.
#[test]
fn rev4_manifest_carries_c3negfold_and_era_labels_make_distinct_ids() {
    const SENTINEL: &str = "D6_C3NEGFOLD_OK";
    if !at_revision(
        Some("4"),
        "rev4_manifest_carries_c3negfold_and_era_labels_make_distinct_ids",
        SENTINEL,
    ) {
        return;
    }
    assert!(
        zensim::feature_set_id::is_valid_token(REV4_ERA_LABEL),
        "{REV4_ERA_LABEL} must be a valid era token"
    );
    assert!(
        !zensim::feature_set_id::is_valid_token("tiercanon-c3negfold"),
        "the hyphenated spelling is not a token"
    );
    let (w, h) = (64, 64);
    let (src, dst) = pair(w, h);
    let go = |req: Request| {
        research::extract(&req, &RgbSlice::new(&src, w, h), &RgbSlice::new(&dst, w, h))
            .expect("extract")
    };
    let x = go(Request::everything().with_era_label(REV4_ERA_LABEL));
    let manifest = x.manifest_json();
    assert!(
        manifest.contains("\"c3negfold\""),
        "manifest lacks the c3negfold era"
    );
    let (mut bin, mut bin_tagged, mut max, mut max_tagged) = (0, 0, 0, 0);
    for line in manifest
        .lines()
        .filter(|l| l.contains("\"family\": \"tailhist\""))
    {
        let tagged = line.contains("\"proposed_revision\": \"c3negfold\"");
        if line.contains("\"statistic\": \"bin\"") {
            bin += 1;
            bin_tagged += tagged as usize;
        } else if line.contains("\"statistic\": \"max\"") {
            max += 1;
            max_tagged += tagged as usize;
        }
    }
    assert!(bin > 0 && max > 0, "no tailhist rows found in the manifest");
    assert_eq!(
        bin_tagged, bin,
        "every tailhist Bin slot must name c3negfold"
    );
    assert_eq!(max_tagged, 0, "tailhist Max slots must not carry c3negfold");
    println!("tailhist rows: {bin} Bin (all c3negfold), {max} Max (none)");

    let id_a = x.feature_set_id().expect("id").clone();
    let id_b = go(Request::everything().with_era_label("tiercanon"))
        .feature_set_id()
        .expect("id")
        .clone();
    let id_c = go(Request::everything())
        .feature_set_id()
        .expect("id")
        .clone();
    assert_ne!(id_a, id_b, "two era labels must give two ids");
    assert_ne!(
        id_a, id_c,
        "the default label must differ from the stamped one"
    );
    assert_eq!(id_a.era(), REV4_ERA_LABEL);
    println!("ids: {id_a} | {id_b} | {id_c}");
    println!("{SENTINEL}");
}
