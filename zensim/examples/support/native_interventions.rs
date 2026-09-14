//! Saved native intervention replay for diffmap_block_coherence. No controller,
//! synthetic repair, feature formula, or statistics implementation lives here.
//! Canonical split admission belongs to the hash-bound experiment registration.

#[path = "zen_io.rs"]
mod zen_io;
use serde_json::{Value, json};
use std::path::Path;
use zensim::{BakeScorer, RgbSlice};

fn array(v: &Value) -> &[Value] {
    v.as_array().expect("array")
}
fn text(v: &Value) -> &str {
    v.as_str().expect("string")
}
fn number(v: &Value) -> usize {
    v.as_u64().expect("nonnegative integer").try_into().unwrap()
}
fn checked_file(v: &Value) -> Vec<u8> {
    let bytes = std::fs::read(text(&v["path"])).expect("read bound file");
    assert_eq!(super::sha(&bytes), text(&v["sha256"]), "file hash mismatch");
    bytes
}
fn pixels(v: &Value, w: usize, h: usize) -> (Vec<[u8; 3]>, String) {
    let bytes = checked_file(v);
    assert!(
        bytes.len() >= 26
            && &bytes[..8] == b"\x89PNG\r\n\x1a\n"
            && bytes[24] == 8
            && bytes[25] == 2,
        "native replay requires RGB8 PNG"
    );
    let (p, pw, ph) = zen_io::decode_rgb8(Path::new(text(&v["path"])));
    assert_eq!((pw, ph), (w, h), "pixel dimensions");
    assert_eq!(
        super::sha(&std::fs::read(text(&v["path"])).unwrap()),
        text(&v["sha256"])
    );
    let hash = super::sha(p.as_flattened());
    if let Some(expected) = v["pixels_sha256"].as_str() {
        assert_eq!(hash, expected, "pixel hash mismatch");
    }
    (p, hash)
}
fn rect(v: &Value, w: usize, h: usize) -> (usize, usize, usize, usize) {
    let a = array(v);
    assert_eq!(a.len(), 4, "rectangle arity");
    let (x0, y0, x1, y1) = (number(&a[0]), number(&a[1]), number(&a[2]), number(&a[3]));
    assert!(x0 < x1 && y0 < y1 && x1 <= w && y1 <= h, "rectangle bounds");
    (x0, y0, x1, y1)
}

pub(super) fn run(manifest: &str, hash: &str, output: &str) {
    assert!(
        !Path::new(output).exists(),
        "refusing to overwrite evidence"
    );
    let bytes = std::fs::read(manifest).expect("manifest");
    assert_eq!(super::sha(&bytes), hash, "manifest hash mismatch");
    let input: Value = serde_json::from_slice(&bytes).expect("manifest JSON");
    assert_eq!(input["schema"], "zensim-native-map-replay-v1");
    assert_eq!(input["role"], "train", "TRAIN replay only");
    let revision = number(&input["formula_revision"]);
    assert!(matches!(revision, 1 | 3), "replay revision");
    assert_eq!(
        std::env::var("ZENSIM_FORMULA_REV").as_deref(),
        Ok(revision.to_string().as_str()),
        "explicit matching revision required"
    );
    let admission_bytes = std::fs::read(text(&input["admission_path"])).expect("admission");
    assert_eq!(
        super::sha(&admission_bytes),
        text(&input["admission_sha256"]),
        "admission hash mismatch"
    );
    let admission: Value = serde_json::from_slice(&admission_bytes).unwrap();
    assert_eq!(admission["role"], "train", "TRAIN admission only");
    let cells = array(&input["cases"]);
    assert!(!cells.is_empty(), "empty replay");
    let mut ids = std::collections::BTreeSet::new();
    // Validate every role, source binding and union before the first pixel read.
    for c in cells {
        assert!(ids.insert(text(&c["id"])), "duplicate cell");
        assert_eq!(c["role"], "train", "TRAIN cell only");
        assert!(
            array(&admission["allowed_sources"])
                .iter()
                .any(|s| s["origin"] == c["origin"]
                    && s["family"] == c["family"]
                    && s["role"] == "train"),
            "unadmitted source"
        );
        let (w, h) = (number(&c["width"]), number(&c["height"]));
        assert!(
            w >= 8 && h >= 8 && w <= 16384 && h <= 16384,
            "supported geometry"
        );
        let mut coverage = vec![false; w.checked_mul(h).unwrap()];
        let regions = array(&c["regions"]);
        assert!(!regions.is_empty(), "empty regions");
        for (i, g) in regions.iter().enumerate() {
            assert_eq!(number(&g["id"]), i, "region order");
            let mut area = 0;
            for q in array(&g["rects"]) {
                let (x0, y0, x1, y1) = rect(q, w, h);
                area += (x1 - x0) * (y1 - y0);
                for y in y0..y1 {
                    for x in x0..x1 {
                        assert!(!coverage[y * w + x], "overlapping regions");
                        coverage[y * w + x] = true;
                    }
                }
            }
            assert_eq!(area, number(&g["area"]), "union area");
        }
        assert!(coverage.iter().all(|x| *x), "incomplete regions");
        let probes = array(&c["probes"]);
        assert_eq!(probes.len(), 2 + 2 * regions.len(), "probe count");
        assert_eq!(probes[0]["name"], "baseline");
        assert_eq!(probes[1]["name"], "neutral");
        for (i, p) in probes.iter().enumerate() {
            assert_eq!(p["role"], "train", "TRAIN probe only");
            if i >= 2 {
                assert_eq!(number(&p["region"]), (i - 2) / 2, "probe region order");
                assert_eq!(
                    p["arm"],
                    if i % 2 == 0 { "down" } else { "up" },
                    "probe arm order"
                );
            }
        }
    }
    let recipes = array(&input["models"]);
    assert!(!recipes.is_empty(), "empty candidates");
    let models: Vec<Vec<zenpredict::Model>> = recipes
        .iter()
        .map(|m| {
            array(&m["members"])
                .iter()
                .map(|b| zenpredict::Model::from_bytes(&checked_file(b)).unwrap())
                .collect()
        })
        .collect();
    let weights: Vec<Vec<f64>> = recipes
        .iter()
        .map(|m| {
            array(&m["weights"])
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect()
        })
        .collect();
    let mut scorers: Vec<_> = models
        .iter()
        .zip(&weights)
        .map(|(m, w)| {
            BakeScorer::ensemble(m, Some(w))
                .unwrap()
                .with_parallel(false)
        })
        .collect();
    let mut results = Vec::new();
    for c in cells {
        let (w, h) = (number(&c["width"]), number(&c["height"]));
        let (reference, rhash) = pixels(&c["reference"], w, h);
        let rs = RgbSlice::new(&reference, w, h);
        let probes = array(&c["probes"]);
        let (base, bhash) = pixels(&probes[0], w, h);
        let bs = RgbSlice::new(&base, w, h);
        let mut predictions = Vec::new();
        let mut base_features = Vec::new();
        let mut sensitivities = Vec::new();
        let mut base_scores = Vec::new();
        // Every prediction sees only the base pair and region geometry. Probe
        // pixels, independent judges and response derivatives are read later.
        for (mi, scorer) in scorers.iter_mut().enumerate() {
            let mut worker = scorer.prepare_steering(&rs, 8).expect("prepared contract");
            let map = worker.compute(&bs, None).expect("complete base map");
            assert!(
                map.result().score().is_finite()
                    && map.sensitivities().iter().all(|v| v.is_finite())
            );
            let groups: Vec<_> = array(&c["regions"])
                .iter()
                .map(|g| {
                    let mass: f64 = array(&g["rects"])
                        .iter()
                        .map(|v| {
                            let (x0, y0, x1, y1) = rect(v, w, h);
                            map.attribution().query_rect(x0, y0, x1, y1)
                        })
                        .sum();
                    assert!(mass.is_finite());
                    json!({"region":g["id"],"mass":mass,"density":mass/number(&g["area"]) as f64})
                })
                .collect();
            predictions.push(json!({"model":recipes[mi]["name"],"score":map.result().score(),
                "density_missing":map.unsupported_feature_ids(),"refinement_missing":map.unsupported_refinement_feature_ids(),"regions":groups}));
            base_features.push(map.result().features().to_vec());
            sensitivities.push(map.sensitivities().to_vec());
            base_scores.push(map.result().score());
        }
        let bsrc: Vec<_> = reference
            .iter()
            .map(|p| butteraugli::RGB8::new(p[0], p[1], p[2]))
            .collect();
        let mut outputs = Vec::new();
        for p in probes {
            let (image, phash) = pixels(p, w, h);
            let ds = RgbSlice::new(&image, w, h);
            if p["name"] == "neutral" {
                assert_eq!(phash, bhash, "neutral pixels");
            }
            let ssim2 = fast_ssim2::compute_ssimulacra2(
                imgref::Img::new(reference.as_slice(), w, h),
                imgref::Img::new(image.as_slice(), w, h),
            )
            .unwrap();
            let bdst: Vec<_> = image
                .iter()
                .map(|p| butteraugli::RGB8::new(p[0], p[1], p[2]))
                .collect();
            let ba = butteraugli::butteraugli(
                imgref::Img::new(bsrc.as_slice(), w, h),
                imgref::Img::new(bdst.as_slice(), w, h),
                &butteraugli::ButteraugliParams::default(),
            )
            .unwrap()
            .pnorm_3;
            assert!(
                ssim2.is_finite() && ba.is_finite(),
                "finite independent judges"
            );
            let mut scores = Vec::new();
            for (mi, scorer) in scorers.iter_mut().enumerate() {
                let value = scorer.compute(&rs, &ds, None).unwrap();
                assert!(value.score().is_finite());
                assert_eq!(value.features().len(), base_features[mi].len());
                if p["name"] == "baseline" || p["name"] == "neutral" {
                    assert_eq!(
                        value.score().to_bits(),
                        base_scores[mi].to_bits(),
                        "base/neutral score parity"
                    );
                    for (a, b) in value.features().iter().zip(&base_features[mi]) {
                        assert_eq!(a.to_bits(), b.to_bits(), "base/neutral feature parity");
                    }
                }
                let linearized: f64 = value
                    .features()
                    .iter()
                    .zip(&base_features[mi])
                    .zip(&sensitivities[mi])
                    .map(|((after, before), s)| s * (after - before))
                    .sum();
                assert!(linearized.is_finite());
                scores.push(json!({"model":recipes[mi]["name"],"score":value.score(),"delta":value.score()-base_scores[mi],"linearized_delta":linearized,"features":value.features()}));
            }
            outputs.push(json!({"name":p["name"],"pixels_sha256":phash,"ssim2":ssim2,"butteraugli_pnorm3":ba,"models":scores}));
        }
        results.push(json!({"id":c["id"],"reference_pixels_sha256":rhash,"baseline_pixels_sha256":bhash,"predictions":predictions,"probes":outputs}));
        eprintln!("native replay complete: {}", text(&c["id"]));
    }
    let comparisons: usize = cells.iter().map(|c| array(&c["probes"]).len()).sum();
    let result = json!({"schema":"zensim-native-map-replay-result-v1","manifest_sha256":hash,"formula_revision":revision,"role":"train","qualified":false,
        "work":{"candidate_maps":cells.len()*models.len(),"candidate_pixel_scores":comparisons*models.len(),"peer_comparisons":comparisons*2},"cases":results});
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .expect("new evidence file");
    serde_json::to_writer_pretty(file, &result).unwrap();
}
