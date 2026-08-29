//! End-to-end gate for the issue #33 curation tool: a synthetic corpus
//! of 3 sources × 3 resample variants (incl. a non-uniform aspect
//! change, like the real `_769x513` renditions) + 1 singleton is
//! clustered by the `corpus_content_clusters` binary at d ≤ 3, and the
//! three CSV outputs (cull / reweight / split) are checked row-for-row.

use image::{DynamicImage, ImageBuffer, Rgb};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Seed-specific content: three 9×9 random lattices (one per channel),
/// bilinearly upsampled — smooth enough that dHash is stable across
/// resampling, random enough that different seeds are ~32 bits apart.
fn render(seed: u64, w: u32, h: u32) -> DynamicImage {
    let lattice = |ch: u64| -> Vec<f64> {
        (0..81u64)
            .map(|i| (splitmix64(seed * 1000 + ch * 100 + i) & 0xFFFF) as f64 / 65535.0)
            .collect()
    };
    let lat = [lattice(0), lattice(1), lattice(2)];
    let img = ImageBuffer::from_fn(w, h, |x, y| {
        let fx = x as f64 / w as f64 * 8.0;
        let fy = y as f64 / h as f64 * 8.0;
        let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
        let (tx, ty) = (fx - x0 as f64, fy - y0 as f64);
        let sample = |l: &Vec<f64>| {
            let at = |i: usize, j: usize| l[j.min(8) * 9 + i.min(8)];
            let v = at(x0, y0) * (1.0 - tx) * (1.0 - ty)
                + at(x0 + 1, y0) * tx * (1.0 - ty)
                + at(x0, y0 + 1) * (1.0 - tx) * ty
                + at(x0 + 1, y0 + 1) * tx * ty;
            (v.clamp(0.0, 1.0) * 255.0) as u8
        };
        Rgb([sample(&lat[0]), sample(&lat[1]), sample(&lat[2])])
    });
    DynamicImage::ImageRgb8(img)
}

fn write_png(img: &DynamicImage, path: &Path) {
    img.save(path)
        .unwrap_or_else(|e| panic!("save {}: {e}", path.display()));
}

struct Fixture {
    root: PathBuf,
    sources: PathBuf,
    csv: PathBuf,
    /// file name → (base hint, pixels)
    files: BTreeMap<String, (String, u64)>,
}

fn build_fixture(tag: &str) -> Fixture {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("cc_e2e_{tag}"));
    let _ = std::fs::remove_dir_all(&root);
    let sources = root.join("sources");
    std::fs::create_dir_all(&sources).unwrap();
    let mut files = BTreeMap::new();
    // Three multi-variant sources: 256sq (canonical), 192x128 (aspect
    // change), 128sq.
    for (i, seed) in [11u64, 22, 33].iter().enumerate() {
        let base = format!("src{i:02}aaaa");
        let full = render(*seed, 256, 256);
        for (suffix, w, h) in [
            ("256sq", 256u32, 256u32),
            ("192x128", 192, 128),
            ("128sq", 128, 128),
        ] {
            let img = full.resize_exact(w, h, image::imageops::FilterType::Lanczos3);
            let name = format!("{base}_{suffix}.png");
            write_png(&img, &sources.join(&name));
            files.insert(name, (base.clone(), u64::from(w) * u64::from(h)));
        }
    }
    // One singleton source.
    let single = render(44, 200, 150);
    write_png(&single, &sources.join("single99_200x150.png"));
    files.insert(
        "single99_200x150.png".into(),
        ("single99".into(), 200 * 150),
    );

    // Training CSV: 2 rows per file, column 0 = source path (as the
    // synthetic corpus CSVs are laid out), plus a decoy `ref_basename`-free
    // header so the column-0 fallback is exercised.
    let csv = root.join("train.csv");
    let mut body = String::from("source,human_score,f0,f1\n");
    for (k, name) in files.keys().enumerate() {
        for r in 0..2 {
            body.push_str(&format!(
                "{},{:.3},{},{}\n",
                sources.join(name).display(),
                0.5 + 0.01 * (k * 2 + r) as f64,
                k,
                r
            ));
        }
    }
    std::fs::write(&csv, body).unwrap();
    Fixture {
        root,
        sources,
        csv,
        files,
    }
}

fn read_tsv(path: &Path) -> (Vec<String>, Vec<Vec<String>>) {
    let text = std::fs::read_to_string(path).unwrap();
    let mut lines = text.lines();
    let header: Vec<String> = lines
        .next()
        .unwrap()
        .split('\t')
        .map(String::from)
        .collect();
    let rows = lines
        .map(|l| l.split('\t').map(String::from).collect())
        .collect();
    (header, rows)
}

fn col<'a>(header: &[String], row: &'a [String], name: &str) -> &'a str {
    let i = header
        .iter()
        .position(|h| h == name)
        .unwrap_or_else(|| panic!("no column {name}"));
    &row[i]
}

fn csv_rows(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .skip(1)
        .map(String::from)
        .collect()
}

#[test]
fn clusters_resample_variants_and_emits_cull_reweight_split() {
    let fx = build_fixture("full");
    let out_tsv = fx.root.join("clusters.tsv");
    let cull = fx.root.join("culled.csv");
    let reweight = fx.root.join("reweight");
    let split = fx.root.join("split");
    let status = Command::new(env!("CARGO_BIN_EXE_corpus_content_clusters"))
        .args([
            "--training-csv",
            fx.csv.to_str().unwrap(),
            "--source-root",
            fx.sources.to_str().unwrap(),
            "--max-dist",
            "3",
            "--val-frac",
            "0.3",
            "--seed",
            "5",
            "--out-tsv",
            out_tsv.to_str().unwrap(),
            "--cull-csv",
            cull.to_str().unwrap(),
            "--reweight-dir",
            reweight.to_str().unwrap(),
            "--split-dir",
            split.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(status.success(), "corpus_content_clusters failed: {status}");

    // --- per-file TSV: 10 files, 4 clusters, hash clusters == name hints.
    let (header, rows) = read_tsv(&out_tsv);
    assert_eq!(rows.len(), fx.files.len());
    let mut cluster_of_hint: BTreeMap<&str, BTreeSet<String>> = BTreeMap::new();
    let mut hints_of_cluster: BTreeMap<String, BTreeSet<&str>> = BTreeMap::new();
    let mut canonical_names = BTreeSet::new();
    let mut side_of_cluster: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for r in &rows {
        let name = col(&header, r, "basename");
        let (hint, px) = &fx.files[name];
        assert_eq!(col(&header, r, "base_hint"), hint);
        assert_eq!(col(&header, r, "pixels"), px.to_string());
        let cid = col(&header, r, "cluster_id").to_string();
        cluster_of_hint.entry(hint).or_default().insert(cid.clone());
        hints_of_cluster
            .entry(cid.clone())
            .or_default()
            .insert(hint);
        let size: usize = col(&header, r, "cluster_size").parse().unwrap();
        let expected_size = if hint == "single99" { 1 } else { 3 };
        assert_eq!(size, expected_size, "{name}: cluster_size");
        let w: f64 = col(&header, r, "content_weight").parse().unwrap();
        assert!(
            (w - 1.0 / expected_size as f64).abs() < 1e-6,
            "{name}: weight {w}"
        );
        if col(&header, r, "canonical") == "1" {
            canonical_names.insert(name.to_string());
        }
        side_of_cluster
            .entry(cid)
            .or_default()
            .insert(col(&header, r, "split").to_string());
    }
    // Every base hint is exactly one cluster and every cluster exactly one hint.
    assert_eq!(cluster_of_hint.len(), 4);
    assert!(
        cluster_of_hint.values().all(|s| s.len() == 1),
        "{cluster_of_hint:?}"
    );
    assert_eq!(hints_of_cluster.len(), 4);
    assert!(
        hints_of_cluster.values().all(|s| s.len() == 1),
        "{hints_of_cluster:?}"
    );
    // Canonical = the 256sq variant of each multi-variant source + the singleton.
    let expected_canon: BTreeSet<String> = [
        "src00aaaa_256sq.png",
        "src01aaaa_256sq.png",
        "src02aaaa_256sq.png",
        "single99_200x150.png",
    ]
    .into_iter()
    .map(String::from)
    .collect();
    assert_eq!(canonical_names, expected_canon);
    // No cluster straddles the split; ≥ 30% of members are val.
    assert!(
        side_of_cluster.values().all(|s| s.len() == 1),
        "{side_of_cluster:?}"
    );
    let n_val = rows
        .iter()
        .filter(|r| col(&header, r, "split") == "val")
        .count();
    assert!(n_val >= 3, "val members {n_val} < ceil(0.3 * 10)");
    assert!(n_val < 10);

    // --- option 3: cull keeps the 2 rows of each canonical source.
    let culled = csv_rows(&cull);
    assert_eq!(culled.len(), 8, "{culled:?}");
    for l in &culled {
        let src = Path::new(l.split(',').next().unwrap());
        let name = src.file_name().unwrap().to_string_lossy().into_owned();
        assert!(
            expected_canon.contains(&name),
            "non-canonical row survived cull: {l}"
        );
    }

    // --- option 2: one CSV per cluster size + the group specs.
    let k3 = csv_rows(&reweight.join("cluster_size_3.csv"));
    let k1 = csv_rows(&reweight.join("cluster_size_1.csv"));
    assert_eq!((k3.len(), k1.len()), (18, 2));
    let groups = std::fs::read_to_string(reweight.join("groups.txt")).unwrap();
    // raw weights 18/3 = 6 and 2/1 = 2 → 0.75 / 0.25.
    let weight_of = |k: &str| -> f64 {
        let line = groups
            .lines()
            .find(|l| l.starts_with(&format!("--group k{k}:")))
            .unwrap_or_else(|| panic!("no group k{k} in {groups}"));
        line.rsplit(':').nth(1).unwrap().parse().unwrap()
    };
    assert!((weight_of("3") - 0.75).abs() < 1e-5, "{groups}");
    assert!((weight_of("1") - 0.25).abs() < 1e-5, "{groups}");

    // --- option 4: split CSVs partition the rows, whole clusters per side.
    let train = csv_rows(&split.join("train.csv"));
    let val = csv_rows(&split.join("val.csv"));
    assert_eq!(train.len() + val.len(), 20);
    assert_eq!(val.len(), 2 * n_val);
    let hint_of_row = |l: &String| -> String {
        let src = Path::new(l.split(',').next().unwrap());
        let name = src.file_name().unwrap().to_string_lossy().into_owned();
        fx.files[&name].0.clone()
    };
    let train_hints: BTreeSet<String> = train.iter().map(hint_of_row).collect();
    let val_hints: BTreeSet<String> = val.iter().map(hint_of_row).collect();
    assert!(
        train_hints.is_disjoint(&val_hints),
        "content leaked across the split: train={train_hints:?} val={val_hints:?}"
    );
}

/// `--corpus-dir` mode (no CSV) and the contamination-threshold refusal.
#[test]
fn corpus_dir_mode_and_loose_threshold_refused() {
    let fx = build_fixture("dir");
    let out_tsv = fx.root.join("clusters_dir.tsv");
    let status = Command::new(env!("CARGO_BIN_EXE_corpus_content_clusters"))
        .args([
            "--corpus-dir",
            fx.sources.to_str().unwrap(),
            "--out-tsv",
            out_tsv.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(status.success());
    let (header, rows) = read_tsv(&out_tsv);
    assert_eq!(rows.len(), 10);
    let clusters: BTreeSet<&str> = rows.iter().map(|r| col(&header, r, "cluster_id")).collect();
    assert_eq!(clusters.len(), 4);

    let status = Command::new(env!("CARGO_BIN_EXE_corpus_content_clusters"))
        .args([
            "--corpus-dir",
            fx.sources.to_str().unwrap(),
            "--max-dist",
            "16",
            "--out-tsv",
            fx.root.join("never.tsv").to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(
        !status.success(),
        "d ≤ 16 must be refused as a contamination-screen threshold"
    );
    assert!(!fx.root.join("never.tsv").exists());
}

/// Validation step 2 of issue #33 — the EYEBALL pass — is only executable
/// if the tool hands the reviewer pictures. This gates that it does, and
/// that the pictures show the right members in the right places: every
/// cell is cropped back out of the montage and dHash-compared to the file
/// the index claims is there, so a shuffled or mis-scaled placement fails.
#[test]
fn montage_dir_renders_reviewable_clusters_with_correct_members() {
    use zensim_validate::content_clusters::{dhash_64, hamming, montage_layout};

    let fx = build_fixture("montage");
    let out_tsv = fx.root.join("clusters_montage.tsv");
    let montage = fx.root.join("montage");
    let cell = 192u32;
    let status = Command::new(env!("CARGO_BIN_EXE_corpus_content_clusters"))
        .args([
            "--training-csv",
            fx.csv.to_str().unwrap(),
            "--source-root",
            fx.sources.to_str().unwrap(),
            "--max-dist",
            "3",
            "--out-tsv",
            out_tsv.to_str().unwrap(),
            "--montage-dir",
            montage.to_str().unwrap(),
            "--montage-all",
            "--montage-cell",
            &cell.to_string(),
            "--montage-cols",
            "6",
        ])
        .status()
        .unwrap();
    assert!(status.success());

    let (header, rows) = read_tsv(&out_tsv);
    // Members of each cluster, in the order the tool emitted them — which
    // is the left-to-right order of the montage cells.
    let mut members: BTreeMap<String, Vec<(PathBuf, u64)>> = BTreeMap::new();
    for r in &rows {
        let id = col(&header, r, "cluster_id").to_string();
        let path = PathBuf::from(col(&header, r, "path"));
        let hash = u64::from_str_radix(col(&header, r, "dhash"), 16).unwrap();
        members.entry(id).or_default().push((path, hash));
    }
    // 3 multi-variant sources + 1 singleton; only the former get montages
    // (--montage-all still skips size-1 clusters — a montage of one image
    // is not a comparison).
    let multi: Vec<&String> = members
        .iter()
        .filter(|(_, m)| m.len() > 1)
        .map(|(k, _)| k)
        .collect();
    assert_eq!(multi.len(), 3, "expected 3 multi-variant clusters");

    let index = std::fs::read_to_string(montage.join("index.html")).unwrap();
    assert!(index.contains("issue #33"), "index must name the issue");
    let singleton = members
        .iter()
        .find(|(_, m)| m.len() == 1)
        .map(|(k, _)| k.clone())
        .unwrap();
    assert!(
        !index.contains(&format!(
            "cluster_{:05}_n1",
            singleton.parse::<usize>().unwrap()
        )),
        "a one-member cluster is not a comparison and must not be rendered"
    );

    let mut rendered = 0usize;
    for id in multi {
        let group = &members[id];
        let file = format!(
            "cluster_{:05}_n{}.png",
            id.parse::<usize>().unwrap(),
            group.len()
        );
        let path = montage.join(&file);
        assert!(path.exists(), "missing montage {}", path.display());
        assert!(index.contains(&file), "index does not link {file}");
        rendered += 1;

        let sheet = image::open(&path).unwrap().to_rgb8();
        let imgs: Vec<DynamicImage> = group.iter().map(|(p, _)| image::open(p).unwrap()).collect();
        let sizes: Vec<(u32, u32)> = imgs.iter().map(|i| (i.width(), i.height())).collect();
        let layout = montage_layout(&sizes, cell, 6);
        assert_eq!(
            (sheet.width(), sheet.height()),
            (layout.width, layout.height)
        );
        for (k, (&(x, y, w, h), (_, member_hash))) in
            layout.cells.iter().zip(group.iter()).enumerate()
        {
            let crop =
                DynamicImage::ImageRgb8(image::imageops::crop_imm(&sheet, x, y, w, h).to_image());
            let d = hamming(dhash_64(&crop), *member_hash);
            assert!(
                d <= 6,
                "cluster {id} cell {k}: montage content is {d} bits from the member it claims"
            );
        }
        // The mid-grey ground must survive around a member smaller than the
        // cell, or flat white screen content would be invisible in review.
        let small = layout
            .cells
            .iter()
            .position(|&(_, _, w, h)| w < cell || h < cell);
        if let Some(i) = small {
            let (x, y, _, _) = layout.cells[i];
            let (bx, by) = ((x / cell) * cell, (y / cell) * cell);
            assert_eq!(sheet.get_pixel(bx + 1, by + 1).0, [128, 128, 128]);
        }
    }
    assert_eq!(rendered, 3);
}
