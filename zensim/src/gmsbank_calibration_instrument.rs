// Explicit, label-free producer of the frozen GMSBANK ratio and XYB dumps.
//
// Compile and run only through scripts/gmsbank/calibration_instrument.sh.
// This file is included inside feature_v2::tests under a custom cfg so it
// can call the exact private XYB conversion owner used during calibration.
// The input directory is mandatory; ordinary cargo test never runs this tool.

/// Label-free calibration instrument. The zensim owner produces the
/// exact scale-1 XYB Y plane; the 2x-box Prewitt side is the GMSD domain.
/// Only interior co-sited pixels enter the median, so neither operator's
/// border extension can change the units mapping. Set GMSBANK_CALIB_DIR
/// to a directory made by gmsbank_decode_dump; it is required.
#[test]
fn gmsbank_calibration_scale1_y_dump() {
    let dir = std::env::var("GMSBANK_CALIB_DIR")
        .expect("GMSBANK_CALIB_DIR is required for the explicit calibration tool");
    use std::io::Write;
    let root = std::path::Path::new(&dir);
    let manifest = std::fs::read_to_string(root.join("planes.tsv")).unwrap();
    let mut out = std::fs::File::create(root.join("ratios.tsv")).unwrap();
    writeln!(out, "pair_key\tratio\teligible_sites").unwrap();
    let mut dump_index = std::fs::File::create(root.join("xyb8.tsv")).unwrap();
    writeln!(
        dump_index,
        "pair_key\tside\tscale\tchannel\twidth\theight\tfile"
    )
    .unwrap();
    for (n, line) in manifest.lines().enumerate() {
        let f: Vec<_> = line.split('\t').collect();
        assert_eq!(f.len(), 5);
        let w: usize = f[1].parse().unwrap();
        let h: usize = f[2].parse().unwrap();
        assert!(w >= 64 && h >= 64);
        let mut ratios = Vec::new();
        for (side, name) in [f[3], f[4]].into_iter().enumerate() {
            let raw = std::fs::read(root.join(name)).unwrap();
            assert_eq!(raw.len(), w * h * 3);
            let rgb: Vec<[u8; 3]> = raw.chunks_exact(3).map(|v| [v[0], v[1], v[2]]).collect();
            let image = RgbSlice::new(&rgb, w, h);
            let mut xyb = crate::streaming::convert_source_to_xyb(&image, w, false);
            let dump = |scale: usize,
                        dw: usize,
                        dh: usize,
                        planes: &[Vec<f32>; 3],
                        index: &mut std::fs::File| {
                if n >= 8 {
                    return;
                }
                for (ch, plane) in planes.iter().enumerate() {
                    let filename = format!("{}_{}_{}_{}.f32", f[0], side, scale, ch);
                    let mut raw = Vec::with_capacity(plane.len() * 4);
                    for sample in plane {
                        raw.extend_from_slice(&sample.to_le_bytes());
                    }
                    std::fs::write(root.join(&filename), raw).unwrap();
                    writeln!(
                        index,
                        "{}\t{side}\t{scale}\t{ch}\t{dw}\t{dh}\t{filename}",
                        f[0]
                    )
                    .unwrap();
                }
            };
            dump(0, w, h, &xyb, &mut dump_index);
            let (hw, hh) = crate::streaming::downscale_3_planes(&mut xyb, w, h, false);
            dump(1, hw, hh, &xyb, &mut dump_index);
            if n < 8 {
                let mut deeper = xyb.clone();
                let (w2, h2) = crate::streaming::downscale_3_planes(&mut deeper, hw, hh, false);
                dump(2, w2, h2, &deeper, &mut dump_index);
                let (w3, h3) = crate::streaming::downscale_3_planes(&mut deeper, w2, h2, false);
                dump(3, w3, h3, &deeper, &mut dump_index);
            }
            let y_plane = &xyb[1];
            let gray: Vec<f64> = rgb
                .iter()
                .map(|p| {
                    (0.299 * f64::from(p[0]) + 0.587 * f64::from(p[1]) + 0.114 * f64::from(p[2]))
                        .round()
                })
                .collect();
            let mut box2 = vec![0.0f64; hw * hh];
            for yy in 0..hh {
                for xx in 0..hw {
                    let i = (yy * 2) * w + xx * 2;
                    box2[yy * hw + xx] =
                        (gray[i] + gray[i + 1] + gray[i + w] + gray[i + w + 1]) / (4.0 * 255.0);
                }
            }
            for yy in 1..hh - 1 {
                for xx in 1..hw - 1 {
                    let i = yy * hw + xx;
                    let px = ((box2[i + 1 - hw] + box2[i + 1] + box2[i + 1 + hw])
                        - (box2[i - 1 - hw] + box2[i - 1] + box2[i - 1 + hw]))
                        / 3.0;
                    let py = ((box2[i + hw - 1] + box2[i + hw] + box2[i + hw + 1])
                        - (box2[i - hw - 1] + box2[i - hw] + box2[i - hw + 1]))
                        / 3.0;
                    let pm = (px * px + py * py).sqrt();
                    let dx = f64::from(y_plane[i + 1] - y_plane[i - 1]);
                    let dy = f64::from(y_plane[i + hw] - y_plane[i - hw]);
                    let ym = (dx * dx + dy * dy).sqrt();
                    if pm > 1e-6 && ym > 1e-6 {
                        ratios.push(pm / ym);
                    }
                }
            }
        }
        ratios.sort_unstable_by(f64::total_cmp);
        assert!(
            !ratios.is_empty(),
            "no co-sited nonflat gradients: {}",
            f[0]
        );
        let median = if ratios.len() % 2 == 0 {
            (ratios[ratios.len() / 2 - 1] + ratios[ratios.len() / 2]) * 0.5
        } else {
            ratios[ratios.len() / 2]
        };
        writeln!(out, "{}\t{median:.17e}\t{}", f[0], ratios.len()).unwrap();
        if n % 25 == 0 {
            eprintln!("gmsbank ratio {n}");
        }
    }
}

/// C8 chroma revision: the preregistered four independent units ratios.
/// The source directory contains verified native/box-derived TRAIN RGB8.
#[test]
fn gmsbank_chroma_calibration_dump() {
    use std::io::Write;
    let directory = std::env::var("GMSBANK_CHROMA_CALIB_DIR")
        .expect("GMSBANK_CHROMA_CALIB_DIR required for the chroma calibration tool");
    let root = std::path::Path::new(&directory);
    let manifest = std::fs::read_to_string(root.join("planes.tsv")).unwrap();
    let create = |name: &str| std::fs::OpenOptions::new().write(true).create_new(true)
        .open(root.join(name)).unwrap();
    let mut output = create("chroma_ratios.tsv");
    let mut index = create("chroma_xyb8.tsv");
    writeln!(output, "pair_key\tratio_kind\tmedian\teligible_sites").unwrap();
    writeln!(index, "pair_key\tside\tscale\tchannel\twidth\theight\tfile").unwrap();
    for (pair_number, line) in manifest.lines().enumerate() {
        let fields: Vec<_> = line.split('\t').collect();
        assert_eq!(fields.len(), 5);
        let width: usize = fields[1].parse().unwrap();
        let height: usize = fields[2].parse().unwrap();
        assert!(width >= 64 && height >= 64);
        let mut ratios: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
        for (side, name) in [fields[3], fields[4]].into_iter().enumerate() {
            let raw = std::fs::read(root.join(name)).unwrap();
            assert_eq!(raw.len(), width * height * 3);
            let pixels: Vec<[u8; 3]> = raw.as_chunks::<3>().0.to_vec();
            let source = RgbSlice::new(&pixels, width, height);
            let mut xyb = crate::streaming::convert_source_to_xyb(&source, width, false);
            if pair_number < 8 {
                let mut planes = xyb.clone();
                let (mut dw, mut dh) = (width, height);
                for scale in 0..4 {
                    if scale > 0 {
                        (dw, dh) = crate::streaming::downscale_3_planes(&mut planes, dw, dh, false);
                    }
                    for (channel, plane) in planes.iter().enumerate() {
                        let filename = format!("{}_{}_{}_{}.f32", fields[0], side, scale, channel);
                        let mut file = create(&filename);
                        for value in plane { file.write_all(&value.to_le_bytes()).unwrap(); }
                        writeln!(index, "{}\t{side}\t{scale}\t{channel}\t{dw}\t{dh}\t{filename}",fields[0]).unwrap();
                    }
                }
            }
            let (hw, hh) = crate::streaming::downscale_3_planes(&mut xyb, width, height, false);
            assert_eq!((hw, hh), (width / 2, height / 2));
            let mut opponents = [vec![0.0; hw * hh], vec![0.0; hw * hh]];
            for y in 0..hh {
                for x in 0..hw {
                    let i = 2 * y * width + 2 * x;
                    let mut average = [0.0; 3];
                    for (channel, value) in average.iter_mut().enumerate() {
                        *value = [i, i + 1, i + width, i + width + 1].into_iter()
                            .map(|site| f64::from(pixels[site][channel])).sum::<f64>() / 4.0;
                    }
                    let [r, g, b] = average;
                    opponents[0][y * hw + x] = 0.34 * r - 0.60 * g + 0.17 * b;
                    opponents[1][y * hw + x] = 0.30 * r + 0.04 * g - 0.35 * b;
                }
            }
            for (opponent, channel, offset, kind) in [
                (&opponents[0], 0, f64::from(0.42_f32), 0),
                (&opponents[1], 2, f64::from(0.55_f32), 2),
            ] {
                let plane = &xyb[channel];
                for y in 1..hh - 1 {
                    for x in 1..hw - 1 {
                        let i = y * hw + x;
                        let numerator = opponent[i].abs();
                        let denominator = (f64::from(plane[i]) - offset).abs();
                        if numerator > 1e-6 && denominator > 1e-6 {
                            ratios[kind].push(numerator / denominator);
                        }
                        let gx = ((opponent[i + 1 - hw] + opponent[i + 1] + opponent[i + 1 + hw])
                            - (opponent[i - 1 - hw] + opponent[i - 1] + opponent[i - 1 + hw])) / 3.0;
                        let gy = ((opponent[i + hw - 1] + opponent[i + hw] + opponent[i + hw + 1])
                            - (opponent[i - hw - 1] + opponent[i - hw] + opponent[i - hw + 1])) / 3.0;
                        let px = f64::from(plane[i + 1] - plane[i - 1]);
                        let py = f64::from(plane[i + hw] - plane[i - hw]);
                        let numerator = (gx * gx + gy * gy).sqrt();
                        let denominator = (px * px + py * py).sqrt();
                        if numerator > 1e-6 && denominator > 1e-6 {
                            ratios[kind + 1].push(numerator / denominator);
                        }
                    }
                }
            }
        }
        for (name, values) in ["x_value", "x_gradient", "b_value", "b_gradient"].into_iter().zip(&mut ratios) {
            values.sort_unstable_by(f64::total_cmp);
            if values.is_empty() {
                writeln!(output, "{}\t{name}\tNA\t0", fields[0]).unwrap();
            } else {
                let n = values.len();
                let median = if n % 2 == 0 { (values[n / 2 - 1] + values[n / 2]) * 0.5 } else { values[n / 2] };
                writeln!(output, "{}\t{name}\t{median:.17e}\t{n}", fields[0]).unwrap();
            }
        }
        if pair_number % 25 == 0 { eprintln!("chroma_calibration_pair {pair_number}"); }
    }
}

/// Differential use of the unchanged Octave author's H/M and CS maps.
/// This exercises the same per-pixel helper used by the fused C8 walk.
#[test]
fn gmsbank_chroma_author_map() {
    let path = std::env::var("GMSBANK_CHROMA_ORACLE_DIR")
        .expect("GMSBANK_CHROMA_ORACLE_DIR is required for the explicit author-map tool");
    let output = std::env::var("GMSBANK_CHROMA_ORACLE_REPORT")
        .expect("GMSBANK_CHROMA_ORACLE_REPORT is required");
    let root = std::path::Path::new(&path);
    let scores = std::fs::read_to_string(root.join("scores.tsv")).unwrap();
    let mut records = Vec::new();
    let mut negative_rejections = 0;
    for line in scores.lines().skip(1) {
        let id = line.split('\t').next().unwrap();
        let read = |name: &str| -> Vec<f64> {
            let bytes = std::fs::read(root.join(format!("{id}.{name}.f64"))).unwrap();
            let (values, remainder) = bytes.as_chunks::<8>();
            assert!(remainder.is_empty());
            values.iter().map(|v| f64::from_le_bytes(*v)).collect()
        };
        let h1 = read("h1");
        let h2 = read("h2");
        let m1 = read("m1");
        let m2 = read("m2");
        let expected = read("cs");
        assert!(!expected.is_empty());
        assert!([h1.len(),h2.len(),m1.len(),m2.len()].into_iter().all(|n|n==expected.len()));
        let (mut max_abs,mut max_rel) = (0.0_f64,0.0_f64);
        let mut wrong_rejected = false;
        for i in 0..expected.len() {
            let got = 1.0 - chromaticity_loss([h1[i],m1[i]],[h2[i],m2[i]],[550.0;2]);
            let wrong = 1.0 - chromaticity_loss([h1[i],m1[i]],[h2[i],m2[i]],[55.0;2]);
            let error = (got-expected[i]).abs();
            max_abs = max_abs.max(error);
            if expected[i].abs()>1e-12 {
                max_rel = max_rel.max(error/expected[i].abs());
            }
            wrong_rejected |= (wrong-expected[i]).abs()>1e-12
                || (expected[i].abs()>1e-12 && (wrong-expected[i]).abs()/expected[i].abs()>1e-9);
        }
        negative_rejections += usize::from(wrong_rejected);
        records.push(serde_json::json!({"id":id,"samples":expected.len(),
            "max_abs":max_abs,"max_rel":max_rel,"passed":max_abs<=1e-12 && max_rel<=1e-9}));
    }
    let passed = records.len()==116 && negative_rejections>0
        && records.iter().all(|r|r["passed"].as_bool()==Some(true));
    let report = serde_json::json!({"passed":passed,"pairs":records.len(),
        "negative_control_rejections":negative_rejections,"records":records});
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new().write(true).create_new(true).open(output).unwrap();
    writeln!(file,"{}",serde_json::to_string_pretty(&report).unwrap()).unwrap();
    println!("gmsbank_author_cs {}",report);
    assert!(passed,"author CS differential failed; report preserved");
}

/// Run the normal streaming owner on the exact eight native TRAIN pairs
/// whose independent XYB planes were frozen by calibration.
#[test]
fn gmsbank_chroma_features_dump() {
    use std::io::Write;
    let root = std::path::PathBuf::from(std::env::var("GMSBANK_CHROMA_CALIB_DIR").unwrap());
    let output = std::env::var("GMSBANK_CHROMA_FEATURES").unwrap();
    let mut file = std::io::BufWriter::new(std::fs::OpenOptions::new()
        .write(true).create_new(true).open(output).unwrap());
    write!(file, "pair_index,pair_key").unwrap();
    for i in 0..1502 { write!(file, ",f{i}").unwrap(); }
    writeln!(file).unwrap();
    for (i, line) in std::fs::read_to_string(root.join("planes.tsv")).unwrap().lines().take(8).enumerate() {
        let fields: Vec<_> = line.split('\t').collect();
        assert_eq!(fields.len(), 5);
        let width: usize = fields[1].parse().unwrap();
        let height: usize = fields[2].parse().unwrap();
        let pixels = |name: &str| {
            let bytes = std::fs::read(root.join(name)).unwrap();
            assert_eq!(bytes.len(), width * height * 3);
            bytes.as_chunks::<3>().0.to_vec()
        };
        let values = gmsbank_extract(&pixels(fields[3]), &pixels(fields[4]), width, height, false);
        assert_eq!(values.len(), 1502);
        write!(file, "{i},{}", fields[0]).unwrap();
        for value in values { write!(file, ",{value:.17e}").unwrap(); }
        writeln!(file).unwrap();
    }
}

include!("gmsbank_matrix_instrument.rs");
