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
