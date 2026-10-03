//! Phase-2 measurements for the `satsign` family (pixels only, TRAIN-role inputs, no labels).
//!
//! (i) the neutral axis: every gray 0..=255 through the XYB conversion owner; reports `Xc = X-0.42`,
//!     `Bc = B-0.55` and the normalised chroma magnitude `m = sqrt(Xc²/cx + Bc²/cb)` over the ramp.
//! (ii) quantiles of `m` over natural reference images (raw `<name>_<W>x<H>.rgb` files in a directory).
//!
//! `cargo run --release -p zensim --example signedfeat_sat_probe -- <dir of .rgb files>`
//! `(cx, cb)` = `GMSBANK_CS_C[2]` (C8's mid constant), duplicated here as literals because that table is
//! crate-private; the Phase-2 implementation reads the table itself.
use zensim::__bench_stages::srgb_to_positive_xyb_planar_into;

const CX: f64 = 0.0014875777316638384;
const CB: f64 = 0.7739603828506174;

fn neutral_b() -> f64 {
    f64::from(0.55f32) + f64::from(0.003_793_073_4f32).cbrt()
}

fn m_of(x: f32, b: f32, b_centre: f64) -> f64 {
    let xc = f64::from(x) - f64::from(0.42f32);
    let bc = f64::from(b) - b_centre;
    (xc * xc / CX + bc * bc / CB).sqrt()
}

fn quantiles(v: &mut [f64], qs: &[f64]) -> Vec<f64> {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    qs.iter()
        .map(|q| v[((v.len() - 1) as f64 * q) as usize])
        .collect()
}

fn main() {
    let dir = std::env::args().nth(1).expect("dir of .rgb files");
    // (i) gray ramp, padded to a multiple of 8 and run as one bulk call.
    let ramp: Vec<[u8; 3]> = (0..=255u8).map(|g| [g, g, g]).collect();
    let n = ramp.len();
    let (mut x, mut y, mut b) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
    srgb_to_positive_xyb_planar_into(&ramp, &mut x, &mut y, &mut b);
    let (mut xmin, mut xmax, mut bmin, mut bmax) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
    let mut ms = Vec::new();
    for i in 0..n {
        let xc = f64::from(x[i]) - f64::from(0.42f32);
        let bc = f64::from(b[i]) - f64::from(0.55f32);
        xmin = xmin.min(xc);
        xmax = xmax.max(xc);
        bmin = bmin.min(bc);
        bmax = bmax.max(bc);
        ms.push(m_of(x[i], b[i], neutral_b()));
    }
    let (mmin, mmax) = ms
        .iter()
        .fold((f64::MAX, f64::MIN), |a, &v| (a.0.min(v), a.1.max(v)));
    println!(
        "(i) gray ramp, 256 levels, raw Xc/Bc (C8 centring X-0.42, B-0.55); m below uses the registered neutral-axis centring:"
    );
    println!("    Xc range [{xmin:.3e}, {xmax:.3e}]   Bc range [{bmin:.6}, {bmax:.6}]");
    println!(
        "    m range [{mmin:.6}, {mmax:.6}]  spread (max-min) = {:.3e}",
        mmax - mmin
    );
    println!(
        "    Y plane at 0/128/255: {:.4} {:.4} {:.4}",
        y[0], y[128], y[255]
    );

    // (ii) natural content.
    let mut all = Vec::new();
    let mut all_n = Vec::new();
    let mut per_img = Vec::new();
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    files.sort();
    for p in files {
        let name = p.file_name().unwrap().to_str().unwrap().to_string();
        let Some(dims) = name.strip_suffix(".rgb").and_then(|s| s.rsplit('_').next()) else {
            continue;
        };
        let (w, h): (usize, usize) = {
            let (a, b2) = dims.split_once('x').expect("WxH");
            (a.parse().unwrap(), b2.parse().unwrap())
        };
        let raw = std::fs::read(&p).unwrap();
        assert_eq!(raw.len(), w * h * 3);
        let px: Vec<[u8; 3]> = raw.as_chunks::<3>().0.to_vec();
        let (mut xx, mut yy, mut bb) = (vec![0f32; w * h], vec![0f32; w * h], vec![0f32; w * h]);
        srgb_to_positive_xyb_planar_into(&px, &mut xx, &mut yy, &mut bb);
        let mut m: Vec<f64> = (0..w * h)
            .map(|i| m_of(xx[i], bb[i], neutral_b()))
            .collect();
        let q = quantiles(&mut m.clone(), &[0.5]);
        per_img.push((name, q[0]));
        all_n.extend((0..w * h).map(|i| m_of(xx[i], bb[i], f64::from(0.55f32))));
        all.append(&mut m);
    }
    let q = quantiles(&mut all, &[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]);
    println!(
        "(ii) m over {} images, {} pixels (scale 0, neutral-axis centring = registered):",
        per_img.len(),
        all.len()
    );
    println!(
        "    q01 {:.4} q05 {:.4} q25 {:.4} q50 {:.4} q75 {:.4} q95 {:.4} q99 {:.4}",
        q[0], q[1], q[2], q[3], q[4], q[5], q[6]
    );
    let mut med: Vec<f64> = per_img.iter().map(|p| p.1).collect();
    let mq = quantiles(&mut med, &[0.0, 0.5, 1.0]);
    println!(
        "    per-image median m: min {:.4} median {:.4} max {:.4}",
        mq[0], mq[1], mq[2]
    );
    let qn = quantiles(&mut all_n, &[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]);
    println!(
        "    legacy C8 centring (B - 0.55): q01 {:.4} q05 {:.4} q25 {:.4} q50 {:.4} q75 {:.4} q95 {:.4} q99 {:.4}",
        qn[0], qn[1], qn[2], qn[3], qn[4], qn[5], qn[6]
    );
    println!(
        "    leak (ramp spread / pooled median) = {:.3e}",
        (mmax - mmin) / q[3]
    );
}
