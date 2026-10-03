//! Per-feature-ID steering support probe: unit sensitivity on one ID at a time,
//! |density| mass through the buffered and fused-944 attribution entries.
use zensim::profile::ProfileParams;
use zensim::{Fused944Session, RgbSlice, Zensim, ZensimProfile};
fn pair(w: usize, h: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
    let mut s = Vec::new();
    let mut d = Vec::new();
    let mut st = 0x9e3779b97f4a7c15u64;
    let mut rnd = || { st ^= st << 13; st ^= st >> 7; st ^= st << 17; (st >> 33) as u32 };
    for y in 0..h {
        for x in 0..w {
            let base = ((x * 255) / w) as u8;
            let tex = (((x * 7 + y * 13) % 32) * 3) as u8;
            let edge = if (y / 16) % 2 == 0 { 40 } else { 0 };
            let n = (rnd() % 24) as u8;
            let px = [base.wrapping_add(tex), base.wrapping_add(edge).wrapping_add(n), (255 - base).wrapping_add(tex / 2)];
            s.push(px);
            let q = |v: u8| (v / 12) * 12;
            let mut dd = [q(px[0]), q(px[1]), q(px[2])];
            if x < w / 2 && y < h / 2 { dd[0] = dd[0].saturating_add(18); }
            if (x / 8 + y / 8) % 5 == 0 { dd[2] = dd[2].saturating_sub(30); }
            // soft ringing/banding-ish: smooth ramp in lower right
            if x > w / 2 && y > h / 2 { dd[1] = (dd[1] / 40) * 40; }
            d.push(dd);
        }
    }
    (s, d)
}
fn main() {
    let (w, h) = (192usize, 160usize);
    let (s, d) = pair(w, h);
    let rs = RgbSlice::new(&s, w, h);
    let ds = RgbSlice::new(&d, w, h);
    let params: &'static ProfileParams = Box::leak(Box::new(ProfileParams::builder().extended_features(true).build()));
    let z = Zensim::new(ZensimProfile::Custom { params, name: "probe" });
    let pre = z.precompute_reference(&rs).unwrap();
    let mut sess = Fused944Session::new();
    println!("id\tbuffered_mass\tfused_mass");
    for k in 0..944usize {
        let mut v = vec![0.0f64; 944];
        v[k] = -1.0;
        let m1: f64 = match z.compute_attribution_density_full(&rs, &ds, &v) {
            Ok(r) => r.density().iter().map(|x| x.abs() as f64).sum(),
            Err(e) => { eprintln!("buf {k}: {e}"); f64::NAN }
        };
        let m2: f64 = match z.compute_folded944_score_and_attribution(&rs, &pre, &ds, &v, &mut sess) {
            Ok((_, _, r)) => r.density().iter().map(|x| x.abs() as f64).sum(),
            Err(e) => { eprintln!("fused {k}: {e}"); f64::NAN }
        };
        println!("{k}\t{m1:e}\t{m2:e}");
    }
}
