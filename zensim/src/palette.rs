//! Research-only palette_v2. Contract and bounds: benchmarks/palette_design_2026-10-07.md.
// The research error and extraction surface exist only with this feature.
#![cfg(feature = "feature-regime-v2")]

use crate::source::{ColorPrimaries, ImageSource, PixelFormat};

pub(crate) const BASE: usize = 1825;
pub(crate) const WIDTH: usize = 42;
type Lab = [f64; 3];
#[derive(Clone, Debug, PartialEq)]
struct Colour {
    lab: Lab,
    weight: f64,
}

fn distance(a: Lab, b: Lab) -> f64 {
    libm::sqrt((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2))
}
fn oklab(rgb: [u8; 3]) -> Lab {
    let [r, g, b] = rgb.map(|v| f64::from(crate::color::srgb_u8_to_linear(v)));
    let l = libm::cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
    let m = libm::cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
    let s = libm::cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
    [
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    ]
}
fn lab_cmp(a: Lab, b: Lab) -> core::cmp::Ordering {
    a[0].total_cmp(&b[0])
        .then(a[1].total_cmp(&b[1]))
        .then(a[2].total_cmp(&b[2]))
}
fn samples(source: &impl ImageSource) -> Result<Vec<Colour>, crate::research::ResearchError> {
    if source.is_hdr()
        || source.pixel_format() != PixelFormat::Srgb8Rgb
        || source.color_primaries() != ColorPrimaries::Srgb
    {
        return Err(crate::research::ResearchError::Plan("palette_v2 requires legacy RGB8 sRGB/BT.709 D65; other colour contracts are unsupported".into()));
    }
    let (w, h) = (source.width(), source.height());
    if w == 0 || h == 0 {
        return Err(crate::research::ResearchError::Plan(
            "palette_v2 requires nonempty images".into(),
        ));
    }
    let (nx, ny) = (w.min(32), h.min(32));
    let mut out = Vec::with_capacity(nx * ny);
    for y in 0..ny {
        let row = source.row_bytes(((2 * y + 1) * h / (2 * ny)).min(h - 1));
        for x in 0..nx {
            let x = ((2 * x + 1) * w / (2 * nx)).min(w - 1) * 3;
            out.push(Colour {
                lab: oklab([row[x], row[x + 1], row[x + 2]]),
                weight: 1.0,
            });
        }
    }
    out.sort_by(|a, b| lab_cmp(a.lab, b.lab));
    let mut compact: Vec<Colour> = Vec::new();
    for c in out {
        if let Some(last) = compact.last_mut()
            && last.lab == c.lab
        {
            last.weight += c.weight;
        } else {
            compact.push(c);
        }
    }
    Ok(compact)
}
fn range_axis(points: &[Colour]) -> (f64, usize) {
    let mut ranges = [0.0f64; 3];
    for (axis, range) in ranges.iter_mut().enumerate() {
        let lo = points
            .iter()
            .map(|c| c.lab[axis])
            .fold(f64::INFINITY, f64::min);
        let hi = points
            .iter()
            .map(|c| c.lab[axis])
            .fold(f64::NEG_INFINITY, f64::max);
        *range = hi - lo;
    }
    let axis = (0..3)
        .max_by(|&a, &b| ranges[a].total_cmp(&ranges[b]).then(b.cmp(&a)))
        .unwrap();
    (ranges[axis], axis)
}
fn palettes(points: Vec<Colour>) -> Vec<Vec<Colour>> {
    let total = points.iter().map(|c| c.weight).sum::<f64>();
    let mut boxes = vec![points];
    let mut out = Vec::new();
    for n in 2..=8 {
        if let Some((i, _, axis)) = boxes
            .iter()
            .enumerate()
            .filter(|(_, b)| b.len() > 1)
            .map(|(i, b)| {
                let (range, axis) = range_axis(b);
                (i, range * b.iter().map(|c| c.weight).sum::<f64>(), axis)
            })
            .max_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(&a.0)))
        {
            let mut b = boxes.remove(i);
            b.sort_by(|a, b| {
                a.lab[axis]
                    .total_cmp(&b.lab[axis])
                    .then(lab_cmp(a.lab, b.lab))
            });
            let half = b.iter().map(|c| c.weight).sum::<f64>() * 0.5;
            let mut sum = 0.0;
            let split = (1..b.len())
                .find(|&j| {
                    sum += b[j - 1].weight;
                    sum >= half
                })
                .unwrap_or(b.len() - 1);
            let right = b.split_off(split);
            boxes.insert(i, b);
            boxes.insert(i + 1, right);
        }
        let mut palette: Vec<Colour> = boxes
            .iter()
            .map(|b| {
                let mass = b.iter().map(|c| c.weight).sum::<f64>();
                let mut lab = [0.0; 3];
                for c in b {
                    for (i, v) in lab.iter_mut().enumerate() {
                        *v += c.lab[i] * c.weight;
                    }
                }
                Colour {
                    lab: lab.map(|v| v / mass),
                    weight: mass / total,
                }
            })
            .collect();
        palette.sort_by(|a, b| b.weight.total_cmp(&a.weight).then(lab_cmp(a.lab, b.lab)));
        while palette.len() < n {
            palette.push(Colour {
                lab: palette[0].lab,
                weight: 0.0,
            });
        }
        out.push(palette);
    }
    out
}
fn canonical(mut p: Vec<Colour>) -> Vec<Colour> {
    p.sort_by(|a, b| lab_cmp(a.lab, b.lab).then(a.weight.total_cmp(&b.weight)));
    p
}
/// Exact subset-DP weighted assignment, with deterministic ties.
fn assignment(a: &[Colour], b: &[Colour]) -> Vec<usize> {
    let n = a.len();
    let end = 1usize << n;
    let mut dp = vec![f64::INFINITY; end];
    let mut choice = vec![0; end];
    dp[0] = 0.0;
    for mask in 1..end {
        let i = mask.count_ones() as usize - 1;
        for (j, c) in b.iter().enumerate() {
            if mask & (1 << j) == 0 {
                continue;
            }
            let cost =
                dp[mask ^ (1 << j)] + 0.5 * (a[i].weight + c.weight) * distance(a[i].lab, c.lab);
            if cost < dp[mask] {
                dp[mask] = cost;
                choice[mask] = j;
            }
        }
    }
    let mut answer = vec![0; n];
    let mut mask = end - 1;
    for i in (0..n).rev() {
        answer[i] = choice[mask];
        mask ^= 1 << answer[i];
    }
    answer
}
/// Balanced transportation via residual shortest augmenting paths. Reverse
/// edges carry negative cost and allow earlier flows to be rerouted.
fn emd(centres: &[Colour], weights: &[f64]) -> f64 {
    let n = centres.len();
    let nodes = 2 * n + 2;
    let sink = nodes - 1;
    let mut cap = vec![vec![0.0; nodes]; nodes];
    let mut cost = cap.clone();
    for i in 0..n {
        cap[0][1 + i] = centres[i].weight;
        cap[1 + n + i][sink] = weights[i];
        for j in 0..n {
            let (u, v) = (1 + i, 1 + n + j);
            cap[u][v] = 1.0;
            cost[u][v] = distance(centres[i].lab, centres[j].lab);
            cost[v][u] = -cost[u][v];
        }
    }
    let mut value = 0.0;
    // Each path saturates an edge; bounded network, positive residual mass only.
    loop {
        let mut dist = vec![f64::INFINITY; nodes];
        let mut prev = vec![usize::MAX; nodes];
        dist[0] = 0.0;
        for _ in 1..nodes {
            let mut changed = false;
            for u in 0..nodes {
                for v in 0..nodes {
                    if cap[u][v] > 1e-15 && dist[u] + cost[u][v] < dist[v] - 1e-15 {
                        dist[v] = dist[u] + cost[u][v];
                        prev[v] = u;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        if prev[sink] == usize::MAX {
            break;
        }
        let mut amount = f64::INFINITY;
        let mut v = sink;
        while v != 0 {
            let u = prev[v];
            amount = amount.min(cap[u][v]);
            v = u;
        }
        v = sink;
        while v != 0 {
            let u = prev[v];
            cap[u][v] -= amount;
            cap[v][u] += amount;
            v = u;
        }
        value += amount * dist[sink];
    }
    value.max(0.0)
}
fn compare(a: Vec<Colour>, b: Vec<Colour>) -> [f64; 6] {
    let (a, b) = (canonical(a), canonical(b));
    if a == b {
        return [0.0; 6];
    }
    let pairs = assignment(&a, &b);
    let mut out = [0.0f64; 6];
    let mut weights = Vec::new();
    for (i, &j) in pairs.iter().enumerate() {
        let (r, d) = (&a[i], &b[j]);
        let mass = (r.weight + d.weight) * 0.5;
        let shift = distance(r.lab, d.lab);
        out[0] += mass * shift;
        out[1] += d.weight * d.lab[0] - r.weight * r.lab[0];
        let rc = libm::hypot(r.lab[1], r.lab[2]);
        let dc = libm::hypot(d.lab[1], d.lab[2]);
        out[2] += d.weight * dc - r.weight * rc;
        if rc > 1e-12 && dc > 1e-12 {
            let angle = libm::atan2(
                r.lab[1] * d.lab[2] - r.lab[2] * d.lab[1],
                r.lab[1] * d.lab[1] + r.lab[2] * d.lab[2],
            );
            out[3] += mass * rc.min(dc) * angle / core::f64::consts::TAU;
        }
        if mass > 0.0 {
            out[5] = out[5].max(shift);
        }
        weights.push(d.weight);
    }
    out[4] = emd(&a, &weights);
    out
}
pub(crate) fn extract(
    a: &impl ImageSource,
    b: &impl ImageSource,
) -> Result<[f64; WIDTH], crate::research::ResearchError> {
    if (a.width(), a.height()) != (b.width(), b.height()) {
        return Err(crate::research::ResearchError::Plan(
            "palette dimensions differ".into(),
        ));
    }
    let pa = palettes(samples(a)?);
    let pb = palettes(samples(b)?);
    let mut out = [0.0; WIDTH];
    for (i, (a, b)) in pa.into_iter().zip(pb).enumerate() {
        out[6 * i..6 * i + 6].copy_from_slice(&compare(a, b));
    }
    Ok(out)
}
#[cfg(test)]
mod tests {
    use super::*;
    fn colours() -> Vec<Colour> {
        vec![
            Colour {
                lab: [0.4, 0.1, 0.02],
                weight: 0.6,
            },
            Colour {
                lab: [0.7, -0.02, 0.1],
                weight: 0.4,
            },
        ]
    }
    #[test]
    fn identity_permutation_and_direction() {
        let a = colours();
        let mut perm = a.clone();
        perm.reverse();
        assert!(compare(a.clone(), perm).iter().all(|v| v.to_bits() == 0));
        for scale in [0.5, 1.5] {
            let mut b = a.clone();
            for c in &mut b {
                c.lab[1] *= scale;
                c.lab[2] *= scale;
            }
            let out = compare(a.clone(), b);
            assert!(out[0] > 0.0);
            assert!(out[2] * (scale - 1.0) > 0.0);
        }
        for delta in [-0.05, 0.05] {
            let mut b = a.clone();
            for c in &mut b {
                c.lab[0] += delta;
            }
            let out = compare(a.clone(), b);
            assert!((out[1] - delta).abs() < 1e-14);
            assert!(out[2].abs() < 1e-14);
        }
        for theta in [-0.2f64, 0.2] {
            let mut b = a.clone();
            for c in &mut b {
                let [_, x, y] = c.lab;
                c.lab[1] = x * libm::cos(theta) - y * libm::sin(theta);
                c.lab[2] = x * libm::sin(theta) + y * libm::cos(theta);
            }
            let out = compare(a.clone(), b);
            assert!(out[3] * theta > 0.0);
            assert!(out[2].abs() < 1e-14);
        }
    }
    #[test]
    fn transport_population_and_assignment_oracle() {
        let a = colours();
        let mut b = a.clone();
        b[0].weight = 0.3;
        b[1].weight = 0.7;
        let out = compare(a.clone(), b);
        assert!((out[4] - 0.3 * distance(a[0].lab, a[1].lab)).abs() < 1e-14);
        // Independent exhaustive permutation oracle on all 8-colour assignments.
        let a: Vec<_> = (0..8)
            .map(|i| Colour {
                lab: [i as f64 / 10.0, (i % 3) as f64 / 8.0, 0.1],
                weight: 1.0 / 8.0,
            })
            .collect();
        let mut b = a.clone();
        b.rotate_left(3);
        b[0].lab[0] += 0.01;
        fn exhaustive(a: &[Colour], b: &[Colour], i: usize, used: usize, sum: f64, best: &mut f64) {
            if i == a.len() {
                *best = best.min(sum);
                return;
            }
            for j in 0..b.len() {
                if used & (1 << j) == 0 {
                    exhaustive(
                        a,
                        b,
                        i + 1,
                        used | (1 << j),
                        sum + 0.5 * (a[i].weight + b[j].weight) * distance(a[i].lab, b[j].lab),
                        best,
                    );
                }
            }
        }
        let mut best = f64::INFINITY;
        exhaustive(&a, &b, 0, 0, 0.0, &mut best);
        let p = assignment(&a, &b);
        let got = (0..8)
            .map(|i| 0.5 * (a[i].weight + b[p[i]].weight) * distance(a[i].lab, b[p[i]].lab))
            .sum::<f64>();
        assert_eq!(got.to_bits(), best.to_bits());
        // Independent transport oracle: split integer populations into four
        // equal atoms and enumerate every bijection. This covers zero-mass
        // centres and rearranged populations without sharing the flow code.
        for n in 2..=8 {
            for seed in 0..32usize {
                let positions: Vec<_> = (0..n)
                    .map(|i| {
                        [
                            0.2 + i as f64 * 0.06,
                            ((i * 7) % 5) as f64 * 0.03,
                            ((i * 11) % 7) as f64 * 0.02,
                        ]
                    })
                    .collect();
                let source_indices: Vec<_> =
                    (0..4).map(|i| (seed * 7 + i * 3 + i * i) % n).collect();
                let target_indices: Vec<_> = (0..4)
                    .map(|i| (seed * 11 + i * 5 + i * i * i + 1) % n)
                    .collect();
                let centres: Vec<_> = (0..n)
                    .map(|i| Colour {
                        lab: positions[i],
                        weight: source_indices.iter().filter(|&&j| i == j).count() as f64 / 4.0,
                    })
                    .collect();
                let weights: Vec<_> = (0..n)
                    .map(|i| target_indices.iter().filter(|&&j| i == j).count() as f64 / 4.0)
                    .collect();
                let source_atoms: Vec<_> = source_indices
                    .iter()
                    .map(|&i| Colour {
                        lab: positions[i],
                        weight: 0.25,
                    })
                    .collect();
                let target_atoms: Vec<_> = target_indices
                    .iter()
                    .map(|&i| Colour {
                        lab: positions[i],
                        weight: 0.25,
                    })
                    .collect();
                let mut oracle = f64::INFINITY;
                exhaustive(&source_atoms, &target_atoms, 0, 0, 0.0, &mut oracle);
                let transported = emd(&centres, &weights);
                assert!(
                    (transported - oracle).abs() < 1e-14,
                    "transport N={n} seed={seed}: {transported} != {oracle}"
                );
            }
        }
    }
    // Diagnostic RGB8 edits are generated here through the same extraction
    // owner. The inverse is a test-only mirror of Ottosson's published matrix.
    fn rgb_of_lab([l, a, b]: Lab) -> [u8; 3] {
        let ll = (l + 0.3963377774 * a + 0.2158037573 * b).powi(3);
        let mm = (l - 0.1055613458 * a - 0.0638541728 * b).powi(3);
        let ss = (l - 0.0894841775 * a - 1.2914855480 * b).powi(3);
        [
            4.0767416621 * ll - 3.3077115913 * mm + 0.2309699292 * ss,
            -1.2684380046 * ll + 2.6097574011 * mm - 0.3413193965 * ss,
            -0.0041960863 * ll - 0.7034186147 * mm + 1.7076147010 * ss,
        ]
        .map(|v| {
            let v = v.clamp(0.0, 1.0);
            let s = if v <= 0.0031308 {
                12.92 * v
            } else {
                1.055 * libm::pow(v, 1.0 / 2.4) - 0.055
            };
            (s * 255.0).round() as u8
        })
    }
    #[test]
    fn global_rgb8_edits_have_direction_at_every_n() {
        let labs: Vec<_> = (0..8)
            .map(|i| {
                [
                    0.3 + i as f64 * 0.055,
                    0.018 * libm::cos(0.4),
                    0.018 * libm::sin(0.4),
                ]
            })
            .collect();
        let image = |colours: &[Lab]| {
            (0..1024)
                .map(|i| rgb_of_lab(colours[(i % 32) / 4]))
                .collect::<Vec<_>>()
        };
        let src = image(&labs);
        let source = crate::RgbSlice::new(&src, 32, 32);
        for (signal, delta) in [
            (1, -0.03),
            (1, 0.03),
            (2, -0.5),
            (2, 0.5),
            (3, -0.2),
            (3, 0.2),
        ] {
            let mut edited = labs.clone();
            for lab in &mut edited {
                match signal {
                    1 => lab[0] += delta,
                    2 => {
                        lab[1] *= 1.0 + delta;
                        lab[2] *= 1.0 + delta;
                    }
                    3 => {
                        let [_, a, b] = *lab;
                        lab[1] = a * libm::cos(delta) - b * libm::sin(delta);
                        lab[2] = a * libm::sin(delta) + b * libm::cos(delta);
                    }
                    _ => unreachable!(),
                }
            }
            let dst = image(&edited);
            let values = extract(&source, &crate::RgbSlice::new(&dst, 32, 32)).unwrap();
            for n in 2..=8 {
                assert!(
                    values[(n - 2) * 6 + signal] * delta > 0.0,
                    "global edit signal {signal} delta {delta}, N={n}: {:?}",
                    &values[(n - 2) * 6..(n - 1) * 6]
                );
            }
        }
    }
    #[test]
    fn unlabelled_colour_edits() {
        let labs: Vec<Lab> = (0..8)
            .map(|i| {
                let theta = i as f64 * core::f64::consts::TAU / 8.0;
                [
                    0.55 + 0.02 * (i % 3) as f64,
                    0.045 * libm::cos(theta),
                    0.045 * libm::sin(theta),
                ]
            })
            .collect();
        let src: Vec<_> = (0..1024).map(|i| rgb_of_lab(labs[i % 8])).collect();
        let source = crate::source::RgbSlice::new(&src, 32, 32);
        let mut rows = String::from(
            "edit,level,n,shift,lightness_signed,chroma_signed,hue_signed,weight_emd,largest_shift\n",
        );
        for (kind, levels) in [
            (
                "chroma",
                (0..=20).map(|i| i as f64 / 10.0).collect::<Vec<_>>(),
            ),
            (
                "hue",
                (-12..=12)
                    .map(|i| i as f64 * core::f64::consts::PI / 12.0)
                    .collect(),
            ),
            ("lightness", (-4..=4).map(|i| i as f64 * 0.025).collect()),
        ] {
            for level in levels {
                let dst: Vec<_> = (0..1024)
                    .map(|i| {
                        let [l, a, b] = labs[i % 8];
                        rgb_of_lab(match kind {
                            "chroma" => [l, a * level, b * level],
                            "hue" => [
                                l,
                                a * libm::cos(level) - b * libm::sin(level),
                                a * libm::sin(level) + b * libm::cos(level),
                            ],
                            _ => [l + level, a, b],
                        })
                    })
                    .collect();
                let dist = crate::source::RgbSlice::new(&dst, 32, 32);
                let values = extract(&source, &dist).unwrap();
                for n in 2..=8 {
                    let f = &values[(n - 2) * 6..(n - 1) * 6];
                    if kind == "chroma" && (level - 1.0).abs() > 0.2 {
                        assert!(f[2] * (level - 1.0) > 0.0);
                    }
                    if kind == "lightness" && level != 0.0 {
                        assert!(f[1] * level > 0.0);
                    }
                    rows.push_str(&format!(
                        "{kind},{level},{n},{},{},{},{},{},{}\n",
                        f[0], f[1], f[2], f[3], f[4], f[5]
                    ));
                }
            }
        }
        if let Some(path) = std::env::var_os("PALETTE_DIAGNOSTIC_OUT") {
            use std::io::Write;
            std::fs::OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(path)
                .unwrap()
                .write_all(rows.as_bytes())
                .unwrap();
        }
    }

    #[test]
    fn extraction_determinism_and_all_n() {
        let rgb: Vec<_> = (0..1024)
            .map(|i| [(i % 251) as u8, (i * 13 % 251) as u8, (i * 31 % 251) as u8])
            .collect();
        let source = crate::source::RgbSlice::new(&rgb, 32, 32);
        assert!(
            extract(&source, &source)
                .unwrap()
                .iter()
                .all(|v| v.to_bits() == 0)
        );
        let changed: Vec<_> = rgb.iter().map(|p| [p[0] / 2, p[1], p[2]]).collect();
        let dist = crate::source::RgbSlice::new(&changed, 32, 32);
        assert_eq!(
            extract(&source, &dist).unwrap(),
            extract(&source, &dist).unwrap()
        );
        let ps = palettes(samples(&source).unwrap());
        for (i, p) in ps.iter().enumerate() {
            assert_eq!(p.len(), i + 2);
            assert!((p.iter().map(|c| c.weight).sum::<f64>() - 1.0).abs() < 1e-14);
        }
    }
}
