//! Interleaved A/B wall timing of one layer-1 pair step at screen_main width (1853 inputs):
//!   OLD: fused kernel (g loaded and stored) + separate per-row group-lasso pass over all of w1
//!   NEW: fused kernel with the prox folded into the row loop and `g_zero_in`
//! Alternates OLD/NEW every step, reports min and median ms/step per shape. Pin with `taskset`; run alone
//! on a quiet core pair. Not a gate (the bit-identity gates are the adam_simd tests and the trainer cells).

mod tier_cap {
    pub fn avx512_allowed() -> bool {
        false
    }
}

#[allow(dead_code)]
#[path = "../src/simd_mlp.rs"]
mod simd_mlp;

#[allow(dead_code)]
#[path = "../src/adam_simd.rs"]
mod adam_simd;

use std::hint::black_box;
use std::time::Instant;

use adam_simd::{AdamW1FusedArgs, adam_update_w1_fused, group_l1_row};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

struct State {
    w: Vec<f64>,
    g: Vec<f64>,
    m: Vec<f64>,
    v: Vec<f64>,
}

fn fresh(n: usize, seed: u64) -> State {
    let mut r = Rng(seed | 1);
    State {
        w: (0..n).map(|_| r.next() * 0.05).collect(),
        g: vec![0.0; n],
        m: (0..n).map(|_| r.next() * 1e-3).collect(),
        v: (0..n).map(|_| r.next().abs() * 1e-6).collect(),
    }
}

#[allow(clippy::too_many_arguments)]
fn step(
    s: &mut State,
    nh: usize,
    xa: &[f64],
    xb: &[f64],
    dha: &[f64],
    dhb: &[f64],
    tau: Option<f64>,
    gz: bool,
) {
    adam_update_w1_fused(&mut AdamW1FusedArgs {
        w: &mut s.w,
        g: &mut s.g,
        m: &mut s.m,
        v: &mut s.v,
        xa: black_box(xa),
        dha,
        xb: black_box(xb),
        dhb,
        l2_scale: 1e-5,
        l2_mult: None,
        n_hidden: nh,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        bc1: 0.1,
        bc2: 0.001,
        lr: 0.005,
        active_rows: None,
        group_l1_tau: tau,
        g_zero_in: gz,
    });
}

fn main() {
    let nf = 1853usize;
    let reps = 120usize;
    for &nh in &[128usize, 64] {
        for &lambda in &[0.0f64, 3.0] {
            let n = nf * nh;
            let mut r = Rng(0x5eed);
            let xa: Vec<f64> = (0..nf).map(|_| r.next()).collect();
            let xb: Vec<f64> = (0..nf).map(|_| r.next()).collect();
            let dha: Vec<f64> = (0..nh).map(|_| r.next() * 1e-2).collect();
            let dhb: Vec<f64> = (0..nh).map(|_| r.next() * 1e-2).collect();
            let (mut old, mut new) = (fresh(n, 7), fresh(n, 7));
            let tau = (lambda > 0.0).then_some(0.005 * lambda);
            let (mut t_old, mut t_new) = (Vec::new(), Vec::new());
            for rep in 0..reps {
                for first_old in [rep % 2 == 0, rep % 2 != 0] {
                    if first_old {
                        let t = Instant::now();
                        step(&mut old, nh, &xa, &xb, &dha, &dhb, None, false);
                        if let Some(tau) = tau {
                            for row in old.w.chunks_mut(nh) {
                                group_l1_row(row, tau);
                            }
                        }
                        t_old.push(t.elapsed().as_secs_f64() * 1e3);
                    } else {
                        let t = Instant::now();
                        step(&mut new, nh, &xa, &xb, &dha, &dhb, tau, true);
                        t_new.push(t.elapsed().as_secs_f64() * 1e3);
                    }
                }
            }
            assert_eq!(
                old.w.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                new.w.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
            let stat = |v: &mut Vec<f64>| {
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                (v[0], v[v.len() / 2])
            };
            let (o, nw) = (stat(&mut t_old), stat(&mut t_new));
            println!(
                "nf={nf} nh={nh} lambda={lambda}: old min/med {:.3}/{:.3} ms  new min/med {:.3}/{:.3} ms  speedup(min) {:.2}x (med) {:.2}x",
                o.0,
                o.1,
                nw.0,
                nw.1,
                o.0 / nw.0,
                o.1 / nw.1
            );
        }
    }
}
