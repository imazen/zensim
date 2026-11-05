//! Isolated path audit: exact output fingerprints, zenbench latency, or fresh-process RSS.
//! Usage: path_audit <check|bench|memory> <profile> <compute|cached|scratch|map|planar>
//!        <width> <height> <threads> <auto|scalar> [zenbench rounds]
//! Inputs and cached references are prepared outside the timed calls. RSS includes setup.
#![forbid(unsafe_code)]

#[path = "path_audit/fixture.rs"]
mod fixture;
#[path = "path_audit/profile.rs"]
mod profile;

use std::hint::black_box;
use std::time::Duration;
use zensim::{DiffmapOptions, PrecomputedReference, RgbSlice, Zensim, ZensimResult, ZensimScratch};

struct Case {
    z: Zensim,
    src: Vec<[u8; 3]>,
    dst: Vec<[u8; 3]>,
    pre: Option<PrecomputedReference>,
    planes: Option<[Vec<f32>; 3]>,
    scratch: ZensimScratch,
    width: usize,
    height: usize,
    route: String,
}

fn hash(words: impl Iterator<Item = u64>) -> u64 {
    words.fold(0xcbf2_9ce4_8422_2325, |mut h, w| {
        for b in w.to_le_bytes() {
            h = (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
        }
        h
    })
}

fn score_digest(r: &ZensimResult) -> [u64; 4] {
    [
        r.score().to_bits(),
        r.raw_distance().to_bits(),
        hash(r.features().iter().map(|v| v.to_bits())),
        0,
    ]
}

impl Case {
    fn new(name: &str, route: &str, width: usize, height: usize, threads: usize) -> Self {
        let (src, dst) = fixture::pair(width, height);
        let z = Zensim::new(profile::profile(name)).with_parallel(threads > 1);
        let pre = (route != "compute").then(|| {
            z.precompute_reference(&RgbSlice::new(&src, width, height))
                .unwrap()
        });
        let planes = (route == "planar").then(|| {
            let stride = width + 7;
            let mut planes = [
                vec![f32::NAN; stride * height],
                vec![f32::NAN; stride * height],
                vec![f32::NAN; stride * height],
            ];
            for y in 0..height {
                for x in 0..width {
                    for c in 0..3 {
                        let v = f64::from(dst[y * width + x][c]) / 255.0;
                        planes[c][y * stride + x] = (if v <= 0.04045 {
                            v / 12.92
                        } else {
                            ((v + 0.055) / 1.055).powf(2.4)
                        }) as f32;
                    }
                }
            }
            planes
        });
        Self {
            z,
            src,
            dst,
            pre,
            planes,
            scratch: ZensimScratch::new(),
            width,
            height,
            route: route.into(),
        }
    }

    fn run(&mut self, digest: bool) -> [u64; 4] {
        let dst = RgbSlice::new(&self.dst, self.width, self.height);
        match self.route.as_str() {
            "compute" | "cached" => {
                let result = if self.route == "compute" {
                    self.z
                        .compute(&RgbSlice::new(&self.src, self.width, self.height), &dst)
                        .unwrap()
                } else {
                    self.z
                        .compute_with_ref(self.pre.as_ref().unwrap(), &dst)
                        .unwrap()
                };
                let out = if digest {
                    score_digest(&result)
                } else {
                    [0; 4]
                };
                black_box(result);
                out
            }
            "scratch" => {
                let result = self
                    .z
                    .compute_with_ref_into(self.pre.as_ref().unwrap(), &dst, &mut self.scratch)
                    .unwrap();
                let out = if digest {
                    [
                        result.score().to_bits(),
                        result.raw_distance().to_bits(),
                        hash(result.features().iter().map(|v| v.to_bits())),
                        0,
                    ]
                } else {
                    [0; 4]
                };
                black_box(result);
                out
            }
            "map" | "planar" => {
                let options = DiffmapOptions {
                    masking_strength: Some(4.0),
                    sqrt: true,
                    include_edge_mse: true,
                    include_hf: true,
                    ..Default::default()
                };
                let result = if self.route == "map" {
                    self.z
                        .compute_with_ref_and_diffmap(self.pre.as_ref().unwrap(), &dst, options)
                        .unwrap()
                } else {
                    let p = self.planes.as_ref().unwrap();
                    self.z
                        .compute_with_ref_and_diffmap_linear_planar(
                            self.pre.as_ref().unwrap(),
                            [&p[0], &p[1], &p[2]],
                            self.width,
                            self.height,
                            self.width + 7,
                            options,
                        )
                        .unwrap()
                };
                let out = if digest {
                    let mut out = score_digest(result.result());
                    out[3] = hash(result.diffmap().iter().map(|v| u64::from(v.to_bits())));
                    out
                } else {
                    [0; 4]
                };
                black_box(result);
                out
            }
            _ => panic!("unknown route: {}", self.route),
        }
    }
}

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    assert!(
        args.len() >= 7,
        "see example module documentation for arguments"
    );
    let (width, height, threads) = (
        args[3].parse().unwrap(),
        args[4].parse().unwrap(),
        args[5].parse().unwrap(),
    );
    let _tokens = archmage::testing::lock_token_testing();
    if args[6] == "scalar" {
        #[cfg(target_arch = "x86_64")]
        {
            use archmage::{SimdToken, X64V3Token, X64V4Token, X64V4xToken};
            X64V3Token::dangerously_disable_token_process_wide(true).unwrap();
            X64V4Token::dangerously_disable_token_process_wide(true).unwrap();
            X64V4xToken::dangerously_disable_token_process_wide(true).unwrap();
            assert!(
                X64V3Token::summon().is_none()
                    && X64V4Token::summon().is_none()
                    && X64V4xToken::summon().is_none()
            );
        }
        #[cfg(target_arch = "aarch64")]
        {
            use archmage::{NeonToken, SimdToken};
            NeonToken::dangerously_disable_token_process_wide(true).unwrap();
            assert!(NeonToken::summon().is_none());
        }
    } else {
        assert_eq!(args[6], "auto");
    }
    #[cfg(feature = "threads")]
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();
    #[cfg(not(feature = "threads"))]
    assert_eq!(
        threads, 1,
        "multi-thread audit requires the threads feature"
    );
    let mut case = Case::new(&args[1], &args[2], width, height, threads);
    let fingerprint = case.run(true);
    if args[0] == "bench" {
        let rounds = args.get(7).map_or(20, |s| s.parse().unwrap());
        let result = zenbench::run(move |suite| {
            suite.group("path", |g| {
                g.config()
                    .min_rounds(rounds)
                    .max_rounds(rounds)
                    .warmup_time(Duration::from_millis(100))
                    .max_time(Duration::from_secs(60));
                g.bench("call", move |b| b.iter(|| case.run(false)));
            });
        });
        println!(
            "{}",
            serde_json::json!({"fingerprint": fingerprint, "timing": result})
        );
    } else {
        assert!(args[0] == "check" || args[0] == "memory");
        println!("{}", serde_json::json!({"fingerprint": fingerprint}));
    }
}
