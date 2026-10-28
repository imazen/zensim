//! Per-call wall time of one profile on one generated pair, single-threaded.
//!
//! Half of the slim-versus-research paired timing: zensimx has a twin example
//! with the same arguments and output, and a driver alternates the two binaries.
//!
//! ```text
//! split_timing <E|ECodec|A|PreviewV0_2> <pair-id> <warmup> <calls>
//! ```
//! Prints one line per timed call: nanoseconds.

include!("split_parity/pairs.rs");

use std::hint::black_box;
use std::time::Instant;
use zensim::{RgbSlice, Zensim, ZensimProfile};

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let profile = match a[0].as_str() {
        "E" => ZensimProfile::E,
        "ECodec" => ZensimProfile::ECodec,
        "A" => ZensimProfile::A,
        "PreviewV0_2" => ZensimProfile::PreviewV0_2,
        other => panic!("unknown profile {other}"),
    };
    let pair = split_parity_pairs().into_iter().find(|p| p.id == a[1]).expect("pair id");
    let (warmup, calls): (usize, usize) = (a[2].parse().unwrap(), a[3].parse().unwrap());
    let src = RgbSlice::new(bytemuck::cast_slice(&pair.src), pair.width, pair.height);
    let dst = RgbSlice::new(bytemuck::cast_slice(&pair.dst), pair.width, pair.height);
    let z = Zensim::new(profile).with_parallel(false);
    for _ in 0..warmup {
        black_box(z.compute(&src, &dst).unwrap());
    }
    for _ in 0..calls {
        let t = Instant::now();
        let r = z.compute(&src, &dst).unwrap();
        let dt = t.elapsed();
        black_box(r);
        println!("{}", dt.as_nanos());
    }
}
