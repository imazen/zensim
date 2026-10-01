//! Full-width feature-vector dump for the SIGNEDFEAT before/after gate: writes `research::extract(everything())`
//! as little-endian f64 bits for every pair in a `scripts/signedfeat/prep_parity_pairs.py` directory.
//!
//! `ZENSIM_FORMULA_REV=<1..4> cargo run --release -p zensim --features training --example signedfeat_dump_vector -- <parity dir> <out dir>`
//! The same example builds on trees with and without the new families; `scripts/signedfeat/compare_dumps.py`
//! compares the common prefix cell by cell.
use zensim::RgbSlice;
use zensim::research::{self, Request};

fn main() {
    let mut a = std::env::args().skip(1);
    let dir = a.next().expect("parity dir");
    let out = a.next().expect("out dir");
    std::fs::create_dir_all(&out).unwrap();
    let index = std::fs::read_to_string(format!("{dir}/index.tsv")).expect("index.tsv");
    for line in index.lines().skip(1) {
        let f: Vec<&str> = line.split('\t').collect();
        let (name, n): (&str, usize) = (f[0], f[3].parse().unwrap());
        let rd = |s: &str| -> Vec<[u8; 3]> {
            std::fs::read(format!("{dir}/{name}_{s}.rgb"))
                .unwrap()
                .as_chunks::<3>()
                .0
                .to_vec()
        };
        let (src, dst) = (rd("ref"), rd("dst"));
        let e = research::extract(
            &Request::everything(),
            &RgbSlice::new(&src, n, n),
            &RgbSlice::new(&dst, n, n),
        )
        .expect("extract");
        let mut bytes = Vec::with_capacity(e.values().len() * 8);
        for v in e.values() {
            bytes.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        std::fs::write(format!("{out}/{name}.f64"), bytes).unwrap();
        eprintln!("{name}: {} slots", e.values().len());
    }
}
