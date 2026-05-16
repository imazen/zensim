//! Rust port of `scripts/v_next/affine_calibrate_znpr_v2.py`.
//!
//! Applies `y' = α + β · y` calibration to a ZNPR v2/v3 bake by
//! modifying the final layer's weights + bias in-place:
//!
//!   W' = β · W
//!   b' = β · b + α
//!
//! Eliminates the Python step from the V_X ship pipeline so the
//! entire reproduce chain is `cargo run`. v2 and v3 share the same
//! layer-table layout for the first 96 header bytes; v3.1's reserved
//! fields (offset 96..128) are zero for the F32 bakes we calibrate.
//!
//! ## Usage
//!
//! ```sh
//! cargo run --release -p zensim-validate --bin affine_calibrate -- \
//!   --in-bake benchmarks/v0_X_concat_3way.bin \
//!   --out-bake zensim/weights/v0_X_$(date -u +%Y-%m-%d).bin \
//!   --alpha 28.0366 --beta -5.0738
//! ```

use std::fs;
use std::path::PathBuf;

use clap::Parser;

const HEADER_SIZE: usize = 128;
const LAYER_ENTRY_SIZE: usize = 48;

#[derive(Parser)]
#[command(about = "Affine-calibrate the final layer of a ZNPR v2/v3 bake (y' = α + β·y)")]
struct Args {
    /// Input bake (ZNPR v2 or v3 F32).
    #[arg(long)]
    in_bake: PathBuf,
    /// Output bake path (will be overwritten).
    #[arg(long)]
    out_bake: PathBuf,
    /// Affine offset α in `y' = α + β · y`.
    #[arg(long)]
    alpha: f32,
    /// Affine scale β in `y' = α + β · y`.
    #[arg(long)]
    beta: f32,
}

fn read_u32(bytes: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
}

fn read_u16(bytes: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([bytes[off], bytes[off + 1]])
}

fn main() {
    let args = Args::parse();
    let mut data = fs::read(&args.in_bake).expect("read in-bake");
    assert!(data.len() >= HEADER_SIZE, "bake too small");
    assert_eq!(&data[0..4], b"ZNPR", "bad magic");
    let version = read_u16(&data, 4);
    assert!(
        version == 2 || version == 3,
        "expected v2 or v3, got {version}"
    );
    let n_outputs = read_u32(&data, 12) as usize;
    let n_layers = read_u32(&data, 16) as usize;
    let layer_table_off = read_u32(&data, 48) as usize;
    assert_eq!(
        n_outputs, 1,
        "calibration assumes scalar output; got {n_outputs}"
    );
    assert!(n_layers >= 1, "no layers");

    // Read the last layer's weights + biases section offsets.
    let last_idx = n_layers - 1;
    let entry_off = layer_table_off + last_idx * LAYER_ENTRY_SIZE;
    let in_dim = read_u32(&data, entry_off) as usize;
    let out_dim = read_u32(&data, entry_off + 4) as usize;
    assert_eq!(out_dim, 1, "final layer must have out_dim=1");
    let weight_dtype = data[entry_off + 9];
    assert_eq!(
        weight_dtype, 0,
        "calibrate requires F32 weights (dtype=0); got dtype={weight_dtype}"
    );

    let w_off = read_u32(&data, entry_off + 12) as usize;
    let w_len = read_u32(&data, entry_off + 16) as usize;
    let b_off = read_u32(&data, entry_off + 28) as usize;
    let b_len = read_u32(&data, entry_off + 32) as usize;
    assert_eq!(w_len, in_dim * 4, "weights section size mismatch");
    assert_eq!(b_len, 4, "bias section size mismatch (scalar output)");

    // Multiply final-layer weights by β.
    for i in 0..in_dim {
        let off = w_off + i * 4;
        let w = f32::from_le_bytes(data[off..off + 4].try_into().unwrap());
        let new_w = w * args.beta;
        data[off..off + 4].copy_from_slice(&new_w.to_le_bytes());
    }
    // b' = β · b + α.
    let b = f32::from_le_bytes(data[b_off..b_off + 4].try_into().unwrap());
    let new_b = args.beta * b + args.alpha;
    data[b_off..b_off + 4].copy_from_slice(&new_b.to_le_bytes());

    fs::write(&args.out_bake, &data).expect("write out-bake");
    eprintln!(
        "affine_calibrate: y' = {} + {} · y applied to final layer; wrote {} ({} bytes)",
        args.alpha,
        args.beta,
        args.out_bake.display(),
        data.len()
    );
}
