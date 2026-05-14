//! Content-addressable cache for trained MLP bakes.
//!
//! The trainer hashes (input CSV md5s + all hyperparameter values +
//! binary version) into a single 32-hex-char key. If a cached bake
//! exists at `$ZENSIM_TRAIN_CACHE/<key>.bin`, the trainer copies it
//! to `--out` and returns in milliseconds. Otherwise it trains
//! normally and writes BOTH `--out` AND the cache entry.
//!
//! Designed for the Dockerfile / peer-review flow described by the
//! user 2026-05-14: "we want dockerfiles to be the easy place to
//! edit flags, rust rebuilds suck — we want peer reviews and
//! explorers to tweak params in the dockerfile with as many cached
//! stages as possible so it is trivial to play with without losing
//! progress." Docker layer caching invalidates downstream of any
//! ARG change; the Rust-side cache makes those reruns instant when
//! the actual (inputs, flags) tuple hasn't changed (e.g. you tweaked
//! a flag that only affects a different RUN line).
//!
//! ## Cache key
//!
//! ```text
//! SHA-256(
//!   binary_version || "\0" ||
//!   csv1_md5 || "\0" || csv2_md5 || ... || "\0" ||
//!   serialize(hyperparams) || "\0"
//! ) | first 16 bytes → 32 hex chars
//! ```
//!
//! `binary_version` is the trainer's git rev (or
//! `CARGO_PKG_VERSION + build_timestamp` fallback). Including it
//! means a code change automatically invalidates the cache.
//!
//! ## Cache location
//!
//! `$ZENSIM_TRAIN_CACHE` env var (default: `$HOME/.cache/zensim-train`).
//! Dockerfile mounts this as a volume so cache persists across
//! container rebuilds.
//!
//! ## Sidecar
//!
//! Alongside `<key>.bin` we write `<key>.json` with:
//! - cache key inputs (hyperparams, input md5s, binary version)
//! - timestamp
//! - training-side metrics (best val_mean SROCC, final epoch)
//!
//! This lets a reviewer `cat <key>.json` to understand WHICH config
//! produced a cached bake without re-running.

extern crate alloc;

use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Get the cache directory, creating it if needed.
pub fn cache_dir() -> PathBuf {
    let dir = std::env::var("ZENSIM_TRAIN_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
            PathBuf::from(home).join(".cache").join("zensim-train")
        });
    let _ = fs::create_dir_all(&dir);
    dir
}

/// Compute md5 of a file, streaming. Returns 32-hex-char lowercase.
pub fn md5_of_file(path: &Path) -> std::io::Result<String> {
    let mut f = fs::File::open(path)?;
    let mut h = Md5::new();
    let mut buf = [0u8; 8192];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex_lower(&h.finalize()))
}

/// Compute the cache key for a training run. The key is the SHA-256
/// of `binary_version || \0 || sorted_inputs_md5s || \0 || sorted_flags`
/// truncated to 16 bytes (32 hex chars).
pub fn cache_key(
    binary_version: &str,
    input_files: &[(&str, &Path)],
    flags: &BTreeMap<&'static str, String>,
) -> std::io::Result<String> {
    let mut h = Sha256::new();
    h.update(binary_version.as_bytes());
    h.update(b"\0");
    let mut input_lines: Vec<String> = Vec::with_capacity(input_files.len());
    for (label, path) in input_files {
        let md5 = md5_of_file(path)?;
        input_lines.push(format!("{label}={md5}"));
    }
    input_lines.sort();
    for line in &input_lines {
        h.update(line.as_bytes());
        h.update(b"\0");
    }
    for (k, v) in flags {
        h.update(k.as_bytes());
        h.update(b"=");
        h.update(v.as_bytes());
        h.update(b"\0");
    }
    let digest = h.finalize();
    Ok(hex_lower(&digest[..16]))
}

/// Check cache. If hit, copy `<key>.bin` → `out_path` and return true.
pub fn cache_lookup(key: &str, out_path: &Path) -> std::io::Result<bool> {
    let cached = cache_dir().join(format!("{key}.bin"));
    if cached.exists() {
        fs::copy(&cached, out_path)?;
        eprintln!(
            "train_cache: HIT key={key} → copied to {} ({} bytes)",
            out_path.display(),
            fs::metadata(out_path)?.len()
        );
        Ok(true)
    } else {
        eprintln!("train_cache: miss key={key} (will train)");
        Ok(false)
    }
}

/// Insert into cache: write bake bytes to `<key>.bin` + metadata sidecar.
pub fn cache_insert(
    key: &str,
    bake_bytes: &[u8],
    binary_version: &str,
    input_files: &[(&str, &Path)],
    flags: &BTreeMap<&'static str, String>,
    extra_metadata: &BTreeMap<&'static str, String>,
) -> std::io::Result<()> {
    let dir = cache_dir();
    let bin_path = dir.join(format!("{key}.bin"));
    let json_path = dir.join(format!("{key}.json"));
    fs::write(&bin_path, bake_bytes)?;
    let mut sidecar = String::new();
    sidecar.push_str("{\n");
    sidecar.push_str(&format!("  \"key\": \"{key}\",\n"));
    sidecar.push_str(&format!("  \"binary_version\": \"{binary_version}\",\n"));
    sidecar.push_str(&format!(
        "  \"timestamp_utc\": \"{}\",\n",
        current_utc_iso()
    ));
    sidecar.push_str("  \"inputs\": {\n");
    for (i, (label, path)) in input_files.iter().enumerate() {
        let comma = if i + 1 < input_files.len() { "," } else { "" };
        let md5 = md5_of_file(path).unwrap_or_else(|_| "READ_ERROR".into());
        sidecar.push_str(&format!(
            "    \"{label}\": {{ \"path\": \"{}\", \"md5\": \"{md5}\" }}{comma}\n",
            path.display()
        ));
    }
    sidecar.push_str("  },\n");
    sidecar.push_str("  \"flags\": {\n");
    let n = flags.len();
    for (i, (k, v)) in flags.iter().enumerate() {
        let comma = if i + 1 < n { "," } else { "" };
        sidecar.push_str(&format!("    \"{k}\": \"{v}\"{comma}\n"));
    }
    sidecar.push_str("  }");
    if !extra_metadata.is_empty() {
        sidecar.push_str(",\n  \"metrics\": {\n");
        let n = extra_metadata.len();
        for (i, (k, v)) in extra_metadata.iter().enumerate() {
            let comma = if i + 1 < n { "," } else { "" };
            sidecar.push_str(&format!("    \"{k}\": \"{v}\"{comma}\n"));
        }
        sidecar.push_str("  }");
    }
    sidecar.push_str("\n}\n");
    fs::write(&json_path, &sidecar)?;
    eprintln!(
        "train_cache: INSERT key={key} → {} ({} bytes) + {}",
        bin_path.display(),
        bake_bytes.len(),
        json_path.display()
    );
    Ok(())
}

// ── tiny in-tree md5 + sha256 (no external deps) ───────────────────
//
// We use small hand-rolled hashers to keep zensim-validate free of
// extra Cargo deps. These are NOT designed for cryptographic strength;
// they're cache-keying hash functions where the only adversary is
// accidental collision.

struct Md5 {
    state: [u32; 4],
    buffer: Vec<u8>,
    total_len: u64,
}

impl Md5 {
    fn new() -> Self {
        Self {
            state: [0x6745_2301, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476],
            buffer: Vec::new(),
            total_len: 0,
        }
    }
    fn update(&mut self, data: &[u8]) {
        self.total_len += data.len() as u64;
        self.buffer.extend_from_slice(data);
        let mut consumed = 0;
        while self.buffer.len() - consumed >= 64 {
            let chunk = &self.buffer[consumed..consumed + 64];
            md5_compress(&mut self.state, chunk);
            consumed += 64;
        }
        self.buffer.drain(..consumed);
    }
    fn finalize(mut self) -> [u8; 16] {
        let bit_len = self.total_len.wrapping_mul(8);
        self.buffer.push(0x80);
        while self.buffer.len() % 64 != 56 {
            self.buffer.push(0);
        }
        self.buffer.extend_from_slice(&bit_len.to_le_bytes());
        let mut consumed = 0;
        while self.buffer.len() - consumed >= 64 {
            md5_compress(&mut self.state, &self.buffer[consumed..consumed + 64]);
            consumed += 64;
        }
        let mut out = [0u8; 16];
        for (i, w) in self.state.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
        out
    }
}

fn md5_compress(state: &mut [u32; 4], chunk: &[u8]) {
    const K: [u32; 64] = [
        0xd76a_a478, 0xe8c7_b756, 0x2420_70db, 0xc1bd_ceee,
        0xf57c_0faf, 0x4787_c62a, 0xa830_4613, 0xfd46_9501,
        0x6980_98d8, 0x8b44_f7af, 0xffff_5bb1, 0x895c_d7be,
        0x6b90_1122, 0xfd98_7193, 0xa679_438e, 0x49b4_0821,
        0xf61e_2562, 0xc040_b340, 0x265e_5a51, 0xe9b6_c7aa,
        0xd62f_105d, 0x0244_1453, 0xd8a1_e681, 0xe7d3_fbc8,
        0x21e1_cde6, 0xc337_07d6, 0xf4d5_0d87, 0x455a_14ed,
        0xa9e3_e905, 0xfcef_a3f8, 0x676f_02d9, 0x8d2a_4c8a,
        0xfffa_3942, 0x8771_f681, 0x6d9d_6122, 0xfde5_380c,
        0xa4be_ea44, 0x4bde_cfa9, 0xf6bb_4b60, 0xbebf_bc70,
        0x289b_7ec6, 0xeaa1_27fa, 0xd4ef_3085, 0x0488_1d05,
        0xd9d4_d039, 0xe6db_99e5, 0x1fa2_7cf8, 0xc4ac_5665,
        0xf429_2244, 0x432a_ff97, 0xab94_23a7, 0xfc93_a039,
        0x655b_59c3, 0x8f0c_cc92, 0xffef_f47d, 0x8584_5dd1,
        0x6fa8_7e4f, 0xfe2c_e6e0, 0xa301_4314, 0x4e08_11a1,
        0xf753_7e82, 0xbd3a_f235, 0x2ad7_d2bb, 0xeb86_d391,
    ];
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
        5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20,
        4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
        6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    let mut m = [0u32; 16];
    for i in 0..16 {
        m[i] = u32::from_le_bytes(chunk[i * 4..i * 4 + 4].try_into().unwrap());
    }
    let (mut a, mut b, mut c, mut d) = (state[0], state[1], state[2], state[3]);
    for i in 0..64 {
        let (f, g): (u32, usize) = match i {
            0..=15 => ((b & c) | (!b & d), i),
            16..=31 => ((d & b) | (!d & c), (5 * i + 1) % 16),
            32..=47 => (b ^ c ^ d, (3 * i + 5) % 16),
            _ => (c ^ (b | !d), (7 * i) % 16),
        };
        let temp = d;
        d = c;
        c = b;
        b = b.wrapping_add(
            a.wrapping_add(f)
                .wrapping_add(K[i])
                .wrapping_add(m[g])
                .rotate_left(S[i]),
        );
        a = temp;
    }
    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
}

// ── SHA-256 (small, in-tree) ──────────────────────────────────────

struct Sha256 {
    state: [u32; 8],
    buffer: Vec<u8>,
    total_len: u64,
}

impl Sha256 {
    fn new() -> Self {
        Self {
            state: [
                0x6a09_e667, 0xbb67_ae85, 0x3c6e_f372, 0xa54f_f53a,
                0x510e_527f, 0x9b05_688c, 0x1f83_d9ab, 0x5be0_cd19,
            ],
            buffer: Vec::new(),
            total_len: 0,
        }
    }
    fn update(&mut self, data: &[u8]) {
        self.total_len += data.len() as u64;
        self.buffer.extend_from_slice(data);
        let mut consumed = 0;
        while self.buffer.len() - consumed >= 64 {
            sha256_compress(&mut self.state, &self.buffer[consumed..consumed + 64]);
            consumed += 64;
        }
        self.buffer.drain(..consumed);
    }
    fn finalize(mut self) -> [u8; 32] {
        let bit_len = self.total_len.wrapping_mul(8);
        self.buffer.push(0x80);
        while self.buffer.len() % 64 != 56 {
            self.buffer.push(0);
        }
        self.buffer.extend_from_slice(&bit_len.to_be_bytes());
        let mut consumed = 0;
        while self.buffer.len() - consumed >= 64 {
            sha256_compress(&mut self.state, &self.buffer[consumed..consumed + 64]);
            consumed += 64;
        }
        let mut out = [0u8; 32];
        for (i, w) in self.state.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&w.to_be_bytes());
        }
        out
    }
}

fn sha256_compress(state: &mut [u32; 8], chunk: &[u8]) {
    const K: [u32; 64] = [
        0x428a_2f98, 0x7137_4491, 0xb5c0_fbcf, 0xe9b5_dba5, 0x3956_c25b, 0x59f1_11f1, 0x923f_82a4, 0xab1c_5ed5,
        0xd807_aa98, 0x1283_5b01, 0x2431_85be, 0x550c_7dc3, 0x72be_5d74, 0x80de_b1fe, 0x9bdc_06a7, 0xc19b_f174,
        0xe49b_69c1, 0xefbe_4786, 0x0fc1_9dc6, 0x240c_a1cc, 0x2de9_2c6f, 0x4a74_84aa, 0x5cb0_a9dc, 0x76f9_88da,
        0x983e_5152, 0xa831_c66d, 0xb003_27c8, 0xbf59_7fc7, 0xc6e0_0bf3, 0xd5a7_9147, 0x06ca_6351, 0x1429_2967,
        0x27b7_0a85, 0x2e1b_2138, 0x4d2c_6dfc, 0x5338_0d13, 0x650a_7354, 0x766a_0abb, 0x81c2_c92e, 0x9272_2c85,
        0xa2bf_e8a1, 0xa81a_664b, 0xc24b_8b70, 0xc76c_51a3, 0xd192_e819, 0xd699_0624, 0xf40e_3585, 0x106a_a070,
        0x19a4_c116, 0x1e37_6c08, 0x2748_774c, 0x34b0_bcb5, 0x391c_0cb3, 0x4ed8_aa4a, 0x5b9c_ca4f, 0x682e_6ff3,
        0x748f_82ee, 0x78a5_636f, 0x84c8_7814, 0x8cc7_0208, 0x90be_fffa, 0xa450_6ceb, 0xbef9_a3f7, 0xc671_78f2,
    ];
    let mut w = [0u32; 64];
    for i in 0..16 {
        w[i] = u32::from_be_bytes(chunk[i * 4..i * 4 + 4].try_into().unwrap());
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
    }
    let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h) = (
        state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7],
    );
    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ (!e & g);
        let temp1 = h
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(K[i])
            .wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let temp2 = s0.wrapping_add(maj);
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(temp1);
        d = c;
        c = b;
        b = a;
        a = temp1.wrapping_add(temp2);
    }
    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e);
    state[5] = state[5].wrapping_add(f);
    state[6] = state[6].wrapping_add(g);
    state[7] = state[7].wrapping_add(h);
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0xf) as usize] as char);
    }
    s
}

fn current_utc_iso() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // No chrono dep — emit a simple Unix epoch and let consumers
    // post-process if they want ISO format.
    format!("epoch_{now}")
}
