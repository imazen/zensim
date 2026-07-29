//! Minimal `.npz` (zip-of-`.npy`) reader for the frozen linear-probe fit
//! artifacts (`grams/*.npz`, `val/anchor.npz`, `fits/*.npz`) consumed by
//! `bake_dial_refit fit-lasso` (task #68 — the Rust-native BHdr fit chain).
//!
//! Deliberately NOT a general zip/npy library. Supported subset — anything
//! else fails loudly with a descriptive error instead of guessing:
//!
//! * **zip**: single-disk archives, no zip64 (any `0xFFFF`/`0xFFFFFFFF`
//!   sentinel is rejected), no encryption; entry methods **stored (0)** and
//!   **deflate (8)** only. Entry sizes are taken from the central directory
//!   (correct for `numpy.savez*` output, whose local headers are rewritten
//!   in place by Python's `zipfile`).
//! * **npy**: format versions 1.0/2.0/3.0; little-endian `<f8` / `<f4` /
//!   `<i8` dtypes; C-order only (`fortran_order: False`); 0-d, 1-d, and 2-d
//!   shapes.
//!
//! DEFLATE decompression goes through zenflate (the workspace's pure-Rust
//! DEFLATE owner) — no external zip/npy crates.

use std::path::Path;

/// Typed payload of one `.npy` entry.
pub enum NpyData {
    F64(Vec<f64>),
    F32(Vec<f32>),
    I64(Vec<i64>),
}

/// One decoded `.npy` array: shape (row-major / C-order) + typed data.
pub struct NpyArray {
    /// Dimensions as stored (`[]` for 0-d scalars, `[n]`, or `[rows, cols]`).
    pub shape: Vec<usize>,
    pub data: NpyData,
}

impl NpyArray {
    /// Borrow as `&[f64]`; errors on any other dtype.
    pub fn f64s(&self) -> Result<&[f64], String> {
        match &self.data {
            NpyData::F64(v) => Ok(v),
            NpyData::F32(_) => Err("expected <f8 array, got <f4".into()),
            NpyData::I64(_) => Err("expected <f8 array, got <i8".into()),
        }
    }

    /// Borrow as `&[f32]`; errors on any other dtype.
    pub fn f32s(&self) -> Result<&[f32], String> {
        match &self.data {
            NpyData::F32(v) => Ok(v),
            NpyData::F64(_) => Err("expected <f4 array, got <f8".into()),
            NpyData::I64(_) => Err("expected <f4 array, got <i8".into()),
        }
    }

    /// A 0-d `<f8` scalar (how `np.savez` stores Python floats).
    pub fn scalar_f64(&self) -> Result<f64, String> {
        if !self.shape.is_empty() {
            return Err(format!("expected 0-d scalar, got shape {:?}", self.shape));
        }
        Ok(self.f64s()?[0])
    }
}

struct Entry {
    name: String,
    method: u16,
    csize: usize,
    usize_: usize,
    local_offset: usize,
}

/// An opened `.npz` archive (fully read into memory; the fit inputs are a
/// few MB).
pub struct Npz {
    bytes: Vec<u8>,
    entries: Vec<Entry>,
}

fn le_u16(b: &[u8], off: usize) -> Result<u16, String> {
    b.get(off..off + 2)
        .map(|s| u16::from_le_bytes([s[0], s[1]]))
        .ok_or_else(|| format!("zip: truncated at offset {off}"))
}

fn le_u32(b: &[u8], off: usize) -> Result<u32, String> {
    b.get(off..off + 4)
        .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
        .ok_or_else(|| format!("zip: truncated at offset {off}"))
}

impl Npz {
    /// Open + index an `.npz` file. Errors on anything outside the module's
    /// documented zip subset.
    pub fn open(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("read {path:?}: {e}"))?;
        Self::from_bytes(bytes).map_err(|e| format!("{path:?}: {e}"))
    }

    /// Index an in-memory `.npz` archive (see [`Npz::open`]).
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, String> {
        // End-of-central-directory: scan backwards for PK\x05\x06 within the
        // last 64 KiB + 22 (max comment length).
        const EOCD_SIG: [u8; 4] = [0x50, 0x4b, 0x05, 0x06];
        if bytes.len() < 22 {
            return Err("zip: too short for EOCD".into());
        }
        let scan_from = bytes.len().saturating_sub(22 + 65535);
        let eocd = (scan_from..=bytes.len() - 22)
            .rev()
            .find(|&i| bytes[i..i + 4] == EOCD_SIG)
            .ok_or("zip: EOCD signature not found")?;
        let disk = le_u16(&bytes, eocd + 4)?;
        let cd_disk = le_u16(&bytes, eocd + 6)?;
        let n_entries = le_u16(&bytes, eocd + 10)?;
        let cd_offset = le_u32(&bytes, eocd + 16)?;
        if disk != 0 || cd_disk != 0 {
            return Err("zip: multi-disk archives unsupported".into());
        }
        if n_entries == 0xFFFF || cd_offset == 0xFFFF_FFFF {
            return Err("zip: zip64 archives unsupported".into());
        }

        let mut entries = Vec::with_capacity(n_entries as usize);
        let mut off = cd_offset as usize;
        for _ in 0..n_entries {
            if le_u32(&bytes, off)? != 0x0201_4b50 {
                return Err(format!("zip: bad central-directory signature at {off}"));
            }
            let flags = le_u16(&bytes, off + 8)?;
            if flags & 0x1 != 0 {
                return Err("zip: encrypted entries unsupported".into());
            }
            let method = le_u16(&bytes, off + 10)?;
            let csize = le_u32(&bytes, off + 20)?;
            let usize_ = le_u32(&bytes, off + 24)?;
            let nlen = le_u16(&bytes, off + 28)? as usize;
            let xlen = le_u16(&bytes, off + 30)? as usize;
            let clen = le_u16(&bytes, off + 32)? as usize;
            let local_offset = le_u32(&bytes, off + 42)?;
            if csize == 0xFFFF_FFFF || usize_ == 0xFFFF_FFFF || local_offset == 0xFFFF_FFFF {
                return Err("zip: zip64 entry sizes unsupported".into());
            }
            let name_bytes = bytes
                .get(off + 46..off + 46 + nlen)
                .ok_or("zip: truncated central-directory name")?;
            let name = std::str::from_utf8(name_bytes)
                .map_err(|_| "zip: non-UTF-8 entry name")?
                .to_string();
            entries.push(Entry {
                name,
                method,
                csize: csize as usize,
                usize_: usize_ as usize,
                local_offset: local_offset as usize,
            });
            off += 46 + nlen + xlen + clen;
        }
        Ok(Npz { bytes, entries })
    }

    /// Entry keys (zip names with any `.npy` suffix stripped, matching
    /// `np.load(...)` key semantics).
    pub fn keys(&self) -> Vec<&str> {
        self.entries
            .iter()
            .map(|e| e.name.strip_suffix(".npy").unwrap_or(&e.name))
            .collect()
    }

    /// Decode one entry by key (with or without the `.npy` suffix).
    pub fn get(&self, key: &str) -> Result<NpyArray, String> {
        let entry = self
            .entries
            .iter()
            .find(|e| e.name == key || e.name.strip_suffix(".npy") == Some(key))
            .ok_or_else(|| format!("npz: no entry {key:?} (have: {:?})", self.keys()))?;
        let raw = self.entry_bytes(entry)?;
        parse_npy(&raw).map_err(|e| format!("npz entry {:?}: {e}", entry.name))
    }

    fn entry_bytes(&self, entry: &Entry) -> Result<Vec<u8>, String> {
        let b = &self.bytes;
        let lo = entry.local_offset;
        if le_u32(b, lo)? != 0x0403_4b50 {
            return Err(format!("zip: bad local-header signature at {lo}"));
        }
        let nlen = le_u16(b, lo + 26)? as usize;
        let xlen = le_u16(b, lo + 28)? as usize;
        let data_start = lo + 30 + nlen + xlen;
        let comp = b
            .get(data_start..data_start + entry.csize)
            .ok_or("zip: truncated entry data")?;
        match entry.method {
            0 => {
                if entry.csize != entry.usize_ {
                    return Err("zip: stored entry with csize != usize".into());
                }
                Ok(comp.to_vec())
            }
            8 => {
                let mut out = vec![0u8; entry.usize_];
                let r = zenflate::Decompressor::new()
                    .deflate_decompress(comp, &mut out, zenflate::Unstoppable)
                    .map_err(|e| format!("zip: deflate failed on {:?}: {e:?}", entry.name))?;
                if r.output_written != entry.usize_ {
                    return Err(format!(
                        "zip: {:?} inflated to {} B, central directory says {}",
                        entry.name, r.output_written, entry.usize_
                    ));
                }
                Ok(out)
            }
            m => Err(format!(
                "zip: unsupported compression method {m} on {:?} (stored/deflate only)",
                entry.name
            )),
        }
    }
}

/// Extract the value of a single-quoted `'key': 'value'` pair from an npy
/// header dict.
fn header_quoted(header: &str, key: &str) -> Option<String> {
    let k = format!("'{key}':");
    let rest = &header[header.find(&k)? + k.len()..];
    let q0 = rest.find('\'')?;
    let rest = &rest[q0 + 1..];
    let q1 = rest.find('\'')?;
    Some(rest[..q1].to_string())
}

fn parse_npy(bytes: &[u8]) -> Result<NpyArray, String> {
    const MAGIC: &[u8; 6] = b"\x93NUMPY";
    if bytes.len() < 10 || &bytes[..6] != MAGIC {
        return Err("npy: bad magic".into());
    }
    let major = bytes[6];
    let (hlen, hstart) = match major {
        1 => (le_u16(bytes, 8)? as usize, 10usize),
        2 | 3 => (le_u32(bytes, 8)? as usize, 12usize),
        v => return Err(format!("npy: unsupported format version {v}")),
    };
    let header_bytes = bytes
        .get(hstart..hstart + hlen)
        .ok_or("npy: truncated header")?;
    let header =
        std::str::from_utf8(header_bytes).map_err(|_| "npy: non-UTF-8 header".to_string())?;

    let descr = header_quoted(header, "descr").ok_or("npy: header missing 'descr'")?;
    if header.contains("'fortran_order': True") {
        return Err("npy: fortran_order arrays unsupported (C-order only)".into());
    }
    if !header.contains("'fortran_order': False") {
        return Err("npy: header missing 'fortran_order': False".into());
    }

    // shape tuple: '(372, 372)' / '(2000,)' / '()'
    let sh_key = "'shape':";
    let sh_at = header.find(sh_key).ok_or("npy: header missing 'shape'")?;
    let rest = &header[sh_at + sh_key.len()..];
    let po = rest.find('(').ok_or("npy: shape missing '('")?;
    let pc = rest.find(')').ok_or("npy: shape missing ')'")?;
    let mut shape: Vec<usize> = Vec::new();
    for tok in rest[po + 1..pc].split(',') {
        let t = tok.trim();
        if t.is_empty() {
            continue;
        }
        shape.push(
            t.parse::<usize>()
                .map_err(|_| format!("npy: bad shape token {t:?}"))?,
        );
    }
    if shape.len() > 2 {
        return Err(format!("npy: rank-{} arrays unsupported", shape.len()));
    }
    let n_elems: usize = shape.iter().product::<usize>().max(1); // 0-d scalar = 1 element

    let data = &bytes[hstart + hlen..];
    let need = |itemsize: usize| -> Result<(), String> {
        if data.len() != n_elems * itemsize {
            return Err(format!(
                "npy: payload is {} B, expected {} ({} x {} B)",
                data.len(),
                n_elems * itemsize,
                n_elems,
                itemsize
            ));
        }
        Ok(())
    };
    let payload = match descr.as_str() {
        "<f8" => {
            need(8)?;
            NpyData::F64(
                data.chunks_exact(8)
                    .map(|c| f64::from_le_bytes(c.try_into().expect("chunk of 8")))
                    .collect(),
            )
        }
        "<f4" => {
            need(4)?;
            NpyData::F32(
                data.chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().expect("chunk of 4")))
                    .collect(),
            )
        }
        "<i8" => {
            need(8)?;
            NpyData::I64(
                data.chunks_exact(8)
                    .map(|c| i64::from_le_bytes(c.try_into().expect("chunk of 8")))
                    .collect(),
            )
        }
        other => {
            return Err(format!(
                "npy: dtype {other:?} unsupported (<f8/<f4/<i8 little-endian only)"
            ));
        }
    };
    Ok(NpyArray {
        shape,
        data: payload,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `.npy` byte blob (v1.0 header, same layout numpy writes).
    fn npy_bytes(descr: &str, shape: &str, payload: &[u8]) -> Vec<u8> {
        let dict = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}");
        // pad header (incl. 10-byte preamble) to a multiple of 64, ending \n
        let unpadded = 10 + dict.len() + 1;
        let pad = (64 - unpadded % 64) % 64;
        let header = format!("{dict}{}\n", " ".repeat(pad));
        let mut out = Vec::new();
        out.extend_from_slice(b"\x93NUMPY\x01\x00");
        out.extend_from_slice(&(header.len() as u16).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(payload);
        out
    }

    /// Hand-rolled zip writer (test-only): local headers + central directory,
    /// entries stored (method 0) or deflated via zenflate (method 8).
    fn zip_bytes(entries: &[(&str, &[u8], bool)]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut central = Vec::new();
        let mut n = 0u16;
        for (name, data, deflate) in entries {
            let (method, stored): (u16, Vec<u8>) = if *deflate {
                let mut comp = vec![0u8; zenflate::Compressor::deflate_compress_bound(data.len())];
                let mut c = zenflate::Compressor::new(zenflate::CompressionLevel::balanced());
                let clen = c
                    .deflate_compress(data, &mut comp, zenflate::Unstoppable)
                    .expect("test deflate");
                comp.truncate(clen);
                (8, comp)
            } else {
                (0, data.to_vec())
            };
            let local_offset = out.len() as u32;
            // local header
            out.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
            out.extend_from_slice(&[20, 0, 0, 0]); // version, flags
            out.extend_from_slice(&method.to_le_bytes());
            out.extend_from_slice(&[0; 8]); // time, date, crc (unchecked)
            out.extend_from_slice(&(stored.len() as u32).to_le_bytes());
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            out.extend_from_slice(&(name.len() as u16).to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes());
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(&stored);
            // central record
            central.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
            central.extend_from_slice(&[20, 0, 20, 0, 0, 0]); // made by, need, flags
            central.extend_from_slice(&method.to_le_bytes());
            central.extend_from_slice(&[0; 8]); // time, date, crc
            central.extend_from_slice(&(stored.len() as u32).to_le_bytes());
            central.extend_from_slice(&(data.len() as u32).to_le_bytes());
            central.extend_from_slice(&(name.len() as u16).to_le_bytes());
            central.extend_from_slice(&[0; 12]); // extra, comment, disk, attrs(2+4)
            central.extend_from_slice(&local_offset.to_le_bytes());
            central.extend_from_slice(name.as_bytes());
            n += 1;
        }
        let cd_offset = out.len() as u32;
        let cd_size = central.len() as u32;
        out.extend_from_slice(&central);
        out.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
        out.extend_from_slice(&[0; 4]); // disk numbers
        out.extend_from_slice(&n.to_le_bytes());
        out.extend_from_slice(&n.to_le_bytes());
        out.extend_from_slice(&cd_size.to_le_bytes());
        out.extend_from_slice(&cd_offset.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // comment len
        out
    }

    #[test]
    fn round_trips_f64_f32_scalar_deflate_and_stored() {
        let m: Vec<f64> = vec![1.5, -2.25, 3.0, 0.125, 1e-30, 7410.0];
        let m_le: Vec<u8> = m.iter().flat_map(|v| v.to_le_bytes()).collect();
        let f: Vec<f32> = vec![0.5, -1.0, 3.25, 65504.0];
        let f_le: Vec<u8> = f.iter().flat_map(|v| v.to_le_bytes()).collect();
        let scalar = 7410.0f64.to_le_bytes();

        let mat = npy_bytes("<f8", "(2, 3)", &m_le);
        let vecf = npy_bytes("<f4", "(4,)", &f_le);
        let sc = npy_bytes("<f8", "()", &scalar);
        let zip = zip_bytes(&[
            ("mat.npy", &mat, true),
            ("vecf.npy", &vecf, true),
            ("sc.npy", &sc, false),
        ]);

        let npz = Npz::from_bytes(zip).expect("open");
        assert_eq!(npz.keys(), vec!["mat", "vecf", "sc"]);

        let a = npz.get("mat").expect("mat");
        assert_eq!(a.shape, vec![2, 3]);
        assert_eq!(a.f64s().unwrap(), m.as_slice());

        let b = npz.get("vecf").expect("vecf");
        assert_eq!(b.shape, vec![4]);
        assert_eq!(b.f32s().unwrap(), f.as_slice());

        let c = npz.get("sc").expect("sc");
        assert!(c.shape.is_empty());
        assert_eq!(c.scalar_f64().unwrap(), 7410.0);
    }

    #[test]
    fn i64_scalar_and_missing_key_and_bad_dtype() {
        let dropped = 42i64.to_le_bytes();
        let iarr = npy_bytes("<i8", "()", &dropped);
        // '<U6' unicode entry (numpy string array) must error, not panic.
        let upayload = [0u8; 24];
        let uarr = npy_bytes("<U6", "()", &upayload);
        let zip = zip_bytes(&[("dropped.npy", &iarr, true), ("space.npy", &uarr, true)]);
        let npz = Npz::from_bytes(zip).expect("open");

        let d = npz.get("dropped").expect("dropped");
        match d.data {
            NpyData::I64(ref v) => assert_eq!(v, &[42]),
            _ => panic!("expected i64"),
        }
        assert!(d.scalar_f64().is_err(), "i64 scalar_f64 must error");
        assert!(npz.get("space").is_err(), "unicode dtype must error");
        assert!(npz.get("absent").is_err(), "missing key must error");
    }

    #[test]
    fn reads_npy_v2_header() {
        let payload = 1.25f64.to_le_bytes();
        let dict = "{'descr': '<f8', 'fortran_order': False, 'shape': (), }";
        let unpadded = 12 + dict.len() + 1;
        let pad = (64 - unpadded % 64) % 64;
        let header = format!("{dict}{}\n", " ".repeat(pad));
        let mut npy = Vec::new();
        npy.extend_from_slice(b"\x93NUMPY\x02\x00");
        npy.extend_from_slice(&(header.len() as u32).to_le_bytes());
        npy.extend_from_slice(header.as_bytes());
        npy.extend_from_slice(&payload);
        let zip = zip_bytes(&[("x.npy", &npy, false)]);
        let npz = Npz::from_bytes(zip).expect("open");
        assert_eq!(npz.get("x").unwrap().scalar_f64().unwrap(), 1.25);
    }

    #[test]
    fn rejects_fortran_order() {
        let payload = [0u8; 16];
        let dict = "{'descr': '<f8', 'fortran_order': True, 'shape': (2, 1), }";
        let unpadded = 10 + dict.len() + 1;
        let pad = (64 - unpadded % 64) % 64;
        let header = format!("{dict}{}\n", " ".repeat(pad));
        let mut npy = Vec::new();
        npy.extend_from_slice(b"\x93NUMPY\x01\x00");
        npy.extend_from_slice(&(header.len() as u16).to_le_bytes());
        npy.extend_from_slice(header.as_bytes());
        npy.extend_from_slice(&payload);
        let zip = zip_bytes(&[("f.npy", &npy, false)]);
        let npz = Npz::from_bytes(zip).expect("open");
        assert!(npz.get("f").is_err());
    }
}
