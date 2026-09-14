//! Hash-bound native inspection, using the shared decoder without a color transform.

use crate::zen_decode::{NativeMetadata, decode_native_bytes};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::io::Write;
use std::path::{Path, PathBuf};

fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn icc(bytes: Option<&[u8]>) -> Value {
    match bytes {
        None => Value::Null,
        Some(bytes) => json!({
            "bytes": bytes.len(), "sha256": sha(bytes),
            "identification": zenpixels::icc::identify_common(bytes).map(|id| json!({
                "primaries": format!("{:?}", id.primaries),
                "transfer": format!("{:?}", id.transfer),
                "valid_use": format!("{:?}", id.valid_use),
            })),
        }),
    }
}

fn cicp(value: Option<zenpixels::Cicp>) -> Value {
    value.map_or(Value::Null, |c| {
        json!({
            "primaries": c.color_primaries, "transfer": c.transfer_characteristics,
            "matrix": c.matrix_coefficients, "full_range": c.full_range,
        })
    })
}

fn admitted_list(text: &str) -> Result<Vec<(PathBuf, String)>, String> {
    let mut lines = text.lines();
    if lines.next() != Some("path\tsha256") {
        return Err("inspection requires exactly the path/sha256 TSV header".into());
    }
    let mut paths = HashSet::new();
    let mut inputs = Vec::new();
    for (index, line) in lines.enumerate() {
        let Some((path, hash)) = line.split_once('\t') else {
            return Err(format!("invalid inspection row {}", index + 2));
        };
        let path = PathBuf::from(path);
        if !path.is_absolute()
            || !paths.insert(path.clone())
            || hash.len() != 64
            || !hash
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        {
            return Err(format!("invalid or duplicate inspection row {}", index + 2));
        }
        inputs.push((path, hash.to_owned()));
    }
    if inputs.is_empty() {
        return Err("empty inspection list".into());
    }
    Ok(inputs)
}

pub(super) fn run(args: &[String]) -> Result<(), String> {
    if args.len() != 4 || args[0] != "--inspect-list" || args[2] != "--out" {
        return Err("usage: --inspect-list LIST.tsv --out FRESH.jsonl".into());
    }
    let out = Path::new(&args[3]);
    if out.exists() {
        return Err("inspection output already exists".into());
    }
    let text = std::fs::read_to_string(&args[1]).map_err(|e| e.to_string())?;
    let inputs = admitted_list(&text)?;
    let list_sha = sha(text.as_bytes());
    let mut rows = Vec::with_capacity(inputs.len());
    for (path, expected) in inputs {
        let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        if sha(&bytes) != expected {
            return Err(format!("file hash mismatch: {}", path.display()));
        }
        let native =
            decode_native_bytes(&bytes, &path.display().to_string()).map_err(|e| e.to_string())?;
        let source = match &native.metadata {
            NativeMetadata::Png(p) => json!({
                "format":"PNG", "width":p.width, "height":p.height,
                "bit_depth":p.bit_depth, "has_alpha":p.has_alpha,
                "icc":icc(p.icc_profile.as_deref()), "cicp":cicp(p.cicp),
                "gamma":p.source_gamma, "srgb_intent":p.srgb_intent,
                "chromaticities":p.chromaticities.as_ref().map(|v|format!("{v:?}")),
                "content_light_level":p.content_light_level.as_ref().map(|v|format!("{v:?}")),
                "mastering_display":p.mastering_display.as_ref().map(|v|format!("{v:?}")),
                "exif_sha256":p.exif.as_deref().map(sha),
                "orientation":"not interpreted by this inspector; retain EXIF hash",
            }),
            NativeMetadata::Codec(p) => json!({
                "format":format!("{:?}",p.format), "width":p.width,"height":p.height,
                "bit_depth":p.source_color.bit_depth,"has_alpha":p.has_alpha,
                "orientation":format!("{:?}",p.orientation),
                "icc":icc(p.source_color.icc_profile.as_deref()), "cicp":cicp(p.source_color.cicp),
                "color_authority":format!("{:?}",p.source_color.color_authority),
                "content_light_level":p.source_color.content_light_level.as_ref().map(|v|format!("{v:?}")),
                "mastering_display":p.source_color.mastering_display.as_ref().map(|v|format!("{v:?}")),
                "warnings":p.warnings,
            }),
        };
        let buffer = &native.pixels;
        let context = buffer.color_context().map(|c| {
            json!({
                "icc":icc(c.icc.as_deref()), "cicp":cicp(c.cicp),
                "diffuse_white":c.diffuse_white.as_ref().map(|v|format!("{v:?}")),
            })
        });
        let legacy = match native.to_rgb8(&path.display().to_string()) {
            Ok(rgb) => {
                json!({"status":"ok","pixels_sha256":sha(&rgb.pixels),"bytes":rgb.pixels.len()})
            }
            Err(e) => json!({"status":"unsupported","error":e.to_string()}),
        };
        rows.push(json!({"schema":"native-decoder-inspection-v1", "path":path,
            "file_sha256":expected,"file_bytes":bytes.len(),"list_sha256":list_sha,
            "source":source,"decoded":{
                "descriptor":format!("{:?}",buffer.descriptor()),
                "width":buffer.width(),"height":buffer.height(),"stride_bytes":buffer.stride(),
                "endianness":if cfg!(target_endian="little") {"little"} else {"big"},
                "active_row_bytes_sha256":sha(&buffer.copy_to_contiguous_bytes()),
                "color_context":context,
            }, "legacy_rgb8":legacy,"color_conversion_applied_by_inspector":false}));
    }
    // All input validation and decoding finish before a successful report exists.
    let mut output = Vec::new();
    for row in &rows {
        serde_json::to_writer(&mut output, row).map_err(|e| e.to_string())?;
        output.push(b'\n');
    }
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out)
        .map_err(|e| e.to_string())?;
    if let Err(error) = file.write_all(&output).and_then(|()| file.sync_all()) {
        drop(file);
        let cleanup = std::fs::remove_file(out);
        return Err(format!(
            "inspection write failed: {error}; partial-output removal: {cleanup:?}"
        ));
    }
    eprintln!("native inspection: {} complete files", rows.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn admission_rejects_bad_rows_before_file_access() {
        let row = format!("/does/not/exist\t{}", "a".repeat(64));
        assert!(admitted_list(&format!("path\tsha256\n{row}\n")).is_ok());
        for input in [
            String::new(),
            "path\tsha256\n".into(),
            format!("path\tsha256\n{row}\n{row}\n"),
            format!("path\tsha256\nrelative\t{}\n", "a".repeat(64)),
            "path\tsha256\n/absolute\tnot-a-hash\n".into(),
        ] {
            assert!(admitted_list(&input).is_err(), "{input}");
        }
    }
}
