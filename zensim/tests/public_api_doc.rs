//! Regenerates the committed public-API surface snapshots
//! (`docs/public-api/<crate>.txt`, one per published crate) on every
//! `cargo test` run, so API changes always show up as a git diff next to the
//! code change that caused them, and the surface size stays one glance away.
//!
//! Modes (`ZEN_API_DOC` env var — set per-job in CI workflows, see ci.yml):
//! - unset / `regen` → regenerate the files in place (local default; commit the diff)
//! - `check`         → regenerate to memory, FAIL if a committed file is stale
//! - `off`           → skipped (CI matrix jobs without nightly rustdoc / the tool)
//!
//! File layout per crate:
//! - `## summary` — generated line taxonomy. Raw `cargo public-api` output
//!   counts every auto-trait impl (`Send`/`Sync`/`Freeze`/...) and every
//!   derived-trait method (`clone`/`fmt`/`eq`/...) as a line, which dwarfs
//!   the real API; the summary separates free functions, associated
//!   functions, fields, variants, and impl plumbing so the headline can't
//!   be misread as "N functions".
//! - `## default features (N lines)` — the full surface, verbatim. Auto-trait
//!   impl lines are kept on purpose: losing `Send`/`Sync` on a public type is
//!   a real semver break and must show up in the diff.
//! - `## added by non-default features: ... (N lines)` — DELTA ONLY: lines
//!   not present in the default section. (Before 2026-06-11 this section
//!   repeated the whole surface, double-counting every default item.)
//!   Underscore-prefixed features are internal/research gates and excluded;
//!   the feature list is computed from `cargo metadata`, so new features
//!   appear automatically. A `## removed by non-default features` section
//!   appears only when enabling features removes lines (rare; cfg(not) gates).
//!
//! Requires `cargo-public-api` (0.52+) and a nightly toolchain for rustdoc
//! JSON: `cargo install cargo-public-api --locked && rustup toolchain install nightly`

use std::collections::{BTreeMap, HashSet};
use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::Command;

/// Published crates in this workspace (zensim-validate & friends are
/// internal).
const CRATES: &[&str] = &["zensim", "zensim-regress"];

fn run(args: &[&str]) -> Vec<u8> {
    let out = Command::new("cargo")
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args(args)
        .output()
        .unwrap_or_else(|e| {
            panic!(
                "failed to run cargo {} ({e}); for public-api: install with \
                 `cargo install cargo-public-api --locked` and ensure a nightly \
                 toolchain exists (`rustup toolchain install nightly`), or set \
                 ZEN_API_DOC=off to skip this test",
                args[0]
            )
        });
    assert!(
        out.status.success(),
        "cargo {} failed (set ZEN_API_DOC=off to skip):\n{}",
        args.join(" "),
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

fn surface(package: &str, feature_args: &[&str]) -> Vec<String> {
    let mut args = vec!["public-api", "-p", package, "--simplified"];
    args.extend_from_slice(feature_args);
    String::from_utf8(run(&args))
        .expect("cargo public-api emitted non-UTF8")
        .lines()
        .map(str::to_owned)
        .filter(|l| !l.is_empty())
        .collect()
}

/// All manifest features of `package` except `default` and underscore-
/// prefixed internal gates, sorted for determinism.
fn public_features(package: &str) -> Vec<String> {
    let meta: serde_json::Value =
        serde_json::from_slice(&run(&["metadata", "--no-deps", "--format-version", "1"]))
            .expect("cargo metadata JSON");
    let pkg = meta["packages"]
        .as_array()
        .expect("packages array")
        .iter()
        .find(|p| p["name"] == package)
        .unwrap_or_else(|| panic!("{package} not in workspace metadata"));
    let mut feats: Vec<String> = pkg["features"]
        .as_object()
        .expect("features map")
        .keys()
        .filter(|k| *k != "default" && !k.starts_with('_'))
        .cloned()
        .collect();
    feats.sort();
    feats
}

// ---------------------------------------------------------------------------
// Line taxonomy: classify each `cargo public-api --simplified` line so the
// summary reports honest counts instead of a raw line total.

/// Marker / auto traits whose impl lines are compiler-controlled plumbing
/// (still diff-guarded — losing `Send` is a semver break — but counted apart
/// from hand-written or derived trait impls).
const AUTO_TRAITS: &[&str] = &[
    "core::marker::Freeze",
    "core::marker::Send",
    "core::marker::StructuralPartialEq",
    "core::marker::Sync",
    "core::marker::Unpin",
    "core::marker::UnsafeUnpin",
    "core::panic::unwind_safe::RefUnwindSafe",
    "core::panic::unwind_safe::UnwindSafe",
];

#[derive(Default, Clone, Copy)]
struct Tally {
    modules: usize,
    types: usize,
    consts: usize,
    macros: usize,
    free_fns: usize,
    assoc_fns: usize,
    fields: usize,
    variants: usize,
    impls_auto: usize,
    impls_other: usize,
    reexports: usize,
    other: usize,
}

impl Tally {
    fn rows(&self) -> [(&'static str, usize); 12] {
        [
            ("pub modules", self.modules),
            ("pub types (struct/enum/trait/alias)", self.types),
            ("pub consts/statics", self.consts),
            ("pub macros", self.macros),
            ("free functions", self.free_fns),
            ("associated functions (methods)", self.assoc_fns),
            ("struct fields", self.fields),
            ("enum variants", self.variants),
            ("impl lines (auto traits)", self.impls_auto),
            ("impl lines (derived + manual)", self.impls_other),
            ("re-exports", self.reexports),
            ("other", self.other),
        ]
    }
}

/// For an `impl` line, return the part after `impl` / `impl<...>` — the
/// implemented trait (or inherent type). `None` if not an impl line.
fn impl_body(line: &str) -> Option<&str> {
    let rest = line.strip_prefix("impl")?;
    if let Some(r) = rest.strip_prefix(' ') {
        return Some(r);
    }
    if rest.starts_with('<') {
        let bytes = rest.as_bytes();
        let mut depth = 0usize;
        for (i, &b) in bytes.iter().enumerate() {
            match b {
                b'<' => depth += 1,
                b'>' => {
                    depth -= 1;
                    if depth == 0 {
                        return rest[i + 1..].strip_prefix(' ');
                    }
                }
                _ => {}
            }
        }
    }
    None
}

/// Strip leading `#[...]` attributes (e.g. `#[non_exhaustive] `).
fn strip_attrs(mut line: &str) -> &str {
    while line.starts_with("#[") {
        match line.find("] ") {
            Some(i) => line = &line[i + 2..],
            None => break,
        }
    }
    line
}

/// Split a path on `::` at angle-bracket depth 0 (so `Type<'a>::method`
/// splits into `Type<'a>` + `method`, not inside the generics).
fn path_segments(path: &str) -> Vec<&str> {
    let bytes = path.as_bytes();
    let mut segments = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'<' => depth += 1,
            b'>' => depth = depth.saturating_sub(1),
            b':' if depth == 0 && bytes.get(i + 1) == Some(&b':') => {
                segments.push(&path[start..i]);
                i += 2;
                start = i;
                continue;
            }
            _ => {}
        }
        i += 1;
    }
    segments.push(&path[start..]);
    segments
}

/// The path portion of a classified line body: everything up to the first
/// depth-0 `(`, ` `, or `: ` type annotation.
fn leading_path(body: &str) -> &str {
    let bytes = body.as_bytes();
    let mut depth = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'<' => depth += 1,
            b'>' => depth = depth.saturating_sub(1),
            b'(' | b' ' if depth == 0 => return &body[..i],
            b':' if depth == 0 => {
                if bytes.get(i + 1) == Some(&b':') {
                    i += 2;
                    continue;
                }
                return &body[..i];
            }
            _ => {}
        }
        i += 1;
    }
    body
}

/// `module` bucket for a path like `crate::module::Item::member` — the
/// second segment when it names a module (lowercase), else `(root)`.
fn module_of(path: &str) -> String {
    let segs = path_segments(path);
    if segs.len() >= 3 {
        let m = segs[1].split('<').next().unwrap_or(segs[1]);
        if m.chars().next().is_some_and(|c| c.is_lowercase() || c == '_') {
            return m.to_owned();
        }
    }
    "(root)".to_owned()
}

fn classify(line: &str, tally: &mut Tally, per_module: &mut BTreeMap<String, usize>) {
    let l = strip_attrs(line);
    if let Some(rest) = impl_body(l) {
        // Impl lines carry no `pub` path; not attributed to a module.
        if AUTO_TRAITS.iter().any(|t| rest.starts_with(t)) {
            tally.impls_auto += 1;
        } else {
            tally.impls_other += 1;
        }
        return;
    }
    let Some(body) = l.strip_prefix("pub ") else {
        tally.other += 1;
        return;
    };
    // Track the module bucket for every pub line.
    let keyword_stripped = body
        .strip_prefix("mod ")
        .or_else(|| body.strip_prefix("struct "))
        .or_else(|| body.strip_prefix("enum "))
        .or_else(|| body.strip_prefix("trait "))
        .or_else(|| body.strip_prefix("type "))
        .or_else(|| body.strip_prefix("union "))
        .or_else(|| body.strip_prefix("const "))
        .or_else(|| body.strip_prefix("static "))
        .or_else(|| body.strip_prefix("fn "))
        .or_else(|| body.strip_prefix("use "))
        .unwrap_or(body);
    *per_module.entry(module_of(leading_path(keyword_stripped))).or_default() += 1;

    if body.starts_with("mod ") {
        tally.modules += 1;
    } else if body.starts_with("struct ")
        || body.starts_with("enum ")
        || body.starts_with("trait ")
        || body.starts_with("type ")
        || body.starts_with("union ")
    {
        tally.types += 1;
    } else if body.starts_with("const ") || body.starts_with("static ") {
        tally.consts += 1;
    } else if body.starts_with("macro") {
        tally.macros += 1;
    } else if body.starts_with("use ") {
        tally.reexports += 1;
    } else if let Some(sig) = body.strip_prefix("fn ") {
        let path = leading_path(sig);
        let segs = path_segments(path);
        let parent = if segs.len() >= 2 { segs[segs.len() - 2] } else { "" };
        let parent = parent.split('<').next().unwrap_or(parent);
        if parent.chars().next().is_some_and(char::is_uppercase) {
            tally.assoc_fns += 1;
        } else {
            tally.free_fns += 1;
        }
    } else {
        // Bare path lines: `pub Path::field: Type` (field) or
        // `pub Path::Variant` / `pub Path::Variant(..)` (enum variant).
        let path = leading_path(body);
        let after = &body[path.len()..];
        if after.starts_with(':') {
            tally.fields += 1;
        } else {
            tally.variants += 1;
        }
    }
}

fn tally_section(lines: &[String]) -> (Tally, BTreeMap<String, usize>) {
    let mut tally = Tally::default();
    let mut per_module = BTreeMap::new();
    for line in lines {
        classify(line, &mut tally, &mut per_module);
    }
    (tally, per_module)
}

fn render_summary(
    default_lines: &[String],
    delta_lines: &[String],
    removed_count: usize,
) -> String {
    let (dt, dmods) = tally_section(default_lines);
    let (ft, fmods) = tally_section(delta_lines);
    let mut s = String::from("## summary\n#\n");
    let _ = writeln!(
        s,
        "# {:<38} {:>8} {:>10}",
        "kind", "default", "+features"
    );
    let _ = writeln!(
        s,
        "# {:<38} {:>8} {:>10}",
        "lines total",
        default_lines.len(),
        delta_lines.len()
    );
    for ((label, d), (_, f)) in dt.rows().into_iter().zip(ft.rows()) {
        if d == 0 && f == 0 {
            continue;
        }
        let _ = writeln!(s, "#   {:<36} {:>8} {:>10}", label, d, f);
    }
    if removed_count > 0 {
        let _ = writeln!(s, "#   {:<36} {:>8} {:>10}", "removed by features", "-", removed_count);
    }
    s.push_str("#\n# per-module pub lines (default + feature-additions):\n");
    let modules: BTreeMap<&str, (usize, usize)> = dmods
        .iter()
        .map(|(m, n)| (m.as_str(), (*n, 0)))
        .chain(fmods.iter().map(|(m, n)| (m.as_str(), (0, *n))))
        .fold(BTreeMap::new(), |mut acc, (m, (d, f))| {
            let e = acc.entry(m).or_insert((0, 0));
            e.0 += d;
            e.1 += f;
            acc
        });
    for (module, (d, f)) in modules {
        let _ = writeln!(s, "#   {module:<24} {d:>6} +{f}");
    }
    s
}

#[test]
fn public_api_surface_docs_are_current() {
    match std::env::var("ZEN_API_DOC").as_deref() {
        Ok("off") => {
            eprintln!("ZEN_API_DOC=off — public-API snapshot regen skipped by caller");
            return;
        }
        Ok("check") | Ok("regen") | Err(_) => {}
        Ok(other) => panic!("unknown ZEN_API_DOC value {other:?} (off|check|regen)"),
    }
    let check = std::env::var("ZEN_API_DOC").as_deref() == Ok("check");

    for package in CRATES {
        let features = public_features(package);
        let feature_csv = features.join(",");

        let default_lines = surface(package, &[]);
        let all_lines = if features.is_empty() {
            default_lines.clone()
        } else {
            surface(package, &["--features", &feature_csv])
        };
        let default_set: HashSet<&str> = default_lines.iter().map(String::as_str).collect();
        let all_set: HashSet<&str> = all_lines.iter().map(String::as_str).collect();
        let delta_lines: Vec<String> = all_lines
            .iter()
            .filter(|l| !default_set.contains(l.as_str()))
            .cloned()
            .collect();
        let removed_lines: Vec<String> = default_lines
            .iter()
            .filter(|l| !all_set.contains(l.as_str()))
            .cloned()
            .collect();

        let mut doc = String::new();
        doc.push_str(&format!(
            "# {package} public API surface\n\
             # Generated by zensim/tests/public_api_doc.rs via `cargo public-api --simplified`\n\
             # (regenerated on every `cargo test`; ZEN_API_DOC=check verifies, =off skips).\n\
             # The features section is a DELTA: only lines added relative to the\n\
             # default-features section. Underscore-prefixed features are internal\n\
             # and excluded. Line counts are raw `cargo public-api` lines — see the\n\
             # summary block for the honest item taxonomy.\n\
             # DO NOT EDIT BY HAND — commit regenerated changes together with the code.\n\n"
        ));
        doc.push_str(&render_summary(&default_lines, &delta_lines, removed_lines.len()));

        doc.push_str(&format!(
            "\n## default features ({} lines)\n\n",
            default_lines.len()
        ));
        for line in &default_lines {
            doc.push_str(line);
            doc.push('\n');
        }
        eprintln!("{package} [default features]: {} lines", default_lines.len());

        if !features.is_empty() {
            doc.push_str(&format!(
                "\n## added by non-default features: {feature_csv} ({} lines)\n\n",
                delta_lines.len()
            ));
            for line in &delta_lines {
                doc.push_str(line);
                doc.push('\n');
            }
            eprintln!(
                "{package} [+{feature_csv}]: {} added lines",
                delta_lines.len()
            );
            if !removed_lines.is_empty() {
                doc.push_str(&format!(
                    "\n## removed by non-default features ({} lines)\n\n",
                    removed_lines.len()
                ));
                for line in &removed_lines {
                    doc.push_str(line);
                    doc.push('\n');
                }
                eprintln!(
                    "{package} [+{feature_csv}]: {} REMOVED lines (cfg(not) gate?)",
                    removed_lines.len()
                );
            }
        }

        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../docs/public-api")
            .join(format!("{package}.txt"));
        let existing = std::fs::read_to_string(&path).ok();

        if check {
            assert_eq!(
                existing.as_deref(),
                Some(doc.as_str()),
                "committed public-API snapshot for {package} is stale: run \
                 `cargo test` locally and commit the regenerated {}",
                path.display()
            );
        } else if existing.as_deref() != Some(doc.as_str()) {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, &doc).unwrap();
            eprintln!(
                "regenerated {} — review and commit the diff",
                path.display()
            );
        }
    }
}
