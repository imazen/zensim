//! Within-corpus content clustering + curation driver (issue #33).
//!
//! Clusters a source-image corpus by dHash-64 at a STRICT threshold
//! (default d ≤ 3 — "nearly bit-identical after rescale") so that the
//! deliberate resample variants of one source (`<hex>_512sq.png`,
//! `<hex>_769x513.png`, …) land in one content cluster, then emits the
//! curation artifacts the issue proposes:
//!
//! * `--out-tsv` (always): one row per hashed file —
//!   `path  basename  base_hint  dhash  pixels  cluster_id  cluster_size
//!   content_weight  canonical  split`.
//! * `--cull-csv OUT` (option 3): the training CSV restricted to rows
//!   whose source is its cluster's canonical (highest-resolution) member.
//! * `--reweight-dir DIR` (option 2): the training CSV split into
//!   `cluster_size_<k>.csv` files plus `groups.txt` holding the
//!   `--group` specs (`train_w ∝ n_rows / k`) that realise per-row
//!   `1 / cluster_size` sampling with no trainer changes.
//! * `--split-dir DIR` (option 4): `train.csv` / `val.csv` with whole
//!   content clusters on one side.
//!
//! Input is either `--corpus-dir` (one-level listing, png/jpg/jpeg/webp)
//! or `--training-csv` (+ `--source-root`): the distinct sources named in
//! the CSV's `ref_basename` / `image_path` column (else column 0), which
//! is also what the three CSV outputs key on. Rows whose source could not
//! be hashed are treated as a size-1 cluster (kept, weight 1, train
//! side) and counted in the report.
//!
//! The report also compares the hash clusters with the corpus NAMING
//! (`base_hint` = stem before the first `_`): clusters spanning more than
//! one base hint (a cross-source duplicate, or a flat-content false
//! positive — eyeball these) and base hints spread over more than one
//! cluster (variants the hash did NOT join — crops, or a naming
//! coincidence). Per the 2026-05-14 policy nothing here is a blocklist;
//! `--cull-csv` and friends are explicit, user-invoked curation.
//!
//! ```text
//! cargo run --release -p zensim-validate --bin corpus_content_clusters -- \
//!   --training-csv /mnt/v/output/zensim/synthetic-v2/training_safe_synthetic.csv \
//!   --source-root /mnt/v/input/zensim/sources \
//!   --out-tsv benchmarks/content_clusters_<date>.tsv \
//!   --reweight-dir /mnt/v/output/zensim/synthetic-v2/reweight_<date>/
//! ```

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use zensim_validate::content_clusters::{
    Split, base_hint, canonical_members, cluster_by_hamming, cluster_sizes, content_weights,
    dhash_64, hamming, render_montage, reweight_groups, stratified_split,
};

#[derive(Parser, Debug)]
#[command(
    version,
    about = "dHash-64 within-corpus content clustering + curation (issue #33)"
)]
struct Args {
    /// Corpus directory to hash (one level, png/jpg/jpeg/webp).
    #[arg(long, conflicts_with = "training_csv")]
    corpus_dir: Option<PathBuf>,

    /// Training CSV whose distinct sources are hashed and whose rows the
    /// --cull-csv / --reweight-dir / --split-dir outputs are drawn from.
    #[arg(long)]
    training_csv: Option<PathBuf>,

    /// Directory the CSV's source names resolve against (basename join).
    /// Sources given as existing absolute paths need no root.
    #[arg(long)]
    source_root: Option<PathBuf>,

    /// Maximum Hamming distance for two hashes to share a cluster.
    #[arg(long, default_value_t = 3)]
    max_dist: u32,

    /// Per-file cluster report (TSV).
    #[arg(long)]
    out_tsv: PathBuf,

    /// Fraction of MEMBERS on the validation side of the stratified split.
    #[arg(long, default_value_t = 0.2)]
    val_frac: f64,

    /// Seed for the stratified split ordering.
    #[arg(long, default_value_t = 1)]
    seed: u64,

    /// Option 3: write the training CSV restricted to canonical sources.
    #[arg(long, requires = "training_csv")]
    cull_csv: Option<PathBuf>,

    /// Option 2: write per-cluster-size CSVs + `groups.txt` here.
    #[arg(long, requires = "training_csv")]
    reweight_dir: Option<PathBuf>,

    /// Option 4: write `train.csv` / `val.csv` here.
    #[arg(long, requires = "training_csv")]
    split_dir: Option<PathBuf>,

    /// Validation step 2: write side-by-side montages + `index.html` here
    /// for the EYEBALL pass. The 2026-05-14 revert made montage review the
    /// standing precondition for acting on any dHash result; without this
    /// the reviewer has only file names to judge by.
    #[arg(long)]
    montage_dir: Option<PathBuf>,

    /// Montage cell size in pixels (each member is scaled to fit).
    #[arg(long, default_value_t = 192)]
    montage_cell: u32,

    /// Members per montage row.
    #[arg(long, default_value_t = 6)]
    montage_cols: u32,

    /// Cap on montages written (review-flagged clusters first).
    #[arg(long, default_value_t = 60)]
    montage_max: usize,

    /// Render every multi-member cluster, not just the review-flagged ones.
    #[arg(long)]
    montage_all: bool,
}

/// One rendered montage plus the facts the reviewer signs off against.
struct MontageEntry {
    file: String,
    title: String,
    flag: &'static str,
    members: Vec<usize>,
    max_dist: u32,
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Render one montage PNG from member indices; returns the max pairwise
/// Hamming distance inside the group (the number that says how far the
/// linkage chain stretched).
fn write_montage(
    dir: &Path,
    file: &str,
    members: &[usize],
    hashed: &[Hashed],
    cell: u32,
    cols: u32,
) -> Result<u32> {
    let mut images = Vec::with_capacity(members.len());
    for &i in members {
        let img = image::open(&hashed[i].path)
            .with_context(|| format!("re-open {}", hashed[i].path.display()))?;
        images.push(img);
    }
    let montage = render_montage(&images, cell, cols);
    let path = dir.join(file);
    montage
        .save(&path)
        .with_context(|| format!("write {}", path.display()))?;
    let mut max_dist = 0u32;
    for (a, &i) in members.iter().enumerate() {
        for &j in &members[a + 1..] {
            max_dist = max_dist.max(hamming(hashed[i].hash, hashed[j].hash));
        }
    }
    Ok(max_dist)
}

fn is_image(p: &Path) -> bool {
    p.extension()
        .and_then(|s| s.to_str())
        .map(|e| {
            matches!(
                e.to_ascii_lowercase().as_str(),
                "png" | "jpg" | "jpeg" | "webp"
            )
        })
        .unwrap_or(false)
}

fn walk_image_dir(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let p = entry?.path();
        if p.is_file() && is_image(&p) {
            paths.push(p);
        }
    }
    paths.sort();
    Ok(paths)
}

/// The training CSV: header, the index of the source column, and the
/// raw data lines (kept verbatim so the outputs are byte-faithful).
struct TrainingCsv {
    header: String,
    source_col: usize,
    lines: Vec<String>,
}

fn read_training_csv(path: &Path) -> Result<TrainingCsv> {
    let rdr = BufReader::new(File::open(path).with_context(|| format!("open {}", path.display()))?);
    let mut it = rdr.lines();
    let header = it
        .next()
        .ok_or_else(|| anyhow!("{}: empty file", path.display()))??;
    let cols: Vec<&str> = header.trim_end_matches('\r').split(',').collect();
    let source_col = cols
        .iter()
        .position(|c| *c == "ref_basename")
        .or_else(|| cols.iter().position(|c| *c == "image_path"))
        .unwrap_or(0);
    let mut lines = Vec::new();
    for line in it {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        lines.push(line);
    }
    eprintln!(
        "training csv: {} rows, source column {} ({:?})",
        lines.len(),
        source_col,
        cols.get(source_col).copied().unwrap_or("?")
    );
    Ok(TrainingCsv {
        header,
        source_col,
        lines,
    })
}

fn source_of(line: &str, col: usize) -> &str {
    line.split(',').nth(col).unwrap_or("").trim()
}

fn file_key(p: &Path) -> String {
    p.file_name()
        .map(|f| f.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn resolve_source(name: &str, root: Option<&Path>) -> PathBuf {
    let p = Path::new(name);
    if p.is_absolute() && p.exists() {
        return p.to_path_buf();
    }
    match root {
        Some(r) => {
            let by_name = r.join(Path::new(name).file_name().unwrap_or_default());
            if by_name.exists() { by_name } else { r.join(p) }
        }
        None => p.to_path_buf(),
    }
}

struct Hashed {
    path: PathBuf,
    hash: u64,
    pixels: u64,
}

fn hash_all(paths: &[PathBuf]) -> Vec<Result<Hashed, String>> {
    let total = paths.len();
    let done = std::sync::atomic::AtomicUsize::new(0);
    paths
        .par_iter()
        .map(|p| {
            let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if n.is_multiple_of(500) || n == total {
                eprintln!("  hashed {n} / {total}");
            }
            match image::open(p) {
                Ok(img) => Ok(Hashed {
                    path: p.clone(),
                    hash: dhash_64(&img),
                    pixels: u64::from(img.width()) * u64::from(img.height()),
                }),
                Err(e) => Err(format!("{}: {e}", p.display())),
            }
        })
        .collect()
}

fn write_csv(path: &Path, header: &str, lines: &[&String]) -> Result<()> {
    let mut w =
        BufWriter::new(File::create(path).with_context(|| format!("create {}", path.display()))?);
    writeln!(w, "{header}")?;
    for l in lines {
        writeln!(w, "{l}")?;
    }
    w.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.max_dist > 10 {
        bail!(
            "--max-dist {} is a contamination-screen threshold, not a within-corpus \
             duplicate threshold (benchmarks/dhash_threshold_revert_2026-05-14.md); use ≤ 10",
            args.max_dist
        );
    }

    // 1. Enumerate what to hash.
    let csv = match &args.training_csv {
        Some(p) => Some(read_training_csv(p)?),
        None => None,
    };
    let paths: Vec<PathBuf> = match (&args.corpus_dir, &csv) {
        (Some(dir), _) => walk_image_dir(dir)?,
        (None, Some(csv)) => {
            let distinct: BTreeSet<&str> = csv
                .lines
                .iter()
                .map(|l| source_of(l, csv.source_col))
                .filter(|s| !s.is_empty())
                .collect();
            distinct
                .into_iter()
                .map(|s| resolve_source(s, args.source_root.as_deref()))
                .collect()
        }
        (None, None) => bail!("one of --corpus-dir / --training-csv is required"),
    };
    eprintln!("sources to hash: {}", paths.len());

    // 2. Hash (parallel), dropping undecodable files loudly.
    let mut hashed: Vec<Hashed> = Vec::with_capacity(paths.len());
    let mut n_failed = 0usize;
    for r in hash_all(&paths) {
        match r {
            Ok(h) => hashed.push(h),
            Err(e) => {
                n_failed += 1;
                eprintln!("WARN: {e}");
            }
        }
    }
    eprintln!("hashed: {} ({} failed)", hashed.len(), n_failed);

    // 3. Cluster + derive the per-member curation columns.
    let hashes: Vec<u64> = hashed.iter().map(|h| h.hash).collect();
    let pixels: Vec<u64> = hashed.iter().map(|h| h.pixels).collect();
    let ids = cluster_by_hamming(&hashes, args.max_dist);
    let sizes = cluster_sizes(&ids);
    let weights = content_weights(&ids, &sizes);
    let canonical = canonical_members(&ids, &pixels);
    let split = stratified_split(&ids, &hashes, args.val_frac, args.seed);

    // 4. Per-file TSV.
    {
        let mut w = BufWriter::new(
            File::create(&args.out_tsv)
                .with_context(|| format!("create {}", args.out_tsv.display()))?,
        );
        writeln!(
            w,
            "path\tbasename\tbase_hint\tdhash\tpixels\tcluster_id\tcluster_size\tcontent_weight\tcanonical\tsplit"
        )?;
        for (i, h) in hashed.iter().enumerate() {
            let name = file_key(&h.path);
            writeln!(
                w,
                "{}\t{}\t{}\t{:016x}\t{}\t{}\t{}\t{:.6}\t{}\t{}",
                h.path.display(),
                name,
                base_hint(&name),
                h.hash,
                h.pixels,
                ids[i],
                sizes[ids[i]],
                weights[i],
                u8::from(canonical[i]),
                split[i].as_str()
            )?;
        }
        w.flush()?;
    }

    // 5. Report: size histogram + naming agreement.
    let mut hist: BTreeMap<usize, usize> = BTreeMap::new();
    for &s in &sizes {
        *hist.entry(s).or_default() += 1;
    }
    eprintln!(
        "=== clusters: {} over {} files at d ≤ {} ===",
        sizes.len(),
        hashed.len(),
        args.max_dist
    );
    for (s, n) in &hist {
        eprintln!("  size {s:>3}: {n} clusters");
    }
    let mut hints_per_cluster: Vec<BTreeSet<&str>> = vec![BTreeSet::new(); sizes.len()];
    let mut clusters_per_hint: BTreeMap<&str, BTreeSet<usize>> = BTreeMap::new();
    let names: Vec<String> = hashed.iter().map(|h| file_key(&h.path)).collect();
    for (i, name) in names.iter().enumerate() {
        let hint = base_hint(name);
        hints_per_cluster[ids[i]].insert(hint);
        clusters_per_hint.entry(hint).or_default().insert(ids[i]);
    }
    let multi_hint: Vec<usize> = (0..sizes.len())
        .filter(|&c| hints_per_cluster[c].len() > 1)
        .collect();
    let split_hints: Vec<&str> = clusters_per_hint
        .iter()
        .filter(|(_, cs)| cs.len() > 1)
        .map(|(h, _)| *h)
        .collect();
    eprintln!(
        "=== naming agreement: {} clusters span >1 base hint (cross-source dup or flat-content FP — eyeball); {} base hints spread over >1 cluster (variants the hash did not join) ===",
        multi_hint.len(),
        split_hints.len()
    );
    for &c in multi_hint.iter().take(20) {
        let members: Vec<&str> = names
            .iter()
            .enumerate()
            .filter(|(i, _)| ids[*i] == c)
            .map(|(_, n)| n.as_str())
            .collect();
        eprintln!("  cluster {c}: {}", members.join(" | "));
    }
    for h in split_hints.iter().take(20) {
        eprintln!("  hint {h}: clusters {:?}", clusters_per_hint[h]);
    }

    // 5b. Validation step 2 (the EYEBALL pass) instrument. The 2026-05-14
    // revert's ship policy is "build side-by-side montages, sign off entry
    // by entry" — file names alone are what produced the 149-basename
    // false-positive blocklist. Both halves of the naming-agreement report
    // get a montage: clusters the hash JOINED across base hints, and base
    // hints the hash SPLIT across clusters.
    if let Some(dir) = &args.montage_dir {
        std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
        let flagged: BTreeSet<usize> = multi_hint.iter().copied().collect();
        let mut order: Vec<usize> = multi_hint.clone();
        if args.montage_all {
            let mut rest: Vec<usize> = (0..sizes.len())
                .filter(|c| sizes[*c] >= 2 && !flagged.contains(c))
                .collect();
            rest.sort_by_key(|&c| (std::cmp::Reverse(sizes[c]), c));
            order.extend(rest);
        }
        let mut entries: Vec<MontageEntry> = Vec::new();
        for &c in order.iter().take(args.montage_max) {
            let members: Vec<usize> = (0..ids.len()).filter(|&i| ids[i] == c).collect();
            let is_flagged = flagged.contains(&c);
            let file = format!(
                "cluster_{c:05}_n{}{}.png",
                members.len(),
                if is_flagged { "_multihint" } else { "" }
            );
            let max_dist = write_montage(
                dir,
                &file,
                &members,
                &hashed,
                args.montage_cell,
                args.montage_cols,
            )?;
            entries.push(MontageEntry {
                file,
                title: format!("cluster {c} — {} members", members.len()),
                flag: if is_flagged {
                    "SPANS &gt;1 BASE HINT — cross-source duplicate, or the flat-content false positive the 2026-05-14 revert is about"
                } else {
                    ""
                },
                members,
                max_dist,
            });
        }
        for h in split_hints.iter().take(args.montage_max) {
            let members: Vec<usize> = (0..names.len())
                .filter(|&i| base_hint(&names[i]) == *h)
                .collect();
            let file = format!("hint_{h}.png");
            let max_dist = write_montage(
                dir,
                &file,
                &members,
                &hashed,
                args.montage_cell,
                args.montage_cols,
            )?;
            entries.push(MontageEntry {
                file,
                title: format!("base hint {h} — {} members", members.len()),
                flag: "SPREAD OVER &gt;1 CLUSTER — variants the hash did NOT join (crop, or a naming collision)",
                members,
                max_dist,
            });
        }

        let index = dir.join("index.html");
        let mut w = BufWriter::new(
            File::create(&index).with_context(|| format!("create {}", index.display()))?,
        );
        writeln!(
            w,
            "<!doctype html><meta charset=\"utf-8\"><title>content clusters — eyeball pass</title>\
             <style>body{{font:14px/1.45 system-ui,sans-serif;margin:2rem;max-width:80rem}}\
             img{{max-width:100%;image-rendering:auto;border:1px solid #8888}}\
             table{{border-collapse:collapse;margin:.4rem 0}}td,th{{padding:.15rem .5rem;text-align:left;\
             border-bottom:1px solid #8883;font-variant-numeric:tabular-nums}}\
             .flag{{color:#b40;font-weight:600}}section{{margin:2rem 0;padding-top:1rem;border-top:2px solid #8884}}</style>"
        )?;
        writeln!(
            w,
            "<h1>Within-corpus content clusters — eyeball pass (issue #33)</h1>\
             <p>d &le; {}, {} files, {} clusters. Every montage below is a group this run \
             proposes to treat as ONE content. Sign off entry by entry; a group that is NOT \
             one content is a false positive and must be excluded before any reweight/cull \
             is applied (2026-05-14 revert policy).</p>",
            args.max_dist,
            hashed.len(),
            sizes.len()
        )?;
        for e in &entries {
            writeln!(
                w,
                "<section><h2>{}</h2>{}<p><img src=\"{}\" alt=\"{}\"></p>\
                 <p>max pairwise dHash distance inside the group: <b>{}</b></p>\
                 <table><tr><th>#</th><th>file</th><th>dhash</th><th>pixels</th>\
                 <th>cluster</th><th>canonical</th><th>split</th></tr>",
                html_escape(&e.title),
                if e.flag.is_empty() {
                    String::new()
                } else {
                    format!("<p class=\"flag\">{}</p>", e.flag)
                },
                html_escape(&e.file),
                html_escape(&e.title),
                e.max_dist
            )?;
            for (n, &i) in e.members.iter().enumerate() {
                writeln!(
                    w,
                    "<tr><td>{}</td><td>{}</td><td>{:016x}</td><td>{}</td><td>{}</td>\
                     <td>{}</td><td>{}</td></tr>",
                    n,
                    html_escape(&names[i]),
                    hashed[i].hash,
                    hashed[i].pixels,
                    ids[i],
                    if canonical[i] { "yes" } else { "" },
                    split[i].as_str()
                )?;
            }
            writeln!(w, "</table></section>")?;
        }
        w.flush()?;
        eprintln!(
            "=== montages: {} written to {} (open {}) ===",
            entries.len(),
            dir.display(),
            index.display()
        );
    }

    // 6. CSV-derived outputs.
    if let Some(csv) = &csv {
        let by_name: HashMap<&str, usize> = names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i))
            .collect();
        let row_index = |line: &str| -> Option<usize> {
            let src = source_of(line, csv.source_col);
            let key = Path::new(src)
                .file_name()
                .map(|f| f.to_string_lossy().into_owned())
                .unwrap_or_default();
            by_name.get(key.as_str()).copied()
        };
        let row_idx: Vec<Option<usize>> = csv.lines.iter().map(|l| row_index(l)).collect();
        let n_unknown = row_idx.iter().filter(|r| r.is_none()).count();
        eprintln!(
            "csv rows: {} ({} with a source that was not hashed → treated as size-1 / canonical / train)",
            csv.lines.len(),
            n_unknown
        );

        if let Some(out) = &args.cull_csv {
            let kept: Vec<&String> = csv
                .lines
                .iter()
                .zip(&row_idx)
                .filter(|(_, r)| r.is_none_or(|i| canonical[i]))
                .map(|(l, _)| l)
                .collect();
            write_csv(out, &csv.header, &kept)?;
            eprintln!(
                "cull: kept {} / {} rows (one variant per cluster) → {}",
                kept.len(),
                csv.lines.len(),
                out.display()
            );
        }

        if let Some(dir) = &args.reweight_dir {
            std::fs::create_dir_all(dir)?;
            let row_k: Vec<usize> = row_idx
                .iter()
                .map(|r| r.map_or(1, |i| sizes[ids[i]]))
                .collect();
            let groups = reweight_groups(&row_k);
            let mut spec = String::new();
            for g in &groups {
                let fname = format!("cluster_size_{}.csv", g.cluster_size);
                let rows: Vec<&String> = csv
                    .lines
                    .iter()
                    .zip(&row_k)
                    .filter(|&(_, &k)| k == g.cluster_size)
                    .map(|(l, _)| l)
                    .collect();
                write_csv(&dir.join(&fname), &csv.header, &rows)?;
                spec.push_str(&format!(
                    "--group k{}:{}:{:.6}:0\n",
                    g.cluster_size,
                    dir.join(&fname).display(),
                    g.train_weight
                ));
            }
            std::fs::write(dir.join("groups.txt"), &spec)?;
            eprintln!(
                "reweight: {} groups (train_w ∝ n_rows / cluster_size) → {}/groups.txt",
                groups.len(),
                dir.display()
            );
            eprint!("{spec}");
        }

        if let Some(dir) = &args.split_dir {
            std::fs::create_dir_all(dir)?;
            let (mut train, mut val) = (Vec::new(), Vec::new());
            for (l, r) in csv.lines.iter().zip(&row_idx) {
                match r.map_or(Split::Train, |i| split[i]) {
                    Split::Train => train.push(l),
                    Split::Val => val.push(l),
                }
            }
            write_csv(&dir.join("train.csv"), &csv.header, &train)?;
            write_csv(&dir.join("val.csv"), &csv.header, &val)?;
            eprintln!(
                "split: train {} / val {} rows (whole clusters per side) → {}",
                train.len(),
                val.len(),
                dir.display()
            );
        }
    }
    Ok(())
}
