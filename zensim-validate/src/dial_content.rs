//! Content class per dial-grid reference image — the axis the ladder-inversion
//! panel splits on.
//!
//! The table is a HAND REVIEW recorded in
//! `benchmarks/dial_grid_content_classes_2026-08-31.tsv` (every source PNG was
//! rendered into contact sheets and labelled by eye, 2026-08-31). There is no
//! classifier here and none is wanted: a feature-threshold "screen detector"
//! would produce a plausible-looking split that nobody can audit, which is
//! exactly the failure mode the failure-profile work exists to avoid.
//!
//! An image the table does not name is [`UNCLASSIFIED`], and callers must
//! report that bucket rather than fold it into a class.

/// Bucket for an image the review has not labelled. Reported, never merged.
pub const UNCLASSIFIED: &str = "unclassified";

/// The reviewed classes, in report order.
pub const CLASSES: [&str; 3] = ["photo", "text_lineart", "nonphoto"];

const TABLE: &str = include_str!("../../benchmarks/dial_grid_content_classes_2026-08-31.tsv");

/// `(image_id, class, note)` for every reviewed image.
pub fn entries() -> impl Iterator<Item = (&'static str, &'static str, &'static str)> {
    TABLE.lines().filter_map(|l| {
        if l.starts_with('#') || l.trim().is_empty() {
            return None;
        }
        let mut it = l.split('\t');
        let id = it.next()?.trim();
        let class = it.next()?.trim();
        let note = it.next().unwrap_or("").trim();
        if id.is_empty() || class.is_empty() {
            return None;
        }
        Some((id, class, note))
    })
}

/// Content class for a dial-grid `image_id`, or [`UNCLASSIFIED`].
pub fn class_of(image_id: &str) -> &'static str {
    entries()
        .find(|(id, _, _)| *id == image_id)
        .map(|(_, c, _)| c)
        .unwrap_or(UNCLASSIFIED)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_parses_and_covers_the_reviewed_grid() {
        let rows: Vec<_> = entries().collect();
        assert_eq!(rows.len(), 39, "the 2026-08-31 review labelled 39 images");
        let mut ids: Vec<&str> = rows.iter().map(|(i, _, _)| *i).collect();
        ids.sort_unstable();
        let n = ids.len();
        ids.dedup();
        assert_eq!(ids.len(), n, "duplicate image_id in the class table");
        for (id, class, _) in &rows {
            assert!(
                CLASSES.contains(class),
                "{id}: class `{class}` is not one of {CLASSES:?}"
            );
        }
    }

    #[test]
    fn class_counts_match_the_recorded_review() {
        // The doc + the panel caption quote these; if the table is edited the
        // counts must be re-quoted, so lock them here.
        let count = |c: &str| entries().filter(|(_, k, _)| *k == c).count();
        assert_eq!(count("photo"), 27);
        assert_eq!(count("text_lineart"), 9);
        assert_eq!(count("nonphoto"), 3);
    }

    #[test]
    fn unknown_image_is_unclassified_not_silently_a_class() {
        assert_eq!(class_of("no_such_image_9999"), UNCLASSIFIED);
        assert_eq!(class_of("4bb837d2ff0eabc5_513x769"), "nonphoto");
        assert_eq!(class_of("5a9b3b963f852e20_512sq"), "text_lineart");
        assert_eq!(class_of("00b13be94a4867dd_1022x818"), "photo");
    }
}
