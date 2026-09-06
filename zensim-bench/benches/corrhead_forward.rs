//! **What the corruption head's forward pass costs**, against the profile
//! forward it rides alongside.
//!
//! G8 of `docs/PLAN_CORRHEAD_SERVING_2026-09-06.md`; record:
//! `benchmarks/corruption_head_serving_2026-09-06.md`.
//!
//! ## Why this is a separate bench from `ssim2_speed_bar`
//!
//! `ssim2_speed_bar`'s `add156_plus_corrhead` arm prices the head IN SITU —
//! the honest end-to-end question, and the right instrument for "what does a
//! served compare cost". It is the wrong instrument for "what does the tree
//! walk cost", because the head is a low-single-digit-percent rider on a
//! multi-millisecond extraction: the marginal signal sits inside that arm's
//! own round-to-round spread. MEASURED at 576²/1T on a loaded box, that group
//! degenerated to 3 rounds at CV 51-141 % and reported a marginal that is not
//! a measurement (see the record).
//!
//! So this bench removes the extraction entirely. Every arm forwards the SAME
//! pre-built 372-wide feature row, so the only thing that varies is the
//! forward. No image work, no allocation inside the timed region, nothing
//! thread-dependent — which is what makes a microsecond-scale difference
//! resolvable at all.
//!
//! ## Arms
//!
//! | arm | what it forwards |
//! |---|---|
//! | `profile_d_forward` | `ZEN_HY_ADD` — the shipped fast-class dial (ADD156) |
//! | `corrhead_tree_forward` | `ZEN_HY_CORRHEAD` when it is a `ZCTH` gradient-boosted tree |
//! | `corrhead_znpr_forward` | `ZEN_HY_CORRHEAD` when it is the incumbent `ZNPR` logistic |
//!
//! Run (1 thread — the forward is scalar and single-threaded by construction,
//! so the thread count only changes who else is competing for the core):
//!
//! ```sh
//! cd zensim-bench
//! ZEN_HY_ADD=…/d_sdr_add156_id100_negrich_dial_2026-09-05.bin \
//! ZEN_HY_CORRHEAD=…/corrhead_hgb_theoryfit_w372.zcth \
//! RAYON_NUM_THREADS=1 cargo bench --bench corrhead_forward
//! ```

use zenpredict::{Model, Predictor};

/// A pre-built 372-wide feature row.
///
/// Deterministic and in the plausible range of a real extraction — but the
/// VALUES do not matter for a cost measurement of a tree walk in the way they
/// would for a scoring measurement, and pretending otherwise would be the
/// interesting kind of wrong. What DOES matter is that they are not
/// degenerate: an all-zero row would send every tree down one branch and
/// price a best case. These are spread across the standardised range so the
/// walk takes a mix of paths.
fn feature_row() -> Vec<f64> {
    (0..372)
        .map(|i| {
            let t = (i as f64) * 0.37;
            t.sin() * 2.5 + (i as f64 % 7.0) * 0.11
        })
        .collect()
}

fn main() {
    let row: &'static [f64] = Box::leak(feature_row().into_boxed_slice());

    let dial: Option<&'static (Model, bool)> = std::env::var("ZEN_HY_ADD")
        .ok()
        .and_then(|p| std::fs::read(&p).ok().map(|b| (p, b)))
        .and_then(|(p, b)| match Model::from_bytes(&b) {
            Ok(m) => {
                eprintln!(
                    "# ZEN_HY_ADD={p}: ZNPR n_inputs={} caller_width={}",
                    m.n_inputs(),
                    m.caller_input_width()
                );
                let tf = m.has_nontrivial_feature_transforms();
                Some(&*Box::leak(Box::new((m, tf))))
            }
            Err(e) => {
                eprintln!("# ZEN_HY_ADD={p}: not a ZNPR ({e:?}) — arm SKIPPED");
                None
            }
        });

    enum Head {
        Tree(zensim::corruption_head::CorruptionHead),
        Znpr(Model, bool),
    }
    let head: Option<&'static Head> = std::env::var("ZEN_HY_CORRHEAD")
        .ok()
        .and_then(|p| std::fs::read(&p).ok().map(|b| (p, b)))
        .and_then(|(p, b)| {
            if b.get(..4) == Some(&zensim::corruption_head::MAGIC[..]) {
                match zensim::corruption_head::CorruptionHead::from_bytes(&b) {
                    Ok(h) => {
                        eprintln!(
                            "# ZEN_HY_CORRHEAD={p}: ZCTH {} trees / {} nodes, reads {} of {}",
                            h.n_trees(),
                            h.n_nodes(),
                            h.declared_feature_ids().len(),
                            h.caller_input_width()
                        );
                        Some(&*Box::leak(Box::new(Head::Tree(h))))
                    }
                    Err(e) => {
                        eprintln!("# ZEN_HY_CORRHEAD={p}: bad ZCTH ({e}) — arm SKIPPED");
                        None
                    }
                }
            } else {
                match Model::from_bytes(&b) {
                    Ok(m) => {
                        eprintln!("# ZEN_HY_CORRHEAD={p}: ZNPR logistic head");
                        let tf = m.has_nontrivial_feature_transforms();
                        Some(&*Box::leak(Box::new(Head::Znpr(m, tf))))
                    }
                    Err(e) => {
                        eprintln!("# ZEN_HY_CORRHEAD={p}: neither ZCTH nor ZNPR ({e:?})");
                        None
                    }
                }
            }
        });

    if dial.is_none() && head.is_none() {
        eprintln!(
            "corrhead_forward: nothing to measure. Set ZEN_HY_ADD and/or \
             ZEN_HY_CORRHEAD; see this file's header for the exact invocation."
        );
        std::process::exit(2);
    }

    let rounds: usize = std::env::var("ZEN_CH_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(400);
    let wall_s: u64 = std::env::var("ZEN_CH_WALL_S")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(120);

    let result = zenbench::run(|suite| {
        suite.compare("corrhead_forward", |group| {
            group
                .config()
                .max_rounds(rounds)
                .min_rounds(rounds.min(24))
                .max_wall_time(std::time::Duration::from_secs(wall_s));
            if let Some((m, tf)) = dial {
                group.bench("profile_d_forward", move |b| {
                    let mut pred = Predictor::new(m);
                    let mut x: Vec<f32> = row.iter().map(|v| *v as f32).collect();
                    b.iter(move || {
                        let out = if *tf {
                            pred.predict_transformed(&mut x).expect("forward")
                        } else {
                            pred.predict(&x).expect("forward")
                        };
                        zenbench::black_box(out[0])
                    })
                });
            }
            match head {
                Some(Head::Tree(h)) => {
                    group.bench("corrhead_tree_forward", move |b| {
                        b.iter(move || {
                            zenbench::black_box(h.probability_f64(row).expect("forward"))
                        })
                    });
                }
                Some(Head::Znpr(m, tf)) => {
                    group.bench("corrhead_znpr_forward", move |b| {
                        let mut pred = Predictor::new(m);
                        let mut x: Vec<f32> = row.iter().map(|v| *v as f32).collect();
                        b.iter(move || {
                            let out = if *tf {
                                pred.predict_transformed(&mut x).expect("forward")
                            } else {
                                pred.predict(&x).expect("forward")
                            };
                            zenbench::black_box(out[0])
                        })
                    });
                }
                None => {}
            }
        });
    });
    let _ = result;
}
