use zenpsy_api::{
    self as psy, Direction, DynPreparePlanarReference, Erased, ErrorKind, EvalOpts, Evaluation,
    Metric, PlanarWarmReference, PreparePlanarReference, PrepareReference,
    how_far::{Stop, StopOnly, StopReason},
    prepare_planar_reference, prepare_reference, score, score_planar,
    zenpixels::{plane::*, *},
};

struct CancelledStop;
impl Stop for CancelledStop {
    fn check(&self) -> Result<(), StopReason> {
        Err(StopReason::Cancelled)
    }
}
use zensim::{DiffmapOptions, PrecomputedReference, Zensim, ZensimProfile};
use zensim_psy::{DIFFMAP_KEY, SCORE_KEY, ZENSIM_A_UNIT, ZensimMetric};

fn planar_layout() -> PlanarLayout {
    PlanarLayout::new(
        [PlaneRole::Red, PlaneRole::Green, PlaneRole::Blue]
            .map(|r| PlaneSpec::new(r, PlaneEncoding::Float(FloatEncoding::NOMINAL)))
            .to_vec(),
        sample::ChromaSubsampling::Yuv444,
        sample::ChromaSiting::Unknown,
    )
    .unwrap()
}

fn make_linear_planes(w: usize, h: usize, strides: [usize; 3], delta: f32) -> [Vec<f32>; 3] {
    std::array::from_fn(|c| {
        // Minimal active-row storage: no trailing padding on final row, NaN in interior row padding.
        let s = strides[c];
        let len = (h - 1) * s + w;
        let mut buf = vec![f32::NAN; len];
        for y in 0..h {
            for x in 0..w {
                let base = ((x * 7 + y * 13 + c * 29) % 256) as f32 / 255.0;
                buf[y * s + x] = (base + delta * ((x + y + c) % 3) as f32).clamp(0.0, 1.0);
            }
        }
        buf
    })
}

#[test]
fn planar_lending_matches_direct_rev7_zensim_a_bit_exact() {
    let w = 97usize;
    let h = 73usize;
    let ref_strides = [w + 5, w + 9, w];
    let dst_strides = [w + 11, w + 3, w + 7];
    let r_bufs = make_linear_planes(w, h, ref_strides, 0.0);
    let d_bufs = make_linear_planes(w, h, dst_strides, 0.06);

    let layout = planar_layout();
    let color = ColorContext::from_cicp(Cicp::new(1, 8, 0, true));
    let r_planes = [
        AnyPlane::F32(PlaneRef::new(&r_bufs[0], w, h, ref_strides[0]).unwrap()),
        AnyPlane::F32(PlaneRef::new(&r_bufs[1], w, h, ref_strides[1]).unwrap()),
        AnyPlane::F32(PlaneRef::new(&r_bufs[2], w, h, ref_strides[2]).unwrap()),
    ];
    let d_planes = [
        AnyPlane::F32(PlaneRef::new(&d_bufs[0], w, h, dst_strides[0]).unwrap()),
        AnyPlane::F32(PlaneRef::new(&d_bufs[1], w, h, dst_strides[1]).unwrap()),
        AnyPlane::F32(PlaneRef::new(&d_bufs[2], w, h, dst_strides[2]).unwrap()),
    ];
    let ref_slice = PlanarSlice::new(&r_planes, &layout, w as u32, h as u32, &color, None).unwrap();
    let dst_slice = PlanarSlice::new(&d_planes, &layout, w as u32, h as u32, &color, None).unwrap();

    let diff_opts = DiffmapOptions {
        include_edge_mse: true,
        include_hf: true,
        ..DiffmapOptions::default()
    };

    // Direct Rev-7 ZensimProfile::A oracle using tight planes.
    let direct_r: [Vec<f32>; 3] = std::array::from_fn(|c| {
        (0..h)
            .flat_map(|y| {
                r_bufs[c][y * ref_strides[c]..y * ref_strides[c] + w]
                    .iter()
                    .copied()
            })
            .collect()
    });
    let direct_d: [Vec<f32>; 3] = std::array::from_fn(|c| {
        (0..h)
            .flat_map(|y| {
                d_bufs[c][y * dst_strides[c]..y * dst_strides[c] + w]
                    .iter()
                    .copied()
            })
            .collect()
    });
    let z = Zensim::new(ZensimProfile::A);
    let pre = z
        .precompute_reference_linear_planar([&direct_r[0], &direct_r[1], &direct_r[2]], w, h, w)
        .unwrap();
    let direct = z
        .compute_with_ref_and_diffmap_linear_planar(
            &pre,
            [&direct_d[0], &direct_d[1], &direct_d[2]],
            w,
            h,
            w,
            diff_opts,
        )
        .unwrap();

    let mut metric = ZensimMetric::profile_a().with_diffmap_options(diff_opts);
    assert_eq!(metric.headline_score_key(), Some(SCORE_KEY));
    assert_eq!(metric.steering_map_key(), Some(DIFFMAP_KEY));
    assert_eq!(
        PreparePlanarReference::estimated_reference_bytes(&metric, w as u32, h as u32),
        PrecomputedReference::estimated_bytes_for_profile(w, h, ZensimProfile::A)
    );
    assert!(
        PreparePlanarReference::estimated_reference_bytes(&metric, w as u32, h as u32).unwrap() > 0
    );

    let cold = score_planar(
        &mut metric,
        &ref_slice,
        &dst_slice,
        EvalOpts::new().with_map(DIFFMAP_KEY),
    )
    .unwrap();
    let cold_score = cold.scores().next().unwrap();
    assert_eq!(cold_score.spec.unit(), ZENSIM_A_UNIT);
    assert_eq!(cold_score.spec.direction(), Direction::Higher);
    assert_eq!(cold_score.spec.perfect(), Some(100.0));
    assert_eq!(cold_score.value.to_bits(), direct.score().to_bits());
    assert_eq!(
        cold.diffmap()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        direct
            .diffmap()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );

    let mut prepared = prepare_planar_reference(&mut metric, &ref_slice, EvalOpts::new()).unwrap();
    assert!(prepared.is_deterministic());
    let (lent_score_bits, lent_map_bits) = prepared
        .with_evaluation(&dst_slice, EvalOpts::new().with_map(DIFFMAP_KEY), |view| {
            let s = view.score(SCORE_KEY).unwrap();
            let m = view.map(DIFFMAP_KEY).unwrap();
            assert!(m.grid.is_pixel_aligned(m.values.width(), m.values.height()));
            let p = m.as_f32().unwrap();
            let bits: Vec<u32> = p.rows().flatten().map(|v| v.to_bits()).collect();
            Ok((s.value.to_bits(), bits))
        })
        .unwrap();
    assert_eq!(lent_score_bits, direct.score().to_bits());
    assert_eq!(
        lent_map_bits,
        direct
            .diffmap()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );

    // Works through Erased + DynPreparePlanarReference too.
    let mut erased: Box<dyn DynPreparePlanarReference> = Box::new(Erased::new(metric));
    assert_eq!(erased.steering_map_key(), Some(DIFFMAP_KEY));
    let mut dyn_prep = erased.prepare_planar(&ref_slice, EvalOpts::new()).unwrap();
    assert!(dyn_prep.is_deterministic());
    let mut dyn_score = None;
    dyn_prep
        .with_evaluation(
            &dst_slice,
            EvalOpts::new().with_map(DIFFMAP_KEY),
            &mut |view| {
                dyn_score = view.primary_score().map(|s| s.value.to_bits());
            },
        )
        .unwrap();
    assert_eq!(dyn_score, Some(direct.score().to_bits()));
}

#[test]
fn packed_srgb8_and_wide_gamut_planar_and_refusals() {
    // Packed sRGB8 parity.
    let w = 32u32;
    let h = 24u32;
    let mut ref_bytes = vec![0u8; (w * h * 3) as usize];
    let mut dst_bytes = vec![0u8; (w * h * 3) as usize];
    for i in 0..ref_bytes.len() {
        ref_bytes[i] = ((i * 37) % 256) as u8;
        dst_bytes[i] = ref_bytes[i].saturating_add((i % 7) as u8);
    }
    let r_slice = PixelSlice::new(
        &ref_bytes,
        w,
        h,
        (w * 3) as usize,
        PixelDescriptor::RGB8_SRGB,
    )
    .unwrap();
    let d_slice = PixelSlice::new(
        &dst_bytes,
        w,
        h,
        (w * 3) as usize,
        PixelDescriptor::RGB8_SRGB,
    )
    .unwrap();

    let mut metric = ZensimMetric::profile_a();
    assert!(PrepareReference::estimated_reference_bytes(&metric, w, h).unwrap() > 0);
    let cold = score(&mut metric, &r_slice, &d_slice, EvalOpts::new()).unwrap();
    let mut warm = prepare_reference(&mut metric, &r_slice, EvalOpts::new()).unwrap();
    assert!(warm.is_deterministic());
    let warm_res = warm.score(&d_slice, EvalOpts::new()).unwrap();
    assert_eq!(
        cold.scores().next().unwrap().value.to_bits(),
        warm_res.scores().next().unwrap().value.to_bits()
    );

    // Wide gamut Display P3 & BT.2020 planar accepted; HDR & ICC refused.
    let layout = planar_layout();
    let p_bufs = make_linear_planes(16, 16, [16, 16, 16], 0.0);
    let planes = [
        AnyPlane::F32(PlaneRef::new(&p_bufs[0], 16, 16, 16).unwrap()),
        AnyPlane::F32(PlaneRef::new(&p_bufs[1], 16, 16, 16).unwrap()),
        AnyPlane::F32(PlaneRef::new(&p_bufs[2], 16, 16, 16).unwrap()),
    ];
    for cp in [1u8, 12u8, 9u8] {
        let color = ColorContext::from_cicp(Cicp::new(cp, 8, 0, true));
        let slice = PlanarSlice::new(&planes, &layout, 16, 16, &color, None).unwrap();
        let res = score_planar(&mut metric, &slice, &slice, EvalOpts::new()).unwrap();
        assert!((res.scores().next().unwrap().value - 100.0).abs() < 1e-6);
    }

    // HDR PQ refused.
    let hdr = ColorContext::from_cicp(Cicp::new(9, 16, 0, true));
    let hdr_slice = PlanarSlice::new(&planes, &layout, 16, 16, &hdr, None).unwrap();
    assert_eq!(
        prepare_planar_reference(&mut metric, &hdr_slice, EvalOpts::new())
            .err()
            .unwrap()
            .error()
            .kind(),
        ErrorKind::UnsupportedInput
    );

    // ICC refused.
    let icc = ColorContext::from_cicp(Cicp::new(1, 8, 0, true)).with_icc(vec![1u8, 2, 3, 4]);
    let icc_slice = PlanarSlice::new(&planes, &layout, 16, 16, &icc, None).unwrap();
    assert_eq!(
        prepare_planar_reference(&mut metric, &icc_slice, EvalOpts::new())
            .err()
            .unwrap()
            .error()
            .kind(),
        ErrorKind::UnsupportedInput
    );

    // Missing named map refused.
    let srgb = ColorContext::from_cicp(Cicp::new(1, 8, 0, true));
    let ok_slice = PlanarSlice::new(&planes, &layout, 16, 16, &srgb, None).unwrap();
    assert_eq!(
        score_planar(
            &mut metric,
            &ok_slice,
            &ok_slice,
            EvalOpts::new().with_map("other")
        )
        .err()
        .unwrap()
        .error()
        .kind(),
        ErrorKind::NoMap
    );

    // Cancellation stops and subsequent uncanceled call succeeds.
    let mut prepared = prepare_planar_reference(&mut metric, &ok_slice, EvalOpts::new()).unwrap();
    let pulse = StopOnly::new(CancelledStop);
    let err = prepared
        .with_evaluation(&ok_slice, EvalOpts::new().with_pulse(&pulse), |_| Ok(()))
        .unwrap_err();
    assert_eq!(psy::stop_reason(&err), Some(StopReason::Cancelled));
    let s = prepared
        .with_evaluation(&ok_slice, EvalOpts::new(), |v| {
            Ok(v.primary_score().unwrap().value)
        })
        .unwrap();
    assert!((s - 100.0).abs() < 1e-6);
}
