// Disjoint retained output rows let a worker finish A -> retention -> B without
// a channel barrier. Each destination element receives the original copy.
struct RetainedRows<'a> {
    planes: [&'a mut [f32]; 7],
    basic: Option<&'a mut [f32]>,
    bs2: Option<&'a mut [f32]>,
}

impl<'a> RetainedRows<'a> {
    fn take_prefix(&mut self, n: usize) -> RetainedRows<'a> {
        let planes = core::array::from_fn(|i| {
            let (head, tail) = core::mem::take(&mut self.planes[i]).split_at_mut(n);
            self.planes[i] = tail;
            head
        });
        fn take<'a>(value: &mut Option<&'a mut [f32]>, n: usize) -> Option<&'a mut [f32]> {
            value.as_mut().map(|v| {
                let (head, tail) = core::mem::take(v).split_at_mut(n);
                *v = tail;
                head
            })
        }
        RetainedRows {
            planes,
            basic: take(&mut self.basic, n),
            bs2: take(&mut self.bs2, n),
        }
    }

    fn copy(
        &mut self,
        info: &crate::feature_v2_stream::StripInfo,
        scr: &ScratchV2Strip,
        src: &[f32],
        dst: &[f32],
        want_bs2: bool,
    ) {
        let off = HALO_P * info.plane_w;
        let n = info.strip_h * info.plane_w;
        for (out, input) in self.planes.iter_mut().zip([
            src,
            dst,
            &scr.mu1,
            &scr.mu2,
            &scr.ssq,
            &scr.s12,
            &scr.activity,
        ]) {
            out.copy_from_slice(&input[off..off + n]);
        }
        if let Some(sd) = &mut self.basic {
            incant!(
                retain_basic_sd(
                    self.planes[2],
                    self.planes[3],
                    self.planes[4],
                    self.planes[5],
                    sd
                ),
                [v4x, v4, v3, scalar]
            );
        }
        if let Some(bs2) = &mut self.bs2 {
            if want_bs2 {
                bs2.copy_from_slice(&scr.bs2[off..off + n]);
            } else {
                bs2.fill(0.0);
            }
        }
    }
}

impl FoldRetention {
    fn row_sinks(&mut self) -> impl Iterator<Item = [RetainedRows<'_>; 3]> {
        let retain_basic = self.retain_basic;
        let rev5 = self.rev5;
        let mut basic = self.basic_sd.iter_mut();
        self.pyr_src
            .iter_mut()
            .zip(self.pyr_dst.iter_mut())
            .zip(self.planes.iter_mut())
            .map(move |((src, dst), planes)| {
                let mut src = src.iter_mut();
                let mut dst = dst.iter_mut();
                let mut planes = planes.iter_mut();
                let mut sd = retain_basic.then(|| basic.next().unwrap().iter_mut());
                core::array::from_fn(|_| {
                    let p = planes.next().unwrap();
                    RetainedRows {
                        planes: [
                            src.next().unwrap(),
                            dst.next().unwrap(),
                            &mut p.mu1,
                            &mut p.mu2,
                            &mut p.ssq,
                            &mut p.s12,
                            &mut p.act,
                        ],
                        basic: sd.as_mut().map(|it| it.next().unwrap().as_mut_slice()),
                        bs2: (!rev5).then_some(p.bs2.as_mut_slice()),
                    }
                })
            })
    }

    #[cfg(feature = "threads")]
    fn job_sinks<'a>(&'a mut self, jobs: &[Rev5StripJob]) -> Vec<Option<RetainedRows<'a>>> {
        let mut out: Vec<_> = (0..jobs.len()).map(|_| None).collect();
        for (scale, sinks) in self.row_sinks().enumerate() {
            for (ch, mut sink) in sinks.into_iter().enumerate() {
                let mut cursor = 0;
                for (index, job) in jobs.iter().enumerate() {
                    if job.info.scale == scale && job.ch == ch {
                        let start = job.info.y0 * job.info.plane_w;
                        assert!(start >= cursor);
                        sink.take_prefix(start - cursor);
                        let n = job.info.strip_h * job.info.plane_w;
                        out[index] = Some(sink.take_prefix(n));
                        cursor = start + n;
                    }
                }
            }
        }
        out
    }
}
