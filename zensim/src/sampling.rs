//! Model-bound experimental XYB sampling. Kernels belong to zenresize;
//! extraction and attribution remain in their existing owners.
#![cfg_attr(
    not(all(feature = "custom-profiles", feature = "feature-regime-v2")),
    allow(dead_code)
)]

use crate::streaming::XybPyramidLevel;
use crate::{ImageSource, ZensimError};
use zenresize::filter::InterpolationDetails;
use zenresize::weights::F32WeightTable;
use zenresize::{Filter, PixelDescriptor, ResizeConfig, Resizer};

pub(crate) const KEY: &str = "zentrain.sampling";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Sampling {
    pub(crate) keep_y: bool,
    kernel: u8,
    num: usize,
    den: usize,
}

impl Sampling {
    pub(crate) fn from_model(model: &crate::mlp::Model) -> Result<Option<Self>, ZensimError> {
        if model.metadata().get(KEY).is_none() {
            return Ok(None);
        }
        let bad = || ZensimError::ModelLoadFailed {
            reason: "unknown zentrain.sampling contract",
        };
        let value = model.metadata().get_utf8(KEY).map_err(|_| bad())?;
        let parts: Vec<_> = value.split(':').collect();
        if parts.len() != 4 || parts[0] != "v1" {
            return Err(bad());
        }
        let keep_y = match parts[1] {
            "y" => true,
            "xyb" => false,
            _ => return Err(bad()),
        };
        let kernel = match parts[2] {
            "triangle" => 0,
            "mitchell" => 1,
            "robidouxsharp" => 2,
            _ => return Err(bad()),
        };
        let (num, den) = match parts[3] {
            "3/2" => (3, 2),
            "2" => (2, 1),
            "3" => (3, 1),
            _ => return Err(bad()),
        };
        Ok(Some(Self {
            keep_y,
            kernel,
            num,
            den,
        }))
    }

    fn filter(self) -> Filter {
        match self.kernel {
            0 => Filter::Triangle,
            1 => Filter::Mitchell,
            _ => Filter::RobidouxSharp,
        }
    }

    pub(crate) fn min_dim(self) -> usize {
        // Four real levels, each at least eight pixels on both axes.
        (if self.keep_y { 32 } else { 64 }) * self.num / self.den
    }

    pub(crate) fn dims(self, w: usize, h: usize) -> Vec<(usize, usize)> {
        let (mut w, mut h) = (w.max(self.min_dim()), h.max(self.min_dim()));
        let mut dims = Vec::with_capacity(4);
        if !self.keep_y {
            w = w * self.den / self.num;
            h = h * self.den / self.num;
        }
        dims.push((w, h));
        for level in 1..4 {
            if level == 1 && self.keep_y {
                w = w * self.den / self.num;
                h = h * self.den / self.num;
            } else {
                w /= 2;
                h /= 2;
            }
            dims.push((w, h));
        }
        dims
    }

    fn resize(
        self,
        previous: &XybPyramidLevel,
        w: usize,
        h: usize,
        parallel: bool,
    ) -> XybPyramidLevel {
        let config =
            ResizeConfig::builder(previous.1 as u32, previous.2 as u32, w as u32, h as u32)
                .format(PixelDescriptor::GRAYF32_LINEAR)
                .filter(self.filter())
                .build();
        #[cfg(feature = "threads")]
        if parallel && previous.1 * previous.2 >= 65_536 && rayon::current_num_threads() > 1 {
            use rayon::prelude::*;
            let mut planes: [Vec<f32>; 3] = core::array::from_fn(|_| vec![0.0; w * h]);
            planes.par_iter_mut().enumerate().for_each(|(c, out)| {
                Resizer::new(&config).resize_f32_into(&previous.0[c], out);
            });
            return (planes, w, h);
        }
        let _ = parallel;
        let mut resizer = Resizer::new(&config);
        let planes = core::array::from_fn(|c| {
            let mut out = vec![0.0; w * h];
            resizer.resize_f32_into(&previous.0[c], &mut out);
            out
        });
        (planes, w, h)
    }

    pub(crate) fn pyramid(self, source: &impl ImageSource, parallel: bool) -> Vec<XybPyramidLevel> {
        let dims = self.dims(source.width(), source.height());
        let padded;
        let planes = if source.width() < self.min_dim() || source.height() < self.min_dim() {
            padded = crate::metric::reflect_pad_to_size(source, self.min_dim());
            crate::streaming::convert_source_to_xyb(&padded, padded.width(), parallel)
        } else {
            crate::streaming::convert_source_to_xyb(source, source.width(), parallel)
        };
        let original = (
            planes,
            source.width().max(self.min_dim()),
            source.height().max(self.min_dim()),
        );
        let first = if self.keep_y {
            original
        } else {
            self.resize(&original, dims[0].0, dims[0].1, parallel)
        };
        let mut levels = vec![first];
        for &(w, h) in &dims[1..] {
            levels.push(self.resize(levels.last().unwrap(), w, h, parallel));
        }
        levels
    }

    pub(crate) fn reference(
        self,
        source: &impl ImageSource,
        parallel: bool,
    ) -> crate::PrecomputedReference {
        crate::PrecomputedReference {
            scales: self.pyramid(source, parallel),
            ref_width: source.width(),
            ref_height: source.height(),
            sampling: Some(self),
            sampling_geometry: Some(Geometry {
                width: source.width(),
                height: source.height(),
                axes: (0..4)
                    .map(|level| {
                        [
                            self.axis(source.width(), level),
                            self.axis(source.height(), level),
                        ]
                    })
                    .collect(),
            }),
        }
    }

    /// Compose nonnegative, normalized squared-tap ownership back to original
    /// coordinates. This is a removal-mass model, not the resizer's derivative.
    fn axis(self, logical: usize, level: usize) -> Vec<Vec<(usize, f64)>> {
        let extended = logical.max(self.min_dim());
        let mut map: Vec<Vec<(usize, f64)>> = (0..extended)
            .map(|i| vec![(crate::metric::reflect_index(i, logical), 1.0)])
            .collect();
        let dims = self.dims(logical, logical);
        for (i, &(len, _)) in dims.iter().enumerate().take(level + 1) {
            if i == 0 && self.keep_y {
                continue;
            }
            let table = F32WeightTable::new(
                map.len() as u32,
                len as u32,
                &InterpolationDetails::create(self.filter()),
            );
            let next = (0..len)
                .map(|o| {
                    let weights = table.weights(o);
                    let norm: f64 = weights.iter().map(|&w| f64::from(w).powi(2)).sum();
                    let mut sums = std::collections::BTreeMap::<usize, f64>::new();
                    for (j, &weight) in weights.iter().enumerate() {
                        if weight == 0.0 {
                            continue;
                        }
                        let a = f64::from(weight).powi(2) / norm;
                        for &(x, mass) in &map[table.left[o] as usize + j] {
                            *sums.entry(x).or_default() += a * mass;
                        }
                    }
                    sums.into_iter().collect()
                })
                .collect();
            map = next;
        }
        map
    }
}

type Axis = Vec<Vec<(usize, f64)>>;
/// Cache spatial ownership once per reference; it depends only on the model's
/// sampling contract and the logical dimensions, never on image values.
pub(crate) struct Geometry {
    width: usize,
    height: usize,
    axes: Vec<[Axis; 2]>,
}
impl Geometry {
    pub(crate) fn footprints(&self, axis: usize, level: usize) -> Vec<(usize, usize)> {
        self.axes[level][axis]
            .iter()
            .map(|a| (a.first().unwrap().0, a.last().unwrap().0))
            .collect()
    }
    pub(crate) fn project(&self, plane: &[f32], level: usize) -> Vec<f32> {
        let (width, height) = (self.width, self.height);
        let xs = &self.axes[level][0];
        let ys = &self.axes[level][1];
        assert_eq!(plane.len(), xs.len() * ys.len());
        // Separable transpose keeps work linear in axis tap counts rather
        // than expanding every 2D footprint into a Cartesian product.
        let mut tmp = vec![0.0f64; width * ys.len()];
        for y in 0..ys.len() {
            for (sx, weights) in xs.iter().enumerate() {
                let v = f64::from(plane[y * xs.len() + sx]);
                for &(x, weight) in weights {
                    tmp[y * width + x] += v * weight;
                }
            }
        }
        let mut out = vec![0.0f64; width * height];
        for (sy, weights) in ys.iter().enumerate() {
            for &(y, weight) in weights {
                for x in 0..width {
                    out[y * width + x] += tmp[sy * width + x] * weight;
                }
            }
        }
        out.into_iter().map(|v| v as f32).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "threads")]
    #[test]
    fn parallel_pyramids_match_serial_bits() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();
        let (w, h) = (257, 259);
        let pixels: Vec<_> = (0..w * h)
            .map(|i| {
                [
                    (i % 251) as u8,
                    ((i * 7) % 239) as u8,
                    ((i * 13) % 233) as u8,
                ]
            })
            .collect();
        let source = crate::RgbSlice::new(&pixels, w, h);
        for keep_y in [true, false] {
            for kernel in 0..3 {
                for (num, den) in [(3, 2), (2, 1), (3, 1)] {
                    let sampling = Sampling {
                        keep_y,
                        kernel,
                        num,
                        den,
                    };
                    let serial = sampling.pyramid(&source, false);
                    let parallel = pool.install(|| sampling.pyramid(&source, true));
                    assert_eq!(serial, parallel, "{sampling:?}");
                }
            }
        }
    }

    #[test]
    fn spatial_ownership_conserves_signed_mass_and_contains_all_taps() {
        for keep_y in [false, true] {
            for kernel in 0..3 {
                for (num, den) in [(3, 2), (2, 1), (3, 1)] {
                    let sampling = Sampling {
                        keep_y,
                        kernel,
                        num,
                        den,
                    };
                    for (width, height) in [(7, 11), (97, 131)] {
                        let geometry = Geometry {
                            width,
                            height,
                            axes: (0..4)
                                .map(|level| {
                                    [sampling.axis(width, level), sampling.axis(height, level)]
                                })
                                .collect(),
                        };
                        for level in 0..4 {
                            for axis in 0..2 {
                                let bounds = geometry.footprints(axis, level);
                                for (i, weights) in geometry.axes[level][axis].iter().enumerate() {
                                    assert!(
                                        (weights.iter().map(|p| p.1).sum::<f64>() - 1.0).abs()
                                            < 1e-12
                                    );
                                    assert!(weights.iter().all(|&(x, weight)| weight > 0.0
                                        && x >= bounds[i].0
                                        && x <= bounds[i].1));
                                }
                            }
                            let n = geometry.axes[level][0].len() * geometry.axes[level][1].len();
                            let plane: Vec<f32> = (0..n).map(|i| (i % 17) as f32 - 6.0).collect();
                            let sum = plane.iter().map(|&v| f64::from(v)).sum::<f64>();
                            let projected = geometry.project(&plane, level);
                            let total = projected.iter().map(|&v| f64::from(v)).sum::<f64>();
                            assert!(
                                (sum - total).abs() < n as f64 * 2e-6,
                                "{sampling:?} {width}x{height} level {level}: {sum} != {total}"
                            );
                        }
                    }
                }
            }
        }
    }
}
