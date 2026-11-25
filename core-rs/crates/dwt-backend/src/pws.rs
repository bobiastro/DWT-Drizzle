use crate::config::{PwsConfig, RejectionConfig};
use crate::error::{BackendError, BackendResult};
use crate::frame::Frame;

/// Per-frame quality metrics used to build PWS weights.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameMetrics {
    /// Measured background noise sigma.
    pub noise: f32,
    /// Median stellar FWHM in pixels.
    pub fwhm: f32,
    /// Optional normalized stellar flux. Missing values receive the batch median.
    pub flux: Option<f32>,
}

/// Precomputed logarithmic PWS quality terms.
#[derive(Debug, Clone)]
pub struct PwsWeights {
    config: PwsConfig,
    log_clarity: Vec<f64>,
    log_snr: Vec<f64>,
    log_flux: Vec<f64>,
    normalized_detail: Vec<f64>,
    normalized_background: Vec<f64>,
}

impl PwsWeights {
    /// Number of frames represented by these weights.
    pub fn frame_count(&self) -> usize {
        self.log_clarity.len()
    }

    /// Raw PWS weight for one frame at a scalar region score R.
    ///
    /// The weight is `C^(A R + sharp_w) * S^(B (1-R) + beta R) * T^p`.
    /// A common per-pixel normalization cancels from a weighted average and from
    /// effective frame count, so the accumulator keeps raw weights.
    pub fn weight_at(&self, frame_index: usize, region_score: f32) -> BackendResult<f64> {
        if frame_index >= self.frame_count() {
            return Err(BackendError::UnknownFrame {
                frame_index,
                frame_count: self.frame_count(),
            });
        }
        if !region_score.is_finite() {
            return Err(BackendError::InvalidValue {
                name: "region_score",
                value: region_score,
            });
        }
        let r = region_score.clamp(0.0, 1.0);
        let r = f64::from(r);
        let exponent = f64::from(self.config.clarity_exponent) * r * self.log_clarity[frame_index]
            + (f64::from(self.config.snr_exponent) * (1.0 - r)
                + f64::from(self.config.detail_snr) * r)
                * self.log_snr[frame_index]
            + f64::from(self.config.flux_power) * self.log_flux[frame_index]
            + f64::from(self.config.sharp_weight) * self.log_clarity[frame_index];
        let weight = exponent.exp();
        if weight.is_finite() && weight > 0.0 {
            Ok(weight)
        } else {
            Err(BackendError::InvalidValue {
                name: "weight",
                value: weight as f32,
            })
        }
    }

    /// Per-frame weights normalized so their mean is one.
    pub fn normalized_weights_at(&self, region_score: f32) -> BackendResult<Vec<f64>> {
        if !region_score.is_finite() {
            return Err(BackendError::InvalidValue {
                name: "region_score",
                value: region_score,
            });
        }
        if region_score <= 0.0 {
            return Ok(self.normalized_background.clone());
        }
        if region_score >= 1.0 {
            return Ok(self.normalized_detail.clone());
        }
        let mut weights = Vec::with_capacity(self.frame_count());
        for frame_index in 0..self.frame_count() {
            weights.push(self.weight_at(frame_index, region_score)?);
        }
        normalize_mean(&mut weights)?;
        Ok(weights)
    }
}

/// Builds C/S/T quality terms from per-frame metrics.
pub fn derive_pws_weights(
    metrics: &[FrameMetrics],
    config: PwsConfig,
) -> BackendResult<PwsWeights> {
    config.validate()?;
    if metrics.is_empty() {
        return Err(BackendError::EmptyFrames);
    }

    let mut noise_values = Vec::with_capacity(metrics.len());
    let mut fwhm_values = Vec::with_capacity(metrics.len());
    for (index, metric) in metrics.iter().enumerate() {
        validate_positive("noise", index, metric.noise)?;
        validate_positive("fwhm", index, metric.fwhm)?;
        noise_values.push(f64::from(metric.noise));
        fwhm_values.push(f64::from(metric.fwhm));
    }

    let median_noise = median(&mut noise_values).ok_or(BackendError::EmptyFrames)?;
    let median_fwhm = median(&mut fwhm_values).ok_or(BackendError::EmptyFrames)?;

    let raw_flux: Vec<Option<f64>> = metrics
        .iter()
        .map(|metric| {
            metric
                .flux
                .filter(|value| value.is_finite() && *value > 0.0)
                .map(f64::from)
        })
        .collect();
    let mut flux_values: Vec<f64> = raw_flux.iter().flatten().copied().collect();
    let median_flux = median(&mut flux_values).unwrap_or(1.0);

    let mut log_clarity = Vec::with_capacity(metrics.len());
    let mut log_snr = Vec::with_capacity(metrics.len());
    let mut log_flux = Vec::with_capacity(metrics.len());
    let clip_floor = 1.0 / f64::from(config.snr_clip);
    let clip_ceiling = f64::from(config.snr_clip);

    for (index, metric) in metrics.iter().enumerate() {
        let clarity = median_fwhm / f64::from(metric.fwhm);
        let mut snr = median_noise / f64::from(metric.noise);
        if config.snr_clip > 1.0 {
            snr = snr.clamp(clip_floor, clip_ceiling);
        }
        let flux = raw_flux[index].unwrap_or(median_flux);
        log_clarity.push(finite_positive("clarity", clarity)?.ln());
        log_snr.push(finite_positive("snr", snr)?.ln());
        log_flux.push(finite_positive("flux", flux / median_flux)?.ln());
    }

    let mut normalized_detail = Vec::with_capacity(metrics.len());
    let mut normalized_background = Vec::with_capacity(metrics.len());
    for index in 0..metrics.len() {
        normalized_detail.push(
            (f64::from(config.clarity_exponent) * log_clarity[index]
                + f64::from(config.detail_snr) * log_snr[index]
                + f64::from(config.flux_power) * log_flux[index]
                + f64::from(config.sharp_weight) * log_clarity[index])
                .exp(),
        );
        normalized_background.push(
            (f64::from(config.snr_exponent) * log_snr[index]
                + f64::from(config.flux_power) * log_flux[index]
                + f64::from(config.sharp_weight) * log_clarity[index])
                .exp(),
        );
    }
    normalize_mean(&mut normalized_detail)?;
    normalize_mean(&mut normalized_background)?;

    Ok(PwsWeights {
        config,
        log_clarity,
        log_snr,
        log_flux,
        normalized_detail,
        normalized_background,
    })
}

/// Result of one aligned PWS stack.
#[derive(Debug, Clone)]
pub struct StackResult {
    pub image: Frame,
    pub neff: Vec<f32>,
    pub coverage: Vec<u32>,
    pub rejected_fraction: f32,
}

/// Stacks already-aligned frames with PWS weights and robust MAD rejection.
///
/// Rejection follows the source algorithm: a per-frame P99 detail allowance is
/// measured on R > 0.5 pixels, applied over a MAD background threshold, and
/// guarded by an expected-false-rejection-aware maximum rejected fraction.
pub fn stack_aligned(
    frames: &[Frame],
    region_scores: &[f32],
    weights: &PwsWeights,
    rejection: RejectionConfig,
) -> BackendResult<StackResult> {
    rejection.validate()?;
    let first = frames.first().ok_or(BackendError::EmptyFrames)?;
    let shape = first.shape();
    validate_frame_set(frames, shape)?;
    if weights.frame_count() != frames.len() {
        return Err(BackendError::BufferLengthMismatch {
            name: "frame weights",
            expected: frames.len(),
            actual: weights.frame_count(),
        });
    }
    if region_scores.len() != shape.pixel_count() {
        return Err(BackendError::BufferLengthMismatch {
            name: "region_scores",
            expected: shape.pixel_count(),
            actual: region_scores.len(),
        });
    }

    let mut output = vec![0.0_f32; shape.pixel_count()];
    let mut neff = vec![0.0_f32; shape.pixel_count()];
    let mut coverage = vec![0_u32; shape.pixel_count()];
    let mut rejected_total = 0_u64;
    let mut sample_total = 0_u64;
    let mut values = Vec::with_capacity(frames.len());
    let mut deviations = Vec::with_capacity(frames.len());
    let mut keep = vec![true; frames.len()];
    let mut raw_weights = vec![0.0_f64; frames.len()];

    let mut detail_allowance = if rejection.sigma_multiplier > 0.0 {
        estimate_detail_allowance(
            frames,
            region_scores,
            rejection.sky,
            &mut values,
            &mut deviations,
        )?
    } else {
        vec![0.0_f32; frames.len()]
    };
    if detail_allowance.is_empty() {
        detail_allowance = vec![0.0; frames.len()];
    }

    for pixel_index in 0..shape.pixel_count() {
        let region_score = region_scores[pixel_index];
        if !region_score.is_finite() {
            return Err(BackendError::InvalidValue {
                name: "region_score",
                value: region_score,
            });
        }
        for (frame_index, raw_weight) in raw_weights.iter_mut().enumerate() {
            *raw_weight = weights.weight_at(frame_index, region_score)?;
        }

        values.clear();
        deviations.clear();
        keep.fill(true);
        for frame in frames {
            let value = frame.pixels()[pixel_index];
            if value.is_finite() {
                values.push(value);
            }
        }
        sample_total += values.len() as u64;
        if values.is_empty() {
            continue;
        }

        let anchor = median_f32(&values).ok_or(BackendError::EmptyPixels)?;
        let signal = (anchor - rejection.sky).max(0.0);
        if rejection.sigma_multiplier > 0.0 && values.len() >= 3 {
            for value in &values {
                deviations.push((*value - anchor).abs());
            }
            let mad = median_f32(&deviations).ok_or(BackendError::EmptyPixels)?;
            let mut rejected = 0_usize;
            for (frame_index, frame) in frames.iter().enumerate() {
                let value = frame.pixels()[pixel_index];
                let threshold = rejection.sigma_multiplier * 1.4826 * mad
                    + detail_allowance[frame_index] * region_score.max(0.0) * signal;
                if value.is_finite() && (value - anchor).abs() > threshold {
                    keep[frame_index] = false;
                    rejected += 1;
                }
            }
            let expected_false = frames.len() as f64
                * f64::from(erf_complement(
                    rejection.sigma_multiplier / std::f32::consts::SQRT_2,
                ));
            let gate = (rejection.max_fraction * frames.len() as f32).max(1.0);
            let excess = (rejected as f64 - expected_false).max(0.0);
            if excess > f64::from(gate) {
                keep.fill(true);
                rejected = 0;
            }
            rejected_total += rejected as u64;
        }

        let mut numerator = 0.0_f64;
        let mut denominator = 0.0_f64;
        let mut denominator_sq = 0.0_f64;
        let mut used = 0_u32;
        for (frame_index, frame) in frames.iter().enumerate() {
            if !keep[frame_index] {
                continue;
            }
            let value = frame.pixels()[pixel_index];
            if !value.is_finite() {
                continue;
            }
            let weight = raw_weights[frame_index];
            numerator += f64::from(value) * weight;
            denominator += weight;
            denominator_sq += weight * weight;
            used += 1;
        }

        if denominator > 0.0 {
            output[pixel_index] = (numerator / denominator) as f32;
            neff[pixel_index] = (denominator * denominator / denominator_sq) as f32;
            coverage[pixel_index] = used;
        }
    }

    let rejected_fraction = if sample_total == 0 {
        0.0
    } else {
        (rejected_total as f64 / sample_total as f64) as f32
    };
    Ok(StackResult {
        image: Frame::new(shape.width, shape.height, output)?,
        neff,
        coverage,
        rejected_fraction,
    })
}

fn estimate_detail_allowance(
    frames: &[Frame],
    region_scores: &[f32],
    sky: f32,
    values: &mut Vec<f32>,
    deviations: &mut Vec<f32>,
) -> BackendResult<Vec<f32>> {
    frames.first().ok_or(BackendError::EmptyFrames)?;
    let mut ratios_by_frame: Vec<Vec<f32>> = vec![Vec::new(); frames.len()];
    for (pixel_index, region_score) in region_scores.iter().copied().enumerate() {
        if region_score <= 0.5 {
            continue;
        }
        values.clear();
        for frame in frames {
            let value = frame.pixels()[pixel_index];
            if value.is_finite() {
                values.push(value);
            }
        }
        if values.len() < 3 {
            continue;
        }
        let anchor = median_f32(values).ok_or(BackendError::EmptyPixels)?;
        let signal = (anchor - sky).max(0.0);
        if signal <= 0.0 {
            continue;
        }
        for (frame_index, frame) in frames.iter().enumerate() {
            let value = frame.pixels()[pixel_index];
            if value.is_finite() {
                ratios_by_frame[frame_index].push((value - anchor).abs() / signal);
            }
        }
    }
    deviations.clear();
    let mut per_frame = Vec::with_capacity(frames.len());
    for values in &mut ratios_by_frame {
        if values.is_empty() {
            continue;
        }
        values.sort_by(f32::total_cmp);
        let rank = (values.len() - 1) as f32 * 0.99;
        let lower = rank.floor() as usize;
        let upper = rank.ceil() as usize;
        let fraction = rank - lower as f32;
        let low = values.get(lower).copied().unwrap_or(0.0);
        let high = values.get(upper).copied().unwrap_or(low);
        let delta = high - low;
        per_frame.push(low + delta * fraction);
    }
    let allowance = median_f32(&per_frame).unwrap_or(0.0);
    Ok(vec![allowance; frames.len()])
}

pub(crate) fn erf_complement(value: f32) -> f32 {
    // Numerical Recipes approximation; sufficient for the expected false-rejection count.
    let z = value.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let ans = t
        * (-z * z - 1.265_512_2
            + t * (1.000_023_7
                + t * (0.374_091_96
                    + t * (0.096_784_18
                        + t * (-0.186_288_06
                            + t * (0.278_868_07
                                + t * (-1.135_204
                                    + t * (1.488_515_9
                                        + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
            .exp();
    if value >= 0.0 { ans } else { 2.0 - ans }
}

fn validate_positive(name: &'static str, frame_index: usize, value: f32) -> BackendResult<()> {
    if value.is_finite() && value > 0.0 {
        return Ok(());
    }
    let _ = frame_index;
    Err(BackendError::InvalidValue { name, value })
}

fn finite_positive(name: &'static str, value: f64) -> BackendResult<f64> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(BackendError::InvalidValue {
            name,
            value: value as f32,
        })
    }
}

fn normalize_mean(values: &mut [f64]) -> BackendResult<()> {
    if values.is_empty() {
        return Err(BackendError::EmptyFrames);
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    if !mean.is_finite() || mean <= 0.0 {
        return Err(BackendError::InvalidValue {
            name: "weight mean",
            value: mean as f32,
        });
    }
    for value in values {
        *value /= mean;
    }
    Ok(())
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 1 {
        values.get(middle).copied()
    } else {
        let left = values.get(middle - 1)?;
        let right = values.get(middle)?;
        Some((left + right) * 0.5)
    }
}

fn median_f32(values: &[f32]) -> Option<f32> {
    select_median(&mut values.to_vec())
}

pub(crate) fn select_median(values: &mut [f32]) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    let even = values.len() % 2 == 0;
    let (lower, middle, _) = values.select_nth_unstable_by(values.len() / 2, f32::total_cmp);
    if even {
        let previous = lower.iter().copied().max_by(f32::total_cmp)?;
        return Some(previous.midpoint(*middle));
    }
    Some(*middle)
}

/// Robust per-pixel noise from vertical differences on a stride-two sample grid.
pub(crate) fn estimate_noise(frame: &Frame) -> Option<f32> {
    let (width, height) = (frame.width(), frame.height());
    let mut differences = Vec::with_capacity((height.saturating_sub(1) / 2) * width.div_ceil(2));
    for y in (2..height).step_by(2) {
        let current = &frame.pixels()[y * width..(y + 1) * width];
        let previous = &frame.pixels()[(y - 2) * width..(y - 1) * width];
        differences.extend(
            current
                .iter()
                .zip(previous)
                .step_by(2)
                .map(|(value, prior)| value - prior)
                .filter(|value| value.is_finite()),
        );
    }
    let normalizer = if differences.is_empty() {
        differences.extend(
            frame
                .pixels()
                .iter()
                .copied()
                .filter(|value| value.is_finite()),
        );
        1.0
    } else {
        std::f32::consts::SQRT_2
    };
    let center = select_median(&mut differences)?;
    for value in &mut differences {
        *value = (*value - center).abs();
    }
    select_median(&mut differences).map(|mad| (mad / normalizer) * 1.4826)
}

pub(crate) fn validate_frame_set(
    frames: &[Frame],
    shape: crate::frame::ImageShape,
) -> BackendResult<()> {
    for (frame_index, frame) in frames.iter().enumerate() {
        if frame.shape() != shape {
            return Err(BackendError::FrameShapeMismatch {
                frame_index,
                expected_width: shape.width,
                expected_height: shape.height,
                actual_width: frame.width(),
                actual_height: frame.height(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod noise_tests {
    use super::*;

    #[test]
    fn noise_matches_upstream_on_gradient() -> BackendResult<()> {
        let pixels = [
            1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 2.0, 8.0, 4.0, 5.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 2.0, 2.0, 10.0, 4.0, 8.0,
        ];
        for slope in [0.0, 10.0] {
            let frame = Frame::new(
                5,
                5,
                pixels
                    .iter()
                    .enumerate()
                    .map(|(i, value)| value + (i / 5) as f32 * slope)
                    .collect(),
            )?;
            let result = estimate_noise(&frame).ok_or(BackendError::EmptyPixels)?;
            assert!((result - 1.572_534_7).abs() < 1.0e-6);
        }
        Ok(())
    }
}
