//! Per-pixel PWS region field.
//!
//! Luminance-normalized and band-pass envelopes for the integration grid.

use crate::error::{BackendError, BackendResult};
use crate::frame::Frame;
use crate::pws::select_median;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RFieldMode {
    #[default]
    Max,
    Band,
    Ctn,
}

/// Parameters for the region-field construction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RFieldConfig {
    pub sigma_psf: f32,
    pub window_multiplier: f32,
    pub noise_epsilon: f32,
    pub shape_power: f32,
    pub floor: f32,
}

impl Default for RFieldConfig {
    fn default() -> Self {
        Self {
            sigma_psf: 3.0,
            window_multiplier: 2.0,
            noise_epsilon: 16.0,
            shape_power: 1.0,
            floor: 0.0,
        }
    }
}

impl RFieldConfig {
    fn validate(self) -> BackendResult<()> {
        for (name, value) in [
            ("sigma_psf", self.sigma_psf),
            ("window_multiplier", self.window_multiplier),
            ("noise_epsilon", self.noise_epsilon),
            ("shape_power", self.shape_power),
            ("floor", self.floor),
        ] {
            if !value.is_finite() {
                return Err(BackendError::InvalidValue { name, value });
            }
        }
        if self.sigma_psf <= 0.0 {
            return Err(BackendError::InvalidValue {
                name: "sigma_psf",
                value: self.sigma_psf,
            });
        }
        if self.window_multiplier <= 0.0 {
            return Err(BackendError::InvalidValue {
                name: "window_multiplier",
                value: self.window_multiplier,
            });
        }
        if self.noise_epsilon < 0.0 || self.shape_power <= 0.0 {
            return Err(BackendError::Message(
                "R field requires nonnegative noise epsilon and positive shape power".into(),
            ));
        }
        if !(0.0..=0.95).contains(&self.floor) {
            return Err(BackendError::InvalidValue {
                name: "floor",
                value: self.floor,
            });
        }
        Ok(())
    }
}

/// Builds a region field from a reference image.
pub fn build_r_field(reference: &Frame, config: RFieldConfig) -> BackendResult<Frame> {
    build_r_field_with_mode(reference, config, RFieldMode::Max, 0.0, 3.0)
}

/// Builds on the half-resolution grid used by the Python Drizzle pipeline.
pub(crate) fn build_drizzle_r_field(
    reference: &Frame,
    mut config: RFieldConfig,
    mode: RFieldMode,
    band_window: f32,
    contrast: f32,
    blur: f32,
    growth_in_psf: f32,
) -> BackendResult<Frame> {
    let (width, height) = (reference.width(), reference.height());
    if width <= 64 || height <= 64 {
        return build_field(
            reference,
            config,
            mode,
            band_window,
            contrast,
            blur,
            growth_in_psf,
        );
    }
    let sampled = Frame::new(
        width.div_ceil(2),
        height.div_ceil(2),
        reference
            .pixels()
            .chunks_exact(width)
            .step_by(2)
            .flat_map(|row| row.iter().step_by(2).copied())
            .collect(),
    )?;
    if !sampled.pixels().iter().any(|value| value.is_finite()) {
        return build_field(
            reference,
            config,
            mode,
            band_window,
            contrast,
            blur,
            growth_in_psf,
        );
    }
    config.sigma_psf *= 0.5;
    let field = build_field(
        &sampled,
        config,
        mode,
        band_window,
        contrast,
        blur,
        growth_in_psf,
    )?;
    expand_sampled_field(&field, reference)
}

fn expand_sampled_field(field: &Frame, reference: &Frame) -> BackendResult<Frame> {
    // scipy.ndimage.zoom(order=1) aligns the endpoints of the original and
    // doubled grids. With an odd source extent the last output pixel is cropped.
    let coordinate = |size: usize, index: usize| {
        let source = index as f64 * (size - 1) as f64 / (2 * size - 1) as f64;
        let low = source as usize;
        (low, (low + 1).min(size - 1), source.fract())
    };
    let x_positions: Vec<_> = (0..reference.width())
        .map(|x| coordinate(field.width(), x))
        .collect();
    let mut expanded = Vec::with_capacity(reference.shape().pixel_count());
    let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
    for (y, row) in reference
        .pixels()
        .chunks_exact(reference.width())
        .enumerate()
    {
        let (top, bottom, dy) = coordinate(field.height(), y);
        let top = &field.pixels()[top * field.width()..][..field.width()];
        let bottom = &field.pixels()[bottom * field.width()..][..field.width()];
        for (&source, &(left, right, dx)) in row.iter().zip(&x_positions) {
            let upper = lerp(f64::from(top[left]), f64::from(top[right]), dx);
            let lower = lerp(f64::from(bottom[left]), f64::from(bottom[right]), dx);
            expanded.push(if source.is_finite() {
                lerp(upper, lower, dy) as f32
            } else {
                0.0
            });
        }
    }
    Frame::new(reference.width(), reference.height(), expanded)
}

pub(crate) fn build_r_field_with_mode(
    reference: &Frame,
    config: RFieldConfig,
    mode: RFieldMode,
    band_window: f32,
    contrast: f32,
) -> BackendResult<Frame> {
    build_field(reference, config, mode, band_window, contrast, 0.0, 0.0)
}

fn build_field(
    reference: &Frame,
    config: RFieldConfig,
    mode: RFieldMode,
    band_window: f32,
    contrast: f32,
    blur: f32,
    growth_in_psf: f32,
) -> BackendResult<Frame> {
    config.validate()?;
    let width = reference.width();
    let height = reference.height();
    let pixels = reference.pixels();
    let (sky, sigma) = sky_and_noise(pixels)?;
    let mut signal: Vec<_> = pixels
        .iter()
        .map(|&value| {
            if value.is_finite() {
                (value - sky).max(0.0)
            } else {
                0.0
            }
        })
        .collect();
    // The Python smoothing branch blurs the positive signal before taking the local peak.
    let mode = if blur > 0.0 {
        signal = gaussian_blur(&signal, width, height, f64::from(blur));
        RFieldMode::Max
    } else {
        mode
    };
    let radius = (config.window_multiplier * config.sigma_psf.max(0.5))
        .round_ties_even()
        .max(1.0) as usize;
    let band_width = if band_window > 0.0 {
        band_window
    } else {
        (4.0 * config.sigma_psf.max(0.5)).max(9.0)
    };
    let band_radius = ((band_width.round_ties_even() as usize).max(3) | 1) / 2;
    if mode != RFieldMode::Max {
        let smooth = box_mean(&signal, width, height, band_radius);
        for (value, local) in signal.iter_mut().zip(smooth) {
            *value -= local;
        }
    }
    let grow_pixels = f64::from(growth_in_psf) * f64::from(config.sigma_psf.max(0.5));
    let (peaks, peak_noise) = if mode == RFieldMode::Ctn {
        let (_, noise) = sky_and_noise(&signal)?;
        let peaks = if grow_pixels > 0.0 {
            signal
                .iter()
                .map(|value| f64::from(value.max(0.0)))
                .collect()
        } else {
            Vec::new()
        };
        let low = 0.8 * noise;
        let scale = ((contrast.max(1.05) - 0.8) * noise).max(1.0e-30);
        for value in &mut signal {
            *value = (*value - low) / scale;
        }
        (peaks, noise)
    } else {
        let radius = if mode == RFieldMode::Band {
            band_radius
        } else {
            radius
        };
        let peaks = local_maximum(&signal, width, height, radius);
        let epsilon = f64::from(config.noise_epsilon * sigma);
        for (value, &peak) in signal.iter_mut().zip(&peaks) {
            let denominator = peak + epsilon;
            *value = if denominator > 0.0 {
                (f64::from(*value) / denominator) as f32
            } else {
                0.0
            };
        }
        (peaks, sigma)
    };
    for value in &mut signal {
        *value = value.clamp(0.0, 1.0).powf(config.shape_power);
    }
    if grow_pixels > 0.0 {
        signal = grow_support(&signal, &peaks, peak_noise, width, height, grow_pixels);
    }
    let output = pixels
        .iter()
        .zip(signal)
        .map(|(source, value)| {
            if source.is_finite() {
                config.floor + (1.0 - config.floor) * value
            } else {
                0.0
            }
        })
        .collect();
    Frame::new(width, height, output)
}

fn grow_support(
    field: &[f32],
    peaks: &[f64],
    noise: f32,
    width: usize,
    height: usize,
    grow_pixels: f64,
) -> Vec<f32> {
    let low = 20.0_f64.log10();
    let span = 1000.0_f64.log10() - low;
    // Python mixes four growth intervals with a 50% minimum radius.
    let levels: Vec<_> = peaks
        .iter()
        .map(|&peak| {
            let brightness = (peak.max(1e-30) / f64::from(noise)).log10();
            2.0 * (1.0 + ((brightness - low) / span).clamp(0.0, 1.0))
        })
        .collect();
    let mut output = vec![0.0; field.len()];
    let mut blurred = Vec::new();
    let mut previous_radius = 0;
    for level in 2..=4 {
        let radius = (grow_pixels * f64::from(level) / 4.0).round_ties_even() as usize;
        if radius != 0 && radius != previous_radius {
            let peaks = local_maximum(field, width, height, radius);
            blurred = gaussian_blur(&peaks, width, height, (radius as f64 * 0.5).max(0.8));
            previous_radius = radius;
        }
        let expanded = if radius == 0 { field } else { &blurred };
        for ((out, &value), &mix) in output.iter_mut().zip(expanded).zip(&levels) {
            let weight = (1.0 - (mix - f64::from(level)).abs()).max(0.0) as f32;
            *out += value * weight;
        }
    }
    output
}

fn gaussian_blur<T: Copy + Into<f64>>(
    pixels: &[T],
    width: usize,
    height: usize,
    sigma: f64,
) -> Vec<f32> {
    // scipy.ndimage.gaussian_filter uses a four-sigma cutoff and nearest edges.
    let radius = (4.0 * sigma + 0.5) as isize;
    let mut kernel: Vec<_> = (-radius..=radius)
        .map(|offset| (-0.5 * (offset as f64 / sigma).powi(2)).exp())
        .collect();
    let total = kernel.iter().sum::<f64>();
    for weight in &mut kernel {
        *weight /= total;
    }
    let horizontal: Vec<_> = pixels
        .chunks_exact(width)
        .flat_map(|row| {
            let kernel = &kernel;
            (0..width).map(move |x| {
                (-radius..=radius)
                    .zip(kernel)
                    .map(|(offset, weight)| {
                        row[x.saturating_add_signed(offset).min(width - 1)].into() * weight
                    })
                    .sum::<f64>()
            })
        })
        .collect();
    let mut output = vec![0.0_f64; pixels.len()];
    for (y, row) in output.chunks_exact_mut(width).enumerate() {
        for (offset, &weight) in (-radius..=radius).zip(&kernel) {
            let source = y.saturating_add_signed(offset).min(height - 1) * width;
            for (value, &sample) in row.iter_mut().zip(&horizontal[source..][..width]) {
                *value += sample * weight;
            }
        }
    }
    output.into_iter().map(|value| value as f32).collect()
}

/// Median major-axis second moment in the Python trail measurement aperture.
pub(crate) fn measure_trail_variance(
    frame: &Frame,
    positions: impl IntoIterator<Item = (usize, usize)>,
    sigma_psf: f32,
) -> Option<f32> {
    let aperture = 3.7 * f64::from(sigma_psf.max(0.5));
    let half = (aperture.ceil() as usize).saturating_add(3);
    if half > (frame.width().min(frame.height()) - 1) / 2 {
        return None;
    }
    let inner = aperture.powi(2);
    let outer = aperture.max((half - 2) as f64).powi(2);
    let mut edge = Vec::new();
    let mut moments = Vec::new();
    for (x, y) in positions {
        if x < half || y < half || x >= frame.width() - half || y >= frame.height() - half {
            continue;
        }
        let taps = || {
            (y - half..=y + half).flat_map(|row| {
                (x - half..=x + half).map(move |column| {
                    let dx = column as f64 - x as f64;
                    let dy = row as f64 - y as f64;
                    let value = frame.pixels()[row * frame.width() + column];
                    (dx, dy, dx * dx + dy * dy, value)
                })
            })
        };
        edge.clear();
        edge.extend(taps().filter_map(|(_, _, distance, value)| {
            (distance >= outer && value.is_finite()).then_some(value)
        }));
        let Some(sky) = select_median(&mut edge) else {
            continue;
        };
        let (mut total, mut mx, mut my, mut mxy) = (0.0, 0.0, 0.0, 0.0);
        for (dx, dy, _, value) in taps().filter(|tap| tap.2 <= inner && tap.3.is_finite()) {
            let signal = (f64::from(value) - f64::from(sky)).max(0.0);
            total += signal;
            mx += signal * dx * dx;
            my += signal * dy * dy;
            mxy += signal * dx * dy;
        }
        if total > 0.0 {
            let major = ((mx + my) + (mx - my).hypot(2.0 * mxy)) / (2.0 * total);
            moments.push(major as f32);
        }
    }
    select_median(&mut moments)
}

pub(crate) fn trail_growth_factor(
    variances: &[Option<f32>],
    sharp_indices: impl IntoIterator<Item = usize>,
    sigma_psf: f32,
) -> f32 {
    let mut sharp: Vec<_> = sharp_indices
        .into_iter()
        .filter_map(|index| variances[index])
        .collect();
    let Some(reference) = select_median(&mut sharp) else {
        return 0.0;
    };
    let mut lengths: Vec<_> = variances
        .iter()
        .flatten()
        .map(|value| (12.0 * (value - reference).max(0.0)).sqrt())
        .collect();
    lengths.sort_unstable_by(f32::total_cmp);
    let rank = (lengths.len() - 1) as f32 * 0.9;
    let low = lengths[rank.floor() as usize];
    let high = lengths[rank.ceil() as usize];
    (low + (high - low) * rank.fract()) / sigma_psf.max(0.5)
}

fn sky_and_noise(pixels: &[f32]) -> BackendResult<(f32, f32)> {
    let mut finite: Vec<_> = pixels
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    let sky = select_median(&mut finite).ok_or(BackendError::EmptyPixels)?;
    for value in &mut finite {
        *value = (*value - sky).abs();
    }
    let noise = select_median(&mut finite).unwrap_or(0.0);
    Ok((sky, (1.4826 * noise).max(1.0e-30)))
}

fn box_mean(pixels: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
    let mean = |line: &[f32]| {
        let n = line.len();
        let mut sums = Vec::with_capacity(n + 1);
        sums.push(0.0_f64);
        let mut running = 0.0;
        for &value in line {
            running += f64::from(value);
            sums.push(running);
        }
        let radius = radius as f64;
        (0..n)
            .map(|i| {
                let start = i.saturating_sub(radius as usize);
                let end = i.saturating_add(radius as usize).saturating_add(1).min(n);
                let left = (radius - i as f64).max(0.0) * f64::from(line[0]);
                let right = (radius - (n - 1 - i) as f64).max(0.0) * f64::from(line[n - 1]);
                ((sums[end] - sums[start] + left + right) / (2.0 * radius + 1.0)) as f32
            })
            .collect::<Vec<_>>()
    };
    let horizontal: Vec<f32> = pixels.chunks_exact(width).flat_map(mean).collect();
    let mut output = vec![0.0; pixels.len()];
    let mut column = vec![0.0; height];
    for x in 0..width {
        for y in 0..height {
            column[y] = horizontal[y * width + x];
        }
        for (y, value) in mean(&column).into_iter().enumerate() {
            output[y * width + x] = value;
        }
    }
    output
}

fn local_maximum(pixels: &[f32], width: usize, height: usize, radius: usize) -> Vec<f64> {
    let horizontal: Vec<f64> = pixels
        .chunks_exact(width)
        .flat_map(|row| sliding_window_maximum(row, radius))
        .collect();
    let mut vertical = vec![0.0_f64; pixels.len()];
    let mut column = vec![0.0_f64; height];
    for x in 0..width {
        for y in 0..height {
            column[y] = horizontal[y * width + x];
        }
        for (y, value) in sliding_window_maximum(&column, radius)
            .into_iter()
            .enumerate()
        {
            vertical[y * width + x] = value;
        }
    }
    vertical
}

fn sliding_window_maximum<T>(values: &[T], half_window: usize) -> Vec<f64>
where
    T: Copy + Into<f64>,
{
    let length = values.len();
    let mut output = vec![0.0_f64; length];
    if length == 0 {
        return output;
    }
    let half_window = half_window.min(length - 1);
    let mut deque = std::collections::VecDeque::new();
    for index in 0..length + half_window {
        if index < length {
            let value = to_f64(values[index]).max(0.0);
            while deque
                .back()
                .is_some_and(|&back| to_f64(values[back]).max(0.0) <= value)
            {
                deque.pop_back();
            }
            deque.push_back(index);
        }
        if index < half_window {
            continue;
        }
        let center = index - half_window;
        let left = center.saturating_sub(half_window);
        while deque.front().is_some_and(|&front| front < left) {
            deque.pop_front();
        }
        output[center] = deque
            .front()
            .and_then(|&front| values.get(front))
            .copied()
            .map(|value| to_f64(value).max(0.0))
            .unwrap_or(0.0);
    }
    output
}

fn to_f64<T>(value: T) -> f64
where
    T: Copy + Into<f64>,
{
    value.into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_reference_builds_zero_region_field() {
        let reference = Frame::filled(8, 8, 10.0).expect("reference");
        let field = build_r_field(&reference, RFieldConfig::default()).expect("field");
        assert!(field.pixels().iter().all(|value| *value == 0.0));
    }

    #[test]
    fn bright_center_builds_positive_region_field() {
        let mut pixels = vec![0.0_f32; 9 * 9];
        for y in 3..6 {
            for x in 3..6 {
                pixels[y * 9 + x] = 1.0;
            }
        }
        let reference = Frame::new(9, 9, pixels).expect("reference");
        let field = build_r_field(
            &reference,
            RFieldConfig {
                sigma_psf: 1.0,
                noise_epsilon: 0.1,
                ..RFieldConfig::default()
            },
        )
        .expect("field");
        assert!(field.pixel(4, 4).expect("center") > 0.0);
    }

    #[test]
    fn all_envelopes_match_python_reference_samples() -> BackendResult<()> {
        let pixels = (0..9)
            .flat_map(|y| {
                (0..11).map(move |x| {
                    let background = 0.4 + f64::from((x * 17 + y * 29) % 13) * 0.002;
                    let distance = (x - 5) * (x - 5) + (y - 4) * (y - 4);
                    (background + 0.09 * (-f64::from(distance) / 2.0).exp()) as f32
                })
            })
            .collect();
        let reference = Frame::new(11, 9, pixels)?;
        let config = RFieldConfig {
            sigma_psf: 1.0,
            noise_epsilon: 3.0,
            ..RFieldConfig::default()
        };
        for (mode, expected) in [
            (
                RFieldMode::Max,
                [0.01019456, 0.37813818, 0.715247, 0.15445338],
            ),
            (RFieldMode::Band, [0.0, 0.262639, 0.6577093, 0.0]),
            (RFieldMode::Ctn, [0.0, 1.0, 1.0, 0.0]),
        ] {
            let field = build_r_field_with_mode(&reference, config, mode, 5.0, 3.0)?;
            for ((x, y), value) in [(6, 2), (5, 3), (5, 4), (6, 5)].into_iter().zip(expected) {
                let actual = field.pixels()[y * 11 + x];
                assert!(
                    (actual - value).abs() < 5.0e-5,
                    "{mode:?} at {x},{y}: {actual} != {value}"
                );
            }
        }
        Ok(())
    }

    fn sampled_reference() -> BackendResult<Frame> {
        let pixels = (0..69)
            .flat_map(|y| {
                (0..68).map(move |x| {
                    let background = 0.4 + f64::from((x * 17 + y * 29) % 13) * 0.002;
                    let star = |dx: i32, dy: i32, scale: f64| {
                        (-f64::from(dx * dx + dy * dy) / scale).exp()
                    };
                    (background
                        + 0.41 * star(x - 31, y - 29, 5.0)
                        + 0.12 * star(x - 46, y - 37, 9.0)) as f32
                })
            })
            .collect();
        Frame::new(68, 69, pixels)
    }

    #[test]
    fn sampled_field_matches_python_on_an_odd_extent() -> BackendResult<()> {
        let reference = sampled_reference()?;
        let config = RFieldConfig {
            sigma_psf: 1.6,
            floor: 0.08,
            ..RFieldConfig::default()
        };
        for (mode, expected) in [
            (RFieldMode::Max, [0.08587487, 0.67747337, 0.3220492]),
            (RFieldMode::Band, [0.08, 0.619097, 0.23511066]),
            (RFieldMode::Ctn, [0.08, 1.0, 0.76886064]),
        ] {
            let field = build_drizzle_r_field(&reference, config, mode, 5.0, 3.0, 0.0, 0.0)?;
            assert_eq!(field.shape(), reference.shape());
            for ((x, y), value) in [(67, 68), (31, 29), (45, 36)].into_iter().zip(expected) {
                let actual = field.pixel(x, y).unwrap();
                assert!(
                    (actual - value).abs() < 5e-5,
                    "{mode:?} at {x},{y}: {actual} != {value}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn adaptive_growth_matches_python_field() -> BackendResult<()> {
        let reference = sampled_reference()?;
        let config = RFieldConfig {
            sigma_psf: 1.6,
            floor: 0.08,
            ..RFieldConfig::default()
        };
        for (mode, expected) in [
            (RFieldMode::Max, [0.15112758, 0.66814786, 0.43168312]),
            (RFieldMode::Band, [0.13678566, 0.6098097, 0.3554415]),
            (RFieldMode::Ctn, [0.9922362, 0.9783836, 0.9455689]),
        ] {
            let field = build_drizzle_r_field(&reference, config, mode, 5.0, 3.0, 0.0, 2.07)?;
            for ((x, y), value) in [(67, 68), (31, 29), (45, 36)].into_iter().zip(expected) {
                let actual = field.pixel(x, y).unwrap();
                assert!(
                    (actual - value).abs() < 5e-5,
                    "{mode:?} at {x},{y}: {actual} != {value}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn trail_measurement_matches_python_aperture() -> BackendResult<()> {
        let reference = sampled_reference()?;
        let value = measure_trail_variance(&reference, [(31, 29), (46, 37)], 1.6).unwrap();
        assert!((value - 3.580_858_5).abs() < 5e-5, "{value}");
        Ok(())
    }

    #[test]
    fn sliding_window_matches_direct_scan() {
        let values = [0.0_f32, 3.0, 1.0, 4.0, 2.0, 5.0, 0.5, 2.5, 1.5];
        for half_window in 1_usize..4 {
            let actual = sliding_window_maximum(&values, half_window);
            let expected: Vec<f64> = (0..values.len())
                .map(|index| {
                    let start = index.saturating_sub(half_window);
                    let end = (index + half_window).min(values.len() - 1);
                    values[start..=end]
                        .iter()
                        .copied()
                        .map(f64::from)
                        .fold(0.0_f64, f64::max)
                })
                .collect();
            assert_eq!(actual, expected, "half_window={half_window}");
        }
    }
}
