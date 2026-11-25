use crate::frame::Frame;
use crate::geometry::SimilarityTransform;
use crate::pws::select_median;
use crate::register::{Star, match_stars};

const MIN_MATCHED_STARS: usize = 20;
const MAX_TREND_FRACTION: f32 = 0.70;
const MAX_SIDE_FRACTION: f32 = 0.85;

pub(super) struct Throughput {
    pub index: usize,
    pub timestamp: i64,
    pub exposure: f32,
    pub flux: f32,
}

/// Stellar aperture flux with a local median sky from the surrounding annulus.
pub(super) fn aperture_flux(frame: &Frame, star: &Star) -> Option<f32> {
    const OUTER: usize = 10;
    const CORE_SQUARED: usize = 36;
    const SKY_INNER_SQUARED: usize = 64;
    let (x, y) = (star.x as usize, star.y as usize);
    if x < OUTER || y < OUTER || x + OUTER >= frame.width() || y + OUTER >= frame.height() {
        return None;
    }
    let mut sky_samples = Vec::new();
    let mut core_sum = 0.0;
    let mut core_count = 0;
    for row in y - OUTER..=y + OUTER {
        for column in x - OUTER..=x + OUTER {
            let radius = row.abs_diff(y).pow(2) + column.abs_diff(x).pow(2);
            let pixel = frame.pixels()[row * frame.width() + column];
            if radius <= CORE_SQUARED {
                if !pixel.is_finite() || pixel >= 0.9 {
                    return None;
                }
                core_sum += pixel;
                core_count += 1;
            } else if (SKY_INNER_SQUARED..=OUTER * OUTER).contains(&radius) && pixel.is_finite() {
                sky_samples.push(pixel);
            }
        }
    }
    let sky = select_median(&mut sky_samples)?;
    let flux = core_sum - sky * core_count as f32;
    (flux.is_finite() && flux > 0.0).then_some(flux)
}

pub(super) fn matched_throughput(
    frame: &Frame,
    stars: &[Star],
    reference: &[Star],
    reference_fluxes: &[Option<f32>],
    transform: SimilarityTransform,
    radius: f64,
) -> Option<f32> {
    let mut ratios: Vec<_> = match_stars(reference, stars, transform, radius)
        .into_iter()
        .filter_map(|(source, target)| {
            let ratio = aperture_flux(frame, &stars[source])? / reference_fluxes[target]?;
            (ratio.is_finite() && ratio > 0.0).then_some(ratio)
        })
        .collect();
    if ratios.len() < MIN_MATCHED_STARS {
        return None;
    }
    select_median(&mut ratios)
}

/// Reject isolated throughput dips with two nearby measured exposures on each side.
pub(super) fn throughput_outliers(mut frames: Vec<Throughput>) -> Vec<(usize, f32)> {
    frames.sort_by_key(|frame| frame.timestamp);
    frames
        .windows(5)
        .filter_map(|window| {
            let [left, prior, frame, next, right] = window else {
                return None;
            };
            let max_span_ms = (4.0 * frame.exposure).clamp(60.0, 3600.0) * 1000.0;
            if !window
                .windows(2)
                .all(|pair| pair[0].timestamp < pair[1].timestamp)
                || frame.timestamp.saturating_sub(left.timestamp) as f32 > max_span_ms
                || right.timestamp.saturating_sub(frame.timestamp) as f32 > max_span_ms
            {
                return None;
            }
            let before = (left.flux.ln() + prior.flux.ln()) * 0.5;
            let after = (next.flux.ln() + right.flux.ln()) * 0.5;
            let log_flux = frame.flux.ln();
            let fraction = (log_flux - (before + after) * 0.5).exp();
            let weaker_side_fraction = (log_flux - before.min(after)).exp();
            (fraction <= MAX_TREND_FRACTION && weaker_side_fraction <= MAX_SIDE_FRACTION)
                .then_some((frame.index, fraction))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_only_the_isolated_dip_with_same_session_brackets() {
        let series = |flux: [f32; 5], timestamps: [i64; 5]| {
            flux.into_iter()
                .zip(timestamps)
                .enumerate()
                .rev()
                .map(|(index, (flux, timestamp))| Throughput {
                    index,
                    timestamp,
                    exposure: 600.0,
                    flux,
                })
                .collect()
        };
        let cadence = [0, 600_000, 1_200_000, 1_800_000, 2_400_000];
        let dip = [0.66, 0.60, 0.33, 0.48, 0.43];
        let rejected = throughput_outliers(series(dip, cadence));
        assert_eq!(rejected.len(), 1);
        assert_eq!(rejected[0].0, 2);
        assert!((0.60..0.65).contains(&rejected[0].1));
        assert!(throughput_outliers(series([0.66, 0.60, 0.33, 0.31, 0.30], cadence)).is_empty());
        let across_gap = [0, 600_000, 1_200_000, 86_400_000, 87_000_000];
        assert!(throughput_outliers(series(dip, across_gap)).is_empty());
    }
}
