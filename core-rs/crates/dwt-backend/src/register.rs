//! Star-based similarity registration.
//!
//! This deterministic first implementation uses local maxima and mutual-nearest
//! neighbor matching. The transform contract is stable, so a stronger matcher
//! can replace the matching stage later.

use crate::error::{BackendError, BackendResult};
use crate::frame::Frame;
use crate::geometry::SimilarityTransform;
use crate::pws::select_median;

mod catalogs;
mod field;
pub(crate) use catalogs::{register_sequence, verify_transform};
pub(crate) use field::{FieldCorrection, fit_field_correction, guide_centroid};

/// Signature of one detected star.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Star {
    pub x: f64,
    pub y: f64,
    pub peak: f64,
}

/// Registration controls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegistrationConfig {
    pub max_stars: usize,
    pub detection_sigma: f64,
    pub separation: usize,
    pub match_radius: f64,
    pub min_matches: usize,
}

impl Default for RegistrationConfig {
    fn default() -> Self {
        Self {
            max_stars: 80,
            detection_sigma: 5.0,
            separation: 3,
            match_radius: 3.0,
            min_matches: 8,
        }
    }
}

/// Similarity transform result for one frame.
#[derive(Debug, Clone, Copy)]
pub struct RegistrationResult {
    pub transform: SimilarityTransform,
    pub matches: usize,
    pub rms: f64,
}

/// Registers `frame` to `reference`.
pub fn register_frame(
    reference: &Frame,
    frame: &Frame,
    config: RegistrationConfig,
) -> BackendResult<RegistrationResult> {
    register_stars(
        &detect_stars(reference, config)?,
        &detect_stars(frame, config)?,
        config,
    )
}

pub(crate) fn register_stars(
    reference_stars: &[Star],
    frame_stars: &[Star],
    config: RegistrationConfig,
) -> BackendResult<RegistrationResult> {
    if !config.match_radius.is_finite() || config.match_radius <= 0.0 {
        return Err(BackendError::InvalidValue {
            name: "match_radius",
            value: config.match_radius as f32,
        });
    }
    if reference_stars.len() < config.min_matches || frame_stars.len() < config.min_matches {
        return Err(BackendError::Message(format!(
            "insufficient stars: reference {}, frame {}",
            reference_stars.len(),
            frame_stars.len()
        )));
    }
    let bright = |stars: &[Star]| stars.len().min(80);
    if let Some((seed, _)) = estimate_similarity_seed(
        &reference_stars[..bright(reference_stars)],
        &frame_stars[..bright(frame_stars)],
    ) {
        if let Ok(result) = refine_stars(reference_stars, frame_stars, seed, config, 4.0) {
            return Ok(result);
        }
    }
    fit_translation(reference_stars, frame_stars, config)
}

pub(crate) fn refine_stars(
    reference_stars: &[Star],
    frame_stars: &[Star],
    seed: SimilarityTransform,
    config: RegistrationConfig,
    seed_radius: f64,
) -> BackendResult<RegistrationResult> {
    let pairs = match_stars(
        reference_stars,
        frame_stars,
        seed,
        config.match_radius.max(seed_radius),
    );
    if pairs.len() < config.min_matches {
        return Err(BackendError::Message(format!(
            "insufficient matches: {}",
            pairs.len()
        )));
    }
    let source: Vec<(f64, f64)> = pairs
        .iter()
        .map(|&(frame_index, _)| {
            let star = frame_stars[frame_index];
            (star.x, star.y)
        })
        .collect();
    let target: Vec<(f64, f64)> = pairs
        .iter()
        .map(|&(_, reference_index)| {
            let star = reference_stars[reference_index];
            (star.x, star.y)
        })
        .collect();
    let transform = fit_similarity(&source, &target)?;
    let pairs = match_stars(reference_stars, frame_stars, transform, config.match_radius);
    if pairs.len() < config.min_matches {
        return Err(BackendError::Message(format!(
            "fit verification produced insufficient matches: {}",
            pairs.len()
        )));
    }
    let source: Vec<(f64, f64)> = pairs
        .iter()
        .map(|&(frame_index, _)| {
            let star = frame_stars[frame_index];
            (star.x, star.y)
        })
        .collect();
    let target: Vec<(f64, f64)> = pairs
        .iter()
        .map(|&(_, reference_index)| {
            let star = reference_stars[reference_index];
            (star.x, star.y)
        })
        .collect();
    let transform = fit_similarity(&source, &target)?;
    let rms = residual_rms(&source, &target, transform);
    if !verify_transform(reference_stars, frame_stars, transform, config) {
        return Err(BackendError::Message(
            "registration failed full-catalog verification".into(),
        ));
    }
    Ok(RegistrationResult {
        transform,
        matches: pairs.len(),
        rms,
    })
}

/// Refines registration from an external seed such as WBPP XDRZ geometry.
pub fn register_frame_with_seed(
    reference: &Frame,
    frame: &Frame,
    seed: SimilarityTransform,
    config: RegistrationConfig,
) -> BackendResult<RegistrationResult> {
    let reference_stars = detect_stars(reference, config)?;
    let frame_stars = detect_stars(frame, config)?;
    refine_stars(&reference_stars, &frame_stars, seed, config, 12.0)
        .or_else(|_| register_stars(&reference_stars, &frame_stars, config))
}

fn fit_translation(
    reference: &[Star],
    frame: &[Star],
    config: RegistrationConfig,
) -> BackendResult<RegistrationResult> {
    let mut votes = std::collections::BTreeMap::new();
    for source in frame {
        for target in reference {
            let x = target.x - source.x;
            let y = target.y - source.y;
            let bin = ((x / 2.0).round() as i64, (y / 2.0).round() as i64);
            let (count, sum_x, sum_y) = votes.entry(bin).or_insert((0_u32, 0.0, 0.0));
            *count += 1;
            *sum_x += x;
            *sum_y += y;
        }
    }
    let mut candidates: Vec<_> = votes.into_values().collect();
    candidates.sort_by_key(|&(votes, _, _)| std::cmp::Reverse(votes));
    let mut best = None;
    let minimum_matches = config
        .min_matches
        .max(reference.len().min(frame.len()) * 35 / 100);
    for (votes, x, y) in candidates.into_iter().take(16) {
        let seed = SimilarityTransform::new(1.0, 0.0, x / f64::from(votes), y / f64::from(votes))?;
        let pairs = match_stars(reference, frame, seed, config.match_radius);
        if pairs.len() < minimum_matches {
            continue;
        }
        let mut shifts_x = Vec::with_capacity(pairs.len());
        let mut shifts_y = Vec::with_capacity(pairs.len());
        for (source, target) in pairs {
            shifts_x.push(reference[target].x - frame[source].x);
            shifts_y.push(reference[target].y - frame[source].y);
        }
        let tx = median(&mut shifts_x);
        let ty = median(&mut shifts_y);
        let transform = SimilarityTransform::new(1.0, 0.0, tx, ty)?;
        let verified = match_stars(reference, frame, transform, config.match_radius);
        let matches = verified.len();
        if matches < minimum_matches || !verify_transform(reference, frame, transform, config) {
            continue;
        }
        let squared_error: f64 = verified
            .iter()
            .map(|&(source, target)| {
                let predicted = transform.apply(frame[source].x, frame[source].y);
                (predicted.0 - reference[target].x).powi(2)
                    + (predicted.1 - reference[target].y).powi(2)
            })
            .sum();
        let rms = (squared_error / matches as f64).sqrt();
        if best.is_none_or(|best: RegistrationResult| {
            matches > best.matches || (matches == best.matches && rms < best.rms)
        }) {
            best = Some(RegistrationResult {
                transform,
                matches,
                rms,
            });
        }
    }
    best.ok_or_else(|| {
        BackendError::Message("similarity and translation registration failed".into())
    })
}

/// Detects bright, separated local maxima above a robust MAD threshold.
pub fn detect_stars(frame: &Frame, config: RegistrationConfig) -> BackendResult<Vec<Star>> {
    let pixels = frame.pixels();
    let mut finite: Vec<_> = pixels
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    let Some(sky_median) = select_median(&mut finite) else {
        return Ok(Vec::new());
    };
    let peak = finite.iter().copied().fold(sky_median, f32::max) - sky_median;
    for sample in &mut finite {
        *sample = (*sample - sky_median).abs();
    }
    let mad = f64::from(select_median(&mut finite).unwrap_or(0.0));
    let sky_median = f64::from(sky_median);
    let mut threshold = sky_median + config.detection_sigma * 1.4826 * mad.max(1.0e-30);
    if mad <= 1.0e-6 {
        threshold = threshold.max(sky_median + f64::from(peak) * 1.0e-3);
    }
    if !threshold.is_finite() {
        threshold = sky_median;
    }
    let mut stars = Vec::new();
    for y in 1..frame.height().saturating_sub(1) {
        for x in 1..frame.width().saturating_sub(1) {
            let value = f64::from(frame.pixel(x, y).unwrap_or(f32::NAN));
            if !value.is_finite() || value <= threshold {
                continue;
            }
            let mut local_maximum = true;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let neighbour = f64::from(
                        frame
                            .pixel((x as isize + dx) as usize, (y as isize + dy) as usize)
                            .unwrap_or(f32::NAN),
                    );
                    let neighbour_position =
                        ((x as isize + dx) as usize, (y as isize + dy) as usize);
                    if neighbour > value || (neighbour == value && neighbour_position < (x, y)) {
                        local_maximum = false;
                        break;
                    }
                }
                if !local_maximum {
                    break;
                }
            }
            if local_maximum {
                let offset = |left: f32, right: f32| {
                    let (left, right) = (f64::from(left), f64::from(right));
                    let curvature = left - 2.0 * value + right;
                    if curvature.is_finite() && curvature < 0.0 {
                        (0.5 * (left - right) / curvature).clamp(-0.5, 0.5)
                    } else {
                        0.0
                    }
                };
                let index = y * frame.width() + x;
                stars.push(Star {
                    x: x as f64 + 0.5 + offset(pixels[index - 1], pixels[index + 1]),
                    y: y as f64
                        + 0.5
                        + offset(pixels[index - frame.width()], pixels[index + frame.width()]),
                    peak: value - sky_median,
                });
            }
        }
    }
    stars.sort_by(|left, right| right.peak.total_cmp(&left.peak));
    suppress_close(&mut stars, config.separation, config.max_stars);
    Ok(stars)
}

/// Measures the half-maximum diameter of isolated stellar peaks.
pub(crate) fn estimate_fwhm(frame: &Frame, stars: &[Star]) -> Option<f32> {
    const RADIUS: usize = 5;
    let mut widths = Vec::new();
    for star in stars.iter().take(64) {
        let (x, y) = (star.x as usize, star.y as usize);
        if x < RADIUS || y < RADIUS || x + RADIUS >= frame.width() || y + RADIUS >= frame.height() {
            continue;
        }
        let pixel = |x, y| frame.pixels()[y * frame.width() + x];
        let mut corners = [
            pixel(x - RADIUS, y - RADIUS),
            pixel(x + RADIUS, y - RADIUS),
            pixel(x - RADIUS, y + RADIUS),
            pixel(x + RADIUS, y + RADIUS),
        ];
        if corners.iter().any(|value| !value.is_finite()) {
            continue;
        }
        let background = select_median(&mut corners).unwrap_or(0.0);
        let peak = pixel(x, y);
        if peak <= background {
            continue;
        }
        let half = peak.midpoint(background);
        let crossing = |dx: isize, dy: isize| {
            let mut previous = peak;
            for step in 1..=RADIUS {
                let value = pixel(
                    x.wrapping_add_signed(dx * step as isize),
                    y.wrapping_add_signed(dy * step as isize),
                );
                if !value.is_finite() {
                    return None;
                }
                if value <= half {
                    return Some((step - 1) as f32 + (previous - half) / (previous - value));
                }
                previous = value;
            }
            None
        };
        if let (Some(left), Some(right), Some(up), Some(down)) = (
            crossing(-1, 0),
            crossing(1, 0),
            crossing(0, -1),
            crossing(0, 1),
        ) {
            widths.push(((left + right) * (up + down)).sqrt());
        }
    }
    select_median(&mut widths)
}

/// Descriptor of a star pair, used for rotation search.
#[derive(Debug, Clone, Copy)]
pub struct StarPair {
    pub first: usize,
    pub second: usize,
    pub length: f64,
    pub angle: f64,
}

/// Enumerates sufficiently long star pairs.
pub fn star_pairs(stars: &[Star], minimum_length: f64) -> Vec<StarPair> {
    let mut pairs = Vec::new();
    for first in 0..stars.len() {
        for second in (first + 1)..stars.len() {
            let dx = stars[second].x - stars[first].x;
            let dy = stars[second].y - stars[first].y;
            let length = dx.hypot(dy);
            if length >= minimum_length {
                pairs.push(StarPair {
                    first,
                    second,
                    length,
                    angle: dy.atan2(dx),
                });
            }
        }
    }
    pairs
}

/// Estimates rotation by matching pair-length bins and voting on angle differences.
pub fn estimate_rotation_degrees(reference: &[Star], frame: &[Star]) -> Option<f64> {
    estimate_similarity_seed(reference, frame).map(|(transform, _)| transform.theta.to_degrees())
}

/// Estimates an initial scale, rotation, and translation from star geometry.
pub fn estimate_similarity_seed(
    reference: &[Star],
    frame: &[Star],
) -> Option<(SimilarityTransform, usize)> {
    let mut frame_pairs = star_pairs(frame, 40.0);
    frame_pairs.sort_by(|left, right| left.length.total_cmp(&right.length));
    let reference_pairs = star_pairs(reference, 40.0);
    if reference_pairs.len() < 3 || frame_pairs.len() < 3 {
        return None;
    }
    let bin_count = 720_usize;
    let mut best = None;
    for scale in scale_candidates() {
        let mut votes = vec![0_usize; bin_count];
        for reference_pair in &reference_pairs {
            let target_length = reference_pair.length / scale;
            let tolerance = (1.0_f64 / scale).max(reference_pair.length * 0.002);
            let lower = frame_pairs.partition_point(|pair| pair.length < target_length - tolerance);
            let upper =
                frame_pairs.partition_point(|pair| pair.length <= target_length + tolerance);
            for frame_pair in &frame_pairs[lower..upper] {
                let difference =
                    (frame_pair.angle - reference_pair.angle).rem_euclid(std::f64::consts::TAU);
                let bin = ((difference / std::f64::consts::TAU) * bin_count as f64).round()
                    as usize
                    % bin_count;
                votes[bin] += 1;
            }
        }
        let (peak, count) = votes
            .iter()
            .copied()
            .enumerate()
            .max_by_key(|&(_, count)| count)?;
        if best.is_none_or(|(best_count, _, _)| count > best_count) {
            best = Some((count, scale, peak));
        }
    }
    let (votes, scale, peak) = best?;
    if votes < 3 {
        return None;
    }
    let theta = -(peak as f64 / bin_count as f64 * 360.0).to_radians();
    let rotation = SimilarityTransform::new(scale, theta, 0.0, 0.0).ok()?;
    let tolerance = (1.0_f64 / scale).max(2.0);
    let mut translations: Vec<(f64, f64, usize)> = Vec::new();
    for reference_pair in &reference_pairs {
        let target_length = reference_pair.length / scale;
        let lower = frame_pairs.partition_point(|pair| pair.length < target_length - tolerance);
        let upper = frame_pairs.partition_point(|pair| pair.length <= target_length + tolerance);
        for frame_pair in &frame_pairs[lower..upper] {
            let difference =
                (frame_pair.angle - reference_pair.angle).rem_euclid(std::f64::consts::TAU);
            let bin = ((difference / std::f64::consts::TAU) * bin_count as f64).round() as usize
                % bin_count;
            let distance = ((bin as isize - peak as isize + bin_count as isize / 2)
                .rem_euclid(bin_count as isize))
                - bin_count as isize / 2;
            if distance.abs() > 2 {
                continue;
            }
            let frame_first = frame[frame_pair.first];
            let frame_second = frame[frame_pair.second];
            let reference_first = reference[reference_pair.first];
            let reference_second = reference[reference_pair.second];
            let first = rotation.apply(frame_first.x, frame_first.y);
            let second = rotation.apply(frame_second.x, frame_second.y);
            let first_translation = (reference_first.x - first.0, reference_first.y - first.1);
            let second_translation = (reference_second.x - second.0, reference_second.y - second.1);
            for (tx, ty) in [first_translation, second_translation] {
                if let Some(index) = translations
                    .iter()
                    .position(|&(x, y, _)| (x - tx).abs() < 0.75 && (y - ty).abs() < 0.75)
                {
                    translations[index].2 += 1;
                } else {
                    translations.push((tx, ty, 1));
                }
            }
            if translations.len() >= 500 {
                break;
            }
        }
        if translations.len() >= 300 {
            break;
        }
    }
    translations.sort_by_key(|left| std::cmp::Reverse(left.2));
    let mut best_transform = None;
    let mut best_matches = 0;
    for (tx, ty, _) in translations.into_iter().take(32) {
        let transform = SimilarityTransform::new(scale, theta, tx, ty).ok()?;
        let matches = match_stars(reference, frame, transform, 4.0);
        if matches.len() > best_matches {
            best_matches = matches.len();
            best_transform = Some(transform);
        }
    }
    Some((best_transform?, best_matches))
}

fn scale_candidates() -> Vec<f64> {
    let mut candidates = Vec::with_capacity(41);
    for index in 0..=40 {
        candidates.push(1.0 + (index as f64 - 20.0) * 0.005);
    }
    candidates.sort_by(|left, right| (left - 1.0).abs().total_cmp(&(right - 1.0).abs()));
    candidates
}

/// Picks spatially separated nearest pairs between two star lists.
pub fn match_stars(
    reference: &[Star],
    frame: &[Star],
    transform: SimilarityTransform,
    radius: f64,
) -> Vec<(usize, usize)> {
    let mut used_reference = vec![false; reference.len()];
    let mut pairs = Vec::new();
    let radius_squared = radius * radius;
    for (frame_index, star) in frame.iter().enumerate() {
        let (x, y) = transform.apply(star.x, star.y);
        let mut best = None;
        for (reference_index, candidate) in reference.iter().enumerate() {
            if used_reference[reference_index] {
                continue;
            }
            let distance = (candidate.x - x).powi(2) + (candidate.y - y).powi(2);
            if distance <= radius_squared
                && best.is_none_or(|(_, best_distance)| distance < best_distance)
            {
                best = Some((reference_index, distance));
            }
        }
        if let Some((reference_index, _)) = best {
            used_reference[reference_index] = true;
            pairs.push((frame_index, reference_index));
        }
    }
    pairs
}

fn suppress_close(stars: &mut Vec<Star>, separation: usize, limit: usize) {
    if separation == 0 {
        stars.truncate(limit);
        return;
    }
    let minimum = (separation * separation) as f64;
    let mut retained: Vec<Star> = Vec::with_capacity(stars.len().min(limit));
    for star in stars.iter().copied() {
        if retained.len() == limit {
            break;
        }
        if retained.iter().all(|other| {
            let dx = other.x - star.x;
            let dy = other.y - star.y;
            dx * dx + dy * dy >= minimum
        }) {
            retained.push(star);
        }
    }
    *stars = retained;
}

/// Least-squares 2D similarity fit with a positive scale and no reflection.
pub fn fit_similarity(
    source: &[(f64, f64)],
    target: &[(f64, f64)],
) -> BackendResult<SimilarityTransform> {
    if source.len() != target.len() || source.len() < 2 {
        return Err(BackendError::Message(
            "similarity fit needs at least two matched points".to_string(),
        ));
    }
    let source_mean = mean(source);
    let target_mean = mean(target);
    let mut dot = 0.0;
    let mut cross = 0.0;
    let mut source_variance = 0.0_f64;
    for ((sx, sy), (tx, ty)) in source.iter().zip(target) {
        let sx = sx - source_mean.0;
        let sy = sy - source_mean.1;
        let tx = tx - target_mean.0;
        let ty = ty - target_mean.1;
        dot += sx * tx + sy * ty;
        cross += sx * ty - sy * tx;
        source_variance += sx * sx + sy * sy;
    }
    if source_variance <= 1.0e-20 {
        return Err(BackendError::Message(
            "degenerate source star pattern".to_string(),
        ));
    }
    // The constrained matrix is [a -b; b a]. Solve for its two independent
    // coefficients directly, including large rotations and collinear matches.
    let a = dot / source_variance;
    let b = cross / source_variance;
    let scale = a.hypot(b);
    let theta = b.atan2(a);
    let tx = target_mean.0 - (a * source_mean.0 - b * source_mean.1);
    let ty = target_mean.1 - (b * source_mean.0 + a * source_mean.1);
    SimilarityTransform::new(scale, theta, tx, ty)
}

fn mean(points: &[(f64, f64)]) -> (f64, f64) {
    let count = points.len() as f64;
    let sum_x: f64 = points.iter().map(|point| point.0).sum();
    let sum_y: f64 = points.iter().map(|point| point.1).sum();
    (sum_x / count, sum_y / count)
}

fn residual_rms(
    source: &[(f64, f64)],
    target: &[(f64, f64)],
    transform: SimilarityTransform,
) -> f64 {
    let total: f64 = source
        .iter()
        .zip(target)
        .map(|((sx, sy), (tx, ty))| {
            let (px, py) = transform.apply(*sx, *sy);
            (px - tx).powi(2) + (py - ty).powi(2)
        })
        .sum();
    (total / source.len() as f64).sqrt()
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 1 {
        values[middle]
    } else {
        0.5 * (values[middle - 1] + values[middle])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn similarity_fit_recovers_transform() {
        let source = [(10.0, 12.0), (18.0, 13.0), (15.0, 25.0), (30.0, 20.0)];
        let expected = SimilarityTransform::new(1.03, 0.07, 4.0, -3.0).expect("transform");
        let target: Vec<_> = source.iter().map(|&(x, y)| expected.apply(x, y)).collect();
        let fitted = fit_similarity(&source, &target).expect("fit");
        assert!((fitted.scale - expected.scale).abs() < 1.0e-10);
        assert!((fitted.theta - expected.theta).abs() < 1.0e-10);
        assert!((fitted.tx - expected.tx).abs() < 1.0e-9);
        assert!((fitted.ty - expected.ty).abs() < 1.0e-9);
    }

    #[test]
    fn register_frame_recovers_rotation_and_scale() {
        let base = [
            (45.0, 52.0),
            (92.0, 38.0),
            (150.0, 55.0),
            (203.0, 87.0),
            (61.0, 112.0),
            (124.0, 102.0),
            (181.0, 126.0),
            (42.0, 171.0),
            (96.0, 158.0),
            (153.0, 176.0),
            (211.0, 164.0),
            (73.0, 209.0),
            (132.0, 214.0),
            (190.0, 205.0),
        ];
        let expected = SimilarityTransform::new(1.02, 0.12, 15.0, -8.0).expect("transform");
        let inverse = expected.inverse().expect("inverse");
        let frame_points: Vec<(f64, f64)> =
            base.iter().map(|&(x, y)| inverse.apply(x, y)).collect();
        let reference = render_stars(256, 256, &base);
        let frame = render_stars(256, 256, &frame_points);
        let result = register_frame(&reference, &frame, RegistrationConfig::default())
            .expect("registration");
        assert!((result.transform.scale - expected.scale).abs() < 0.015);
        assert!((result.transform.theta - expected.theta).abs() < 0.015);
        assert!((result.transform.tx - expected.tx).abs() < 3.0);
        assert!((result.transform.ty - expected.ty).abs() < 3.0);
    }

    fn render_stars(width: usize, height: usize, points: &[(f64, f64)]) -> Frame {
        let mut pixels = vec![0.01_f32; width * height];
        for &(cx, cy) in points {
            let x0 = (cx - 7.0).floor() as isize;
            let x1 = (cx + 7.0).ceil() as isize;
            let y0 = (cy - 7.0).floor() as isize;
            let y1 = (cy + 7.0).ceil() as isize;
            for y in y0..=y1 {
                for x in x0..=x1 {
                    if x < 0 || y < 0 || x >= width as isize || y >= height as isize {
                        continue;
                    }
                    let dx = x as f64 + 0.5 - cx;
                    let dy = y as f64 + 0.5 - cy;
                    let value = 10.0 * (-(dx * dx + dy * dy) / (2.0 * 1.2 * 1.2)).exp();
                    pixels[y as usize * width + x as usize] += value as f32;
                }
            }
        }
        Frame::new(width, height, pixels).expect("synthetic frame")
    }
}
