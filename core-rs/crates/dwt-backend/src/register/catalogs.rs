use super::{
    RegistrationConfig, Star, fit_similarity, match_stars, median, refine_stars, register_stars,
};
use crate::geometry::SimilarityTransform;
use std::cell::OnceCell;

const TRIANGLE_TOLERANCE: f64 = 0.03;
const INLIER_RADIUS: f64 = 2.5;

#[derive(Default)]
pub(crate) struct RegistrationCounts {
    pub xdrz: usize,
    pub triangle: usize,
    pub chained: usize,
    pub adjusted: usize,
}

pub(crate) fn verify_transform(
    reference: &[Star],
    frame: &[Star],
    transform: SimilarityTransform,
    config: RegistrationConfig,
) -> bool {
    let base = reference.len().min(frame.len());
    let required = (base * 35).div_ceil(100).max(config.min_matches);
    let pairs = match_stars(reference, frame, transform, INLIER_RADIUS);
    if pairs.len() >= required {
        return true;
    }
    if base <= 200 && pairs.len() >= 20 {
        let mut residuals: Vec<_> = pairs
            .iter()
            .map(|&(i, j)| {
                let (x, y) = transform.apply(frame[i].x, frame[i].y);
                (x - reference[j].x).hypot(y - reference[j].y)
            })
            .collect();
        if median(&mut residuals) <= 0.6 * INLIER_RADIUS {
            return true;
        }
    }
    match_stars(reference, frame, transform, 4.0).len() >= required
}

struct Triangle {
    corners: [usize; 3],
    ratios: [f64; 2],
}

fn triangles(stars: &[Star], limit: usize) -> Vec<Triangle> {
    let stars = &stars[..stars.len().min(limit)];
    let length = |i: usize, j: usize| (stars[i].x - stars[j].x).hypot(stars[i].y - stars[j].y);
    let mut result = Vec::new();
    for a in 0..stars.len() {
        for b in (a + 1)..stars.len() {
            for c in (b + 1)..stars.len() {
                let mut edges = [(length(b, c), a), (length(a, c), b), (length(a, b), c)];
                edges.sort_unstable_by(|left, right| right.0.total_cmp(&left.0));
                let [(long, a), (middle, b), (short, c)] = edges;
                if long > 8.0 && short > 0.12 * long && middle > 0.2 * long {
                    result.push(Triangle {
                        corners: [a, b, c],
                        ratios: [middle / long, short / long],
                    });
                }
            }
        }
    }
    result.sort_unstable_by(|a, b| a.ratios[0].total_cmp(&b.ratios[0]));
    result
}

fn solve_triangles(
    reference: &[Star],
    frame: &[Star],
    targets: &[Triangle],
    limit: usize,
    minimum: usize,
) -> Option<SimilarityTransform> {
    if reference.len().min(frame.len()) < minimum {
        return None;
    }
    let sources = triangles(frame, limit);
    let mut candidates = Vec::new();
    for (i, triangle) in sources.iter().enumerate() {
        let [x, y] = triangle.ratios;
        let start = targets.partition_point(|target| target.ratios[0] < x - TRIANGLE_TOLERANCE);
        let end = targets.partition_point(|target| target.ratios[0] <= x + TRIANGLE_TOLERANCE);
        let mut nearest = [(TRIANGLE_TOLERANCE.powi(2), 0); 2];
        for (j, target) in targets.iter().enumerate().take(end).skip(start) {
            let distance = (target.ratios[0] - x).powi(2) + (target.ratios[1] - y).powi(2);
            if distance < nearest[1].0 {
                nearest[1] = (distance, j);
                if nearest[1].0 < nearest[0].0 {
                    nearest.swap(0, 1);
                }
            }
        }
        candidates.extend(
            nearest
                .into_iter()
                .filter(|&(d, _)| d < TRIANGLE_TOLERANCE.powi(2))
                .map(|(d, j)| (d, i, j)),
        );
    }
    candidates.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    let point = |star: Star| (star.x, star.y);
    let mut best = None;
    for (_, i, j) in candidates.into_iter().take(1500) {
        let source = sources[i].corners.map(|i| point(frame[i]));
        let target = targets[j].corners.map(|j| point(reference[j]));
        let Ok(seed) = fit_similarity(&source, &target) else {
            continue;
        };
        if !(0.98..=1.02).contains(&seed.scale) {
            continue;
        }
        let pairs = match_stars(reference, frame, seed, INLIER_RADIUS);
        if pairs.len() < minimum
            || best
                .as_ref()
                .is_some_and(|prior: &Vec<_>| prior.len() >= pairs.len())
        {
            continue;
        }
        let enough = pairs.len() * 2 >= frame.len();
        best = Some(pairs);
        if enough {
            break;
        }
    }
    let pairs = best?;
    let source: Vec<_> = pairs.iter().map(|&(i, _)| point(frame[i])).collect();
    let target: Vec<_> = pairs.iter().map(|&(_, j)| point(reference[j])).collect();
    let transform = fit_similarity(&source, &target).ok()?;
    let rms = super::residual_rms(&source, &target, transform);
    ((0.98..=1.02).contains(&transform.scale) && rms <= 2.0).then_some(transform)
}

pub(crate) fn register_sequence(
    catalogs: &[Vec<Star>],
    seeds: &[Option<SimilarityTransform>],
    reference_index: usize,
    config: RegistrationConfig,
    progress: &mut impl FnMut(&'static str, usize, usize),
) -> (Vec<Option<SimilarityTransform>>, RegistrationCounts) {
    let reference = &catalogs[reference_index];
    let reference_seed = seeds[reference_index].and_then(SimilarityTransform::inverse);
    let reference_triangles = OnceCell::new();
    let mut counts = RegistrationCounts::default();
    let mut transforms: Vec<_> = catalogs
        .iter()
        .enumerate()
        .map(|(index, stars)| {
            progress("Registering frames", index, catalogs.len());
            if index == reference_index {
                return Some(SimilarityTransform::identity());
            }
            let seed = seeds[index]
                .zip(reference_seed)
                .and_then(|(source, inverse)| source.compose(inverse).ok());
            if let Some(result) =
                seed.and_then(|seed| refine_stars(reference, stars, seed, config, 12.0).ok())
            {
                counts.xdrz += 1;
                return Some(result.transform);
            }
            if let Ok(result) = register_stars(reference, stars, config) {
                return Some(result.transform);
            }
            let transform = solve_triangles(
                reference,
                stars,
                reference_triangles.get_or_init(|| triangles(reference, 48)),
                48,
                15,
            )?;
            if verify_transform(reference, stars, transform, config) {
                counts.triangle += 1;
                return Some(transform);
            }
            None
        })
        .collect();
    progress("Registering frames", catalogs.len(), catalogs.len());
    let direct: Vec<_> = (0..catalogs.len())
        .filter(|&i| transforms[i].is_some())
        .collect();
    let neighbors: Vec<_> = catalogs.iter().map(|_| OnceCell::new()).collect();
    let pair = |target: usize, source: usize| {
        solve_triangles(
            &catalogs[target],
            &catalogs[source],
            neighbors[target].get_or_init(|| triangles(&catalogs[target], 40)),
            40,
            12,
        )
    };
    let mut chained = vec![false; catalogs.len()];
    for index in 0..catalogs.len() {
        if transforms[index].is_some() || catalogs[index].len() < 20 {
            continue;
        }
        progress("Recovering unmatched frames", index + 1, catalogs.len());
        let mut nearest = direct.clone();
        nearest.sort_unstable_by_key(|&j| (j.abs_diff(index), j));
        for j in nearest
            .into_iter()
            .filter(|&j| j != reference_index && catalogs[j].len() >= 20)
            .take(3)
        {
            let Some(transform) =
                pair(j, index).and_then(|relative| relative.compose(transforms[j]?).ok())
            else {
                continue;
            };
            if (0.9..1.1).contains(&transform.scale)
                && verify_transform(reference, &catalogs[index], transform, config)
            {
                transforms[index] = Some(transform);
                chained[index] = true;
                counts.chained += 1;
                break;
            }
        }
    }
    if counts.chained > 0 && direct.len() >= 3 {
        let nodes: Vec<_> = transforms
            .iter()
            .enumerate()
            .filter_map(|(i, transform)| {
                let start = i.saturating_sub(2);
                let end = (i + 3).min(chained.len());
                transform
                    .filter(|_| chained[start..end].contains(&true))
                    .map(|transform| (i, transform))
            })
            .collect();
        let edges: Vec<_> = nodes
            .windows(2)
            .map(|window| {
                let (a, b) = (window[0].0, window[1].0);
                if b - a > 3 { None } else { pair(b, a) }
            })
            .collect();
        for (index, transform) in adjust_chain(&nodes, &edges, &chained, reference_index) {
            let Some(prior) = transforms[index] else {
                continue;
            };
            let change = (prior.tx - transform.tx).hypot(prior.ty - transform.ty)
                + (prior.theta - transform.theta).abs()
                + (prior.scale - transform.scale).abs();
            if change < 1.0e-8 {
                continue;
            }
            if (0.95..1.05).contains(&transform.scale)
                && verify_transform(reference, &catalogs[index], transform, config)
            {
                transforms[index] = Some(transform);
                counts.adjusted += 1;
            }
        }
    }
    (transforms, counts)
}

// The chain only connects consecutive nodes. The weighted least-squares normal
// matrix is tridiagonal, and its four right-hand sides share one factorization.
fn adjust_chain(
    nodes: &[(usize, SimilarityTransform)],
    edges: &[Option<SimilarityTransform>],
    chained: &[bool],
    reference_index: usize,
) -> Vec<(usize, SimilarityTransform)> {
    let components = |t: SimilarityTransform| [t.tx, t.ty, t.theta, t.scale.ln()];
    let mut diagonal = Vec::with_capacity(nodes.len());
    let mut rhs = Vec::with_capacity(nodes.len());
    let mut off = vec![0.0; edges.len()];
    for &(i, transform) in nodes {
        let weight = if chained[i] { 0.25_f64.powi(2) } else { 1.0 };
        diagonal.push(weight);
        rhs.push(components(transform).map(|value| value * weight));
    }
    for (position, relative) in edges.iter().enumerate() {
        let Some(relative) = relative else {
            continue;
        };
        let (a, b) = (nodes[position], nodes[position + 1]);
        let (x, y) = b.1.apply(relative.tx, relative.ty);
        let theta = relative.theta
            + ((a.1.theta - b.1.theta - relative.theta) / std::f64::consts::TAU).round()
                * std::f64::consts::TAU;
        let delta = [x - b.1.tx, y - b.1.ty, theta, relative.scale.ln()];
        let weight = 4.0;
        if a.0 != reference_index && b.0 != reference_index {
            off[position] = -weight;
        }
        for (j, sign) in [(position, 1.0), (position + 1, -1.0)] {
            if nodes[j].0 != reference_index {
                diagonal[j] += weight;
                for (value, delta) in rhs[j].iter_mut().zip(delta) {
                    *value += sign * weight * delta;
                }
            }
        }
    }
    for i in 1..nodes.len() {
        let factor = off[i - 1] / diagonal[i - 1];
        diagonal[i] -= factor * off[i - 1];
        let previous = rhs[i - 1];
        for (value, previous) in rhs[i].iter_mut().zip(previous) {
            *value -= factor * previous;
        }
    }
    for i in (0..nodes.len()).rev() {
        let next = rhs.get(i + 1).copied().unwrap_or_default();
        let coefficient = off.get(i).copied().unwrap_or(0.0);
        for (value, next) in rhs[i].iter_mut().zip(next) {
            *value = (*value - coefficient * next) / diagonal[i];
        }
    }
    nodes
        .iter()
        .zip(rhs)
        .filter_map(|(&(i, _), [x, y, theta, log_scale])| {
            (i != reference_index).then_some(())?;
            SimilarityTransform::new(log_scale.exp(), theta, x, y)
                .ok()
                .map(|t| (i, t))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triangular_matching_recovers_rotation_with_reordered_stars() {
        let reference: Vec<_> = (0..24)
            .map(|i| Star {
                x: f64::from((i * i * 17 + i * 31) % 211),
                y: f64::from((i * i * 11 + i * 27) % 193),
                peak: 1.0,
            })
            .collect();
        let expected = SimilarityTransform::new(1.01, 1.2, 36.0, -17.0).expect("transform");
        let inverse = expected.inverse().expect("inverse");
        let frame: Vec<_> = reference
            .iter()
            .rev()
            .map(|star| {
                let (x, y) = inverse.apply(star.x, star.y);
                Star { x, y, peak: 1.0 }
            })
            .collect();
        let fit = solve_triangles(&reference, &frame, &triangles(&reference, 48), 48, 15)
            .expect("triangle solution");
        assert!((fit.tx - expected.tx).abs() < 1.0e-8);
        assert!((fit.ty - expected.ty).abs() < 1.0e-8);
        assert!((fit.theta - expected.theta).abs() < 1.0e-8);
        assert!((fit.scale - expected.scale).abs() < 1.0e-8);
    }

    #[test]
    fn chain_adjustment_uses_neighbor_constraints_with_fixed_reference() {
        let shift = |x| SimilarityTransform::new(1.0, 0.0, x, 0.0).expect("shift");
        let nodes = [(0, shift(0.0)), (1, shift(8.0)), (2, shift(20.0))];
        let edges = [Some(shift(-10.0)), Some(shift(-10.0))];
        let result = adjust_chain(&nodes, &edges, &[false, true, false], 0);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, 1);
        assert!((result[0].1.tx - 10.0).abs() < 0.03);
        assert!((result[1].1.tx - 20.0).abs() < 0.03);
    }
}
