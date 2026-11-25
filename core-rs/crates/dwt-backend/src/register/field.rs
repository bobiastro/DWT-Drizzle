//! Fits a bounded correction to a verified source-to-reference similarity.

use super::{Star, match_stars, median};
use crate::frame::{Frame, ImageShape};
use crate::geometry::SimilarityTransform;
use crate::pws::select_median;

const TERMS: usize = 6;
const MIN_GUIDES: usize = 48;
const HUBER_RADIUS: f64 = 0.55;

#[derive(Clone, Copy, Debug)]
pub(crate) struct FieldCorrection {
    coeffs: [[f64; TERMS]; 2],
    shape: ImageShape,
    pub(crate) baseline: f64,
    pub(crate) corrected: f64,
}

#[derive(Clone, Copy)]
pub(crate) struct FieldRow {
    coeffs: [[f64; 3]; 2],
}

impl FieldRow {
    #[inline]
    pub(crate) fn at(self, x: f64) -> [f64; 2] {
        self.coeffs.map(|[a, b, c]| (c * x + b) * x + a)
    }
}

impl FieldCorrection {
    /// Conservative displacement bound on the whole native image.
    pub(crate) fn bounds(self) -> [f64; 2] {
        self.coeffs.map(|axis| axis.into_iter().map(f64::abs).sum())
    }

    pub(crate) fn output_row(self, y: f64, output_scale: f64) -> FieldRow {
        let v = y * 2.0 / self.shape.height as f64 - 1.0;
        let x_scale = 2.0 / self.shape.width as f64;
        let coeffs = self.coeffs.map(|[a, b, c, d, e, f]| {
            // Substitute u = x * x_scale - 1 into the normalized quadratic.
            let linear = b + e * v;
            let constant = a + (c + f * v) * v;
            [
                (constant - linear + d) * output_scale,
                (linear - 2.0 * d) * x_scale * output_scale,
                d * x_scale * x_scale * output_scale,
            ]
        });
        FieldRow { coeffs }
    }

    pub(crate) fn slope_bounds(self) -> [[f64; 2]; 2] {
        self.coeffs.map(|[_, b, c, d, e, f]| {
            [
                (b.abs() + 2.0 * d.abs() + e.abs()) * 2.0 / self.shape.width as f64,
                (c.abs() + e.abs() + 2.0 * f.abs()) * 2.0 / self.shape.height as f64,
            ]
        })
    }

    fn is_bounded(self) -> bool {
        // Keep the map close to its base similarity throughout the image,
        // including the edges outside the guide-star distribution.
        self.bounds().iter().all(|&bound| bound <= 12.0)
            && self.slope_bounds().iter().all(|[dx, dy]| dx + dy <= 0.035)
    }
}

/// Short-aperture photometric centroid. Reject clipped cores and near-edge stars.
pub(crate) fn guide_centroid(frame: &Frame, star: &Star) -> Option<[f64; 2]> {
    const RADIUS: usize = 5;
    let (x, y) = (star.x as usize, star.y as usize);
    if x < RADIUS || y < RADIUS || x + RADIUS >= frame.width() || y + RADIUS >= frame.height() {
        return None;
    }
    let mut background = Vec::with_capacity(64);
    for row in y - RADIUS..=y + RADIUS {
        for col in x - RADIUS..=x + RADIUS {
            let value = frame.pixels()[row * frame.width() + col];
            if row.abs_diff(y).pow(2) + col.abs_diff(x).pow(2) > 16 && value.is_finite() {
                background.push(value);
            }
        }
    }
    let sky = select_median(&mut background)?;
    let mut total = 0.0;
    let mut mx = 0.0;
    let mut my = 0.0;
    for row in y - 3..=y + 3 {
        for col in x - 3..=x + 3 {
            if row.abs_diff(y).pow(2) + col.abs_diff(x).pow(2) > 9 {
                continue;
            }
            let value = frame.pixels()[row * frame.width() + col];
            if !value.is_finite() || value >= 0.8 {
                return None;
            }
            let flux = f64::from((value - sky).max(0.0));
            total += flux;
            mx += flux * (col as f64 - x as f64);
            my += flux * (row as f64 - y as f64);
        }
    }
    let position = [x as f64 + 0.5 + mx / total, y as f64 + 0.5 + my / total];
    ((position[0] - star.x).hypot(position[1] - star.y) <= 1.5).then_some(position)
}

#[derive(Clone, Copy)]
struct Match {
    basis: [f64; TERMS],
    delta: [f64; 2],
}

fn basis(shape: ImageShape, xy: [f64; 2]) -> [f64; TERMS] {
    let u = xy[0] * 2.0 / shape.width as f64 - 1.0;
    let v = xy[1] * 2.0 / shape.height as f64 - 1.0;
    [1.0, u, v, u * u, u * v, v * v]
}

pub(crate) fn fit_field_correction(
    frame: &Frame,
    catalog: &[Star],
    reference: &[Star],
    guide_positions: &[Option<[f64; 2]>],
    transform: SimilarityTransform,
    radius: f64,
) -> Option<FieldCorrection> {
    let matches = match_stars(reference, catalog, transform, radius)
        .into_iter()
        .filter_map(|(source, target)| {
            let guide = guide_positions[target]?;
            let xy = guide_centroid(frame, &catalog[source])?;
            let (px, py) = transform.apply(xy[0], xy[1]);
            Some(Match {
                basis: basis(frame.shape(), xy),
                delta: [guide[0] - px, guide[1] - py],
            })
        })
        .collect();
    fit_validated(frame.shape(), matches)
}

fn fit_validated(shape: ImageShape, mut matches: Vec<Match>) -> Option<FieldCorrection> {
    if matches.len() < MIN_GUIDES {
        return None;
    }
    // Interleave the validation stars across the native field. Require support
    // in all quadrants before letting the polynomial move the image edges.
    let mut quadrants = [0; 4];
    for m in &matches {
        quadrants[usize::from(m.basis[1] >= 0.0) + 2 * usize::from(m.basis[2] >= 0.0)] += 1;
    }
    if quadrants.iter().any(|&count| count < 5) {
        return None;
    }
    matches.sort_unstable_by(|a, b| a.basis[1].total_cmp(&b.basis[1]));
    let (validation, train): (Vec<_>, Vec<_>) = matches
        .iter()
        .copied()
        .enumerate()
        .partition(|(index, _)| index % 4 == 0);
    let training: Vec<_> = train.into_iter().map(|(_, m)| m).collect();
    let baseline = median(
        &mut validation
            .iter()
            .map(|(_, m)| error(*m, [[0.0; TERMS]; 2]))
            .collect::<Vec<_>>(),
    );
    if baseline < 0.30 {
        return None;
    }
    let checked = fit_robust(&training)?;
    let corrected = median(
        &mut validation
            .iter()
            .map(|(_, m)| error(*m, checked))
            .collect::<Vec<_>>(),
    );
    if corrected > baseline * 0.78 || baseline - corrected < 0.08 {
        return None;
    }
    let fit = FieldCorrection {
        coeffs: fit_robust(&matches)?,
        shape,
        baseline,
        corrected,
    };
    fit.is_bounded().then_some(fit)
}

fn error(m: Match, coeffs: [[f64; TERMS]; 2]) -> f64 {
    let residual = std::array::from_fn::<_, 2, _>(|axis| {
        m.delta[axis]
            - coeffs[axis]
                .iter()
                .zip(m.basis)
                .map(|(a, b)| a * b)
                .sum::<f64>()
    });
    residual[0].hypot(residual[1])
}

fn fit_robust(matches: &[Match]) -> Option<[[f64; TERMS]; 2]> {
    let mut coeffs = [[0.0; TERMS]; 2];
    for iteration in 0..5 {
        let mut matrix = [[0.0; TERMS + 2]; TERMS];
        for &m in matches {
            let weight = if iteration == 0 {
                1.0
            } else {
                HUBER_RADIUS / error(m, coeffs).max(HUBER_RADIUS)
            };
            for (i, row) in matrix.iter_mut().enumerate() {
                let wi = weight * m.basis[i];
                for (dest, value) in row.iter_mut().zip(m.basis.into_iter().chain(m.delta)) {
                    *dest += wi * value;
                }
            }
        }
        for column in 0..TERMS {
            let pivot = (column..TERMS)
                .max_by(|&a, &b| matrix[a][column].abs().total_cmp(&matrix[b][column].abs()))?;
            if matrix[pivot][column].abs() < 1.0e-7 {
                return None;
            }
            matrix.swap(column, pivot);
            let divisor = matrix[column][column];
            for value in &mut matrix[column][column..] {
                *value /= divisor;
            }
            let normalized = matrix[column];
            for (i, row) in matrix.iter_mut().enumerate() {
                if i != column {
                    let rate = row[column];
                    for (dest, value) in row[column..].iter_mut().zip(&normalized[column..]) {
                        *dest -= rate * value;
                    }
                }
            }
        }
        coeffs = std::array::from_fn(|axis| std::array::from_fn(|i| matrix[i][TERMS + axis]));
    }
    Some(coeffs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fits_a_quadratic_with_outliers_and_projects_in_output_pixels() {
        let shape = ImageShape::new(1024, 768).unwrap();
        let truth = [
            [0.55, 0.1, -0.3, 1.3, 0.4, -0.15],
            [-0.82, -0.2, 0.3, 0.1, -0.7, 0.5],
        ];
        let evaluate = |basis: [f64; TERMS]| {
            truth.map(|terms| {
                terms
                    .into_iter()
                    .zip(basis)
                    .map(|(a, b)| a * b)
                    .sum::<f64>()
            })
        };
        let guides: Vec<_> = (0..100)
            .map(|i| {
                let xy = [
                    50.0 + (i % 10) as f64 * 100.0,
                    35.0 + (i / 10) as f64 * 76.0,
                ];
                let basis = basis(shape, xy);
                let mut delta = evaluate(basis);
                if i % 17 == 0 {
                    delta[0] += 2.5;
                    delta[1] -= 1.2;
                }
                Match { basis, delta }
            })
            .collect();
        let fit = fit_validated(shape, guides).expect("bounded correction");
        assert!(fit.corrected < 0.12);
        for xy in [[0.5, 0.5], [511.5, 123.0], [900.5, 677.5]] {
            let expected = evaluate(basis(shape, xy)).map(|v| 2.0 * v);
            let projected = fit.output_row(xy[1], 2.0).at(xy[0]);
            assert!((projected[0] - expected[0]).hypot(projected[1] - expected[1]) < 0.30);
        }
    }

    #[test]
    fn retains_similarity_for_small_residuals_or_one_sided_guides() {
        let shape = ImageShape::new(1024, 768).unwrap();
        let guides = (0..100).map(|i| Match {
            basis: basis(
                shape,
                [
                    50.0 + (i % 10) as f64 * 100.0,
                    35.0 + (i / 10) as f64 * 76.0,
                ],
            ),
            delta: [0.12, -0.1],
        });
        assert!(fit_validated(shape, guides.clone().collect()).is_none());
        assert!(fit_validated(shape, guides.filter(|m| m.basis[2] < 0.0).collect()).is_none());
    }
}
