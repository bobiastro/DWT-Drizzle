use std::{sync::mpsc, thread};

use super::{DrizzleResult, DropProjector, MIN_FRAME_COVERAGE, OutputTile, SimilarityCanvasPlan};
use crate::backend::SimilarityDrizzleRequest;
use crate::config::RejectionConfig;
use crate::error::{BackendError, BackendResult};
use crate::frame::{Frame, ImageShape};
use crate::pws::{erf_complement, select_median};
use crate::register::FieldCorrection;

const BLOCK: usize = 64;
const ANCHOR_FRAMES: usize = 48;
pub(crate) const MAX_STACK_WORKERS: usize = 8;

struct OutputBand<'a> {
    start_y: usize,
    image: &'a mut [f32],
    neff: &'a mut [f32],
    coverage: &'a mut [f32],
    frame_count: &'a mut [u32],
}

/// Projects bounded tiles, rejects outliers on the output grid, and combines the
/// remaining frame samples with the same drop coverage used for effective depth.
pub(crate) fn stack_winsorized(
    request: SimilarityDrizzleRequest<'_>,
    plan: SimilarityCanvasPlan,
    rejection: RejectionConfig,
    worker_limit: usize,
    corrections: &[Option<FieldCorrection>],
    mut progress: impl FnMut(usize, usize),
) -> BackendResult<DrizzleResult> {
    let SimilarityDrizzleRequest {
        transforms,
        region_scores,
        config: drizzle,
        ..
    } = request;
    drizzle.validate()?;
    rejection.validate()?;
    let shape = ImageShape::new(plan.width, plan.height)?;
    let pixels = shape.pixel_count();
    if pixels != region_scores.len() {
        return Err(BackendError::BufferLengthMismatch {
            name: "region_scores",
            expected: pixels,
            actual: region_scores.len(),
        });
    }
    let active: Vec<_> = transforms
        .iter()
        .enumerate()
        .filter_map(|(index, transform)| transform.map(|transform| (index, transform)))
        .map(|(index, transform)| {
            let field = corrections.get(index).copied().flatten();
            Ok((index, DropProjector::new(plan, drizzle, transform, field)?))
        })
        .collect::<BackendResult<_>>()?;
    if active.is_empty() {
        return Err(BackendError::EmptyFrames);
    }
    let mut image = vec![0.0; pixels];
    let mut neff = vec![0.0; pixels];
    let mut coverage = vec![0.0; pixels];
    let mut frame_count = vec![0; pixels];

    let block_rows = plan.height.div_ceil(BLOCK);
    let worker_limit = if worker_limit == 0 {
        MAX_STACK_WORKERS
    } else {
        worker_limit
    };
    let workers = thread::available_parallelism()
        .map_or(1, usize::from)
        .min(worker_limit)
        .min(block_rows);
    let band_height = (block_rows.div_ceil(workers) * BLOCK).min(plan.height);
    let band_pixels = band_height * plan.width;

    // Keep band boundaries on the original tile grid and leave each tile's
    // projection and reduction order intact.
    let bands = image
        .chunks_mut(band_pixels)
        .zip(neff.chunks_mut(band_pixels))
        .zip(coverage.chunks_mut(band_pixels))
        .zip(frame_count.chunks_mut(band_pixels))
        .enumerate()
        .map(
            |(index, (((image, neff), coverage), frame_count))| OutputBand {
                start_y: index * band_height,
                image,
                neff,
                coverage,
                frame_count,
            },
        );
    let mut completed = 0;
    progress(0, plan.height);
    let mut report_rows = |rows| {
        completed += rows;
        progress(completed, plan.height);
    };
    if workers == 1 {
        for band in bands {
            stack_band(
                request,
                plan.width,
                &active,
                rejection,
                band,
                &mut report_rows,
            )?;
        }
    } else {
        thread::scope(|scope| -> BackendResult<()> {
            let (sender, receiver) = mpsc::channel();
            let active = &active;
            let handles: Vec<_> = bands
                .map(|band| {
                    let sender = sender.clone();
                    scope.spawn(move || {
                        stack_band(request, plan.width, active, rejection, band, |rows| {
                            let _ = sender.send(rows);
                        })
                    })
                })
                .collect();
            drop(sender);
            for rows in receiver {
                report_rows(rows);
            }
            for handle in handles {
                handle.join().unwrap()?;
            }
            Ok(())
        })?;
    }

    Ok(DrizzleResult {
        image: Frame::new(plan.width, plan.height, image)?,
        neff,
        coverage,
        frame_count,
    })
}

fn stack_band(
    request: SimilarityDrizzleRequest<'_>,
    width: usize,
    active: &[(usize, DropProjector)],
    rejection: RejectionConfig,
    band: OutputBand<'_>,
    mut rows_done: impl FnMut(usize),
) -> BackendResult<()> {
    let SimilarityDrizzleRequest {
        frames,
        region_scores,
        weights,
        ..
    } = request;
    let OutputBand {
        start_y,
        image,
        neff,
        coverage,
        frame_count,
    } = band;
    let height = image.len() / width;
    let band_start = start_y * width;
    let region_scores = &region_scores[band_start..band_start + image.len()];
    let reject = rejection.sigma_multiplier > 0.0 && active.len() >= 3;
    let expected =
        active.len() as f32 * erf_complement(rejection.sigma_multiplier / std::f32::consts::SQRT_2);
    let gate = (rejection.max_fraction * active.len() as f32).max(1.0) + expected;

    let mut numerator = vec![0.0; BLOCK * BLOCK];
    let mut areas = vec![0.0; BLOCK * BLOCK];
    let mut samples = vec![vec![Sample::MISSING; active.len()]; BLOCK * BLOCK];
    let mut regions = vec![0.0; BLOCK * BLOCK];
    let mut deviations = Vec::with_capacity(active.len());
    for y in (0..height).step_by(BLOCK) {
        let tile_height = BLOCK.min(height - y);
        for x in (0..width).step_by(BLOCK) {
            let tile = OutputTile {
                x,
                y: start_y + y,
                width: BLOCK.min(width - x),
                height: tile_height,
            };
            let size = tile.width * tile.height;
            let samples = &mut samples[..size];
            let regions = &mut regions[..size];
            for (column, (index, projector)) in active.iter().enumerate() {
                numerator[..size].fill(0.0);
                areas[..size].fill(0.0);
                projector.scatter(&frames[*index], tile, &mut numerator, &mut areas);
                for pixel in 0..size {
                    let area = areas[pixel];
                    samples[pixel][column] = if area > MIN_FRAME_COVERAGE {
                        Sample {
                            value: (numerator[pixel] / area) as f32,
                            area: area as f32,
                        }
                    } else {
                        Sample::MISSING
                    };
                }
            }
            let output_index = |pixel| (y + pixel / tile.width) * width + x + pixel % tile.width;
            for (pixel, region) in regions.iter_mut().enumerate() {
                *region = region_scores[output_index(pixel)];
            }
            let anchors = if reject {
                compute_anchors(samples, active.len())
            } else {
                vec![0.0; size]
            };
            let (noise, detail) = if reject {
                (
                    winsor_scale(samples, &anchors, tile.width, rejection.sigma_multiplier),
                    detail_allowance(samples, &anchors, regions, rejection.sky, active.len()),
                )
            } else {
                (0.0, 0.0)
            };
            for (pixel, row) in samples.iter().enumerate() {
                let output = output_index(pixel);
                let region = regions[pixel];
                let anchor = anchors[pixel];
                let threshold = rejection.sigma_multiplier * noise
                    + detail * region * (anchor - rejection.sky).max(0.0);
                deviations.clear();
                if reject {
                    deviations.extend(row.iter().map(|sample| {
                        if sample.area > 0.0 {
                            (sample.value - anchor).abs()
                        } else {
                            f32::INFINITY
                        }
                    }));
                }
                let rejected = deviations
                    .iter()
                    .filter(|value| value.is_finite() && **value > threshold)
                    .count();
                let apply = rejected > 0 && rejected as f32 <= gate;
                // A pixel covered by real samples keeps the closest one if all are outliers.
                let closest = if apply {
                    deviations
                        .iter()
                        .enumerate()
                        .min_by(|(_, a), (_, b)| a.total_cmp(b))
                        .map(|(i, _)| i)
                } else {
                    None
                };
                let mut pixel_sum = 0.0;
                let mut weight_sum = 0.0;
                let mut weight_square_sum = 0.0;
                for (column, sample) in row.iter().enumerate() {
                    if sample.area == 0.0
                        || (apply && Some(column) != closest && deviations[column] > threshold)
                    {
                        continue;
                    }
                    let weight =
                        weights.weight_at(active[column].0, region)? * f64::from(sample.area);
                    pixel_sum += weight * f64::from(sample.value);
                    weight_sum += weight;
                    weight_square_sum += weight * weight;
                    coverage[output] += sample.area;
                    frame_count[output] += 1;
                }
                if weight_sum > 0.0 {
                    image[output] = (pixel_sum / weight_sum) as f32;
                    neff[output] = (weight_sum * weight_sum / weight_square_sum) as f32;
                }
            }
        }
        rows_done(tile_height);
    }

    Ok(())
}

#[derive(Clone, Copy)]
struct Sample {
    value: f32,
    area: f32,
}

impl Sample {
    const MISSING: Self = Self {
        value: 0.0,
        area: 0.0,
    };
}

fn compute_anchors(samples: &[Vec<Sample>], frame_count: usize) -> Vec<f32> {
    let count = frame_count.min(ANCHOR_FRAMES);
    let indices: Vec<_> = (0..count)
        .map(|index| index * (frame_count - 1) / (count - 1))
        .collect();
    let mut values = Vec::with_capacity(count);
    samples
        .iter()
        .map(|row| {
            values.clear();
            values.extend(indices.iter().filter_map(|&index| {
                let sample = row[index];
                (sample.area > 0.0).then_some(sample.value)
            }));
            select_median(&mut values).unwrap_or(0.0)
        })
        .collect()
}

fn winsor_scale(samples: &[Vec<Sample>], anchors: &[f32], width: usize, sigma: f32) -> f32 {
    let mut stds = Vec::new();
    let sampled = samples
        .iter()
        .enumerate()
        .filter(|(pixel, _)| (pixel / width) % 3 == 0 && (pixel % width) % 3 == 0);
    for (_, row) in sampled.clone() {
        let (mut count, mut mean, mut m2) = (0, 0.0_f64, 0.0);
        for sample in row.iter().filter(|sample| sample.area > 0.0) {
            count += 1;
            let value = f64::from(sample.value);
            let delta = value - mean;
            mean += delta / f64::from(count);
            m2 += delta * (value - mean);
        }
        if m2 > 0.0 {
            stds.push((m2 / f64::from(count)).sqrt() as f32);
        }
    }
    let limit = select_median(&mut stds).map_or(f32::INFINITY, |value| value * sigma.max(2.0));
    let (mut total, mut count) = (0.0_f64, 0);
    for (pixel, row) in sampled {
        for sample in row.iter().filter(|sample| sample.area > 0.0) {
            total += f64::from((sample.value - anchors[pixel]).abs().min(limit));
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        (1.15 * total / f64::from(count)) as f32
    }
}

fn detail_allowance(
    samples: &[Vec<Sample>],
    anchors: &[f32],
    regions: &[f32],
    sky: f32,
    frames: usize,
) -> f32 {
    let mut percentiles = Vec::with_capacity(frames);
    let mut ratios = Vec::new();
    for index in 0..frames {
        ratios.clear();
        for ((row, &anchor), &region) in samples.iter().zip(anchors).zip(regions) {
            let sample = row[index];
            if region > 0.5 && anchor > sky && sample.area > 0.0 {
                let ratio = (sample.value - anchor).abs() / (anchor - sky);
                if ratio.is_finite() {
                    ratios.push(ratio);
                }
            }
        }
        if ratios.len() >= 5 {
            ratios.sort_unstable_by(f32::total_cmp);
            let rank = (ratios.len() - 1) as f32 * 0.99;
            let (lo, hi) = (ratios[rank.floor() as usize], ratios[rank.ceil() as usize]);
            percentiles.push(lo + (hi - lo) * rank.fract());
        }
    }
    select_median(&mut percentiles).unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{DrizzleConfig, PwsConfig};
    use crate::geometry::SimilarityTransform;
    use crate::pws::{FrameMetrics, derive_pws_weights};

    #[test]
    fn winsor_rejects_a_hot_pixel_beyond_the_first_tile() -> BackendResult<()> {
        let mut frames = vec![Frame::filled(70, 69, 1.0)?; 6];
        let hot_index = 66 * 70 + 67;
        let mut pixels = frames[0].pixels().to_vec();
        pixels[hot_index] = 50.0;
        frames[5] = Frame::new(70, 69, pixels)?;
        let transforms = [Some(SimilarityTransform::identity()); 6];
        let plan = super::super::plan_similarity_canvas(frames[0].shape(), &transforms, 1.0, 0.0)?;
        let metrics = [FrameMetrics {
            fwhm: 1.0,
            noise: 1.0,
            flux: None,
        }; 6];
        let weights = derive_pws_weights(&metrics, PwsConfig::default())?;
        let config = DrizzleConfig {
            scale: 1.0,
            pixfrac: 1.0,
            margin: 0.0,
        };
        let regions = vec![0.0; plan.width * plan.height];
        let mut progress = Vec::new();
        let combined = stack_winsorized(
            SimilarityDrizzleRequest {
                frames: &frames,
                transforms: &transforms,
                region_scores: &regions,
                weights: &weights,
                config,
            },
            plan,
            RejectionConfig::default(),
            0,
            &[],
            |done, total| progress.push((done, total)),
        )?;
        assert_eq!(progress.first(), Some(&(0, plan.height)));
        assert_eq!(progress.last(), Some(&(plan.height, plan.height)));
        assert!(progress.windows(2).all(|pair| pair[0].0 < pair[1].0));
        assert_eq!(combined.image.pixels()[hot_index], 1.0);
        assert_eq!(combined.frame_count[hot_index], 5);
        assert_eq!(combined.neff[hot_index], 5.0);
        assert_eq!(combined.frame_count[hot_index - 1], 6);
        Ok(())
    }
}
