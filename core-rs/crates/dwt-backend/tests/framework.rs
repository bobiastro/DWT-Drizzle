use dwt_backend::{
    AlignedStackRequest, Backend, BackendResult, DrizzleConfig, DrizzleRequest, Frame,
    FrameMetrics, PwsConfig, RejectionConfig, RustBackend, Shift, SimilarityDrizzleRequest,
    SimilarityTransform, derive_pws_weights,
};

const RELATIVE_TOLERANCE: f64 = 1.0e-5;

fn uniform_frame(width: usize, height: usize, value: f32) -> BackendResult<Frame> {
    Frame::filled(width, height, value)
}

#[test]
fn pws_weights_follow_detail_and_background_terms() -> BackendResult<()> {
    let metrics = [
        FrameMetrics {
            noise: 1.0,
            fwhm: 1.0,
            flux: Some(1.0),
        },
        FrameMetrics {
            noise: 2.0,
            fwhm: 2.0,
            flux: Some(1.0),
        },
    ];
    let weights = derive_pws_weights(metrics.as_slice(), PwsConfig::default())?;
    let detail = weights.normalized_weights_at(1.0)?;
    let background = weights.normalized_weights_at(0.0)?;
    assert!(detail[0] > detail[1]);
    assert!(background[0] > background[1]);
    Ok(())
}

#[test]
fn pws_weight_matches_reference_formula() -> BackendResult<()> {
    let metrics = [
        FrameMetrics {
            noise: 1.0,
            fwhm: 1.2,
            flux: Some(1.5),
        },
        FrameMetrics {
            noise: 2.0,
            fwhm: 2.4,
            flux: Some(0.5),
        },
    ];
    let config = PwsConfig {
        clarity_exponent: 20.0,
        snr_exponent: 2.0,
        detail_snr: 0.75,
        snr_clip: 3.0,
        flux_power: 2.0,
        sharp_weight: 0.5,
    };
    let weights = derive_pws_weights(metrics.as_slice(), config)?;

    for region_score in [0.0_f32, 0.25, 0.5, 0.75, 1.0] {
        let clarity = [1.0_f32, 0.5];
        let snr = [1.0_f32, 0.5];
        let flux = [1.5_f32, 0.5];
        let mut expected: Vec<f64> = clarity
            .iter()
            .map(|value| f64::from(*value))
            .zip(snr.into_iter().map(f64::from))
            .zip(flux.into_iter().map(f64::from))
            .map(|((clarity, snr), flux)| {
                let region_score = f64::from(region_score);
                ((20.0 * region_score + 0.5) * clarity.ln()
                    + (2.0 * (1.0 - region_score) + 0.75 * region_score) * snr.ln()
                    + 2.0 * flux.ln())
                .exp()
            })
            .collect();
        let mean = expected.iter().sum::<f64>() / expected.len() as f64;
        for value in &mut expected {
            *value /= mean;
        }
        let actual = weights.normalized_weights_at(region_score)?;
        for (actual_value, expected_value) in actual.iter().zip(expected) {
            let difference = (actual_value - expected_value).abs();
            assert!(
                difference <= RELATIVE_TOLERANCE * expected_value.abs().max(1.0),
                "R={region_score}: actual={actual_value}, expected={expected_value}"
            );
        }
    }
    Ok(())
}

#[test]
fn aligned_stack_averages_equal_quality_frames() -> BackendResult<()> {
    let frames = [uniform_frame(2, 2, 10.0)?, uniform_frame(2, 2, 12.0)?];
    let metrics = [
        FrameMetrics {
            noise: 1.0,
            fwhm: 1.0,
            flux: Some(1.0),
        },
        FrameMetrics {
            noise: 1.0,
            fwhm: 1.0,
            flux: Some(1.0),
        },
    ];
    let weights = derive_pws_weights(metrics.as_slice(), PwsConfig::default())?;
    let region = vec![0.5; 4];
    let result = RustBackend.stack_aligned(AlignedStackRequest {
        frames: frames.as_slice(),
        region_scores: region.as_slice(),
        weights: &weights,
        rejection: RejectionConfig::default(),
    })?;
    for value in result.image.pixels() {
        assert!((*value - 11.0).abs() < RELATIVE_TOLERANCE as f32);
    }
    Ok(())
}

#[test]
fn aligned_stack_rejects_a_single_outlier() -> BackendResult<()> {
    let frames = [
        uniform_frame(2, 2, 10.0)?,
        uniform_frame(2, 2, 10.0)?,
        uniform_frame(2, 2, 10.0)?,
        uniform_frame(2, 2, 100.0)?,
    ];
    let metrics = vec![
        FrameMetrics {
            noise: 1.0,
            fwhm: 1.0,
            flux: Some(1.0),
        };
        4
    ];
    let weights = derive_pws_weights(metrics.as_slice(), PwsConfig::default())?;
    let region = vec![0.0; 4];
    let result = RustBackend.stack_aligned(AlignedStackRequest {
        frames: frames.as_slice(),
        region_scores: region.as_slice(),
        weights: &weights,
        rejection: RejectionConfig::default(),
    })?;
    for value in result.image.pixels() {
        assert!((*value - 10.0).abs() < RELATIVE_TOLERANCE as f32);
    }
    assert!(result.rejected_fraction > 0.0);
    Ok(())
}

#[test]
fn aligned_stack_keeps_real_detail_variation() -> BackendResult<()> {
    let frames = [
        uniform_frame(2, 2, 10.0)?,
        uniform_frame(2, 2, 11.0)?,
        uniform_frame(2, 2, 12.0)?,
        uniform_frame(2, 2, 13.0)?,
    ];
    let metrics = vec![
        FrameMetrics {
            noise: 1.0,
            fwhm: 1.0,
            flux: Some(1.0),
        };
        frames.len()
    ];
    let weights = derive_pws_weights(metrics.as_slice(), PwsConfig::default())?;
    let region = vec![1.0; 4];
    let result = RustBackend.stack_aligned(AlignedStackRequest {
        frames: frames.as_slice(),
        region_scores: region.as_slice(),
        weights: &weights,
        rejection: RejectionConfig::default(),
    })?;
    let value = result.image.pixels()[0];
    assert!(value > 10.0 && value < 14.0);
    assert!(result.coverage[0] >= 2);
    Ok(())
}

#[test]
fn drizzle_identity_preserves_a_constant_image() -> BackendResult<()> {
    let frame = uniform_frame(3, 2, 7.0)?;
    let frames = [frame];
    let shifts = [Some(Shift::new(0.0, 0.0)?)];
    let metrics = [FrameMetrics {
        noise: 1.0,
        fwhm: 1.0,
        flux: Some(1.0),
    }];
    let weights = derive_pws_weights(metrics.as_slice(), PwsConfig::default())?;
    let region = vec![0.5; 6];
    let result = RustBackend.drizzle(DrizzleRequest {
        frames: frames.as_slice(),
        shifts: shifts.as_slice(),
        region_scores: region.as_slice(),
        weights: &weights,
        config: DrizzleConfig {
            scale: 1.0,
            pixfrac: 1.0,
            margin: 0.0,
        },
    })?;
    assert_eq!(result.image.shape().width, 3);
    assert_eq!(result.image.shape().height, 2);
    for value in result.image.pixels() {
        assert!((*value - 7.0).abs() < RELATIVE_TOLERANCE as f32);
    }
    assert!(result.frame_count.iter().all(|count| *count == 1));
    Ok(())
}

#[test]
fn similarity_drizzle_rotates_a_marker_into_place() -> BackendResult<()> {
    let mut pixels = vec![0.0_f32; 4 * 4];
    pixels[5] = 10.0;
    let frame = Frame::new(4, 4, pixels)?;
    let shape = frame.shape();
    let frames = [frame];
    let transforms = [Some(SimilarityTransform::new(
        1.0,
        std::f64::consts::FRAC_PI_2,
        0.0,
        0.0,
    )?)];
    let metrics = [FrameMetrics {
        noise: 1.0,
        fwhm: 1.0,
        flux: Some(1.0),
    }];
    let weights = derive_pws_weights(metrics.as_slice(), PwsConfig::default())?;
    let plan =
        dwt_backend::drizzle::plan_similarity_canvas(shape, transforms.as_slice(), 1.0, 0.0)?;
    let region = vec![0.5; plan.width * plan.height];
    let result = RustBackend.drizzle_similarity(SimilarityDrizzleRequest {
        frames: frames.as_slice(),
        transforms: transforms.as_slice(),
        region_scores: region.as_slice(),
        weights: &weights,
        config: DrizzleConfig {
            scale: 1.0,
            pixfrac: 1.0,
            margin: 0.0,
        },
    })?;
    let peak_index = result
        .image
        .pixels()
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
        .unwrap_or(0);
    let peak_x = peak_index % result.image.width();
    let peak_y = peak_index / result.image.width();
    assert!(peak_x > 0);
    assert!(peak_y < result.image.height());
    Ok(())
}
