use std::hint::black_box;
use std::time::Instant;

use dwt_backend::drizzle::plan_shift_canvas;
use dwt_backend::{
    AlignedStackRequest, Backend, DrizzleConfig, DrizzleRequest, Frame, FrameMetrics, PwsConfig,
    RejectionConfig, RustBackend, Shift, derive_pws_weights,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let frame_count = parse_arg(args.next(), 16)?;
    let width = parse_arg(args.next(), 512)?;
    let height = parse_arg(args.next(), 512)?;
    let repetitions = parse_arg(args.next(), 3)?;

    let mut frames = Vec::with_capacity(frame_count);
    let mut metrics = Vec::with_capacity(frame_count);
    for frame_index in 0..frame_count {
        let mut pixels = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                let value = ((x * 17 + y * 31 + frame_index * 43) % 257) as f32 * 0.01
                    + frame_index as f32 * 0.02;
                pixels.push(value);
            }
        }
        frames.push(Frame::new(width, height, pixels)?);
        metrics.push(FrameMetrics {
            noise: 1.0 + (frame_index % 7) as f32 * 0.05,
            fwhm: 1.2 + (frame_index % 5) as f32 * 0.1,
            flux: Some(1.0 + (frame_index % 3) as f32 * 0.1),
        });
    }

    let mut region_scores = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            region_scores.push(((x * 37 + y * 17) % 1000) as f32 / 1000.0);
        }
    }

    let weights_start = Instant::now();
    let weights = derive_pws_weights(metrics.as_slice(), PwsConfig::default())?;
    let weights_ms = weights_start.elapsed().as_secs_f64() * 1000.0;
    let backend = RustBackend;
    let mut stack_ms = Vec::with_capacity(repetitions);
    let mut rejection_stack_ms = Vec::with_capacity(repetitions);
    for _ in 0..repetitions {
        let start = Instant::now();
        let result = backend.stack_aligned(AlignedStackRequest {
            frames: frames.as_slice(),
            region_scores: region_scores.as_slice(),
            weights: &weights,
            rejection: RejectionConfig {
                sigma_multiplier: 0.0,
                ..RejectionConfig::default()
            },
        })?;
        black_box(&result);
        stack_ms.push(start.elapsed().as_secs_f64() * 1000.0);

        let start = Instant::now();
        let result = backend.stack_aligned(AlignedStackRequest {
            frames: frames.as_slice(),
            region_scores: region_scores.as_slice(),
            weights: &weights,
            rejection: RejectionConfig::default(),
        })?;
        black_box(&result);
        rejection_stack_ms.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    let shifts: Vec<Option<Shift>> = (0..frame_count)
        .map(|index| Shift::new((index % 3) as f32 * 0.25, (index % 5) as f32 * 0.2).map(Some))
        .collect::<Result<Vec<_>, _>>()?;
    let drizzle_config = DrizzleConfig {
        scale: 1.0,
        pixfrac: 1.0,
        margin: 0.0,
    };
    let canvas = plan_shift_canvas(frames[0].shape(), shifts.as_slice(), drizzle_config)?;
    let mut canvas_scores = Vec::with_capacity(canvas.width * canvas.height);
    for y in 0..canvas.height {
        for x in 0..canvas.width {
            canvas_scores.push(((x * 37 + y * 17) % 1000) as f32 / 1000.0);
        }
    }
    let drizzle_start = Instant::now();
    let drizzle = backend.drizzle(DrizzleRequest {
        frames: frames.as_slice(),
        shifts: shifts.as_slice(),
        region_scores: canvas_scores.as_slice(),
        weights: &weights,
        config: drizzle_config,
    })?;
    black_box(&drizzle);
    let drizzle_ms = drizzle_start.elapsed().as_secs_f64() * 1000.0;

    println!(
        "{{\"frames\":{frame_count},\"width\":{width},\"height\":{height},\
         \"repetitions\":{repetitions},\"weights_ms\":{weights_ms:.3},\
         \"stack_ms\":{stack_ms:?},\"rejection_stack_ms\":{rejection_stack_ms:?},\
         \"drizzle_ms\":{drizzle_ms:.3}}}"
    );
    Ok(())
}

fn parse_arg(value: Option<String>, default: usize) -> Result<usize, Box<dyn std::error::Error>> {
    value
        .map(|text| {
            text.parse::<usize>()
                .map_err(|error| format!("invalid integer {text:?}: {error}").into())
        })
        .unwrap_or(Ok(default))
}
