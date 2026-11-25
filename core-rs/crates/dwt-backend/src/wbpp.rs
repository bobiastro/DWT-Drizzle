//! WBPP output discovery and request generation.
//!
//! WBPP produces calibrated light frames, registered previews, and XDRZ
//! geometry files. This module locates those artifacts without requiring the
//! Python application or a PixInsight process instance.

mod quality;

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::config::DrizzleConfig;
use crate::frame::Frame;
use crate::geometry::SimilarityTransform;
use crate::pws::{FrameMetrics, estimate_noise, select_median};
use crate::register::{
    RegistrationConfig, Star, detect_stars, estimate_fwhm, fit_field_correction, guide_centroid,
    match_stars, register_sequence,
};
use crate::rfield::RFieldMode;

/// Errors emitted by the WBPP adapter.
#[derive(Debug)]
pub enum WbppError {
    Io(std::io::Error),
    Json(serde_json::Error),
    Parse(String),
    Message(String),
}

impl Display for WbppError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "filesystem error: {error}"),
            Self::Json(error) => write!(formatter, "JSON error: {error}"),
            Self::Parse(error) => formatter.write_str(error),
            Self::Message(message) => formatter.write_str(message),
        }
    }
}

impl Error for WbppError {}

impl From<std::io::Error> for WbppError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for WbppError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

/// Discovered WBPP artifacts, grouped by source file stem.
#[derive(Debug, Clone, Serialize)]
pub struct WbppScan {
    pub directory: PathBuf,
    pub calibrated: Vec<PathBuf>,
    pub registered: Vec<PathBuf>,
    pub xdrz: Vec<PathBuf>,
    pub masters: Vec<PathBuf>,
    pub frames_with_xdrz: usize,
    pub matched_frames: usize,
    pub missing_xdrz: Vec<String>,
}

/// One calibrated frame with the WBPP products resolved by frame stem.
#[derive(Debug, Clone, Serialize)]
pub struct WbppFramePair {
    pub source: PathBuf,
    pub stem: String,
    pub registered: Option<PathBuf>,
    pub xdrz: Option<PathBuf>,
}

/// A single WBPP `.xdrz` record.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct XdrzRecord {
    pub path: PathBuf,
    pub source_image: String,
    pub stem: String,
    pub alignment_matrix: [[f64; 3]; 3],
    pub reference_width: Option<u32>,
    pub reference_height: Option<u32>,
}

/// Request describing a WBPP result that can be submitted to the Rust backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WbppCommand {
    pub schema: String,
    pub source: String,
    pub calibrated_directory: PathBuf,
    pub registered_directory: Option<PathBuf>,
    pub output_directory: PathBuf,
    pub frames: Vec<WbppFrame>,
    #[serde(default)]
    pub raster: WbppRaster,
    pub options: WbppOptions,
}

/// One WBPP frame and its geometry sidecar.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WbppFrame {
    pub enabled: bool,
    pub source: PathBuf,
    pub registered: Option<PathBuf>,
    pub xdrz: Option<PathBuf>,
    pub has_xdrz: bool,
    #[serde(default)]
    pub fwhm: Option<f32>,
    #[serde(default)]
    pub noise: Option<f32>,
    #[serde(default)]
    pub signal: Option<f32>,
    #[serde(default)]
    pub exposure: Option<f32>,
    #[serde(default)]
    pub observation_timestamp_ms: Option<i64>,
    #[serde(default)]
    pub raster: Option<PathBuf>,
    #[serde(default)]
    pub background: Option<f32>,
}

/// PJSR-exported raster layout used to hand pixels to the Rust backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WbppRaster {
    pub directory: PathBuf,
    pub file_extension: String,
    pub width: usize,
    pub height: usize,
    pub dtype: String,
}

impl Default for WbppRaster {
    fn default() -> Self {
        Self {
            directory: PathBuf::from("DWT/rasters"),
            file_extension: "f32le".to_string(),
            width: 0,
            height: 0,
            dtype: "float32-le".to_string(),
        }
    }
}

/// Result of running the calibrated-frame PWS path.
#[derive(Debug, Clone, Serialize)]
pub struct WbppRunResult {
    pub status: String,
    pub frame_count: usize,
    pub raster_shape: [usize; 2],
    pub output_raster: PathBuf,
    pub output_neff: PathBuf,
    pub output_coverage: PathBuf,
    #[serde(default)]
    pub registered_frames: usize,
    pub integrated_frames: usize,
    pub quality_rejected_frames: usize,
    pub field_corrected_frames: usize,
    pub field_median_residual_before: Option<f32>,
    pub field_median_residual_after: Option<f32>,
    #[serde(default)]
    pub xdrz_transforms: usize,
    pub triangle_frames: usize,
    pub chained_frames: usize,
    pub adjusted_frames: usize,
    pub sharp_reference_frames: usize,
    pub warnings: Vec<String>,
}

/// Backend controls copied from a WBPP pipeline invocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WbppOptions {
    pub clarity_exponent: f32,
    pub snr_exponent: f32,
    #[serde(default)]
    pub detail_snr: f32,
    pub snr_clip: f32,
    pub flux_power: f32,
    pub sharp_weight: f32,
    pub rejection_sigma: f32,
    pub rejection_fraction: f32,
    pub drizzle_scale: f32,
    pub drizzle_pixfrac: f32,
    #[serde(default = "default_registration_stars")]
    pub registration_stars: usize,
    #[serde(default)]
    pub fwhm_max: f32,
    #[serde(default = "default_auto_reject_frames")]
    pub auto_reject_frames: bool,
    #[serde(default = "default_field_correction")]
    pub field_correction: bool,
    #[serde(default)]
    pub stack_workers: usize,
    #[serde(default = "default_allow_missing_xdrz")]
    /// Retained for compatibility with older request files. Geometry sidecars are optional.
    pub allow_missing_xdrz: bool,
    #[serde(default = "default_env_window_multiplier")]
    pub env_window_multiplier: f32,
    #[serde(default = "default_env_epsilon")]
    pub env_epsilon: f32,
    #[serde(default = "default_env_power")]
    pub env_power: f32,
    #[serde(default)]
    pub r_floor: f32,
    #[serde(default)]
    pub r_blur: f32,
    #[serde(default)]
    pub r_mode: RFieldMode,
    #[serde(default)]
    pub r_window: f32,
    #[serde(default = "default_r_contrast")]
    pub r_contrast: f32,
    #[serde(default)]
    pub ref_frames: usize,
}

impl Default for WbppOptions {
    fn default() -> Self {
        Self {
            clarity_exponent: 20.0,
            snr_exponent: 2.0,
            detail_snr: 0.0,
            snr_clip: 3.0,
            flux_power: 0.0,
            sharp_weight: 0.0,
            rejection_sigma: 3.5,
            rejection_fraction: 1.0 / 3.0,
            drizzle_scale: 2.0,
            drizzle_pixfrac: 1.0,
            registration_stars: default_registration_stars(),
            fwhm_max: 0.0,
            auto_reject_frames: default_auto_reject_frames(),
            field_correction: default_field_correction(),
            stack_workers: 0,
            allow_missing_xdrz: true,
            env_window_multiplier: default_env_window_multiplier(),
            env_epsilon: default_env_epsilon(),
            env_power: default_env_power(),
            r_floor: 0.0,
            r_blur: 0.0,
            r_mode: RFieldMode::Max,
            r_window: 0.0,
            r_contrast: default_r_contrast(),
            ref_frames: 0,
        }
    }
}

impl WbppOptions {
    fn drizzle_config(&self) -> DrizzleConfig {
        DrizzleConfig {
            scale: self.drizzle_scale,
            pixfrac: self.drizzle_pixfrac,
            ..DrizzleConfig::default()
        }
    }
}

fn default_allow_missing_xdrz() -> bool {
    true
}

fn default_registration_stars() -> usize {
    240
}

fn default_auto_reject_frames() -> bool {
    true
}

fn default_field_correction() -> bool {
    true
}

fn default_r_contrast() -> f32 {
    3.0
}

/// Output plan returned to a WBPP-side script.
#[derive(Debug, Clone, Serialize)]
pub struct WbppPlan {
    pub status: String,
    pub frame_count: usize,
    pub matched_xdrz: usize,
    pub missing_xdrz: usize,
    pub unsupported: Vec<String>,
    pub next: Vec<String>,
}

/// Scans a WBPP output root for calibrated frames, registered frames, sidecars,
/// and master products.
pub fn scan_wbpp_dir(root: &Path) -> Result<WbppScan, WbppError> {
    if !root.is_dir() {
        return Err(WbppError::Message(format!(
            "WBPP output directory does not exist: {}",
            root.display()
        )));
    }
    let mut calibrated = Vec::new();
    let mut registered = Vec::new();
    let mut xdrz = Vec::new();
    let mut masters = Vec::new();
    walk(root, &mut |path| {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        let extension = path
            .extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        if extension == "xdrz" {
            xdrz.push(path.to_path_buf());
        } else if is_image_extension(&extension) {
            if name.starts_with("master") {
                masters.push(path.to_path_buf());
            } else if path_has_segment(path, "registered") {
                registered.push(path.to_path_buf());
            } else {
                calibrated.push(path.to_path_buf());
            }
        }
    })?;
    calibrated.sort();
    registered.sort();
    xdrz.sort();
    masters.sort();

    let mut geometry_by_stem = BTreeMap::new();
    for path in &xdrz {
        if let Ok(record) = parse_xdrz(path) {
            geometry_by_stem.insert(record.stem.clone(), record);
        }
    }
    let mut matched_frames = 0;
    let mut missing_xdrz = Vec::new();
    for frame in &calibrated {
        let stem = file_stem(frame);
        if geometry_for_frame(&geometry_by_stem, &stem).is_some() {
            matched_frames += 1;
        } else {
            missing_xdrz.push(stem);
        }
    }

    Ok(WbppScan {
        directory: root.to_path_buf(),
        frames_with_xdrz: geometry_by_stem.len(),
        calibrated,
        registered,
        xdrz,
        masters,
        matched_frames,
        missing_xdrz,
    })
}

/// Builds a deterministic backend plan from a WBPP output directory.
pub fn plan_from_wbpp(root: &Path) -> Result<WbppPlan, WbppError> {
    let scan = scan_wbpp_dir(root)?;
    let mut unsupported = Vec::new();
    if scan.calibrated.is_empty() {
        unsupported.push("no calibrated light frames were found".to_string());
    }
    Ok(WbppPlan {
        status: if unsupported.is_empty() {
            "ready".to_string()
        } else {
            "partial".to_string()
        },
        frame_count: scan.calibrated.len(),
        matched_xdrz: scan.matched_frames,
        missing_xdrz: scan.missing_xdrz.len(),
        unsupported,
        next: vec![
            "enable the DWT Pipeline Builder in WBPP to export calibrated rasters and run PWS"
                .to_string(),
        ],
    })
}

/// Builds a request template from files in a WBPP output directory.
pub fn request_from_wbpp(root: &Path) -> Result<WbppCommand, WbppError> {
    let scan = scan_wbpp_dir(root)?;
    let mut registered_by_stem: BTreeMap<String, PathBuf> = BTreeMap::new();
    for path in &scan.registered {
        let stem = registered_stem(path);
        registered_by_stem
            .entry(stem.clone())
            .or_insert_with(|| path.clone());
        registered_by_stem
            .entry(without_registration_suffix(&stem).to_string())
            .or_insert_with(|| path.clone());
    }
    let xdrz_by_stem: BTreeMap<String, PathBuf> = scan
        .xdrz
        .iter()
        .filter_map(|path| {
            parse_xdrz(path)
                .ok()
                .map(|record| (record.stem, path.clone()))
        })
        .collect();
    let frames = scan
        .calibrated
        .iter()
        .map(|source| {
            let stem = file_stem(source);
            WbppFrame {
                enabled: true,
                source: source.clone(),
                registered: registered_by_stem.get(&stem).cloned(),
                xdrz: geometry_path_for(&xdrz_by_stem, &stem),
                has_xdrz: geometry_path_for(&xdrz_by_stem, &stem).is_some(),
                fwhm: None,
                noise: None,
                signal: None,
                exposure: None,
                observation_timestamp_ms: None,
                raster: None,
                background: None,
            }
        })
        .collect();
    Ok(WbppCommand {
        schema: "dwt-wbpp-command/1".to_string(),
        source: "PixInsight/WBPP".to_string(),
        calibrated_directory: root.to_path_buf(),
        registered_directory: Some(root.join("registered")),
        output_directory: root.join("DWT"),
        frames,
        raster: WbppRaster {
            directory: root.join("DWT/rasters"),
            ..WbppRaster::default()
        },
        options: WbppOptions::default(),
    })
}

/// Validates a request before handing it to a compute backend.
pub fn validate_request(request: &WbppCommand) -> Result<(), WbppError> {
    if request.schema != "dwt-wbpp-command/1" {
        return Err(WbppError::Message(format!(
            "unsupported request schema: {}",
            request.schema
        )));
    }
    if request.frames.is_empty() {
        return Err(WbppError::Message(
            "WBPP request contains no frames".to_string(),
        ));
    }
    request
        .options
        .drizzle_config()
        .validate()
        .map_err(|error| WbppError::Message(error.to_string()))?;
    let options = &request.options;
    if !(8..=4096).contains(&options.registration_stars)
        || !options.fwhm_max.is_finite()
        || options.fwhm_max < 0.0
    {
        return Err(WbppError::Message(
            "invalid registration star count or FWHM limit".into(),
        ));
    }
    if options.stack_workers > crate::drizzle::MAX_STACK_WORKERS {
        return Err(WbppError::Message("invalid Drizzle worker limit".into()));
    }
    if !options.r_window.is_finite()
        || options.r_window < 0.0
        || !options.r_contrast.is_finite()
        || options.r_contrast <= 0.8
        || !(0.0..=20.0).contains(&options.r_blur)
        || options.ref_frames > 64
    {
        return Err(WbppError::Message(
            "invalid R-field window, contrast, blur, or reference frame count".into(),
        ));
    }
    crate::config::RejectionConfig {
        sigma_multiplier: request.options.rejection_sigma,
        max_fraction: request.options.rejection_fraction,
        sky: 0.0,
    }
    .validate()
    .map_err(|error| WbppError::Message(error.to_string()))?;
    Ok(())
}

/// Runs the calibrated-frame PWS+Drizzle path.
///
/// The raster files must contain unregistered calibrated samples. Geometry is
/// obtained from `xdrz` when available, otherwise measured by Rust star
/// registration.
pub fn run_calibrated_pws(request: &WbppCommand) -> Result<WbppRunResult, WbppError> {
    run_calibrated_pws_with_progress(request, |_, _, _| {})
}

/// Reports progress within each phase. Counts are local to the named phase.
pub fn run_calibrated_pws_with_progress(
    request: &WbppCommand,
    mut progress: impl FnMut(&'static str, usize, usize),
) -> Result<WbppRunResult, WbppError> {
    validate_request(request)?;
    if request.raster.width == 0 || request.raster.height == 0 {
        return Err(WbppError::Message(
            "raster width and height must be positive".to_string(),
        ));
    }
    if request.raster.dtype != "float32-le" {
        return Err(WbppError::Message(format!(
            "unsupported raster dtype: {}",
            request.raster.dtype
        )));
    }
    let shape = crate::frame::ImageShape::new(request.raster.width, request.raster.height)
        .map_err(|error| WbppError::Message(error.to_string()))?;
    let enabled: Vec<&WbppFrame> = request
        .frames
        .iter()
        .filter(|frame| frame.enabled)
        .collect();
    if enabled.is_empty() {
        return Err(WbppError::Message(
            "no enabled WBPP frames in request".to_string(),
        ));
    }
    let frame_bytes = (shape.pixel_count() as u64).saturating_mul(4);
    // Allow scratch storage for decoding and star detection alongside resident frames.
    ensure_memory_available(
        "loading calibrated frames",
        frame_bytes.saturating_mul(enabled.len() as u64 + 3),
    )?;

    let mut seen_paths = std::collections::BTreeSet::new();
    let mut raster_paths = Vec::with_capacity(enabled.len());
    for frame in &enabled {
        let raster_path = frame.raster.clone().unwrap_or_else(|| {
            request.raster.directory.join(format!(
                "{}.{}",
                file_stem(&frame.source),
                request.raster.file_extension
            ))
        });
        if !seen_paths.insert(raster_path.clone()) {
            return Err(WbppError::Message(format!(
                "duplicate raster path in request: {}",
                raster_path.display()
            )));
        }
        raster_paths.push(raster_path);
    }

    let positive = |value: Option<f32>| value.filter(|value| value.is_finite() && *value > 0.0);
    let mut exposures: Vec<f32> = enabled
        .iter()
        .filter_map(|frame| positive(frame.exposure))
        .collect();
    let median_exposure = select_median(&mut exposures).unwrap_or(1.0);
    let registration = RegistrationConfig {
        max_stars: request.options.registration_stars,
        ..RegistrationConfig::default()
    };
    let input_count = enabled.len();
    let mut frames = Vec::with_capacity(input_count);
    let mut metrics = Vec::with_capacity(input_count);
    let mut catalogs = Vec::with_capacity(input_count);
    let mut measured_fwhm = Vec::with_capacity(input_count);
    let mut usable = Vec::with_capacity(input_count);
    let mut warnings = Vec::new();
    progress("Reading and measuring frames", 0, input_count);
    for (index, (frame, raster_path)) in enabled.iter().zip(&raster_paths).enumerate() {
        let mut pixels = read_f32_le(raster_path, shape.pixel_count())?;
        let exposure_scale = median_exposure / positive(frame.exposure).unwrap_or(median_exposure);
        for pixel in &mut pixels {
            *pixel *= exposure_scale;
        }
        let raster = Frame::new(shape.width, shape.height, pixels)
            .map_err(|error| WbppError::Message(error.to_string()))?;
        let stars = detect_stars(&raster, registration)
            .map_err(|error| WbppError::Message(error.to_string()))?;
        // The WBPP PSF measurement is less sensitive to isolated pixel peaks
        // than the local half-maximum estimate. Use the latter as a fallback.
        let fwhm = positive(frame.fwhm).or_else(|| positive(estimate_fwhm(&raster, &stars)));
        let limit = request.options.fwhm_max;
        if limit > 0.0 && !fwhm.is_some_and(|value| value <= limit) {
            let measured =
                fwhm.map_or_else(|| "unavailable".into(), |value| format!("{value:.2} px"));
            warnings.push(format!(
                "Skipping {}: FWHM {measured}, limit {limit:.2} px",
                frame.source.display()
            ));
            progress("Reading and measuring frames", index + 1, input_count);
            continue;
        }
        measured_fwhm.push(fwhm);
        metrics.push(FrameMetrics {
            noise: positive(estimate_noise(&raster))
                .or_else(|| positive(frame.noise.map(|noise| noise * exposure_scale)))
                .unwrap_or(1.0e-30),
            fwhm: fwhm.unwrap_or(1.0),
            flux: frame.signal,
        });
        frames.push(raster);
        catalogs.push(stars);
        usable.push(*frame);
        progress("Reading and measuring frames", index + 1, input_count);
    }
    if usable.is_empty() {
        return Err(WbppError::Message(format!(
            "No frames met the FWHM limit of {:.2} px",
            request.options.fwhm_max
        )));
    }
    let config = crate::config::PwsConfig {
        clarity_exponent: request.options.clarity_exponent,
        snr_exponent: request.options.snr_exponent,
        detail_snr: request.options.detail_snr,
        snr_clip: request.options.snr_clip,
        flux_power: request.options.flux_power,
        sharp_weight: request.options.sharp_weight,
    };
    let seeds: Vec<_> = usable
        .iter()
        .map(|frame| {
            let path = frame.xdrz.as_deref()?;
            match load_xdrz_transform(path) {
                Ok(transform) => Some(transform),
                Err(error) => {
                    warnings.push(format!("Ignoring XDRZ seed: {error}"));
                    None
                }
            }
        })
        .collect();
    let reference_candidates =
        reference_candidates(&measured_fwhm, &catalogs, registration, &seeds);
    let mut reference_index = reference_candidates[0];
    let (mut transforms, mut registration_counts) = register_sequence(
        &catalogs,
        &seeds,
        reference_index,
        registration,
        &mut progress,
    );
    for &candidate in reference_candidates.iter().skip(1).take(2) {
        let registered = transforms.iter().flatten().count();
        if registered * 2 >= usable.len() || catalogs[candidate].len() < registration.min_matches {
            break;
        }
        let (alternative, counts) =
            register_sequence(&catalogs, &seeds, candidate, registration, &mut progress);
        if alternative.iter().flatten().count() > registered {
            transforms = alternative;
            registration_counts = counts;
            reference_index = candidate;
        }
    }
    for (frame, transform) in usable.iter().zip(&transforms) {
        if transform.is_none() {
            warnings.push(format!(
                "Skipping {}: registration could not be verified",
                frame.source.display()
            ));
        }
    }
    let registered_count = transforms.iter().flatten().count();
    let mut corrections = vec![None; frames.len()];
    if request.options.field_correction {
        let guide = &catalogs[reference_index];
        let guide_positions: Vec<_> = guide
            .iter()
            .map(|star| guide_centroid(&frames[reference_index], star))
            .collect();
        progress("Refining field geometry", 0, frames.len());
        for (i, correction) in corrections.iter_mut().enumerate() {
            if i != reference_index {
                *correction = transforms[i].and_then(|transform| {
                    fit_field_correction(
                        &frames[i],
                        &catalogs[i],
                        guide,
                        &guide_positions,
                        transform,
                        registration.match_radius,
                    )
                });
            }
            progress("Refining field geometry", i + 1, frames.len());
        }
    }
    let trail_stars = trail_reference_stars(&catalogs[reference_index]);
    let mut quality_rejected_frames = 0;
    if request.options.auto_reject_frames
        && usable
            .iter()
            .filter(|frame| frame.observation_timestamp_ms.is_some())
            .count()
            >= 5
    {
        progress("Checking frame throughput", 0, 1);
        let reference_fluxes: Vec<_> = trail_stars
            .iter()
            .map(|star| quality::aperture_flux(&frames[reference_index], star))
            .collect();
        let throughputs: Vec<_> = usable
            .iter()
            .zip(&transforms)
            .enumerate()
            .filter_map(|(index, (input, transform))| {
                let timestamp = input.observation_timestamp_ms?;
                let exposure = positive(input.exposure)?;
                let flux = quality::matched_throughput(
                    &frames[index],
                    &catalogs[index],
                    &trail_stars,
                    &reference_fluxes,
                    (*transform)?,
                    registration.match_radius,
                )?;
                Some(quality::Throughput {
                    index,
                    timestamp,
                    exposure,
                    flux,
                })
            })
            .collect();
        let outliers = quality::throughput_outliers(throughputs);
        quality_rejected_frames = outliers.len();
        for (index, fraction) in outliers {
            transforms[index] = None;
            warnings.push(format!(
                "Skipping {}: matched-star throughput was {:.0}% of nearby frames",
                usable[index].source.display(),
                fraction * 100.0
            ));
        }
        progress("Checking frame throughput", 1, 1);
    }
    let mut image_reference_index = reference_index;
    if transforms.iter().any(Option::is_none) {
        image_reference_index = if transforms[reference_index].is_some() {
            transforms[..reference_index].iter().flatten().count()
        } else {
            0
        };
        let retained: Vec<_> = transforms.iter().map(Option::is_some).collect();
        retain_frames(&mut frames, &retained);
        retain_frames(&mut catalogs, &retained);
        retain_frames(&mut metrics, &retained);
        retain_frames(&mut measured_fwhm, &retained);
        retain_frames(&mut transforms, &retained);
        retain_frames(&mut corrections, &retained);
    }
    let mut registered_fwhm: Vec<_> = measured_fwhm.iter().flatten().copied().collect();
    let median_fwhm = select_median(&mut registered_fwhm).unwrap_or(1.0);
    for (metric, measured) in metrics.iter_mut().zip(measured_fwhm) {
        metric.fwhm = measured.unwrap_or(median_fwhm);
    }
    let weights = crate::pws::derive_pws_weights(metrics.as_slice(), config)
        .map_err(|error| WbppError::Message(error.to_string()))?;
    let drizzle = request.options.drizzle_config();
    let output_scale = f64::from(drizzle.scale);
    let output_plan = crate::drizzle::plan_similarity_canvas(
        shape,
        transforms.as_slice(),
        output_scale,
        f64::from(drizzle.margin),
    )
    .and_then(|plan| crate::drizzle::pad_canvas_for_field(plan, &corrections))
    .map_err(|error| WbppError::Message(error.to_string()))?;
    let output_shape = crate::frame::ImageShape::new(output_plan.width, output_plan.height)
        .map_err(|error| WbppError::Message(error.to_string()))?;
    // The reference, R-field filters, integration image, and diagnostic arrays
    // all operate on the enlarged canvas. Resident input memory is already
    // reflected in the refreshed available-memory reading.
    ensure_memory_available(
        "allocating the Drizzle canvas",
        (output_shape.pixel_count() as u64).saturating_mul(32),
    )?;
    let mut sharp_frames: Vec<_> = transforms
        .iter()
        .enumerate()
        .filter_map(|(i, transform)| transform.map(|t| (i, t)))
        .collect();
    sharp_frames.sort_by(|&(a, _), &(b, _)| metrics[a].fwhm.total_cmp(&metrics[b].fwhm));
    let sharp_count = match request.options.ref_frames {
        0 => 8,
        count => count,
    };
    sharp_frames.truncate(sharp_count);
    let reference_for_r = crate::drizzle::stack_sharp_reference(
        &frames,
        &sharp_frames,
        &corrections,
        output_plan,
        drizzle,
        |done, total| progress("Building sharp reference (rows)", done, total),
    )
    .map_err(|error| WbppError::Message(error.to_string()))?;
    progress("Measuring reference and building R field", 0, 1);
    let sharp_stars = detect_stars(&reference_for_r, registration)
        .map_err(|error| WbppError::Message(error.to_string()))?;
    let sharp_fwhm = estimate_fwhm(&reference_for_r, &sharp_stars)
        .unwrap_or(metrics[image_reference_index].fwhm * output_scale as f32);
    let native_sigma = median_fwhm / 2.3548;
    let trail_variances: Vec<_> = frames
        .iter()
        .zip(&catalogs)
        .zip(&transforms)
        .map(|((frame, catalog), transform)| {
            let pairs = match_stars(
                &trail_stars,
                catalog,
                (*transform)?,
                registration.match_radius,
            );
            let positions = pairs
                .into_iter()
                .map(|(index, _)| (catalog[index].x as usize, catalog[index].y as usize));
            crate::rfield::measure_trail_variance(frame, positions, native_sigma)
        })
        .collect();
    let growth_in_psf = crate::rfield::trail_growth_factor(
        &trail_variances,
        sharp_frames.iter().map(|&(index, _)| index),
        native_sigma,
    );
    let region_field = crate::rfield::build_drizzle_r_field(
        &reference_for_r,
        crate::rfield::RFieldConfig {
            sigma_psf: sharp_fwhm / 2.3548,
            window_multiplier: request.options.env_window_multiplier,
            noise_epsilon: request.options.env_epsilon,
            shape_power: request.options.env_power,
            floor: request.options.r_floor,
        },
        request.options.r_mode,
        request.options.r_window,
        request.options.r_contrast,
        request.options.r_blur,
        growth_in_psf,
    )
    .map_err(|error| WbppError::Message(error.to_string()))?;
    progress("Measuring reference and building R field", 1, 1);
    drop(reference_for_r);
    let region_scores = region_field.pixels();
    let mut sky_samples: Vec<_> = frames[image_reference_index]
        .pixels()
        .iter()
        .step_by(13)
        .copied()
        .filter(|pixel| pixel.is_finite())
        .collect();
    let result = crate::drizzle::stack_winsorized(
        crate::backend::SimilarityDrizzleRequest {
            frames: &frames,
            transforms: &transforms,
            region_scores,
            weights: &weights,
            config: drizzle,
        },
        output_plan,
        crate::config::RejectionConfig {
            sigma_multiplier: request.options.rejection_sigma,
            max_fraction: request.options.rejection_fraction,
            sky: select_median(&mut sky_samples).unwrap_or(0.0),
        },
        request.options.stack_workers,
        &corrections,
        |done, total| progress("Drizzle stacking (rows)", done, total),
    )
    .map_err(|error| WbppError::Message(error.to_string()))?;

    let live_pixels = result
        .frame_count
        .iter()
        .zip(result.image.pixels())
        .filter(|(count, value)| **count > 0 && value.is_finite())
        .count();
    let live_fraction = live_pixels as f64 / output_shape.pixel_count() as f64;
    if live_fraction < 0.05 {
        return Err(WbppError::Message(format!(
            "Only {:.2}% of the Drizzle canvas has valid data",
            live_fraction * 100.0
        )));
    }
    if !result
        .image
        .pixels()
        .iter()
        .any(|value| value.is_finite() && *value != 0.0)
    {
        return Err(WbppError::Message(
            "The integrated image contains only zero or invalid samples. Check the calibrated inputs.".into(),
        ));
    }
    if live_fraction < 0.5 {
        warnings.push(format!(
            "Only {:.1}% of the Drizzle canvas has valid data",
            live_fraction * 100.0
        ));
    }

    fs::create_dir_all(&request.output_directory)?;
    let output_raster = request.output_directory.join("dwt-pws.f32le");
    let output_neff = request.output_directory.join("dwt-pws-neff.f32le");
    let output_coverage = request.output_directory.join("dwt-pws-coverage.u32le");
    progress("Writing output rasters", 0, 3);
    write_f32_le(&output_raster, result.image.pixels())?;
    progress("Writing output rasters", 1, 3);
    write_f32_le(&output_neff, result.neff.as_slice())?;
    progress("Writing output rasters", 2, 3);
    write_u32_le(&output_coverage, &result.frame_count)?;
    progress("Writing output rasters", 3, 3);
    Ok(WbppRunResult {
        status: "ok".to_string(),
        frame_count: input_count,
        raster_shape: [output_shape.height, output_shape.width],
        output_raster,
        output_neff,
        output_coverage,
        registered_frames: registered_count,
        integrated_frames: transforms.iter().flatten().count(),
        quality_rejected_frames,
        field_corrected_frames: corrections.iter().flatten().count(),
        field_median_residual_before: select_median(
            &mut corrections
                .iter()
                .flatten()
                .map(|fit| fit.baseline as f32)
                .collect::<Vec<_>>(),
        ),
        field_median_residual_after: select_median(
            &mut corrections
                .iter()
                .flatten()
                .map(|fit| fit.corrected as f32)
                .collect::<Vec<_>>(),
        ),
        xdrz_transforms: registration_counts.xdrz,
        triangle_frames: registration_counts.triangle,
        chained_frames: registration_counts.chained,
        adjusted_frames: registration_counts.adjusted,
        sharp_reference_frames: sharp_frames.len(),
        warnings,
    })
}

fn retain_frames<T>(frames: &mut Vec<T>, retained: &[bool]) {
    let mut index = 0;
    frames.retain(|_| {
        let keep = retained[index];
        index += 1;
        keep
    });
}

fn trail_reference_stars(catalog: &[Star]) -> Vec<Star> {
    let peak_limit = catalog.first().map_or(0.0, |star| 0.8 * star.peak);
    catalog
        .iter()
        .enumerate()
        .filter(|&(index, star)| {
            star.peak < peak_limit
                && catalog.iter().enumerate().all(|(other, candidate)| {
                    index == other
                        || candidate.peak >= peak_limit
                        || (star.x - candidate.x).powi(2) + (star.y - candidate.y).powi(2)
                            >= 25.0_f64.powi(2)
                })
        })
        .take(120)
        .map(|(_, star)| *star)
        .collect()
}

fn reference_candidates(
    fwhm: &[Option<f32>],
    catalogs: &[Vec<Star>],
    config: RegistrationConfig,
    seeds: &[Option<SimilarityTransform>],
) -> Vec<usize> {
    let seed_distance = |seed: Option<SimilarityTransform>| {
        seed.map_or(f64::INFINITY, |transform| {
            (transform.scale - 1.0).abs()
                + transform.theta.abs()
                + transform.tx.abs()
                + transform.ty.abs()
        })
    };
    let mut indices: Vec<_> = (0..catalogs.len()).collect();
    indices.sort_by(|&a, &b| {
        (catalogs[a].len() < config.min_matches)
            .cmp(&(catalogs[b].len() < config.min_matches))
            .then_with(|| seed_distance(seeds[a]).total_cmp(&seed_distance(seeds[b])))
            .then_with(|| {
                fwhm[a]
                    .unwrap_or(f32::INFINITY)
                    .total_cmp(&fwhm[b].unwrap_or(f32::INFINITY))
            })
            .then_with(|| catalogs[b].len().cmp(&catalogs[a].len()))
    });
    indices
}

fn load_xdrz_transform(path: &Path) -> Result<crate::geometry::SimilarityTransform, WbppError> {
    let record = parse_xdrz(path)
        .map_err(|error| WbppError::Message(format!("{}: {error}", path.display())))?;
    let inverse = crate::geometry::invert_affine(record.alignment_matrix)
        .map_err(|error| WbppError::Message(format!("{}: {error}", path.display())))?;
    crate::geometry::SimilarityTransform::from_affine(inverse)
        .map_err(|error| WbppError::Message(format!("{}: {error}", path.display())))
}

/// Parses a PixInsight XDRZ 1.0 matrix record.
pub fn parse_xdrz(path: &Path) -> Result<XdrzRecord, WbppError> {
    let text = fs::read_to_string(path)?;
    let source = xml_text(&text, "SourceImage")
        .ok_or_else(|| WbppError::Parse(format!("{} has no SourceImage", path.display())))?;
    let matrix_text = xml_text(&text, "AlignmentMatrix")
        .ok_or_else(|| WbppError::Parse(format!("{} has no AlignmentMatrix", path.display())))?;
    let values = matrix_text
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| value.parse::<f64>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| WbppError::Parse(format!("invalid XDRZ matrix: {error}")))?;
    if values.len() != 9 {
        return Err(WbppError::Parse(format!(
            "XDRZ AlignmentMatrix must contain 9 values, found {}",
            values.len()
        )));
    }
    let matrix = [
        [values[0], values[1], values[2]],
        [values[3], values[4], values[5]],
        [values[6], values[7], values[8]],
    ];
    let normalized_source = source.replace('/', "\\");
    let stem = Path::new(&normalized_source)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or_default()
        .to_string();
    Ok(XdrzRecord {
        path: path.to_path_buf(),
        source_image: normalized_source,
        stem,
        alignment_matrix: matrix,
        reference_width: xml_attribute(&text, "ReferenceGeometry", "width")
            .and_then(|value| value.parse().ok()),
        reference_height: xml_attribute(&text, "ReferenceGeometry", "height")
            .and_then(|value| value.parse().ok()),
    })
}

fn walk(root: &Path, visit: &mut impl FnMut(&Path)) -> Result<(), WbppError> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            walk(&path, visit)?;
        } else if path.is_file() {
            visit(&path);
        }
    }
    Ok(())
}

fn is_image_extension(extension: &str) -> bool {
    matches!(extension, "xisf" | "fit" | "fits" | "fts")
}

fn path_has_segment(path: &Path, segment: &str) -> bool {
    path.components().any(|component| {
        component
            .as_os_str()
            .to_string_lossy()
            .eq_ignore_ascii_case(segment)
    })
}

fn file_stem(path: &Path) -> String {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or_default()
        .to_string()
}

fn registered_stem(path: &Path) -> String {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .map(without_registration_suffix)
        .unwrap_or_default()
        .to_string()
}

fn without_registration_suffix(stem: &str) -> &str {
    stem.strip_suffix("_r").unwrap_or(stem)
}

fn geometry_for_frame<'a>(
    geometry: &'a BTreeMap<String, XdrzRecord>,
    stem: &str,
) -> Option<&'a XdrzRecord> {
    geometry
        .get(stem)
        .or_else(|| geometry.get(without_registration_suffix(stem)))
}

fn geometry_path_for(geometry: &BTreeMap<String, PathBuf>, stem: &str) -> Option<PathBuf> {
    geometry
        .get(stem)
        .or_else(|| geometry.get(without_registration_suffix(stem)))
        .cloned()
}

fn xml_text(text: &str, tag: &str) -> Option<String> {
    let start_tag = format!("<{tag}>");
    let end_tag = format!("</{tag}>");
    let start = text.find(&start_tag)? + start_tag.len();
    let end = text[start..].find(&end_tag)? + start;
    Some(text[start..end].trim().to_string())
}

fn xml_attribute(text: &str, tag: &str, attribute: &str) -> Option<String> {
    let start = text.find(&format!("<{tag}"))?;
    let end = text[start..].find('>')? + start;
    let element = &text[start..=end];
    let marker = format!("{attribute}=\"");
    let value_start = element.find(&marker)? + marker.len();
    let value_end = element[value_start..].find('"')? + value_start;
    Some(element[value_start..value_end].to_string())
}

fn read_f32_le(path: &Path, pixel_count: usize) -> Result<Vec<f32>, WbppError> {
    let expected_bytes = pixel_count
        .checked_mul(4)
        .ok_or_else(|| WbppError::Message("raster size overflow".to_string()))?;
    let mut file = fs::File::open(path)?;
    let actual_bytes = file.metadata()?.len();
    if actual_bytes != expected_bytes as u64 {
        return Err(WbppError::Message(format!(
            "{} has {} bytes; expected {}",
            path.display(),
            actual_bytes,
            expected_bytes
        )));
    }
    let mut bytes = vec![0; expected_bytes];
    file.read_exact(&mut bytes)?;
    if file.read(&mut [0])? != 0 {
        return Err(WbppError::Message(format!(
            "{} grew while reading the raster",
            path.display()
        )));
    }
    let mut values = Vec::with_capacity(pixel_count);
    for chunk in bytes.chunks_exact(4) {
        values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(values)
}

fn write_f32_le(path: &Path, values: &[f32]) -> Result<(), WbppError> {
    write_le(path, values.iter().map(|value| value.to_le_bytes()))
}

fn write_u32_le(path: &Path, values: &[u32]) -> Result<(), WbppError> {
    write_le(path, values.iter().map(|value| value.to_le_bytes()))
}

fn write_le(path: &Path, values: impl Iterator<Item = [u8; 4]>) -> Result<(), WbppError> {
    let mut file = BufWriter::with_capacity(128 * 1024, fs::File::create(path)?);
    for value in values {
        file.write_all(&value)?;
    }
    file.flush()?;
    Ok(())
}

fn ensure_memory_available(phase: &str, required: u64) -> Result<(), WbppError> {
    let mut system = sysinfo::System::new();
    system.refresh_memory_specifics(sysinfo::MemoryRefreshKind::nothing().with_ram());
    check_memory_budget(phase, required, system.available_memory())
}

fn check_memory_budget(phase: &str, required: u64, available: u64) -> Result<(), WbppError> {
    if available > 0 && required > available - available / 8 {
        const GIB: f64 = 1_073_741_824.0;
        return Err(WbppError::Message(format!(
            "Insufficient memory for {phase}: estimated {:.2} GiB, {:.2} GiB currently available. Reduce the light group size or Drizzle scale.",
            required as f64 / GIB,
            available as f64 / GIB,
        )));
    }
    Ok(())
}

fn default_env_window_multiplier() -> f32 {
    2.0
}

fn default_env_epsilon() -> f32 {
    16.0
}

fn default_env_power() -> f32 {
    1.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn memory_budget_reserves_headroom() {
        const GIB: u64 = 1 << 30;
        assert!(check_memory_budget("test allocation", 7 * GIB, 8 * GIB).is_ok());
        assert!(check_memory_budget("test allocation", 7 * GIB + 1, 8 * GIB).is_err());
        assert!(check_memory_budget("test allocation", GIB, 0).is_ok());
    }

    #[test]
    fn parses_xdrz_matrix_and_source() {
        let text = r#"<xdrz xmlns="http://www.pixinsight.com/xdrz" version="1.0">
<SourceImage>D:/WBPP/calibrated/Light_001.xisf</SourceImage>
<ReferenceGeometry width="1024" height="768"/>
<AlignmentMatrix>1,0,0,0,1,0,0,0,1</AlignmentMatrix>
</xdrz>"#;
        let matrix = xml_text(text, "AlignmentMatrix").expect("matrix");
        assert_eq!(matrix, "1,0,0,0,1,0,0,0,1");
        assert_eq!(
            xml_text(text, "SourceImage").expect("source"),
            "D:/WBPP/calibrated/Light_001.xisf"
        );
        assert_eq!(
            xml_attribute(text, "ReferenceGeometry", "width").expect("width"),
            "1024"
        );
    }

    #[test]
    fn normalizes_exposure_and_ignores_an_unreadable_xdrz() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("dwt-wbpp-test-{suffix}"));
        let rasters = root.join("DWT/rasters");
        fs::create_dir_all(&rasters).expect("create raster directory");
        let first = vec![1.0_f32; 4];
        let second = vec![2.0_f32; 4];
        write_f32_le(&rasters.join("Light_001.f32le"), first.as_slice()).expect("first raster");
        write_f32_le(&rasters.join("Light_002.f32le"), second.as_slice()).expect("second raster");
        let request = WbppCommand {
            schema: "dwt-wbpp-command/1".to_string(),
            source: "test".to_string(),
            calibrated_directory: root.clone(),
            registered_directory: Some(root.join("registered")),
            output_directory: root.join("DWT"),
            frames: vec![
                WbppFrame {
                    enabled: true,
                    source: PathBuf::from("Light_001.xisf"),
                    registered: None,
                    xdrz: None,
                    has_xdrz: false,
                    fwhm: Some(1.0),
                    noise: Some(1.0),
                    signal: Some(1.0),
                    exposure: Some(1.0),
                    observation_timestamp_ms: None,
                    raster: Some(rasters.join("Light_001.f32le")),
                    background: Some(0.0),
                },
                WbppFrame {
                    enabled: true,
                    source: PathBuf::from("Light_002.xisf"),
                    registered: None,
                    xdrz: Some(root.join("unreadable.xdrz")),
                    has_xdrz: true,
                    fwhm: Some(1.0),
                    noise: Some(1.0),
                    signal: Some(1.0),
                    exposure: Some(2.0),
                    observation_timestamp_ms: None,
                    raster: Some(rasters.join("Light_002.f32le")),
                    background: Some(0.0),
                },
            ],
            raster: WbppRaster {
                directory: rasters,
                file_extension: "f32le".to_string(),
                width: 2,
                height: 2,
                dtype: "float32-le".to_string(),
            },
            options: WbppOptions::default(),
        };
        let result = run_calibrated_pws(&request).expect("run PWS");
        let output = read_f32_le(
            &result.output_raster,
            result.raster_shape[0] * result.raster_shape[1],
        )
        .expect("read output");
        assert_eq!(result.frame_count, 2);
        assert_eq!(result.registered_frames, 1);
        assert_eq!(result.xdrz_transforms, 0);
        assert!(
            result
                .warnings
                .iter()
                .any(|warning| warning.contains("Ignoring XDRZ seed"))
        );
        assert!(output.iter().any(|value| (*value - 1.5).abs() < 1.0e-5));
        fs::remove_dir_all(root).expect("remove test directory");
    }

    #[test]
    fn calibrated_path_self_registers_translated_stars() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("dwt-calibrated-test-{suffix}"));
        let rasters = root.join("DWT/rasters");
        fs::create_dir_all(&rasters).expect("create raster directory");
        let base = [
            (42.0, 45.0),
            (91.0, 56.0),
            (147.0, 72.0),
            (196.0, 43.0),
            (58.0, 111.0),
            (114.0, 132.0),
            (171.0, 106.0),
            (213.0, 151.0),
            (37.0, 181.0),
            (92.0, 204.0),
            (151.0, 188.0),
            (199.0, 217.0),
        ];
        let frame_points: Vec<(f64, f64)> = base.iter().map(|&(x, y)| (x - 7.0, y + 5.0)).collect();
        write_f32_le(
            &rasters.join("a.f32le"),
            render_star_field(256, 256, &base).pixels(),
        )
        .expect("write reference");
        let shifted_field = render_star_field(256, 256, &frame_points);
        let shifted_pixels: Vec<_> = shifted_field
            .pixels()
            .iter()
            .map(|value| value + 0.02)
            .collect();
        write_f32_le(&rasters.join("b.f32le"), &shifted_pixels).expect("write frame");
        let mut request = WbppCommand {
            schema: "dwt-wbpp-command/1".to_string(),
            source: "test".to_string(),
            calibrated_directory: root.clone(),
            registered_directory: None,
            output_directory: root.join("DWT"),
            frames: vec![
                WbppFrame {
                    enabled: true,
                    source: PathBuf::from("a.xisf"),
                    registered: None,
                    xdrz: None,
                    has_xdrz: false,
                    fwhm: Some(2.8),
                    noise: Some(1.0),
                    signal: Some(1.0),
                    exposure: Some(10.0),
                    observation_timestamp_ms: None,
                    raster: Some(rasters.join("a.f32le")),
                    background: Some(0.0),
                },
                WbppFrame {
                    enabled: true,
                    source: PathBuf::from("b.xisf"),
                    registered: None,
                    xdrz: None,
                    has_xdrz: false,
                    fwhm: Some(2.8),
                    noise: Some(8.0),
                    signal: Some(1.0),
                    exposure: Some(10.0),
                    observation_timestamp_ms: None,
                    raster: Some(rasters.join("b.f32le")),
                    background: Some(0.0),
                },
            ],
            raster: WbppRaster {
                directory: rasters,
                file_extension: "f32le".to_string(),
                width: 256,
                height: 256,
                dtype: "float32-le".to_string(),
            },
            options: WbppOptions {
                rejection_sigma: 0.0,
                drizzle_scale: 1.0,
                ..WbppOptions::default()
            },
        };
        request.options.drizzle_pixfrac = 0.0;
        assert!(
            validate_request(&request)
                .expect_err("zero pixfrac must be rejected")
                .to_string()
                .contains("pixfrac")
        );
        request.options.drizzle_pixfrac = 1.0;
        let result = run_calibrated_pws(&request).expect("run calibrated PWS");
        assert_eq!(result.frame_count, 2);
        assert_eq!(result.registered_frames, 2);
        assert_eq!(result.xdrz_transforms, 0);
        assert_eq!(result.sharp_reference_frames, 2);
        assert!(result.warnings.is_empty());
        let output = read_f32_le(
            &result.output_raster,
            result.raster_shape[0] * result.raster_shape[1],
        )
        .expect("read output");
        assert!(output.iter().any(|value| *value > 1.0));

        let failed_raster = request.raster.directory.join("no-stars.f32le");
        write_f32_le(&failed_raster, &vec![0.01; 256 * 256]).expect("write starless frame");
        let mut failed_frame = request.frames[1].clone();
        failed_frame.source = PathBuf::from("no-stars.xisf");
        failed_frame.raster = Some(failed_raster);
        failed_frame.noise = Some(20.0);
        request.frames.insert(1, failed_frame);
        request.output_directory = root.join("with-unregistered");
        let with_failed = run_calibrated_pws(&request).expect("run with unregistered frame");
        assert_eq!(with_failed.frame_count, 3);
        assert_eq!(with_failed.registered_frames, 2);
        assert_eq!(with_failed.integrated_frames, 2);
        assert!(
            with_failed.warnings.iter().any(
                |warning| warning.contains("no-stars.xisf: registration could not be verified")
            )
        );
        for (actual, expected) in [
            (&with_failed.output_raster, &result.output_raster),
            (&with_failed.output_neff, &result.output_neff),
            (&with_failed.output_coverage, &result.output_coverage),
        ] {
            assert!(
                fs::read(actual).expect("read run with unregistered frame")
                    == fs::read(expected).expect("read two-frame run"),
                "unregistered frame changed {}",
                actual.display()
            );
        }
        request.frames.remove(1);

        request.frames[1].fwhm = Some(4.0);
        request.options.registration_stars = 120;
        request.options.fwhm_max = 3.0;
        request.options.r_blur = 2.0;
        request.options.stack_workers = 1;
        let filtered = run_calibrated_pws(&request).expect("run with frame and R-field controls");
        assert_eq!(filtered.frame_count, 2);
        assert_eq!(filtered.registered_frames, 1);
        assert_eq!(filtered.sharp_reference_frames, 1);
        assert!(filtered.warnings.iter().any(|w| w.contains("b.xisf: FWHM")));

        request.options.fwhm_max = 1.0;
        let error = run_calibrated_pws(&request).expect_err("FWHM limit should reject every frame");
        assert!(error.to_string().contains("No frames met the FWHM limit"));
        fs::remove_dir_all(root).expect("remove test directory");
    }

    fn render_star_field(
        width: usize,
        height: usize,
        points: &[(f64, f64)],
    ) -> crate::frame::Frame {
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
        crate::frame::Frame::new(width, height, pixels).expect("synthetic field")
    }
}
