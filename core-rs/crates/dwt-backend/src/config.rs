use crate::error::{BackendError, BackendResult};

/// PWS weight parameters from `core/pws.py` and `core/pws_params.py`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PwsConfig {
    /// Detail-region clarity exponent. The Python backend defaults to 20.
    pub clarity_exponent: f32,
    /// Background-region signal-to-noise exponent. The Python backend defaults to 2.
    pub snr_exponent: f32,
    /// Signal-to-noise exponent in detail regions (β).
    pub detail_snr: f32,
    /// Optional clipping bound for S weights. Values at or below one disable clipping.
    pub snr_clip: f32,
    /// Exponent for normalized star flux.
    pub flux_power: f32,
    /// Additional global clarity exponent.
    pub sharp_weight: f32,
}

impl Default for PwsConfig {
    fn default() -> Self {
        Self {
            clarity_exponent: 20.0,
            snr_exponent: 2.0,
            detail_snr: 0.0,
            snr_clip: 3.0,
            flux_power: 0.0,
            sharp_weight: 0.0,
        }
    }
}

impl PwsConfig {
    pub(crate) fn validate(self) -> BackendResult<()> {
        for (name, value) in [
            ("clarity_exponent", self.clarity_exponent),
            ("snr_exponent", self.snr_exponent),
            ("detail_snr", self.detail_snr),
            ("snr_clip", self.snr_clip),
            ("flux_power", self.flux_power),
            ("sharp_weight", self.sharp_weight),
        ] {
            if !value.is_finite() {
                return Err(BackendError::InvalidValue { name, value });
            }
        }
        Ok(())
    }
}

/// Rejection parameters for aligned stacking.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RejectionConfig {
    /// MAD multiplier. Zero disables rejection.
    pub sigma_multiplier: f32,
    /// Skip rejection at a pixel when more than this fraction is rejected.
    pub max_fraction: f32,
    /// Sky level used to measure the detail-signal allowance.
    pub sky: f32,
}

impl Default for RejectionConfig {
    fn default() -> Self {
        Self {
            sigma_multiplier: 3.5,
            max_fraction: 1.0 / 3.0,
            sky: 0.0,
        }
    }
}

impl RejectionConfig {
    pub(crate) fn validate(self) -> BackendResult<()> {
        if !self.sigma_multiplier.is_finite() || self.sigma_multiplier < 0.0 {
            return Err(BackendError::InvalidValue {
                name: "sigma_multiplier",
                value: self.sigma_multiplier,
            });
        }
        if !self.max_fraction.is_finite() || !(0.0..=1.0).contains(&self.max_fraction) {
            return Err(BackendError::InvalidValue {
                name: "max_fraction",
                value: self.max_fraction,
            });
        }
        if !self.sky.is_finite() {
            return Err(BackendError::InvalidValue {
                name: "sky",
                value: self.sky,
            });
        }
        Ok(())
    }
}

/// Translation-only drizzle parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DrizzleConfig {
    /// Output pixels per input pixel.
    pub scale: f32,
    /// Drop size as a fraction of one input pixel, in (0, 1].
    pub pixfrac: f32,
    /// Canvas padding in input pixels.
    pub margin: f32,
}

impl Default for DrizzleConfig {
    fn default() -> Self {
        Self {
            scale: 2.0,
            pixfrac: 1.0,
            margin: 1.0,
        }
    }
}

impl DrizzleConfig {
    pub(crate) fn validate(self) -> BackendResult<()> {
        if !self.scale.is_finite() || self.scale <= 0.0 {
            return Err(BackendError::InvalidValue {
                name: "scale",
                value: self.scale,
            });
        }
        if !self.pixfrac.is_finite() || self.pixfrac <= 0.0 || self.pixfrac > 1.0 {
            return Err(BackendError::InvalidValue {
                name: "pixfrac",
                value: self.pixfrac,
            });
        }
        if !self.margin.is_finite() || self.margin < 0.0 {
            return Err(BackendError::InvalidValue {
                name: "margin",
                value: self.margin,
            });
        }
        Ok(())
    }
}
