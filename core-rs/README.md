# core-rs

Rust backend for DWT-Drizzle in PixInsight WBPP.

## Scope

`core-rs` implements the standalone CPU backend called by the WBPP Pipeline
Builder. It measures frame quality and registration from calibrated pixels,
projects those pixels into the Drizzle canvas, and combines them with PWS,
Winsorized rejection, and per-frame effective depth.

The WBPP path follows the 2.7.102-D defaults for throughput weighting
(`flux_power = 0`) and detail-region noise weighting (`detail_snr = 0`).
It uses the upstream stride-two noise estimator and Winsorized scale on
64×64 output tiles. Rust measures FWHM from stellar half-maximum profiles.
The registration uses subpixel star positions, full-catalog verification,
triangle matching, and neighbor-frame adjustment. The region field uses the
median of the sharpest registered frames on the Drizzle grid.

Open implementation areas include wgpu, raw CFA Drizzle, radial distortion,
adaptive star-support dilation, background normalization, and alternate
rejection modes.

WBPP integration scripts and usage are under [`wbpp/`](wbpp/README.md).

## Layout

```text
core-rs/
  Cargo.toml
  crates/dwt-backend/
    src/backend.rs    backend trait and request types
    src/config.rs     algorithm parameters
    src/drizzle.rs    similarity-transform canvas and drop accumulation
    src/frame.rs      owned image buffer
    src/geometry.rs   similarity transforms and XDRZ inversion
    src/pws.rs        PWS weights and aligned stacking
    src/register.rs   star detection and self-registration
```

## Build and test

```powershell
cargo test --manifest-path core-rs/Cargo.toml
cargo clippy --manifest-path core-rs/Cargo.toml --all-targets -- -D warnings
```

## Benchmark

Run the local PWS comparison against the source functions in `core/pws.py`:

```powershell
python core-rs/scripts/benchmark_compare.py --frames 16 --width 512 --height 512 --repetitions 11
```

The benchmark is a local CPU function benchmark. It excludes file decoding,
registration, export, process startup, and GPU execution.

The [WBPP adapter](wbpp/README.md) calls the standalone CLI with calibrated
rasters and a JSON request.
