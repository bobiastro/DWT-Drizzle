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

## LSPR (large-scale structure rejection)

`crates/dwt-rs/src/lsp/` is a port of `core/lsp.py`: mask field, shear-scan
cell map, coarse rho sweep, strip/profile measurement, accept criteria,
additive profile subtraction, and the frame-domain veto gate. Thresholds are
the Python values (`_A_ZSUM 14`, `_A_CV 0.45`, `_A_POS 0.90`, `_A_COVER 0.55`,
`_A_WMIN 0.9`, `_A_WMAX 8`, `_A_AMP 1.0`, `_A_AMPMAX 6`, `_A_STRAIGHT 2`,
`_LSP_*` unchanged) with the same `DWT_LSPR_*` environment overrides.
`DWT_LSPR=0` disables the stage; `DWT_LSPR_FRAMEGATE=0` skips only the
frame-domain gate, exactly like Python.

The frame gate keeps Python's asymmetric logic: **any** of the three pieces of
evidence failing vetoes the line (`veto = z_bad or f_bad or a_bad`,
`core/lsp.py:1596`). The gate can only reject, never confirm.

Known deviation, deliberate: Python re-measures each accepted line and
subtracts a second pass (`二次扣除`), which overshoots — on the M81 reference
stack it leaves `-0.146 sigma` along the line. Rust subtracts once; the first
pass already reproduces Python's first-pass pixel count exactly (73,100 px on
that stack) and leaves the line at the `-0.05 sigma` level.

To compare the detector against Python stage by stage, run it on an existing
stack instead of a frame list:

```text
dwt-rs --lsp-only path-to-stack.xisf
```

It prints the mask field, seed counts, the coarse-pass count, one line per
measured seed and the accepted count, without touching registration.

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
