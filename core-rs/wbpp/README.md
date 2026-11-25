# WBPP integration

This directory contains the PixInsight WBPP-side adapter for the Rust backend.
It does not modify the Python application.

## Scope

WBPP calibrates the light frames and can debayer them into separate color
channels. The Pipeline Builder exports each supported group from
`processed[DEBAYER]` or `processed[CALIBRATION]`. Rust measures frame quality
and star positions, registers the frames, and integrates their pixels with
PWS and Drizzle. PixInsight saves the result as XISF.

For a DWT-only integration, enable the Pipeline Builder and turn off WBPP
Image Registration, Local Normalization, Image Integration, and Drizzle
Integration. Rust self-registration is active by default. The project can
reuse existing XDRZ sidecars as registration seeds with `"useXdrzSeeds": true`
in its `DWT/dwt-backend.json`. WBPP Image Registration can produce those
sidecars when you choose to use them.

## Install

The WBPP adapter uses the **Pipeline Builder** and operation queue APIs in
WBPP 3.1.0. The verified combination is PixInsight 1.9.5 with WBPP 3.1.0 on
Windows. The installers build the native Rust executable and copy it together
with the Pipeline Builder scripts. Run the commands from the repository root.
A source build requires [Rust and Cargo](https://rustup.rs/). With an MSVC Rust toolchain
on Windows, install Visual Studio Build Tools with the C++ workload. On
macOS, install Apple's command-line tools with `xcode-select --install`.

### Windows

```powershell
.\core-rs\wbpp\install.ps1
```

The default destination is
`C:\Program Files\PixInsight\src\scripts\DWT-Drizzle`. The installer also
checks registered PixInsight installation paths. Run PowerShell as an
administrator if the PixInsight directory requires it, or select an existing
writable script directory.

```powershell
.\core-rs\wbpp\install.ps1 -PixInsightScripts "D:\PixInsight Scripts"
```

### macOS

```bash
bash core-rs/wbpp/install.sh
```

The default destination is `/Applications/PixInsight/src/scripts/DWT-Drizzle`.
Run the installer on the Mac that will run PixInsight. Cargo builds for the
installed Rust toolchain. Use the native macOS Rust toolchain for Apple Silicon
or Intel, as appropriate. For a custom PixInsight location or a writable
personal script directory, pass that directory explicitly.

```bash
mkdir -p "$HOME/PixInsight Scripts"
bash core-rs/wbpp/install.sh --pixinsight-scripts "$HOME/PixInsight Scripts"
```

Both installers print the path to the interactive builder. They preserve
existing `dwt-backend.json` files when updating. If you already have a compatible
executable, pass `-BackendPath` on Windows or `--backend` on macOS to install it
directly.

## Use in WBPP

Under **Pipeline → Pipeline scripting → Pipeline builder**, enable the builder
and select the installed `DWT-Drizzle/pipeline-builder-ui.js`. WBPP can load the
script by its full path, including from a personal script directory.

Click **Run**. The **DWT-Drizzle (Rust) Settings** dialog opens once before the
first WBPP processing step. **Stack** contains Drizzle, frame and registration
settings. **PWS** and **R field** contain weights, clipping and envelope settings.
**Save and continue** applies the settings to all DWT groups and saves them in
`<WBPP output>/DWT/dwt-backend.json` for the next run.
**Cancel run** stops the pipeline and leaves the saved settings intact.

`DWT-Drizzle/pipeline-builder.js` runs with the saved settings. WBPP automation
mode also uses the saved settings when the interactive builder is selected.

The Pipeline operation list and execution monitor show `DWT-Drizzle (Rust)`.
The **Est. disk space** column and WBPP disk-space diagnostics include the
estimated size of the exported input rasters, master XISF and three Rust
output rasters. The preview uses the active light groups and saved settings.
The output estimate uses the nominal Drizzle canvas. Registration offsets
and rotations can make the actual canvas larger.
The Process Console logs the builder script path when the operation starts.
The console and the operation monitor report frame measurement, registration,
reference construction, Drizzle row progress, and output writing.
Canceling in the execution monitor stops export between frames or terminates
the active Rust process before the builder saves that group's master.

Optionally select `DWT-Drizzle/event-script.js` in the Event Script field.
The builder reads `dwt-backend.json` next to the selected Pipeline Builder,
then applies an optional project override at
`<WBPP output>/DWT/dwt-backend.json`. See the [configuration example](dwt-backend.json.example).
It selects the installed executable for the current platform. Set an
`executable` path in the JSON if you want to use a different binary.
WBPP provides the input files, output directory, filter groups, and debayering.

| Setting | Default | Effect |
| --- | --- | --- |
| `drizzleScale` | `1.0` | Output pixels per input pixel. Set `2.0` for a 2× Drizzle canvas. |
| `drizzlePixfrac` | `1.0` | Drop width in input pixels, greater than zero and at most one. |
| `drizzleEnabled` | `true` | Disabling this uses 1× and pixfrac 1 while keeping registration active. |
| `clarityExponent` | `20.0` | PWS clarity parameter A. Lower values reduce the influence of frame FWHM on detail weights. |
| `registrationStars` | `240` | Maximum catalog stars per frame, from 8 to 4096. |
| `frameLimit` | `0` | Use the first N active frames in each WBPP group. Zero uses all. |
| `fwhmMax` | `0` | FWHM limit in input pixels. Positive values reject frames above the limit or without a valid measurement. |
| `autoRejectFrames` | `true` | Reject isolated low-throughput frames after registration and before reference construction. |
| `fieldCorrection` | `true` | Refine registration with a bounded quadratic correction when held-out stars confirm lower residuals. |
| `excludeFrames` | `[]` | Legacy JSON setting for paths to omit before raster export. |
| `referenceFrames` | `0` | Number of sharp registered frames used for the median reference. Zero uses up to eight. |
| `rejectionSigma` | `3.5` | Rejection threshold multiplier. Zero disables rejection. |
| `rejectionFraction` | `1/3` | Maximum rejected fraction at a pixel. If more samples are flagged, the stack keeps all samples at that pixel. |
| `stackWorkers` | `0` | Drizzle worker limit, up to eight. Zero uses the automatic limit. |
| `snrExponent`, `snrClip` | `2`, `3` | Background SNR exponent B and SNR weight clipping bound. |
| `detailSnr`, `fluxPower`, `sharpWeight` | `0`, `0`, `0` | Detail SNR exponent β, flux exponent and global sharpness exponent. |
| `rFieldMode` | `"max"` | Local peak envelope. `"band"` weights band-pass structures. `"ctn"` uses contrast relative to noise. |
| `rFieldBlur`, `rFieldFloor` | `0`, `0` | Gaussian sigma on the R-field grid and minimum R-field value. Positive blur selects the smoothed local-peak envelope. |
| `rFieldWindow` | `0.0` | Band-pass window in output pixels. Zero sizes the window from the reference FWHM. |
| `rFieldContrast` | `3.0` | Signal-to-noise level for full detail weight in `"ctn"` mode. |
| `envWindowMultiplier`, `envEpsilon`, `envPower` | `2`, `16`, `1` | Envelope window multiplier, noise term and exponent. |

The Python application also exposes GPU, raw CFA and RGB controls, gradient and
radial background correction, crop and reference-grid outputs, alternate rejection
scales, PSF-matched drops, two-pass weights and automatic A tuning. These options
require additional support in the WBPP Rust path.

## Handoff contract

The pipeline-builder exports each active regular light group's calibrated or
debayered frames to:

```text
<WBPP output>/DWT/runs/<group>~<group-id-hash>/<unique-frame-id>.f32le
```

The directory name includes a hash of WBPP's group ID. Groups whose names
sanitize to the same text get separate directories.

Each raster is row-major float32 little-endian. The builder then runs:

```text
dwt-backend wbpp-pws <request.json> <result.json>
```

The Pipeline Builder request uses the `dwt-wbpp-command/1` schema. It records
the calibrated frame paths, float32 rasters, exposure times, and available WBPP
quality measurements and observation timestamps. For automatic frame rejection,
Rust compares local-sky-subtracted apertures of at least 20 matching stars.
It rejects a frame when the median relative flux is at most 70% of the local
trend and at most 85% of each side's baseline. Two measured exposures on either
side must be nearby in time. The process log identifies each rejected frame and
reports separate registration and integration counts. Pixel rejection still
handles local outliers within retained frames. XDRZ 1.0 matrices map reference to source.

```text
source = M * reference
reference = inverse(M) * source
```

When XDRZ seeds are enabled, Rust inverts the matrices and converts them to the
chosen reference frame's coordinates. It refines the initial transforms with
stars from the calibrated frames. Invalid files produce a warning and the
affected frames use Rust self-registration.

Field correction uses unsaturated guide-star centroids across the image. A
robust fit is checked on stars held out from its training set. Both the sharp
reference and final Drizzle stack apply the accepted per-frame corrections as
continuous displacements of drop centers. The Process Console reports the
corrected-frame count and validation residuals.

## Current behavior

The WBPP pipeline-builder currently:

1. Runs the configured WBPP light-frame preprocessing.
2. Exports each supported group as single-channel float32 rasters.
3. Runs `dwt-backend wbpp-pws`. The backend normalizes exposure, measures
   noise and FWHM, verifies each transform against the star catalog, and
   retries failed matches with triangle and neighbor-frame registration.
   When neighbor-frame recovery succeeds, the backend can refine those
   transforms with weighted least squares. It builds the R field from the
   median of the sharpest registered frames, then applies PWS with
   Winsorized rejection on the Drizzle grid.
4. Loads the Rust output raster and saves
   `<WBPP output>/DWT/runs/<group>~<group-id-hash>/master_dwt_<group>.xisf`.

Current support covers regular single-channel groups and separate debayered
channels. Raw CFA Drizzle, wgpu, and parity with the Python radial distortion
model are open implementation areas. The Rust result JSON records the
registered-frame count, fallback counts, and the number of sharp reference frames.
Calibrated frames remain in memory. The backend checks available RAM before
loading a group and again before allocating the Drizzle canvas. Reduce the
group size or Drizzle scale if the estimated allocation exceeds the budget.
