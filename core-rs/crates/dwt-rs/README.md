# dwt-rs — DWT-Drizzle stacking core in Rust

A faithful port of the mono (non-CFA) stacking path of `core/pws.py` +
`core/drizzle.py`, with a CPU path and a wgpu (DX12 / Vulkan) path for the
drizzle/weight/rejection kernels.

Branch: `branch-E-rust`. The Python sources are **not** modified by this work;
everything lives under `core-rs/`.

## Build

```bash
export PATH="$HOME/.cargo/bin:$PATH"
cd core-rs
cargo build --release -p dwt-rs          # CPU + GPU (wgpu, default feature)
cargo build --release -p dwt-rs --no-default-features   # CPU only
cargo test  --release -p dwt-rs
```

> On this machine the Rust GNU toolchain needs a `dlltool.exe` for the
> `raw-dylib` crates pulled in by wgpu. A binutils `bin` directory on `PATH` is
> enough; without it the build fails with
> `error: error calling dlltool 'dlltool.exe': program not found`.

## Run

```bash
./target/release/dwt-rs.exe \
  --photos "G:/HFG1-3/master3/calibrated" --out /tmp/out --tag t \
  --pick O --limit 8 --crop 400 --drizzle --drizzle-scale 1 --A 6 --aligned 1
```

Useful switches:

| flag | meaning |
|---|---|
| `--gpu on\|off` | wgpu kernels vs CPU (default: auto if an adapter exists) |
| `--weights-from <txt>` | inject `idx fwhm C sigma S` table (parity harness) |
| `--star-grow-fixed <v>` | bypass the automatic star-grow estimate |
| `--save-aux` | write `PWS_t_R_q4.npy`, `PWS_t_neff_q4.npy`, `PWS_t_weights.txt` |
| `--dump-frames/--dump-w/--dump-r <dir>` | dump intermediates as `.npy` |
| `--report` | print the timing/statistics report |
| `--bgdeg <n>` | per-frame background polynomial degree (default 1) |

Environment: `DWT_GPU_TIMING=1`, `DWT_GPU_REPS=N`, `DWT_BENCH_CPU=N`,
`DWT_RS_DEBUG=1`.

## Parity vs the Python reference

Same input (`G:/HFG1-3/master3/calibrated`, `--pick O --limit 8 --crop 400
--drizzle --drizzle-scale 1 --A 6 --aligned 1`), 400×400 output.

| run | corr | max\|diff\| | max\|diff\| / median(sky) | rms\|diff\| | wall |
|---|---|---|---|---|---|
| Rust CPU, C/S injected from the Python table | **1.00000000** | 1.024e-08 | 1.8e-06 | 6.32e-10 | 1.04 s |
| Rust GPU, same inputs | **1.00000000** | 9.31e-09 | 1.6e-06 | 6.48e-10 | 2.35 s |
| Rust CPU, native PSF/sky estimators | 0.98543842 | 4.882e-03 | 0.845 | 3.98e-05 | 1.01 s |
| Python reference | — | — | — | — | 25.13 s |

The first two rows show the whole pipeline (sky normalisation, weights,
rejection, weighted mean, xisf I/O) is bit-exact: 1e-8 is float32 rounding of a
~1e-2 signal. The third row differs only through `C_i`, the per-frame PSF-size
term.

### Why the native `C_i` differs

`C_i = median(fwhm) / fwhm_i`. Python measures the FWHM with `photutils.fit_fwhm`
on a fixed 33×33 window anchored at the star positions of one *seed* frame, for
every frame. These frames are dithered by 3–16 px, so on frames far from the
seed the star sits at the window edge and the fit inflates: the Python table
reads `8.41, 8.34, 3.21, 2.84, 2.49, 2.27, 2.14, 2.15` where the physical PSF is
~2.6 px on every frame (photutils itself returns 2.5–2.8 px when re-centred).

dwt-rs measures each frame against its own star table (and refuses fits below
`3σ`), giving `2.13–2.75 px, median 2.64 px`, i.e. the physically correct value.
That is a deliberate difference; injecting Python's own `C/S` reproduces its
output to 1e-8. Sensitivity: perturbing Python's `C` by ±5 % still yields
corr 0.99994, so the remaining 1.5 % is the whole effect of this term.

## GPU

wgpu 26 / DX12 on an RTX 4070 Ti. Three compute passes mirror the CPU maths:
`k_weights`, `k_reject`, `k_accum` (drizzle gather). Results match the CPU path
to corr 1.00000000 / max|diff| 2.8e-09 at both `--drizzle-scale 1` and `2`.

Kernel throughput (8 frames, `DWT_GPU_REPS=50` vs `DWT_BENCH_CPU=5`):

| crop | CPU `stack_strip` | GPU per dispatch | speed-up |
|---|---|---|---|
| 400² (160 kpx) | 0.1116 s | 0.0007 s | **159×** |
| 1600² (2.56 Mpx) | 1.8596 s | 0.0054 s | **344×** |

One-off GPU start-up is ~1.12 s (0.27 s device init + 0.85 s shader/pipeline
setup), so a single 400² run is still faster on the CPU; end-to-end at 1600² the
GPU run takes 8.83 s vs 18.30 s on the CPU.

## Tests

`cargo test --release -p dwt-rs` → 10 passed:

`weights_plain_formula`, `weights_r_mix`, `weights_cap_limits_share`,
`reject_single_outlier`, `reject_disabled`, `drizzle_point_source_scale2`,
`drizzle_known_shift`, `drizzle_scatter_equals_gather`, `xisf_roundtrip`,
`gpu_matches_cpu_stack` (skipped automatically when no adapter exists).

## Known limits

* `--drizzle-scale > 1` is numerically valid and matches the CPU path, but does
  **not** reproduce Python's `stack_drizzle`, which re-derives the reference
  image, the R field and the rejection on the *output* grid (its `σ_psf`
  doubles: 9.35 px vs 0.57 px). dwt-rs keeps weights/R/rejection on the input
  crop grid and drizzles the result. At ×1 the two agree bit-exactly.
* Mono only (no CFA/OSC), no registration (input is assumed aligned, as with
  `--aligned 1`).
