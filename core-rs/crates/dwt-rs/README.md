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
| `--fwhm-search N` | per-frame FWHM recentring radius in px (default 8, `0` = Python's pre-2.7.116 behaviour) |
| `--per-frame-stars` | detect stars per frame instead of reusing the seed table |
| `--save-aux` | write `PWS_t_R_q4.npy`, `PWS_t_neff_q4.npy`, `PWS_t_weights.txt` |
| `--dump-frames/--dump-w/--dump-r <dir>` | dump intermediates as `.npy` |
| `--report` | print the timing/statistics report |
| `--bgdeg <n>` | per-frame background polynomial degree (default 1) |

Environment: `DWT_GPU_TIMING=1`, `DWT_GPU_REPS=N`, `DWT_BENCH_CPU=N`,
`DWT_RS_DEBUG=1`.

## Parity vs the Python reference

Same input (`G:/HFG1-3/master3/calibrated`, `--pick O --limit 8 --crop 400
--drizzle --drizzle-scale 1 --A 6 --aligned 1`), 400×400 output, Python run with
`--fwhm-search 8` (its new per-frame FWHM recentring).

| run | corr | max\|diff\| | max\|diff\| per median(sky) | rms\|diff\| | wall |
|---|---|---|---|---|---|
| Rust CPU, native estimators | **0.99991824** | 1.339e-04 | 0.02316 | 2.97e-06 | 1.02 s |
| Rust CPU, C/S injected from the Python table | **1.00000000** | 2.980e-08 | 5.2e-06 | 1.2e-10 | 1.02 s |
| Rust GPU, same inputs | **1.00000000** | 2.794e-09 | 4.8e-07 | 5.2e-10 | 2.53 s |
| Python reference | — | — | — | — | 15.0 s (10.5 s internal) |

Criterion: `max|diff| / median(sky) <= 1e-3` (sky = 5.78e-3, sky noise
1.15e-4). The native run lands at 0.023, the injected run is float32 rounding
of a ~1e-2 signal. Everything except `C_i` is bit-exact; injecting Python's
`C/S` reproduces its output to 2.98e-08.

### Per-frame PSF size (`C_i`)

`C_i = median(fwhm) / fwhm_i`, so only the *shape* of the per-frame PSF series
matters. Python measures it with `photutils.fit_fwhm` at a fixed 33×33 window
anchored on one seed frame's star table; `--fwhm-search 8` re-centres that
window on the brightest pixel first. dwt-rs mirrors this exactly
(`detect::recenter_brightest` + `measure_fwhm_search`, `--fwhm-search N`,
default 8), with a σ-grid fit instead of a 7-parameter LM fit.

| frame | py fwhm | py C | rs fwhm | rs C | rs C / py C |
|---|---|---|---|---|---|
| 0 | 2.707 | 0.9391 | 2.846 | 0.9612 | 1.024 |
| 1 | 2.624 | 0.9689 | 2.795 | 0.9785 | 1.010 |
| 2 | 2.734 | 0.9300 | 2.916 | 0.9380 | 1.009 |
| 3 | 2.699 | 0.9420 | 2.840 | 0.9633 | 1.023 |
| 4 | 2.461 | 1.0332 | 2.675 | 1.0225 | 0.990 |
| 5 | 2.232 | 1.1392 | 2.425 | 1.1278 | 0.990 |
| 6 | 1.971 | 1.2899 | 2.307 | 1.1855 | 0.919 |
| 7 | 2.078 | 1.2234 | 2.291 | 1.1937 | 0.976 |

The absolute FWHM carries a +8 % bias (σ-grid fit on a fixed window), which
cancels in the ratio; the residual is concentrated on the two sharpest frames.
Isolation runs: injecting Python's `C/S` gives corr 1.00000000 and forcing
`--star-grow-fixed 2.59` (Python's auto value) changes nothing (0.99992027), so
C is the only divergent term - the S table is identical to the last digit and
the flux term is off by default (`flux_pow=0`). Perturbing Python's C by ±5 %
still yields corr 0.99994.

### CPU wall time

| case | dwt-rs | Python |
|---|---|---|
| 8 frames, crop 400 (160 kpx) | **1.02 s** (FWHM 0.34 s, stack 0.19 s) | 15.0 s wall / 10.5 s internal |
| 60 frames, full frame 2758×2204 | **85.3 s** (FWHM 41 s, stack 40 s, 12 threads) | 138 s wall / 134 s internal |

At 60 frames the full-frame correlation drops to 0.899: the C series is taken
over 120 mostly-faint stars (Python's probe frame yields 66) and the σ-grid fit
is biased on the sharpest frames, which also selects a slightly different
reference image. The reference is built from the k=min(8,n) sharpest frames
(`pws.py:4245`); using *all* frames instead costs 0.865 → fix that made this
0.899.

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
