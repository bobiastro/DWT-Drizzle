//! GPU (wgpu -> DX12/Vulkan) path for the stacking hot loops.
//!
//! Four compute kernels mirror the CPU code exactly:
//!   1. `k_weights` — per input pixel: `W_i = exp(clamp(A·R·lnC + b·lnS))`, normalized
//!      so that `mean_i W = 1`, with the `max_share` log-cap applied first.
//!   2. `k_reject`  — per input pixel: sigma-clip across frames (`stack::reject`),
//!      including the sky-regime one-sided rule and the `_ok`/`_one` gating.
//!   3. `k_mean`    — per output pixel: weighted mean + N_eff (non-drizzle path).
//!   4. `k_accum`   — per output pixel: drizzle gather over all frames and all
//!      source pixels whose drop touches the output pixel.
//!
//! The only host-side quantity is `m` (the global 99th-percentile ratio in
//! `stack_strip`), which is a whole-image reduction and is 0 for this dataset.

use crate::image::Img;

const WGSL: &str = r#"
struct P {
  n: u32,
  np: u32,
  ow: u32,
  oh: u32,
  fh: u32,
  fw: u32,
  a_exp: f32,
  b_exp: f32,
  beta: f32,
  cap_l: f32,
  sky: f32,
  rej_k: f32,
  rej_frac: f32,
  m_ratio: f32,
  s: f32,
  oxp: f32,
  oyp: f32,
  pixfrac: f32,
  expected: f32,
  mode: u32,
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
  _pad3: f32,
};

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> rfield: array<f32>;
@group(0) @binding(2) var<storage, read> cs: array<f32>;      // n C then n S
@group(0) @binding(3) var<storage, read> sig: array<f32>;     // per-frame sigma
@group(0) @binding(4) var<storage, read_write> frames: array<f32>;
@group(0) @binding(5) var<storage, read_write> w: array<f32>;
@group(0) @binding(6) var<storage, read> sh: array<f32>;      // n ux then n uy
@group(0) @binding(7) var<storage, read> dumps: array<f32>;   // unused placeholder
@group(0) @binding(8) var<storage, read_write> outv: array<f32>;
@group(0) @binding(9) var<storage, read_write> neff: array<f32>;

fn srt(v: ptr<function, array<f32, 32>>, n: u32) {
  for (var i = 1u; i < n; i = i + 1u) {
    let key = (*v)[i];
    var j = i32(i) - 1;
    loop {
      if (j < 0) { break; }
      if ((*v)[u32(j)] <= key) { break; }
      (*v)[u32(j) + 1u] = (*v)[u32(j)];
      j = j - 1;
    }
    (*v)[u32(j) + 1u] = key;
  }
}

fn med(v: ptr<function, array<f32, 32>>, n: u32) -> f32 {
  if (n == 0u) { return 0.0; }
  srt(v, n);
  if (n % 2u == 1u) { return (*v)[n / 2u]; }
  return 0.5 * ((*v)[n / 2u - 1u] + (*v)[n / 2u]);
}

fn overlap(fp: f32, d: f32) -> f32 {
  let v = (fp + 0.5) - abs(d);
  return clamp(v, 0.0, min(2.0 * fp, 1.0));
}

@compute @workgroup_size(64)
fn k_weights(@builtin(global_invocation_id) gid: vec3<u32>) {
  let px = gid.x;
  if (px >= p.np) { return; }
  let rp = rfield[px];
  var sum = 0.0;
  for (var i = 0u; i < p.n; i = i + 1u) {
    let lc = log(max(cs[i], 1e-12));
    let ls = log(max(cs[p.n + i], 1e-12));
    var lw = p.a_exp * rp * lc + (p.b_exp * (1.0 - rp) + p.beta * rp) * ls;
    if (p.cap_l < 1e29) { lw = clamp(lw, -p.cap_l, p.cap_l); }
    let val = exp(lw);
    w[i * p.np + px] = val;
    sum = sum + val;
  }
  let mean = max(sum / f32(p.n), 1e-30);
  for (var i = 0u; i < p.n; i = i + 1u) {
    w[i * p.np + px] = w[i * p.np + px] / mean;
  }
}

@compute @workgroup_size(64)
fn k_reject(@builtin(global_invocation_id) gid: vec3<u32>) {
  let px = gid.x;
  if (px >= p.np) { return; }
  if (p.rej_k <= 0.0) { return; }
  let rp = rfield[px];
  var dev: array<f32, 32>;
  var tmp: array<f32, 32>;
  var bad: array<f32, 32>;
  var d0 = array<f32, 32>();
  for (var i = 0u; i < p.n; i = i + 1u) {
    d0[i] = frames[i * p.np + px];
    tmp[i] = d0[i];
  }
  let anchor = med(&tmp, p.n);
  for (var i = 0u; i < p.n; i = i + 1u) {
    dev[i] = abs(d0[i] - anchor) / max(sig[i], 1e-12);
    tmp[i] = dev[i];
  }
  let mad = 1.4826 * med(&tmp, p.n);
  let sig_px = max(anchor - p.sky, 0.0);
  for (var i = 0u; i < p.n; i = i + 1u) {
    let thr = p.rej_k * mad + p.m_ratio * rp * sig_px / max(sig[i], 1e-12);
    bad[i] = select(0.0, 1.0, dev[i] > thr);
  }
  if (rp < 0.1) {
    var dn: array<f32, 32>;
    var s2v: array<f32, 32>;
    for (var i = 0u; i < p.n; i = i + 1u) {
      dn[i] = d0[i] / max(sig[i], 1e-12);
      tmp[i] = dn[i];
    }
    srt(&tmp, p.n);
    // numpy percentile (linear interpolation), same as the CPU path
    let pos = 0.25 * f32(p.n - 1u);
    let lo = u32(floor(pos));
    let fr = pos - f32(lo);
    let hi = min(lo + 1u, p.n - 1u);
    let q25 = tmp[lo] + (tmp[hi] - tmp[lo]) * fr;
    for (var i = 0u; i < p.n; i = i + 1u) { s2v[i] = abs(dn[i] - q25); }
    let s2 = max(1.4826 * med(&s2v, p.n), 1e-12);
    var nb = 0u;
    for (var i = 0u; i < p.n; i = i + 1u) {
      if ((dn[i] - q25) / s2 > p.rej_k) { nb = nb + 1u; }
    }
    let okb = select(0.0, 1.0, f32(nb) <= 0.8 * f32(p.n));
    for (var i = 0u; i < p.n; i = i + 1u) {
      bad[i] = select(0.0, 1.0, (dn[i] - q25) / s2 > p.rej_k) * okb;
    }
  }
  var nbad = 0.0;
  var nhi = 0.0;
  for (var i = 0u; i < p.n; i = i + 1u) {
    if (bad[i] > 0.5) {
      nbad = nbad + 1.0;
      // CPU counts one-sided outliers only where dev>thr as well (sky pixels
      // can be flagged by the one-sided rule while dev<=thr)
      let thr_i = p.rej_k * mad + p.m_ratio * rp * max(anchor - p.sky, 0.0) / max(sig[i], 1e-12);
      if (dev[i] > thr_i && d0[i] > anchor) { nhi = nhi + 1.0; }
    }
  }
  let ok = select(0.0, 1.0, nbad - p.expected <= max(1.0, p.rej_frac * f32(p.n)));
  let one = select(0.0, 1.0,
    (nbad - nhi <= 0.05 * f32(p.n)) && (nhi - p.expected <= 0.7 * f32(p.n)));
  let gate = max(ok, one);
  if (gate < 0.5) {
    for (var i = 0u; i < p.n; i = i + 1u) { w[i * p.np + px] = w[i * p.np + px]; }
    return;
  }
  for (var i = 0u; i < p.n; i = i + 1u) {
    if (bad[i] > 0.5) { w[i * p.np + px] = 0.0; }
  }
}

@compute @workgroup_size(64)
fn k_mean(@builtin(global_invocation_id) gid: vec3<u32>) {
  let px = gid.x;
  if (px >= p.np) { return; }
  var sw = 0.0;
  var sw2 = 0.0;
  var acc = 0.0;
  for (var i = 0u; i < p.n; i = i + 1u) {
    let wt = w[i * p.np + px];
    sw = sw + wt;
    sw2 = sw2 + wt * wt;
    acc = acc + wt * frames[i * p.np + px];
  }
  let sd = max(sw, 1e-30);
  outv[px] = acc / sd;
  neff[px] = (sw * sw) / max(sw2, 1e-30);
}

@compute @workgroup_size(64)
fn k_accum(@builtin(global_invocation_id) gid: vec3<u32>) {
  let q = gid.x;
  if (q >= p.ow * p.oh) { return; }
  let oxi = q % p.ow;
  let oyi = q / p.ow;
  let cx = f32(oxi) + 0.5;
  let cy = f32(oyi) + 0.5;
  let fp = 0.5 * p.s * p.pixfrac;
  let reach = i32(ceil(fp / p.s)) + 1;
  var num = 0.0;
  var den = 0.0;
  var sw2 = 0.0;
  for (var f = 0u; f < p.n; f = f + 1u) {
    let ux = sh[f];
    let uy = sh[p.n + f];
    let sx = (cx - p.oxp) / p.s - 0.5 - ux;
    let sy = (cy - p.oyp) / p.s - 0.5 - uy;
    let i0 = max(i32(round(sx)) - reach, 0);
    let i1 = min(i32(round(sx)) + reach, i32(p.fw) - 1);
    let j0 = max(i32(round(sy)) - reach, 0);
    let j1 = min(i32(round(sy)) + reach, i32(p.fh) - 1);
    if (i0 > i1 || j0 > j1) { continue; }
    for (var y = j0; y <= j1; y = y + 1) {
      let fy = p.s * (f32(y) + 0.5 + uy) + p.oyp;
      let wy = overlap(fp, cy - fy);
      if (wy <= 0.0) { continue; }
      for (var x = i0; x <= i1; x = x + 1) {
        let fx = p.s * (f32(x) + 0.5 + ux) + p.oxp;
        let wx = overlap(fp, cx - fx);
        if (wx <= 0.0) { continue; }
        let idx = f * p.np + u32(y) * p.fw + u32(x);
        let wt = w[idx];
        if (wt <= 0.0) { continue; }
        let a = wx * wy;
        num = num + a * frames[idx] * wt;
        den = den + a * wt;
        sw2 = sw2 + a * a * wt * wt;
      }
    }
  }
  if (den > 0.0) { outv[q] = num / den; } else { outv[q] = -1.0; }
  neff[q] = (den * den) / max(sw2, 1e-30);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n: u32,
    np: u32,
    ow: u32,
    oh: u32,
    fh: u32,
    fw: u32,
    a_exp: f32,
    b_exp: f32,
    beta: f32,
    cap_l: f32,
    sky: f32,
    rej_k: f32,
    rej_frac: f32,
    m_ratio: f32,
    s: f32,
    oxp: f32,
    oyp: f32,
    pixfrac: f32,
    expected: f32,
    mode: u32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}

/// Everything the GPU needs for one stack (frames already sky-shifted).
pub struct GpuInput<'a> {
    pub frames: &'a [Img],
    pub r: &'a Img,
    pub c: &'a [f32],
    pub s: &'a [f32],
    pub sig: &'a [f32],
    pub a_exp: f32,
    pub b_exp: f32,
    pub beta: f32,
    pub cap: f32,
    pub sky: f32,
    pub rej_k: f32,
    pub rej_frac: f32,
    pub m_ratio: f32,
    pub s_scale: f32,
    pub ox: f32,
    pub oy: f32,
    pub pixfrac: f32,
    pub ow: usize,
    pub oh: usize,
    pub ux: &'a [f32],
    pub uy: &'a [f32],
}

fn make_instance() -> wgpu::Instance {
    wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::DX12 | wgpu::Backends::VULKAN,
        ..Default::default()
    })
}

fn pick_adapter() -> Result<wgpu::Adapter, String> {
    let instance = make_instance();
    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .map_err(|e| format!("{e}"))
}

/// Human-readable adapter summary for the run log.
pub fn adapter_info() -> String {
    match pick_adapter() {
        Ok(a) => {
            let i = a.get_info();
            format!("{} [{:?} {:?}]", i.name, i.backend, i.device_type)
        }
        Err(e) => format!("unavailable ({e})"),
    }
}

/// True when a DX12/Vulkan adapter is present.
pub fn available() -> bool {
    pick_adapter().is_ok()
}

/// Run weights -> rejection -> mean/drizzle on the GPU.
pub fn stack_gpu(inp: &GpuInput) -> Result<(Img, Vec<f32>), String> {
    let t_all = std::time::Instant::now();
    let n = inp.frames.len();
    if n == 0 || n > 32 {
        return Err(format!("gpu: frame count {n} out of range 1..=32"));
    }
    let (h, w) = (inp.r.h, inp.r.w);
    let np = h * w;
    let nout = inp.ow * inp.oh;

    let instance = make_instance();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .map_err(|e| format!("request_adapter: {e}"))?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("dwt-rs"),
        required_features: wgpu::Features::empty(),
        required_limits: adapter.limits(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .map_err(|e| format!("request_device: {e}"))?;
    let t_init = t_all.elapsed();

    let mut flat: Vec<f32> = Vec::with_capacity(n * np);
    for f in inp.frames {
        if f.d.len() != np {
            return Err("gpu: frame size mismatch".into());
        }
        flat.extend_from_slice(&f.d);
    }
    let mut cs: Vec<f32> = Vec::with_capacity(2 * n);
    cs.extend_from_slice(inp.c);
    cs.extend_from_slice(inp.s);
    let mut sh: Vec<f32> = Vec::with_capacity(2 * n);
    sh.extend_from_slice(inp.ux);
    sh.extend_from_slice(inp.uy);

    let l_cap = if inp.cap > 0.0 && inp.cap < 1.0 && n > 1 {
        0.5 * (((n - 1) as f32 * inp.cap / (1.0 - inp.cap)).max(1e-9)).ln()
    } else {
        1e30
    };
    let params = Params {
        n: n as u32,
        np: np as u32,
        ow: inp.ow as u32,
        oh: inp.oh as u32,
        fh: h as u32,
        fw: w as u32,
        a_exp: inp.a_exp,
        b_exp: inp.b_exp,
        beta: inp.beta,
        cap_l: l_cap,
        sky: inp.sky,
        rej_k: inp.rej_k,
        rej_frac: inp.rej_frac,
        m_ratio: inp.m_ratio,
        s: inp.s_scale,
        oxp: inp.ox,
        oyp: inp.oy,
        pixfrac: inp.pixfrac,
        expected: n as f32 * erfc(inp.rej_k / std::f32::consts::SQRT_2),
        mode: 0,
        _pad0: 0.0,
        _pad1: 0.0,
        _pad2: 0.0,
        _pad3: 0.0,
    };

    let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
    let mk = |bytes: &[u8], u: wgpu::BufferUsages, label: &str| -> wgpu::Buffer {
        let b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (bytes.len() as u64).max(4),
            usage: u,
            mapped_at_creation: false,
        });
        queue.write_buffer(&b, 0, bytes);
        b
    };
    let b_uni = mk(
        bytemuck::bytes_of(&params),
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        "params",
    );
    let b_r = mk(bytemuck::cast_slice(&inp.r.d), usage, "r");
    let b_cs = mk(bytemuck::cast_slice(&cs), usage, "cs");
    let b_sig = mk(bytemuck::cast_slice(inp.sig), usage, "sig");
    let b_frames = mk(bytemuck::cast_slice(&flat), usage, "frames");
    let b_w = mk(bytemuck::cast_slice(&vec![0.0f32; n * np]), usage, "w");
    let b_sh = mk(bytemuck::cast_slice(&sh), usage, "sh");
    let b_dump = mk(bytemuck::cast_slice(&[0.0f32; 4]), usage, "dump");
    let b_out = mk(bytemuck::cast_slice(&vec![0.0f32; np.max(nout)]), usage, "out");
    let b_neff = mk(bytemuck::cast_slice(&vec![0.0f32; np.max(nout)]), usage, "neff");

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("dwt"),
        source: wgpu::ShaderSource::Wgsl(WGSL.into()),
    });

    let ro = wgpu::BufferBindingType::Storage { read_only: true };
    let rw = wgpu::BufferBindingType::Storage { read_only: false };
    let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            entry(0, wgpu::BufferBindingType::Uniform),
            entry(1, ro),
            entry(2, ro),
            entry(3, ro),
            entry(4, rw),
            entry(5, rw),
            entry(6, ro),
            entry(7, ro),
            entry(8, rw),
            entry(9, rw),
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });
    let pipe = |name: &str| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(name),
            layout: Some(&pl),
            module: &module,
            entry_point: Some(name),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let p_w = pipe("k_weights");
    let p_r = pipe("k_reject");
    let p_m = pipe("k_mean");
    let p_a = pipe("k_accum");

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: b_uni.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: b_r.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: b_cs.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: b_sig.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: b_frames.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: b_w.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: b_sh.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: b_dump.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: b_out.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 9, resource: b_neff.as_entire_binding() },
        ],
    });

    let encode_all = || -> wgpu::CommandBuffer {
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("weights"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &bg, &[]);
        pass.set_pipeline(&p_w);
        pass.dispatch_workgroups((np as u32).div_ceil(64), 1, 1);
    }
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("reject"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &bg, &[]);
        pass.set_pipeline(&p_r);
        pass.dispatch_workgroups((np as u32).div_ceil(64), 1, 1);
    }
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("accum"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &bg, &[]);
        pass.set_pipeline(&p_a);
        pass.dispatch_workgroups((nout as u32).div_ceil(64), 1, 1);
    }
    enc.finish()
    };
    let reps: usize = std::env::var("DWT_GPU_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(1).max(1);
    let t_setup = t_all.elapsed();
    let t_comp = std::time::Instant::now();
    for _ in 0..reps {
        queue.submit(Some(encode_all()));
        // wait for this submission so each rep is a full compute round trip
        let _ = device.poll(wgpu::PollType::wait());
    }

    let read = |buf: &wgpu::Buffer, count: usize| -> Result<Vec<f32>, String> {
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (count * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(buf, 0, &staging, 0, (count * 4) as u64);
        queue.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device
            .poll(wgpu::PollType::wait())
            .map_err(|e| format!("poll: {e}"))?;
        rx.recv()
            .map_err(|e| format!("map channel: {e}"))?
            .map_err(|e| format!("map: {e}"))?;
        let data = slice.get_mapped_range();
        let vals: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        Ok(vals)
    };

    let t_read = std::time::Instant::now();
    let raw = read(&b_out, nout)?;
    let mut out = Img::new(inp.ow, inp.oh);
    for i in 0..nout {
        out.d[i] = if raw[i] < 0.0 { f32::NAN } else { raw[i] };
    }
    // N_eff only exists on the same grid when no drizzle upsampling happened
    let neff = if nout == np { read(&b_neff, np)? } else { vec![f32::NAN; nout] };
    if std::env::var("DWT_GPU_TIMING").is_ok() {
        let t_done = t_comp.elapsed().as_secs_f64();
        println!(
            "[GPU-T] init {:.3}s setup {:.3}s compute {:.3}s/reps={reps} ({:.4}s per rep) readback {:.4}s",
            t_init.as_secs_f64(),
            t_setup.as_secs_f64(),
            t_done,
            t_done / reps as f64,
            t_read.elapsed().as_secs_f64()
        );
    }
    Ok((out, neff))
}

fn erfc(x: f32) -> f32 {
    let z = x.abs();
    if z > 3.0 {
        return if x > 0.0 { 0.0 } else { 2.0 };
    }
    let t = 1.0 / (1.0 + 0.3275911 * z);
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let v = poly * (-z * z).exp();
    if x >= 0.0 { v } else { 2.0 - v }
}
