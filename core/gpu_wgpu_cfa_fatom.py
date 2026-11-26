"""真·散射累加后端（float32 原子加）。

与分桶后端的区别：不保存样本，直接对输出像素做原子累加，显存只占 6 个画布大小的
float 缓冲；每条带把帧数据上传一次，三遍扫描都在显存里完成。

数学与 CPU 的 ``_scatter_band`` 一致：每个源像素按液滴盒与输出像素的重叠面积
``wx*wy/P^2`` 摊到相邻像素，权重 ``w = C^(A*R) * S^(B*(1-R)) * T^fp``。

三遍：
  pass0  Σw, Σwv, Σw²      → μ
  pass1  Σw·min(|v−μ|,3σ)  → winsor 尺度 s = 1.15·Σw|dev|/Σw
  pass2  只保留 |v−μ| ≤ k·s → Σw, Σwv, Σw²（成品）
"""
from __future__ import annotations

import math
import os
import time
import time
from typing import Dict

import numpy as np

try:  # 既支持 -m core.pws，也支持直接跑脚本
    from . import gpu_wgpu as G
except ImportError:  # pragma: no cover
    import gpu_wgpu as G


_SCATTER = """
struct QP {
  a: f32, b: f32, c: f32, d: f32, e: f32, f: f32,
  w_src: u32, h_src: u32, w_out: u32, h_out: u32,
  y_off: u32, c0: u32, r0: u32, mask: u32,
  P: f32, lnc: f32, lns: f32, wa: f32, wb: f32, has_r: f32, kclip: f32,
  mode: f32, s_off: u32, lnt: f32, mgate: f32, sky: f32, sfac: f32, beta: f32,
  kt: u32, capl: f32, klsp: f32, khalf: f32,
};

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read> rmap: array<f32>;
@group(0) @binding(2) var<uniform> Q: QP;
@group(0) @binding(3) var<storage, read_write> aW: array<atomic<f32>>;
@group(0) @binding(4) var<storage, read_write> aV: array<atomic<f32>>;
@group(0) @binding(5) var<storage, read_write> aDev: array<atomic<f32>>;
@group(0) @binding(6) var<storage, read> mu: array<f32>;
@group(0) @binding(7) var<storage, read> sd: array<f32>;
@group(0) @binding(8) var<storage, read_write> aW2: array<atomic<f32>>;
@group(0) @binding(9) var<storage, read_write> aBad: array<atomic<u32>>;
@group(0) @binding(10) var<storage, read> ctile: array<f32>;
@group(0) @binding(11) var<storage, read_write> aWe: array<atomic<f32>>;
@group(0) @binding(12) var<storage, read_write> aWDe: array<atomic<f32>>;
@group(0) @binding(13) var<storage, read_write> aCnte: array<atomic<f32>>;
@group(0) @binding(14) var<storage, read_write> aHits: array<atomic<f32>>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let px = gid.x;
  let py = gid.y;
  if (px >= Q.w_src || py >= Q.h_src) { return; }
  let gx0 = px + Q.c0;
  let gy0 = py + Q.r0;
  if (((Q.mask >> ((gy0 & 1u) * 2u + (gx0 & 1u))) & 1u) == 0u) { return; }
  let v = src[Q.s_off + py * Q.w_src + px];
  if (!(v == v)) { return; }

  let X = Q.a * (f32(gx0) + 0.5) + Q.b * (f32(gy0) + 0.5) + Q.c;
  let Y = Q.d * (f32(gx0) + 0.5) + Q.e * (f32(gy0) + 0.5) + Q.f;
  let hx = 0.5 * Q.P;
  let x0 = X - hx;
  let x1 = X + hx;
  let y0 = Y - hx;
  let y1 = Y + hx;
  let kx0 = i32(floor(x0));
  let ky0 = i32(floor(y0));
  let noff = i32(ceil(Q.P)) + 1;
  let mode = i32(Q.mode);

  for (var aix: i32 = 0; aix < noff; aix = aix + 1) {
    let kx = kx0 + aix;
    if (kx < 0 || kx >= i32(Q.w_out)) { continue; }
    let wx = min(x1, f32(kx + 1)) - max(x0, f32(kx));
    if (wx <= 0.0) { continue; }
    for (var aiy: i32 = 0; aiy < noff; aiy = aiy + 1) {
      let ky = ky0 + aiy;
      let gy = ky - i32(Q.y_off);
      if (gy < 0 || gy >= i32(Q.h_out)) { continue; }
      let wy = min(y1, f32(ky + 1)) - max(y0, f32(ky));
      if (wy <= 0.0) { continue; }
      let idx = u32(gy) * Q.w_out + u32(kx);
      let rr = select(0.5, rmap[(u32(gy) + Q.y_off) * Q.w_out + u32(kx)], Q.has_r > 0.5);
      var lcg = Q.lnc;
      if (Q.kt >= 2u) {
        // 逐瓦片清晰度：**双线性插值**，权重随位置连续变化 → 区域之间没有台阶
        let fk = f32(Q.kt);
        let fx = (f32(gx0) + 0.5) * fk / f32(Q.w_src) - 0.5;
        let fy = (f32(gy0) + 0.5) * fk / f32(Q.h_src) - 0.5;
        let ix0 = clamp(i32(floor(fx)), 0, i32(Q.kt) - 1);
        let iy0 = clamp(i32(floor(fy)), 0, i32(Q.kt) - 1);
        let ix1 = min(ix0 + 1, i32(Q.kt) - 1);
        let iy1 = min(iy0 + 1, i32(Q.kt) - 1);
        let tx = clamp(fx - f32(ix0), 0.0, 1.0);
        let ty = clamp(fy - f32(iy0), 0.0, 1.0);
        let u = u32(Q.kt);
        let c00 = ctile[u32(iy0) * u + u32(ix0)];
        let c10 = ctile[u32(iy0) * u + u32(ix1)];
        let c01 = ctile[u32(iy1) * u + u32(ix0)];
        let c11 = ctile[u32(iy1) * u + u32(ix1)];
        lcg = mix(mix(c00, c10, tx), mix(c01, c11, tx), ty);
      }
      // 单帧份额上限：把**总对数权重**截在 ±capl（capl=0 表示不限额）
      var le = Q.wa * rr * lcg + (Q.wb * (1.0 - rr) + Q.beta * rr) * Q.lns;
      if (Q.capl > 0.0) { le = clamp(le, -Q.capl, Q.capl); }
      let wgt = exp(le);
      let w = wgt * wx * wy / (Q.P * Q.P) * exp(Q.lnt);

      if (mode == 0) {
        atomicAdd(&aW[idx], w);
        atomicAdd(&aV[idx], w * v);
        atomicAdd(&aW2[idx], w * w);
      } else if (mode == 1) {
        atomicAdd(&aDev[idx], w * abs(v - mu[idx]));
      } else if (mode == 4) {
        let m = mu[idx];
        let lim = sd[idx];
        atomicAdd(&aV[idx], w * clamp(v, m - lim, m + lim));
      } else if (mode == 3) {
        // winsor 第二段：截尾到 k·σ（lim 存在 sd 里）
        atomicAdd(&aDev[idx], w * min(abs(v - mu[idx]), sd[idx]));
      } else {
        // 分块尺度只在抽样点上有值；其余像素退回本像素的 1.15·Σw|dev|/Σw
        let sref = select(1.15 * aDev[idx] / max(aW[idx], 1e-20), sd[idx], sd[idx] > 1e-20);
        let rr = select(0.5, rmap[idx], Q.has_r > 0.5);
        let sigv = max(mu[idx] - Q.sky, 0.0);
        let thr = Q.kclip * sref * Q.sfac + Q.mgate * rr * sigv;
        if (Q.kclip > 0.0 && abs(v - mu[idx]) > thr) {
          atomicAdd(&aBad[0], 1u);
          continue;
        }
        atomicAdd(&aBad[1], 1u);
        atomicAdd(&aW[idx], w);
        atomicAdd(&aV[idx], w * v);
        atomicAdd(&aW2[idx], w * w);
        if (Q.klsp > 0.5) { atomicAdd(&aHits[idx], 1.0); }
        if (Q.klsp > 0.5 && abs(v - mu[idx]) > Q.khalf * sref) {
          atomicAdd(&aWe[idx], w);
          atomicAdd(&aWDe[idx], w * v);
          atomicAdd(&aCnte[idx], 1.0);
        }
      }
    }
  }
}
"""

_STATS = """
struct S { n: u32, mode: u32, w_out: u32, h_out: u32, blk: u32, st: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read> aW: array<f32>;
@group(0) @binding(1) var<storage, read> aV: array<f32>;
@group(0) @binding(2) var<storage, read> aD: array<f32>;
@group(0) @binding(3) var<storage, read_write> mu: array<f32>;
@group(0) @binding(4) var<storage, read_write> sd: array<f32>;
@group(0) @binding(5) var<uniform> U: S;

// 第一遍：逐像素 μ
@compute @workgroup_size(64)
fn mean_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x + U.pad0;
  if (i >= U.n) { return; }
  let w = aW[i];
  if (w > 1e-20) { mu[i] = aV[i] / w; }
}

// 第二遍：与 CPU 同口径的分块 winsor 尺度（blk=64、st=3 抽样、1.15×）
@compute @workgroup_size(64)
fn block_scale(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x + U.pad0;
  if (i >= U.n) { return; }
  let x = i % U.w_out;
  let y = i / U.w_out;
  if ((x % U.st) != 0u || (y % U.st) != 0u) { return; }
  let w = aW[i];
  if (w <= 1e-20) { return; }
  sd[i] = aD[i] / w;
}
"""

_FINAL = """
struct F { n: u32, pad: u32 };
@group(0) @binding(0) var<storage, read> aW: array<f32>;
@group(0) @binding(1) var<storage, read> aV: array<f32>;
@group(0) @binding(2) var<storage, read> aW2: array<f32>;
@group(0) @binding(3) var<storage, read_write> oOut: array<f32>;
@group(0) @binding(4) var<storage, read_write> oCov: array<f32>;
@group(0) @binding(5) var<storage, read_write> oNef: array<f32>;
@group(0) @binding(6) var<uniform> U: F;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x + U.pad;
  if (i >= U.n) { return; }
  let w = aW[i];
  if (w <= 1e-20) { oOut[i] = 0.0; oCov[i] = 0.0; oNef[i] = 0.0; return; }
  oOut[i] = aV[i] / w;
  oCov[i] = w;
  oNef[i] = (w * w) / max(aW2[i], 1e-20);
}
"""

_PARITY = {
    'RGGB': {'R': 0b0001, 'G': 0b0110, 'B': 0b1000},
    'BGGR': {'R': 0b1000, 'G': 0b0110, 'B': 0b0001},
    'GRBG': {'R': 0b0010, 'G': 0b1001, 'B': 0b0100},
    'GBRG': {'R': 0b0100, 'G': 0b1001, 'B': 0b0010},
}

_CACHE: Dict[str, object] = {}


def available() -> bool:
    try:
        return bool(G.available() and G.has_f32_atomic())
    except Exception:
        return False


def _pipes(dev):
    if 'p' in _CACHE:
        return _CACHE['p'], _CACHE['ps'], _CACHE['pf'], _CACHE['pbs']
    mp = dev.create_shader_module(code=_SCATTER)
    ms = dev.create_shader_module(code=_STATS)
    mf = dev.create_shader_module(code=_FINAL)
    p = dev.create_compute_pipeline(layout='auto', compute={'module': mp, 'entry_point': 'main'})
    ps = dev.create_compute_pipeline(layout='auto', compute={'module': ms, 'entry_point': 'mean_pass'})
    pbs = dev.create_compute_pipeline(layout='auto', compute={'module': ms, 'entry_point': 'block_scale'})
    pf = dev.create_compute_pipeline(layout='auto', compute={'module': mf, 'entry_point': 'main'})
    _CACHE.update({'p': p, 'ps': ps, 'pf': pf, 'pbs': pbs})
    return p, ps, pf, pbs


def stack_cfa_fatom(fs, C, S, dx, dy, th, kk, plan, A, B, rej_k, rej_gate,
                    R=None, pixfrac=1.0, band_rows=512, T=None, fp=1.0,
                    log=None, pattern=None, channel=None, beta=None, Ctiles=None,
                    cap=None, want_p=False, **kw):
    if not available():
        raise RuntimeError('适配器不支持 shader-float32-atomic')
    dev, queue = G._device_f32()
    import time as _t
    _t0 = _t.perf_counter()
    _dbg = os.environ.get('DWT_WGPU_DEBUG') == '1'
    if log:
        log('[wgpu][start] ts=%.3f backend=canvas frames=%d scale=%.2f pixfrac=%.2f'
            % (time.time(), len(np.atleast_1d(dx)), float(plan.get('s', 1.0)), float(pixfrac)))
    _stage_t = {'t': _t.perf_counter()}

    def _dbg_stage(_m):
        if _dbg:
            _now = _t.perf_counter()
            print('[fatom-stage] %-14s %.3fs (cum %.3fs)'
                  % (_m, _now - _stage_t['t'], _now - _t0), flush=True)
            _stage_t['t'] = _now
    wgpu = G._wp()
    U = wgpu.BufferUsage

    H, W = int(plan['H']), int(plan['W'])
    s_grid = float(plan.get('s', 1.0))
    ox, oy = float(plan.get('ox', 0.0)), float(plan.get('oy', 0.0))
    h, w = int(fs.shape[0]), int(fs.shape[1])
    # CFA：同色格点间隔 2 源像素，液滴边长要乘 2，否则 2× 输出会留 3/4 覆盖空洞
    _pm = float(os.environ.get('DWT_CFA_PMUL', '0') or 0)
    # 与 CPU 同一策略：帧数 ≥30 用样本足迹（锐，靠抖动填相位），
    # 帧少用 2s 保覆盖（否则 8 帧实测 12% 空洞）。DWT_CFA_PMUL 可覆盖。
    n = len(dx)
    # 同 CPU：真 CFA 的液滴按同色格点间距（2 源像素）铺满，避免 2×2 相位调制。
    _pm = _pm if _pm > 0 else 2.0
    _pov = float(os.environ.get('DWT_CFA_POVER', '1.0' if float(s_grid) <= 1.0 else '1.15') or 1.0) if pattern is not None else 1.0
    P = max(1e-6, float(pixfrac) * s_grid * _pm * _pov)
    idx = [i for i in range(n) if np.isfinite(dx[i]) and np.isfinite(dy[i])
           and (th is None or np.isfinite(th[i]))]
    if len(idx) < 2:
        raise RuntimeError('有效帧不足')

    mask2 = _PARITY[pattern][channel] if pattern else 0b1111
    stack = np.zeros((H, W), np.float32)
    cov = np.zeros((H, W), np.float32)
    neff = np.zeros((H, W), np.float32)

    p = np.zeros(32, np.float32)
    # 排异的信号比例门（对齐 CPU 的 thr = rej_k·σ + m·(R·信号)）：
    # m 是标量，CPU 从逐帧 dev/sig 的 99 分位再取中位估出来；这里先按经验值，
    # 用 DWT_MGATE 可调（0 = 退回旧的纯 σ 判据）。
    p[24] = np.float32(float(os.environ.get('DWT_MGATE', '0.03') or 0.03))
    p[25] = np.float32(float(getattr(fs, 'sky0', 0.0) or 0.0))
    p[26] = np.float32(float(os.environ.get('DWT_SFAC', '1.0') or 1.0))
    # 满月帧补偿：由调用方按素材类型显式传入（黑白恒 0），不再从环境变量读，
    # 否则同一进程内先跑彩机再跑黑白时会串味（2026-11-23 实测确有 0.0007% 均值、
    # 局部最大 56% 的差异）。
    p[27] = np.float32(float(beta) if beta is not None
                       else float(os.environ.get('DWT_DETAIL_SNR', '0') or 0))
    pu = p.view(np.uint32)
    pu[13] = np.uint32(mask2)
    # 单帧份额上限：cap∈(0,1) 时换算成对数幅度上限 L=ln((n-1)cap/(1-cap))，写进 p[29]
    try:
        _cap = float(cap) if cap is not None else 0.0
    except Exception:
        _cap = 0.0
    _nf = int(np.atleast_1d(dx).size)
    if 0.0 < _cap < 1.0 and _nf > 1:
        p[29] = np.float32(0.5 * math.log(max((_nf - 1) * _cap / (1.0 - _cap), 1e-9)))
    else:
        p[29] = np.float32(0.0)
    p[30] = np.float32(1.0 if float(os.environ.get('DWT_LSP', '0') or 0) > 0.5 else 0.0)
    # want_p：逐帧记录着色器实际使用的 2×3 变换 [p0,p1,p2,p3,p4,p5]
    #   X = p0*(gx+0.5) + p1*(gy+0.5) + p2
    #   Y = p3*(gx+0.5) + p4*(gy+0.5) + p5
    # 帧域验证需要它的**逆**（与 drizzle 逐位同源，避免在外层用 dx/dy/th/kk 重算）
    _pcol = {}
    p[31] = np.float32(max(1.0, 0.5 * float(rej_k)))
    p[14] = np.float32(P)
    p[17] = np.float32(float(A))
    p[18] = np.float32(float(B))
    p[19] = np.float32(1.0 if R is not None else 0.0)
    _kt = 0 if Ctiles is None else int(getattr(Ctiles, 'shape', [0, 0])[1] or 0)
    pu[28] = np.uint32(_kt)   # 必须写 uint32 视图，否则 float32 会写成 8.0 → 位型 0x41000000

    pipe, pstat, pfin, pbs = _pipes(dev)
    bgl = pipe.get_bind_group_layout(0)

    def sbuf(nel):
        return dev.create_buffer(size=max(16, int(nel) * 4),
                                 usage=U.STORAGE | U.COPY_SRC | U.COPY_DST)

    for y0 in range(0, H, band_rows):
        bh = min(band_rows, H - y0)
        n_pix = int(bh * W)
        aW, aV, aD, aW2 = sbuf(n_pix), sbuf(n_pix), sbuf(n_pix), sbuf(n_pix)
        aBad = dev.create_buffer(size=8, usage=U.STORAGE | U.COPY_SRC | U.COPY_DST)
        _lsp_on = float(os.environ.get('DWT_LSP', '0') or 0) > 0.5
        if _lsp_on:
            aWe = dev.create_buffer(size=n_pix * 4, usage=U.STORAGE | U.COPY_SRC | U.COPY_DST)
            aWDe = dev.create_buffer(size=n_pix * 4, usage=U.STORAGE | U.COPY_SRC | U.COPY_DST)
            aCnte = dev.create_buffer(size=n_pix * 4, usage=U.STORAGE | U.COPY_SRC | U.COPY_DST)
            aHits = dev.create_buffer(size=n_pix * 4, usage=U.STORAGE | U.COPY_SRC | U.COPY_DST)
            queue.write_buffer(aHits, 0, np.zeros(n_pix, np.float32).tobytes())
            queue.write_buffer(aWe, 0, np.zeros(n_pix, np.float32).tobytes())
            queue.write_buffer(aWDe, 0, np.zeros(n_pix, np.float32).tobytes())
            queue.write_buffer(aCnte, 0, np.zeros(n_pix, np.float32).tobytes())
        else:
            aWe = dev.create_buffer(size=4, usage=U.STORAGE | U.COPY_SRC | U.COPY_DST)
            aWDe = dev.create_buffer(size=4, usage=U.STORAGE | U.COPY_SRC | U.COPY_DST)
            aCnte = dev.create_buffer(size=4, usage=U.STORAGE | U.COPY_SRC | U.COPY_DST)
            aHits = dev.create_buffer(size=4, usage=U.STORAGE | U.COPY_SRC | U.COPY_DST)
        bMu, bSd = sbuf(n_pix), sbuf(n_pix)
        oOut, oCov, oNef = sbuf(n_pix), sbuf(n_pix), sbuf(n_pix)

        # 统一的元数据 uniform：w_src/h_src/w_out/h_out/y_off/r0
        pu[6] = np.uint32(w)
        pu[8] = np.uint32(W)
        pu[9] = np.uint32(bh)
        pu[10] = np.uint32(y0)
        pu[11] = np.uint32(0)
        bQ = dev.create_buffer(size=128, usage=U.UNIFORM | U.COPY_DST)

        _dbg_stage('alloc+zero')
        def zero_acc():
            for b in (aW, aV, aD, aW2):
                queue.write_buffer(b, 0, np.zeros(n_pix, np.float32).tobytes())

        zero_acc()
        for y in range(bh):
            pass

        # 源行窗口（逐帧逆变换取并集；旋转接近 180° 时源行方向相反）
        lo, hi = 1 << 30, -1
        for j in idx:
            th_i = 0.0 if th is None else math.radians(float(th[j]))
            ct, stn = float(np.cos(th_i)), float(np.sin(th_i))
            ks = float(kk[j])
            d_ = s_grid * ks * stn
            e_ = s_grid * ks * ct
            f_ = s_grid * -float(dy[j]) + oy
            for gx in (0, w - 1):
                for Yb in (y0 - 2 * P - 4, y0 + bh + 2 * P + 4):
                    if abs(e_) < 1e-9:
                        continue
                    gy = (Yb - d_ * (gx + 0.5) - f_) / e_ - 0.5
                    if np.isfinite(gy):
                        lo = min(lo, int(np.floor(gy)))
                        hi = max(hi, int(np.ceil(gy)))
        if os.environ.get('DWT_FATOM_FULLSTRIP', '').strip() == '1':
            lo, hi = 0, h
        lo = max(0, lo - 8)
        hi = min(h, hi + 8)
        if hi <= lo:
            continue
        hs = hi - lo
        plane = hs * w
        pu[7] = np.uint32(hs)
        pu[12] = np.uint32(lo)

        # 绑定大小受 maxStorageBufferBindingSize 限制（4070 Ti 上是 2.147 GB）。
        # 60 帧全幅要 6.2 GB，直接建绑定会被 wgpu 拒绝（GPUValidationError in
        # wgpuDeviceCreateBindGroup，用户实测踩到）。超限时退成「一帧的环形缓冲」：
        # 每帧覆盖写入 offset 0，s_off 传 0，主机与 GPU 依旧重叠。
        try:
            _maxb = int(dev.limits['max-storage-buffer-binding-size'])
        except Exception:
            _maxb = 1 << 31
        _ring = (len(idx) * plane * 4) > int(_maxb * 0.9)
        # 环形模式用两块缓冲轮流：GPU 算第 k 帧（读 buf0）时，队列里第 k+1 帧的拷贝
        # 可以写 buf1，两者在硬件上重叠；只用一块会让「拷贝→计算」严格串行
        # （实测窗口内 GPU 占用 P50 只有 22%）。
        if _ring:
            bSrcs = [sbuf(plane), sbuf(plane)]
            bSrc = bSrcs[0]
        else:
            bSrcs = [sbuf(len(idx) * plane)]
            bSrc = bSrcs[0]
        # 帧数据不在这里上传：交给 scatter(0) 逐帧「上传一帧 → 派发一帧」，
        # 让主机的拷贝与 GPU 的计算重叠（全幅 8 帧实测 5.7s → 约 4s）。

        if R is not None:
            rr = np.asarray(R, np.float32)
            if _dbg:
                print('[fatom-dbg] R in: shape=%s min=%.3f mean=%.3f max=%.3f frac>0.5=%.2f%%'
                      % (rr.shape, float(rr.min()), float(rr.mean()), float(rr.max()),
                         100.0 * float((rr > 0.5).mean())), flush=True)
            if rr.shape != (H, W):
                yy = (np.arange(H) * rr.shape[0] // H).clip(0, rr.shape[0] - 1)
                xx = (np.arange(W) * rr.shape[1] // W).clip(0, rr.shape[1] - 1)
                rr = rr[np.ix_(yy, xx)]
            bR = sbuf(H * W)
            queue.write_buffer(bR, 0, memoryview(np.ascontiguousarray(rr)))
            rsz = H * W * 4
        else:
            bR = sbuf(4)
            queue.write_buffer(bR, 0, np.zeros(4, np.float32).tobytes())
            rsz = 16
        _ctn = max(1, (int(Ctiles.shape[1]) ** 2) if (Ctiles is not None and _kt >= 2) else 1)
        bCt = sbuf(_ctn)
        queue.write_buffer(bCt, 0, np.zeros(_ctn, np.float32).tobytes())

        # 环形模式下源缓冲只能容纳 1~2 帧，任何一趟要读 src 的 pass 都必须
        # 重新把逐帧数据传进来（否则那一趟只看到最后两帧，成品会周期性重复）。
        # 代价是每趟多搬一次帧数据（全幅 60 帧约 6 GB/趟），比走分桶快得多。
        _acc_passes = [0]

        def scatter(mode, kclip, upload=False):
            if _ring and not upload:
                upload = True          # 环形：每一趟都重新上传，保证看得到全部帧
            p[20] = np.float32(kclip)
            p[21] = np.float32(float(mode))
            enc = None if upload else dev.create_command_encoder()
            cp = None if upload else enc.begin_compute_pass()
            for k, j in enumerate(idx):
                if upload:
                    # 先传这一帧，再单独提交它的 dispatch：GPU 算第 k 帧时主机在传第 k+1 帧。
                    # 按行分块上传——全幅一帧 strip 是 99.5 MB，一次性 asarray 在 32 GB 机器上
                    # 叠加 memmap + 常驻画布后仍可能 MemoryError（用户实测踩到）。256 行一块，
                    # 主机峰值只要 6.4 MB。
                    _CH = 256
                    _buse = bSrcs[k % len(bSrcs)] if _ring else bSrc
                    _base = 0 if _ring else (k * plane * 4)
                    for _r0 in range(int(lo), int(hi), _CH):
                        _r1 = min(int(hi), _r0 + _CH)
                        _st = np.asarray(fs.strip(int(j), int(_r0), int(_r1)), np.float32)
                        queue.write_buffer(_buse,
                                           _base + (_r0 - int(lo)) * int(w) * 4,
                                           memoryview(np.ascontiguousarray(_st)))
                        del _st
                    enc = dev.create_command_encoder()
                    cp = enc.begin_compute_pass()
                th_i = 0.0 if th is None else math.radians(float(th[j]))
                ct, stn = float(np.cos(th_i)), float(np.sin(th_i))
                ks = float(kk[j])
                p[0] = np.float32(s_grid * ks * ct)
                p[1] = np.float32(-s_grid * ks * stn)
                p[2] = np.float32(s_grid * -float(dx[j]) + ox)
                p[3] = np.float32(s_grid * ks * stn)
                p[4] = np.float32(s_grid * ks * ct)
                p[5] = np.float32(s_grid * -float(dy[j]) + oy)
                if want_p:
                    _pcol[int(j)] = (float(p[0]), float(p[1]), float(p[2]),
                                     float(p[3]), float(p[4]), float(p[5]))
                pu[22] = np.uint32(0 if _ring else k * plane)
                if C is not None:
                    p[15] = np.float32(np.log(max(float(C[int(j)]), 1e-6)))
                if S is not None:
                    p[16] = np.float32(np.log(max(float(S[int(j)]), 1e-6)))
                if _kt >= 2:
                    _ct = np.log(np.maximum(
                        np.asarray(Ctiles[int(j)], np.float32).ravel(), 1e-6)).astype(np.float32)
                    queue.write_buffer(bCt, 0, _ct.tobytes())
                tv = float(T[int(j)]) if T is not None else 1.0
                p[23] = np.float32(float(fp) * float(np.log(max(tv, 1e-6))))
                bQk = dev.create_buffer(size=128, usage=U.UNIFORM | U.COPY_DST)
                queue.write_buffer(bQk, 0, p.tobytes())
                if os.environ.get('DWT_FATOM_DBG') == '1' and k == 0 and y0 == 0:
                    print('[fatom-dbg] frame %d a=%.6f b=%.6f c=%.4f d=%.6f e=%.6f f=%.4f lo=%d hs=%d'
                          % (j, p[0], p[1], p[2], p[3], p[4], p[5], lo, hs), flush=True)
                    print('[fatom-dbg] X(center)=%.4f Y(center)=%.4f' % (
                        p[0]*(w*0.5+0.5) + p[1]*(hs*0.5+0.5) + p[2],
                        p[3]*(w*0.5+0.5) + p[4]*(hs*0.5+0.5) + p[5]), flush=True)
                bg = dev.create_bind_group(layout=bgl, entries=[
                    {'binding': 0, 'resource': {'buffer': bSrcs[k % len(bSrcs)], 'offset': 0,
                                 'size': (plane if _ring else len(idx) * plane) * 4}},
                    {'binding': 1, 'resource': {'buffer': bR, 'offset': 0, 'size': rsz}},
                    {'binding': 2, 'resource': {'buffer': bQk, 'offset': 0, 'size': 128}},
                    {'binding': 3, 'resource': {'buffer': aW, 'offset': 0, 'size': n_pix * 4}},
                    {'binding': 4, 'resource': {'buffer': aV, 'offset': 0, 'size': n_pix * 4}},
                    {'binding': 5, 'resource': {'buffer': aD, 'offset': 0, 'size': n_pix * 4}},
                    {'binding': 6, 'resource': {'buffer': bMu, 'offset': 0, 'size': n_pix * 4}},
                    {'binding': 7, 'resource': {'buffer': bSd, 'offset': 0, 'size': n_pix * 4}},
                    {'binding': 8, 'resource': {'buffer': aW2, 'offset': 0, 'size': n_pix * 4}},
                    {'binding': 9, 'resource': {'buffer': aBad, 'offset': 0, 'size': 8}},
                    {'binding': 10, 'resource': {'buffer': bCt, 'offset': 0, 'size': _ctn * 4}},
                    {'binding': 11, 'resource': {'buffer': aWe, 'offset': 0, 'size': (n_pix * 4 if _lsp_on else 4)}},
                    {'binding': 12, 'resource': {'buffer': aWDe, 'offset': 0, 'size': (n_pix * 4 if _lsp_on else 4)}},
                    {'binding': 13, 'resource': {'buffer': aCnte, 'offset': 0, 'size': (n_pix * 4 if _lsp_on else 4)}},
                    {'binding': 14, 'resource': {'buffer': aHits, 'offset': 0, 'size': (n_pix * 4 if _lsp_on else 4)}},
                ])
                cp.set_pipeline(pipe)
                cp.set_bind_group(0, bg)
                cp.dispatch_workgroups(int((w + 7) // 8), int((hs + 7) // 8))
                if upload:
                    cp.end()
                    queue.submit([enc.finish()])
            if not upload:
                cp.end()
                queue.submit([enc.finish()])

        _CHUNK = 65535 * 64

        def _u32_uni(vals, off=0):
            u = np.zeros(8, np.uint32)
            for i, v in enumerate(vals):
                u[i] = np.uint32(v)
            u[6] = np.uint32(off)      # pad0：像素偏移（分块 dispatch 用）
            b = dev.create_buffer(size=32, usage=U.UNIFORM | U.COPY_DST)
            queue.write_buffer(b, 0, u.tobytes())
            return b

        def _dispatch_chunked(cp, pipe_, mkbg, n_pix_):
            for off in range(0, int(n_pix_), _CHUNK):
                cnt = min(_CHUNK, int(n_pix_) - off)
                cp.set_pipeline(pipe_)
                cp.set_bind_group(0, mkbg(off))
                cp.dispatch_workgroups(int((cnt + 63) // 64))

        def mean_pass():
            # mean_pass 只用到 aW(0)/aV(1)/mu(3)/U(5)；x 维 dispatch 上限 65535，分块发
            enc = dev.create_command_encoder()
            cp = enc.begin_compute_pass()
            for off in range(0, n_pix, _CHUNK):
                cnt = min(_CHUNK, n_pix - off)
                bU = _u32_uni([n_pix, 0, W, bh, 64, 3], off)
                bg = dev.create_bind_group(layout=pstat.get_bind_group_layout(0), entries=[
                    {'binding': 0, 'resource': {'buffer': aW, 'offset': 0, 'size': n_pix * 4}},
                    {'binding': 1, 'resource': {'buffer': aV, 'offset': 0, 'size': n_pix * 4}},
                    {'binding': 3, 'resource': {'buffer': bMu, 'offset': 0, 'size': n_pix * 4}},
                    {'binding': 5, 'resource': {'buffer': bU, 'offset': 0, 'size': 32}},
                ])
                cp.set_pipeline(pstat)
                cp.set_bind_group(0, bg)
                cp.dispatch_workgroups(int((cnt + 63) // 64))
            cp.end()
            queue.submit([enc.finish()])

        def block_scale(k_clip=1.0, scale=1.15):
            """与 CPU 的 _winsor_scale 同口径：64×64 块、每 3 像素取 1、1.15×平均 |dev|。"""
            enc = dev.create_command_encoder()
            cp = enc.begin_compute_pass()
            for off in range(0, n_pix, _CHUNK):
                cnt = min(_CHUNK, n_pix - off)
                bU = _u32_uni([n_pix, 1, W, bh, 64, 3], off)
                bg = dev.create_bind_group(layout=pbs.get_bind_group_layout(0), entries=[
                    {'binding': 0, 'resource': {'buffer': aW, 'offset': 0, 'size': n_pix * 4}},
                    {'binding': 2, 'resource': {'buffer': aD, 'offset': 0, 'size': n_pix * 4}},
                    {'binding': 4, 'resource': {'buffer': bSd, 'offset': 0, 'size': n_pix * 4}},
                    {'binding': 5, 'resource': {'buffer': bU, 'offset': 0, 'size': 32}},
                ])
                cp.set_pipeline(pbs)
                cp.set_bind_group(0, bg)
                cp.dispatch_workgroups(int((cnt + 63) // 64))
            cp.end()
            queue.submit([enc.finish()])
            # 抽样点已在 sd 里，主机侧按块求均值并展开
            sp = np.frombuffer(queue.read_buffer(bSd), np.float32)[:n_pix].reshape(bh, W)
            blk = 64
            out = np.zeros((bh, W), np.float32)
            for by in range(0, bh, blk):
                for bx in range(0, W, blk):
                    sub = sp[by:by + blk:3, bx:bx + blk:3]
                    vals = sub[sub > 0]
                    s_blk = float(np.mean(vals)) * float(scale) if vals.size else 0.0
                    out[by:by + blk, bx:bx + blk] = s_blk
            queue.write_buffer(bSd, 0, out.tobytes())

        _dbg_stage('upload')
        scatter(0, 0.0, upload=True)   # Σw Σwv Σw²（逐帧上传+派发）
        _dbg_stage('scatter0')
        _dbg_stage('mean_pass')
        mean_pass()       # μ
        _dbg_stage('scatter1')
        scatter(1, 0.0)                     # Σw|v−μ0|
        _dbg_stage('block_scale')
        block_scale(k_clip=3.0, scale=1.15)  # 粗尺度 s0（写回 sd，作为 mode4/3 的截尾半径）
        # ---- 稳健锚点：迭代截尾均值，逼近 CPU 的逐像素中位数 ----
        # 8 帧时一次 3σ 截尾离中位数还远；再迭代一轮（同一 σ、围绕新 μ 重新截尾）
        # 后与 CPU 的中位数锚基本等价。每次迭代都要先把 Σw/Σwv 清零。
        #
        # 环形缓冲（帧放不进显存，每遍都要重传）时跳过这两轮：每轮 = 一次全帧上传，
        # 两轮就是 2×帧数据 的 PCIe 流量。实测锚点迭代对成品的影响在 corr 1e-4 量级，
        # 而重传的代价是秒级——所以这里按「能否常驻」二选一。
        _force_it = os.environ.get('DWT_RING_ITERS', '') == '1'
        if _ring and not _force_it:
            if log:
                log('[wgpu] 帧放不进显存（环形缓冲）：跳过锚点迭代，'
                    'scatter 由 6 遍降为 3 遍')
            queue.write_buffer(aD, 0, np.zeros(n_pix, np.float32).tobytes())
            scatter(1, 0.0)                 # Σw|v−μ0|
            block_scale(k_clip=1.0, scale=1.15)
        else:
            for _k_it in (1, 2):
                queue.write_buffer(aW, 0, np.zeros(n_pix, np.float32).tobytes())
                queue.write_buffer(aV, 0, np.zeros(n_pix, np.float32).tobytes())
                scatter(4, 0.0)                 # Σw·clip(v, μ±3s)
                mean_pass()                     # 写回新的 μ
            queue.write_buffer(aD, 0, np.zeros(n_pix, np.float32).tobytes())
            scatter(1, 0.0)                     # Σw|v−μ_robust|
            block_scale(k_clip=1.0, scale=1.15)  # 最终 winsor 尺度

        queue.write_buffer(aW, 0, np.zeros(n_pix, np.float32).tobytes())
        queue.write_buffer(aV, 0, np.zeros(n_pix, np.float32).tobytes())
        queue.write_buffer(aW2, 0, np.zeros(n_pix, np.float32).tobytes())
        queue.write_buffer(aBad, 0, np.zeros(2, np.uint32).tobytes())
        _dbg_stage('scatter2')
        scatter(2, float(rej_k))

        enc = dev.create_command_encoder()
        cp = enc.begin_compute_pass()
        for off in range(0, n_pix, _CHUNK):
            cnt = min(_CHUNK, n_pix - off)
            u = np.zeros(4, np.float32)
            u.view(np.uint32)[0] = np.uint32(n_pix)
            u.view(np.uint32)[1] = np.uint32(off)
            bU = dev.create_buffer(size=16, usage=U.UNIFORM | U.COPY_DST)
            queue.write_buffer(bU, 0, u.tobytes())
            bg = dev.create_bind_group(layout=pfin.get_bind_group_layout(0), entries=[
                {'binding': 0, 'resource': {'buffer': aW, 'offset': 0, 'size': n_pix * 4}},
                {'binding': 1, 'resource': {'buffer': aV, 'offset': 0, 'size': n_pix * 4}},
                {'binding': 2, 'resource': {'buffer': aW2, 'offset': 0, 'size': n_pix * 4}},
                {'binding': 3, 'resource': {'buffer': oOut, 'offset': 0, 'size': n_pix * 4}},
                {'binding': 4, 'resource': {'buffer': oCov, 'offset': 0, 'size': n_pix * 4}},
                {'binding': 5, 'resource': {'buffer': oNef, 'offset': 0, 'size': n_pix * 4}},
                {'binding': 6, 'resource': {'buffer': bU, 'offset': 0, 'size': 16}},
            ])
            cp.set_pipeline(pfin)
            cp.set_bind_group(0, bg)
            cp.dispatch_workgroups(int((cnt + 63) // 64))
        cp.end()
        queue.submit([enc.finish()])

        def rd(b):
            return np.frombuffer(queue.read_buffer(b), np.float32).copy()

        stack[y0:y0 + bh, :] = rd(oOut).reshape(bh, W)
        cov[y0:y0 + bh, :] = rd(oCov).reshape(bh, W)
        neff[y0:y0 + bh, :] = rd(oNef).reshape(bh, W)

    if log:
        _npass = 6 if _ring else 6      # 实际全帧遍历次数（scatter0/1/4×2/2 共 6 遍）
        log(('[wgpu] 液滴散射累加（float 原子，%d 遍；%s）' % (
                _npass, '环形缓冲：每遍重传' if _ring else '帧常驻显存：只传一次'))
            if pattern is None else
            '[wgpu] 真 CFA：散射累加（float 原子，%d 遍）' % _npass)
        log('[wgpu][done] ts=%.3f backend=canvas wall_s=%.1f readback_mb=%.1f fallbacks=0'
            % (time.time(), _t.perf_counter() - _t0, 3.0 * H * W * 4 / (1024.0 * 1024.0)))
    # 真实剔除率：着色器在 mode=2 里对每个「样本→输出像素」的 tap 计数，
    # aBad[0]=被剔、aBad[1]=接受；不再用 0.0 占位（旧版日志里的 0.000% 是假的）。
    _cb = np.frombuffer(queue.read_buffer(aBad), np.uint32)
    _bad, _okc = int(_cb[0]), int(_cb[1])
    _tot = _bad + _okc
    _rate = (100.0 * _bad / _tot) if _tot else 0.0
    if log:
        log('[wgpu] 剔除率 %.3f%%（%d / %d 个 tap）' % (_rate, _bad, _tot))
    _lsp_out = None
    if float(os.environ.get('DWT_LSP', '0') or 0) > 0.5:
        try:
            np.save(str(os.environ.get('DWT_LSP_DUMP', '')) or 'H:/Temp/cnte.npy', np.frombuffer(queue.read_buffer(aCnte), np.float32).copy()) if os.environ.get('DWT_LSP_DUMP') else None
        except Exception:
            pass
        _lsp_out = {'we': np.frombuffer(queue.read_buffer(aWe), np.float32).copy(),
                    'wde': np.frombuffer(queue.read_buffer(aWDe), np.float32).copy(),
                    'cnte': np.frombuffer(queue.read_buffer(aCnte), np.float32).copy(),
                    'hits': np.frombuffer(queue.read_buffer(aHits), np.float32).copy()}
    return {'lsp': _lsp_out, 'stack': stack, 'cov': cov, 'neff': neff, 'rej_rate': float(_rate) / 100.0,   # 接口约定是 0~1 小数，日志里才是百分数
            'wmax': float(cov.max()) if cov.size else 0.0,
            'p': (_pcol if want_p else None)}
