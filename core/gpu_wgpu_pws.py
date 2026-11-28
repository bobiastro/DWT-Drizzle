"""wgpu（跨平台）PWS 叠加：单通道 + 真 CFA 颜色门控 + 两遍 σ-clipping。

与旧 `gpu_wgpu.stack_pws`（两档权重近似、无排异、不支持真 CFA）相比：

  * 真 CFA：每个输出像素只接受「同色格点上、且该样本的 1×1 液滴正好盖住它」的样本
    —— 与 CPU scatter 的判据一致，故 R/B 覆盖率约 1/4、G 约 1/2；
  * 逐像素权重 `w = exp(A·R·lnC + B·(1-R)·lnS)`，R 场逐像素读取，不再两档近似；
  * 两遍 σ-clipping：第一遍按像素累计 count/Σv/Σv² 得 μ、σ（count≥3 才算），
    第二遍丢弃 `|v-μ| > k·σ` 的样本，剔除率回写，口径与 CPU 的 winsor 一致。

输出按行分带（默认 1024 行），源图按带裁剪上传，4 GB 显存也能跑。
本模块只依赖 numpy + wgpu；由 `core/pws.py` 在「gpu=wgpu 且素材是 CFA」时调用。
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

from gpu_wgpu import _device, _wp, affine_from

# 每个通道的 2×2 站点奇偶位掩码（bit k：k = (y&1)*2 + (x&1)）
PARITY = {
    'RGGB': {'R': 0b0001, 'G': 0b0110, 'B': 0b1000},
    'BGGR': {'R': 0b1000, 'G': 0b0110, 'B': 0b0001},
    'GRBG': {'R': 0b0010, 'G': 0b0101, 'B': 0b1000},
    'GBRG': {'R': 0b0010, 'G': 0b1001, 'B': 0b0100},
}

_WGSL = """
struct Params {
  a: f32, b: f32, c: f32, d: f32, e: f32, f: f32,
  w_src: u32, h_src: u32, w_out: u32, h_out: u32,
  y_off: u32, mode: u32, rej_on: u32, mask: u32,
  lnc: f32, lns: f32, kclip: f32, wa: f32,
  wb: f32, has_r: f32, near: f32, taps: f32,
  a2: f32, b2: f32, c2: f32, d2: f32, e2: f32, f2: f32, pdrop: f32, ofs: f32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> bufA: array<f32>;
@group(0) @binding(3) var<storage, read_write> bufB: array<f32>;
@group(0) @binding(4) var<storage, read> rmap: array<f32>;
@group(0) @binding(5) var<storage, read> mu: array<f32>;
@group(0) @binding(6) var<storage, read> sd: array<f32>;
@group(0) @binding(7) var<storage, read_write> rej: array<f32>;
@group(0) @binding(8) var<storage, read_write> bufC: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let J = gid.x;
  let I = gid.y;
  if (J >= P.w_out || I >= P.h_out) { return; }
  let gy = I + P.y_off;
  let X = f32(J) + 0.5;
  let Y = f32(gy) + 0.5;
  let sx = P.a * X + P.b * Y + P.c;
  let sy = P.d * X + P.e * Y + P.f;

  // 两种取法：
  //   near>0.5 → 最近同色样本（pixfrac*scale≈1 时与 CPU 的 1×1 液滴等价）
  //   否则     → 同色格点 2×2 双线性（pixfrac*scale>1 时才有意义）
  var v = 0.0;
  var warea = 1.0;
  var got = false;
  if (P.taps > 0.5) {
    // 4 抽头 × 盒重叠：与 CPU scatter 的液滴面积权重严格等价
    let Pd = P.pdrop;
    var wsum = 0.0;
    var vsum = 0.0;
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
      if (((P.mask >> k) & 1u) == 0u) { continue; }
      let fx = f32(k & 1u);
      let fy = f32(k >> 1u);
      let bx = 2.0 * floor((sx - fx) * 0.5) + fx;
      let by = 2.0 * floor((sy - fy) * 0.5) + fy;
      for (var t: u32 = 0u; t < 4u; t = t + 1u) {
        let px = bx + 2.0 * f32(t & 1u);
        let py = by + 2.0 * f32(t >> 1u);
        if (px < 0.0 || py < 0.0 || px > f32(P.w_src) - 1.0
            || py > f32(P.h_src) - 1.0) { continue; }
        let ox = px + P.ofs;
        let oy = py + P.ofs;
        let Xc = P.a2 * ox + P.b2 * oy + P.c2;
        let Yc = P.d2 * ox + P.e2 * oy + P.f2;
        let l = max(Xc - 0.5 * Pd, f32(J));
        let rr = min(Xc + 0.5 * Pd, f32(J) + 1.0);
        let tt = max(Yc - 0.5 * Pd, f32(gy));
        let bb = min(Yc + 0.5 * Pd, f32(gy) + 1.0);
        if (rr <= l || bb <= tt) { continue; }
        let ar = (rr - l) * (bb - tt);
        vsum = vsum + ar * src[u32(i32(py)) * P.w_src + u32(i32(px))];
        wsum = wsum + ar;
      }
    }
    if (wsum <= 0.0) { return; }
    v = vsum / wsum;
    warea = wsum;
    got = true;
  } else if (P.near > 0.5) {
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
      if (got || ((P.mask >> k) & 1u) == 0u) { continue; }
      let fx = f32(k & 1u);
      let fy = f32(k >> 1u);
      let px = 2.0 * round((sx - fx) * 0.5) + fx;
      let py = 2.0 * round((sy - fy) * 0.5) + fy;
      if (px < 0.0 || py < 0.0 || px > f32(P.w_src) - 1.0 || py > f32(P.h_src) - 1.0) { continue; }
      v = src[u32(i32(py)) * P.w_src + u32(i32(px))];
      got = true;
    }
  } else {
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
    if (got || ((P.mask >> k) & 1u) == 0u) { continue; }
    let fx = f32(k & 1u);
    let fy = f32(k >> 1u);
    let x0 = 2.0 * floor((sx - fx) * 0.5) + fx;
    let y0 = 2.0 * floor((sy - fy) * 0.5) + fy;
    let tx = clamp((sx - x0) * 0.5, 0.0, 1.0);
    let ty = clamp((sy - y0) * 0.5, 0.0, 1.0);
    let x1 = x0 + 2.0;
    let y1 = y0 + 2.0;
    if (x0 < 0.0 || y0 < 0.0 || x1 > f32(P.w_src) - 1.0 || y1 > f32(P.h_src) - 1.0) { continue; }
    let i0 = u32(i32(x0)) * P.w_src;
    let i1 = u32(i32(y1)) * P.w_src;
    let c00 = src[i0 + u32(i32(x0))];
    let c10 = src[i0 + u32(i32(x1))];
    let c01 = src[i1 + u32(i32(x0))];
    let c11 = src[i1 + u32(i32(x1))];
    v = (1.0 - tx) * (1.0 - ty) * c00 + tx * (1.0 - ty) * c10
      + (1.0 - tx) * ty * c01 + tx * ty * c11;
    got = true;
  }
  }
  if (!got) { return; }

  let idx = I * P.w_out + J;
  if (P.mode == 1u) {
    bufA[idx] = bufA[idx] + warea;
    bufB[idx] = bufB[idx] + warea * v;
    bufC[idx] = bufC[idx] + warea * v * v;
  } else {
    let R = select(0.5, rmap[gy * P.w_out + J], P.has_r > 0.5);
    let wgt = exp(P.wa * R * P.lnc + P.wb * (1.0 - R) * P.lns);
    let s = sd[idx];
    if (P.rej_on == 1u && s >= 0.0) {
      if (abs(v - mu[idx]) > P.kclip * s) {
        rej[idx] = rej[idx] + 1.0;
        return;
      }
    }
    bufA[idx] = bufA[idx] + wgt * warea * v;
    bufB[idx] = bufB[idx] + wgt * warea;
  }
}
"""

_cache: Dict[str, object] = {}

# 采样点 → 输出坐标的常数偏移（源像素单位）。CPU 的 _xy_rows 用 (i+0.5, j+0.5)
# 作中心、affine_from 又把 0.5 从常数项里减掉，两边口径差一个 0.5；实验确定。
_OFS = [float(__import__('os').environ.get('DWT_WGPU_OFS', '0.5'))]


def _pipe():
    if 'pipe' in _cache:
        return _cache['pipe'], _cache['bgl'], _cache['dev'], _cache['queue']
    dev, queue = _device()
    mod = dev.create_shader_module(code=_WGSL)
    pipe = dev.create_compute_pipeline(
        layout='auto', compute={'module': mod, 'entry_point': 'main'})
    _cache.update({'pipe': pipe, 'bgl': pipe.get_bind_group_layout(0),
                   'dev': dev, 'queue': queue})
    return pipe, _cache['bgl'], dev, queue


def _buf(dev, U, nbytes):
    return dev.create_buffer(size=max(16, int(nbytes)),
                             usage=U.STORAGE | U.COPY_SRC | U.COPY_DST)


def _read(dev, queue, buf, n, U):
    wgpu = _wp()
    sz = max(16, int(n) * 4)
    tmp = dev.create_buffer(size=sz, usage=U.COPY_DST | U.MAP_READ)
    enc = dev.create_command_encoder()
    enc.copy_buffer_to_buffer(buf, 0, tmp, 0, sz)
    queue.submit([enc.finish()])
    tmp.map_sync(wgpu.MapMode.READ)
    out = np.frombuffer(tmp.read_mapped(), dtype=np.float32).copy()
    tmp.unmap()
    return out


def _pack(aff, w_src, h_src, W, bh, y_off, mode, rej_on, mask,
          lnc, lns, kclip, wa, wb, has_r, near=1.0, taps=0.0,
          inv6=None, pdrop=1.0):
    p = np.zeros(30, np.float32)
    p[:6] = np.asarray(aff, np.float32)
    p[6:10] = np.array([w_src, h_src, W, bh], np.uint32).view(np.float32)
    p[10:14] = np.array([y_off, mode, rej_on, mask], np.uint32).view(np.float32)
    p[14:20] = np.array([lnc, lns, kclip, wa, wb, has_r], np.float32)
    p[20] = float(near)
    p[21] = float(taps)
    if inv6 is not None:
        p[22:28] = np.asarray(inv6, np.float32)
    p[28] = float(pdrop)
    p[29] = float(_OFS[0])
    return p


def _shift_mask(mask: int, r0: int, c0: int) -> int:
    """把「原图坐标」的颜色奇偶掩膜换算到裁剪后的子图坐标。"""
    out = 0
    for k in range(4):
        if (mask >> k) & 1:
            px, py = k & 1, k >> 1
            nk = ((py - r0) & 1) * 2 + ((px - c0) & 1)
            out |= 1 << nk
    return out


def _strip(aff, y0, y1, W, ws, hs):
    a, b, c, d, e, f = [float(v) for v in aff]
    xs = (0.0, float(W))
    ys = (float(y0), float(y1))
    sx = [a * x + b * y + c for x in xs for y in ys]
    sy = [d * x + e * y + f for x in xs for y in ys]
    c0 = max(0, int(math.floor(min(sx))) - 2)
    r0 = max(0, int(math.floor(min(sy))) - 2)
    c1 = min(ws, int(math.ceil(max(sx))) + 3)
    r1 = min(hs, int(math.ceil(max(sy))) + 3)
    if c1 <= c0 or r1 <= r0:
        return None
    return (r0, r1, c0, c1)


def stack_pws_cfa(fs, C, S, dx, dy, th, kk, plan, A, B, pattern: str,
                  channel: str = 'G', sky: float = 0.0, R=None,
                  rej_k: float = 0.0, log=None, n=None, band_rows: int = 1024,
                  pixfrac: float = 1.0):
    """真 CFA 单通道叠加。返回与 `stack_drizzle` 同结构的字典。"""
    wgpu = _wp()
    U = wgpu.BufferUsage
    dev, queue = _device()
    pipe, bgl, _, _ = _pipe()

    H, W = int(plan['H']), int(plan['W'])
    nn = int(n if n is not None else len(dx))
    idx = [i for i in range(nn) if np.isfinite(dx[i]) and np.isfinite(dy[i])]
    if not idx:
        raise RuntimeError('没有可用帧')
    aff = [affine_from(dx[i], dy[i], th[i], kk[i], plan) for i in idx]

    pat = (pattern or '').upper()
    mask = PARITY.get(pat, {}).get(channel.upper(), 0b1111)
    if pat not in PARITY:
        mask = 0b1111 if channel else 0b1111

    s_w = np.asarray([S[i] for i in idx], np.float64)
    c_w = np.asarray([max(C[i], 1e-12) for i in idx], np.float64)
    lnc = np.log(np.clip(c_w, 1e-6, 1e6))
    lns = np.log(np.clip(s_w, 1e-6, 1e6))

    hs, ws = int(fs.shape[0]), int(fs.shape[1])
    bh = int(min(band_rows, H))
    stride = bh * W
    nb = stride * 4

    bA = _buf(dev, U, nb)
    bB = _buf(dev, U, nb)
    bC = _buf(dev, U, nb)
    bmu = _buf(dev, U, nb)
    bsd = _buf(dev, U, nb)
    brej = _buf(dev, U, nb)
    dummy = _buf(dev, U, 16)
    rmap_buf = _buf(dev, U, max(16, H * W * 4))
    has_r = 0.0
    if R is not None:
        has_r = 1.0
        queue.write_buffer(rmap_buf, 0, np.ascontiguousarray(R, np.float32).tobytes())
    else:
        queue.write_buffer(rmap_buf, 0, np.zeros(4, np.float32).tobytes())

    uni = dev.create_buffer(size=128, usage=U.UNIFORM | U.COPY_DST)
    src_buf = dev.create_buffer(size=max(16, hs * ws * 4), usage=U.STORAGE | U.COPY_DST)

    bg = dev.create_bind_group(layout=bgl, entries=[
        {'binding': 0, 'resource': {'buffer': uni, 'offset': 0, 'size': 128}},
        {'binding': 1, 'resource': {'buffer': src_buf, 'offset': 0,
                                    'size': max(16, hs * ws * 4)}},
        {'binding': 2, 'resource': {'buffer': bA, 'offset': 0, 'size': nb}},
        {'binding': 3, 'resource': {'buffer': bB, 'offset': 0, 'size': nb}},
        {'binding': 4, 'resource': {'buffer': rmap_buf, 'offset': 0,
                                    'size': max(16, H * W * 4)}},
        {'binding': 5, 'resource': {'buffer': bmu, 'offset': 0, 'size': nb}},
        {'binding': 6, 'resource': {'buffer': bsd, 'offset': 0, 'size': nb}},
        {'binding': 7, 'resource': {'buffer': brej, 'offset': 0, 'size': nb}},
        {'binding': 8, 'resource': {'buffer': bC, 'offset': 0, 'size': nb}},
    ])

    gx, gy_ = (W + 7) // 8, (bh + 7) // 8

    near = 1.0 if float(plan.get('s', 1.0)) <= 1.5 else 0.0

    def run(mode, y0, rej_on):
        for j, i in enumerate(idx):
            sp = _strip(aff[j], y0, y0 + bh, W, ws, hs)
            if sp is None:
                continue
            r0, r1, c0, c1 = sp
            frame = np.asarray(fs.frame(int(i)), np.float32)[r0:r1, c0:c1]
            a = list(aff[j])
            a[2] -= c0
            a[5] -= r0
            mk = _shift_mask(mask, r0, c0)
            m3 = np.array([[a[0], a[1], a[2]], [a[3], a[4], a[5]],
                           [0.0, 0.0, 1.0]], np.float64)
            inv = np.linalg.inv(m3)
            inv6 = [inv[0, 0], inv[0, 1], inv[0, 2],
                    inv[1, 0], inv[1, 1], inv[1, 2]]
            # CFA 的同色格点间隔是 2 个源像素，液滴要按 2× 计（与 CPU 的 CFA 路径一致）
            pd_ = 2.0 * float(pixfrac) * float(plan.get('s', 1.0)) * float(kk[j])
            use_taps = 1.0 if abs(pd_ - 1.0) > 0.02 else 0.0
            p = _pack(a, c1 - c0, r1 - r0, W, bh, y0, mode, rej_on, mk,
                      float(lnc[j]), float(lns[j]), float(rej_k),
                      float(A), float(B), has_r, near,
                      taps=use_taps, inv6=inv6, pdrop=pd_)
            queue.write_buffer(uni, 0, p.tobytes())
            queue.write_buffer(src_buf, 0, np.ascontiguousarray(frame).tobytes())
            enc = dev.create_command_encoder()
            cp = enc.begin_compute_pass()
            cp.set_pipeline(pipe)
            cp.set_bind_group(0, bg)
            cp.dispatch_workgroups(gx, gy_)
            cp.end()
            queue.submit([enc.finish()])
        queue.submit([])

    stack = np.zeros((H, W), np.float32)
    cov_out = np.zeros((H, W), np.float32)
    nrej = ntot = 0.0

    for y0 in range(0, H, bh):
        hh = min(bh, H - y0)
        cur = hh * W
        zero = np.zeros(bh * W, np.float32).tobytes()
        for b in (bA, bB, bC, brej):
            queue.write_buffer(b, 0, zero)
        # pass1：count / Σv / Σv²
        run(1, y0, 0)
        cnt = _read(dev, queue, bA, stride, U)[:cur]
        s1 = _read(dev, queue, bB, stride, U)[:cur]
        s2 = _read(dev, queue, bC, stride, U)[:cur]
        nz = cnt > 0
        ok3 = cnt >= 3
        mu = np.zeros_like(cnt)
        sd = np.full_like(cnt, -1.0)
        mu[ok3] = s1[ok3] / cnt[ok3]
        var = np.zeros_like(cnt)
        var[ok3] = np.maximum(s2[ok3] / cnt[ok3] - mu[ok3] ** 2, 0.0)
        sd[ok3] = np.sqrt(var[ok3])
        mu = mu * nz
        tmp = np.zeros(bh * W, np.float32)
        tmp[:cur] = mu
        queue.write_buffer(bmu, 0, tmp.tobytes())
        tmp = np.zeros(bh * W, np.float32)
        tmp[:cur] = sd
        queue.write_buffer(bsd, 0, tmp.tobytes())
        queue.write_buffer(bA, 0, zero)
        queue.write_buffer(bB, 0, zero)
        # pass2：加权（排异在 sd<0 时自动跳过）
        run(2, y0, 1 if rej_k > 0 else 0)
        num = _read(dev, queue, bA, stride, U)[:cur]
        cov = _read(dev, queue, bB, stride, U)[:cur]
        rj = _read(dev, queue, brej, stride, U)[:cur]
        nrej += float(rj.sum())
        ntot += float(cnt.sum())
        d = np.where(cov > 0, num / np.maximum(cov, 1e-20), 0.0)
        stack[y0:y0 + hh, :] = d.reshape(hh, W)
        cov_out[y0:y0 + hh, :] = cov.reshape(hh, W)

    rate = (nrej / ntot) if ntot > 0 else 0.0
    if log:
        log('[wgpu] CFA 收集式：%d 帧，画布 %d×%d，通道 %s，模式 %s，逐像素权重'
            % (len(idx), W, H, channel, pat or '单通道'))
        if rej_k > 0:
            log('[wgpu] 逐像素 σ-clipping：k=%.1f（两遍采样，剔除率见下）' % rej_k)
    stk = (stack - float(sky)).astype(np.float32)
    return {'stack': stk, 'neff': cov_out.astype(np.float32),
            'coverage': cov_out.astype(np.float32),
            'ncov': (cov_out > 0).astype(np.float32),
            'rej_rate': rate, 'wmax': 1.0}
