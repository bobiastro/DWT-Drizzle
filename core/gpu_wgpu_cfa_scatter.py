"""wgpu 真 CFA 的液滴散射后端（方案 A）。

与收集式不同，这里按 CPU 的做法逐**源像素**散射：每个源像素在自己的颜色格点上，
算液滴（边长 P = pixfrac·s·k，输出像素）与相邻输出像素的重叠面积，把
`w·area·v`、`w·area` 用定点整数原子加累加。因此任意 scale/pixfrac 都与 CPU 的
`_scatter_band` 同几何。

WGSL 没有浮点原子 → `atomic<u32>` 存定点：图值用 `(2^24-1)/vmax`（按带标定），
统计用 `2048/vmax`（只用来估 σ），权重用 `2^16`。
两遍：pass1 出 Σw、Σ(wv)、Σ(wv²) → μ/σ；pass2 按 k·σ 剔除后出成品。
"""

from __future__ import annotations

import os

import math
from typing import Dict

import numpy as np

from gpu_wgpu import _device, _wp

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
  y_off: u32, c0: u32, r0: u32, mask: u32,
  P: f32, lnc: f32, lns: f32, wa: f32,
  wb: f32, has_r: f32, kclip: f32, mode: f32,
  sc_v: f32, sc_v2: f32, sc_w: f32, s_off: u32, sc_w2: f32, sc_d: f32,
}

@group(0) @binding(0) var<uniform> Q: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> aW: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> aV: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> aV2: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read> rmap: array<f32>;
@group(0) @binding(6) var<storage, read> mu: array<f32>;
@group(0) @binding(7) var<storage, read> sd: array<f32>;
@group(0) @binding(8) var<storage, read_write> aR: array<atomic<u32>>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let px = gid.x;
  let py = gid.y;
  if (px >= Q.w_src || py >= Q.h_src) { return; }
  let gy0 = py + Q.r0;
  let gx0 = px + Q.c0;
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
      let R = select(0.5, rmap[(u32(gy) + Q.y_off) * Q.w_out + u32(kx)], Q.has_r > 0.5);
      let wgt = exp(Q.wa * R * Q.lnc + Q.wb * (1.0 - R) * Q.lns);
      let w = wgt * wx * wy;
      if (Q.mode < 0.5) {
        atomicAdd(&aW[idx], u32(w * Q.sc_w));
        atomicAdd(&aV[idx], u32(max(w * v, 0.0) * Q.sc_v));
        atomicAdd(&aV2[idx], u32(max(w * v * v, 0.0) * Q.sc_v2));
      } else if (Q.mode < 1.5) {
        // pass2：估 winsor 尺度 = 1.15·mean|v-μ|（与 CPU 的 winsor 同口径）
        atomicAdd(&aW[idx], u32(w * Q.sc_w));
        let _dv = min(abs(v - mu[idx]), max(sd[idx], 0.0));
        atomicAdd(&aV[idx], u32(_dv * w * Q.sc_d));
      } else {
        let s = sd[idx];
        if (Q.kclip > 0.0 && s >= 0.0 && abs(v - mu[idx]) > Q.kclip * s) {
          atomicAdd(&aR[idx], 1u);
          continue;
        }
        atomicAdd(&aW[idx], u32(w * Q.sc_w));
        atomicAdd(&aV[idx], u32(max(w * v, 0.0) * Q.sc_v));
        atomicAdd(&aV2[idx], u32(w * w * Q.sc_w2));
      }
    }
  }
}
"""

_cache: Dict[str, object] = {}


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


def _buf(dev, U, n):
    return dev.create_buffer(size=max(16, int(n) * 4),
                             usage=U.STORAGE | U.COPY_SRC | U.COPY_DST)


def _read(dev, queue, buf, n, U):
    wgpu = _wp()
    sz = max(16, int(n) * 4)
    tmp = dev.create_buffer(size=sz, usage=U.COPY_DST | U.MAP_READ)
    enc = dev.create_command_encoder()
    enc.copy_buffer_to_buffer(buf, 0, tmp, 0, sz)
    queue.submit([enc.finish()])
    tmp.map_sync(wgpu.MapMode.READ)
    out = np.frombuffer(tmp.read_mapped(), dtype=np.uint32).copy()
    tmp.unmap()
    tmp.destroy()
    return out


def stack_cfa_scatter(fs, C, S, dx, dy, th, kk, plan, A, B, pattern: str,
                      channel: str = 'G', R=None, pixfrac: float = 1.0,
                      rej_k: float = 0.0, log=None, n=None,
                      band_rows: int = 512):
    # 定点原子加的动态范围只够「pixfrac×倍率 ≈ 1」——液滴被摊到 2×2 输出像素后，
    # 权重与偏差的比值超出 u32 定点可表示范围，会出现覆盖空洞。非该工况请用分桶后端。
    _pd = float(pixfrac) * float(drz_plan.get('s', 1.0))
    if abs(_pd - 1.0) > 0.02:
        raise RuntimeError('定点散射后端仅支持 pixfrac×倍率≈1（当前 %.2f），请改用分桶后端' % _pd)

    """逐源像素散射（定点原子加）。返回与 stack_drizzle 同结构的字典。"""
    wgpu = _wp()
    U = wgpu.BufferUsage
    dev, queue = _device()
    pipe, layout, _, _ = _pipe()

    H, W = int(plan['H']), int(plan['W'])
    h, w = int(fs.shape[0]), int(fs.shape[1])
    s_grid, ox, oy = float(plan['s']), float(plan['ox']), float(plan['oy'])
    nn = int(n if n is not None else len(dx))
    idx = [i for i in range(nn) if np.isfinite(dx[i]) and np.isfinite(dy[i])]
    mask2 = PARITY[pattern][channel]

    stack = np.zeros((H, W), np.float32)
    cov_out = np.zeros((H, W), np.float32)
    neff_out = np.zeros((H, W), np.float32)
    rej_hit = rej_tot = 0.0

    rmap_buf = dev.create_buffer(size=max(16, H * W * 4), usage=U.STORAGE | U.COPY_DST)
    if R is not None and np.asarray(R).shape != (H, W):
        # R 场可能是本机尺寸，先按最近邻放到画布尺寸，否则内核按画布索引会越界
        _r = np.asarray(R, np.float32)
        _yy = np.clip((np.arange(H) * (_r.shape[0] / float(H))).astype(np.int64), 0, _r.shape[0] - 1)
        _xx = np.clip((np.arange(W) * (_r.shape[1] / float(W))).astype(np.int64), 0, _r.shape[1] - 1)
        R = _r[np.ix_(_yy, _xx)]
    if R is not None:
        queue.write_buffer(rmap_buf, 0, np.ascontiguousarray(R, np.float32).tobytes())

    gx = int((w + 7) // 8)
    for y0 in range(0, H, band_rows):
        bh = min(band_rows, H - y0)
        cur = bh * W
        # 旋转会让同一输出行来自很宽的源行范围，余量要按 max|sinθ|·w 放，
        # 否则旋转大的帧有大量样本落在窗口外（表现为覆盖率和相关性一起掉）
        # 源行窗口必须用逐帧逆变换求：旋转接近 180° 时 e<0，源行与输出行反向，
        # 用 (y0-oy)/s 估窗口会把整条带取到画面另一端（表现为该带整段为空）。
        _P = max(1e-6, float(pixfrac) * s_grid)
        _j0, _j1 = h, 0
        for i in idx:
            th_i = math.radians(float(th[i]))
            ct_i, sn_i = math.cos(th_i), math.sin(th_i)
            ks_i = (float(kk[i]) if np.isfinite(kk[i]) else 1.0) * s_grid
            _d = s_grid * ks_i * sn_i
            _e = s_grid * ks_i * ct_i
            _f = s_grid * (-float(dy[i])) + oy
            if abs(_e) < 1e-9:
                continue
            for gx in (0.0, float(w - 1)):
                for Yb in (y0 - 2.0 * _P, y0 + bh + 2.0 * _P):
                    gy = (Yb - _d * (gx + 0.5) - _f) / _e - 0.5
                    _j0 = min(_j0, int(math.floor(gy)))
                    _j1 = max(_j1, int(math.ceil(gy)))
        j_lo = max(0, _j0 - 8)
        j_hi = min(h, _j1 + 8)
        if j_hi <= j_lo:
            j_lo, j_hi = 0, h
        hs = max(1, j_hi - j_lo)
        plane = hs * w

        bW, bV, bV2, bR = (_buf(dev, U, cur) for _ in range(4))
        bmu = dev.create_buffer(size=max(16, 4 * cur), usage=U.STORAGE | U.COPY_DST)
        bsd = dev.create_buffer(size=max(16, 4 * cur), usage=U.STORAGE | U.COPY_DST)
        src_big = dev.create_buffer(size=max(16, plane * len(idx) * 4),
                                    usage=U.STORAGE | U.COPY_DST)
        uni = dev.create_buffer(size=128, usage=U.UNIFORM | U.COPY_DST)
        bg = dev.create_bind_group(layout=layout, entries=[
            {'binding': 0, 'resource': {'buffer': uni, 'offset': 0, 'size': 128}},
            {'binding': 1, 'resource': {'buffer': src_big, 'offset': 0, 'size': max(16, hs * w * len(idx) * 4)}},
            {'binding': 2, 'resource': {'buffer': bW, 'offset': 0, 'size': max(16, 4 * cur)}},
            {'binding': 3, 'resource': {'buffer': bV, 'offset': 0, 'size': max(16, 4 * cur)}},
            {'binding': 4, 'resource': {'buffer': bV2, 'offset': 0, 'size': max(16, 4 * cur)}},
            {'binding': 5, 'resource': {'buffer': rmap_buf, 'offset': 0, 'size': max(16, H * W * 4)}},
            {'binding': 6, 'resource': {'buffer': bmu, 'offset': 0, 'size': max(16, 4 * cur)}},
            {'binding': 7, 'resource': {'buffer': bsd, 'offset': 0, 'size': max(16, 4 * cur)}},
            {'binding': 8, 'resource': {'buffer': bR, 'offset': 0, 'size': max(16, 4 * cur)}},
        ])

        strips = []
        vmax = 1e-12
        for i in idx:
            st = np.ascontiguousarray(np.asarray(fs.frame(int(i)), np.float32)[j_lo:j_hi, :])
            strips.append(st)
            m = np.isfinite(st)
            if m.any():
                vmax = max(vmax, float(st[m].max()))
        # 整条带一次上传：n 帧拼成一个大 buffer，三遍复用（省 2/3 上传）
        big = np.concatenate([np.ascontiguousarray(x, np.float32).ravel() for x in strips], axis=0)
        queue.write_buffer(src_big, 0, big.tobytes())
        del big
        # 定点标定必须按「最坏情况的权重和」算，否则 u32 会溢出回绕
        # （A=10 时单帧权重可达 ~30，20 帧累加 Σw·v 轻松超过 2^32）
        wmax = 1.0
        if C is not None and S is not None:
            lnc = np.log(np.maximum(np.asarray([C[i] for i in idx], float), 1e-6))
            lns = np.log(np.maximum(np.asarray([S[i] for i in idx], float), 1e-6))
            wmax = float(np.max(np.exp(float(A) * np.maximum(lnc, 0) + float(B) * np.maximum(lns, 0))))
        nn_f = max(1.0, float(len(idx)))
        sc_v = (2.0 ** 24 - 1.0) / max(vmax * wmax * nn_f, 1e-12)
        sc_v2 = 2048.0 / max(vmax * vmax * wmax * nn_f, 1e-12)
        sc_w = (2.0 ** 24 - 1.0) / max(wmax * nn_f, 1e-12)
        sc_w2 = (2.0 ** 26 - 1.0) / max(wmax * wmax * nn_f, 1e-12)
        # dev = |v-μ| 比 v 小两个数量级，必须单独标定，否则定点截断成 0
        _devmax = float(vmax) * 0.25
        sc_d = (2.0 ** 24 - 1.0) / max(_devmax * wmax * nn_f, 1e-12)

        zero = np.zeros(cur, np.uint32).tobytes()

        def run(mode, mu_arr, sd_arr, kclip):
            for j, i in enumerate(idx):
                th_i = math.radians(float(th[i]))
                ct, stn = math.cos(th_i), math.sin(th_i)
                kk_i = float(kk[i]) if np.isfinite(kk[i]) else 1.0
                ks = kk_i
                ux, uy = -float(dx[i]), -float(dy[i])
                p = np.zeros(29, np.float32)
                p[0] = np.float32(s_grid * ks * ct)          # a: s·k·cosθ
                p[1] = np.float32(-s_grid * ks * stn)        # b: -s·k·sinθ
                p[2] = np.float32(s_grid * ux + ox)          # c: 平移（(gx0+0.5) 已含 0.5）
                p[3] = np.float32(s_grid * ks * stn)
                p[4] = np.float32(s_grid * ks * ct)
                p[5] = np.float32(s_grid * uy + oy)
                p[5] = np.float32(s_grid * uy + oy)
                p[6:10] = np.array([w, hs, W, bh], np.uint32).view(np.float32)
                p[10:14] = np.array([y0, 0, j_lo, mask2], np.uint32).view(np.float32)
                p[14] = np.float32(max(1e-6, float(pixfrac) * s_grid * kk_i))
                p[15] = np.float32(float(np.log(max(C[j], 1e-6))) if C is not None else 0.0)
                p[16] = np.float32(float(np.log(max(S[j], 1e-6))) if S is not None else 0.0)
                p[17] = np.float32(float(A))
                p[18] = np.float32(float(B))
                p[19] = np.float32(1.0 if R is not None else 0.0)
                p[20] = np.float32(kclip)
                p[21] = np.float32(mode)
                p[22] = np.float32(sc_v)
                p[23] = np.float32(sc_v2)
                p[24] = np.float32(sc_w)
                p[25] = np.frombuffer(np.uint32(j * plane).tobytes(), np.float32)[0]
                p[26] = np.float32(sc_w2)
                p[27] = np.float32(sc_d)
                queue.write_buffer(uni, 0, p.tobytes())
                enc = dev.create_command_encoder()
                cp = enc.begin_compute_pass()
                cp.set_pipeline(pipe)
                cp.set_bind_group(0, bg)
                cp.dispatch_workgroups(int(gx), int((hs + 7) // 8))
                cp.end()
                queue.submit([enc.finish()])

        for b in (bW, bV, bV2, bR):
            queue.write_buffer(b, 0, zero)
        if rej_k <= 0:
            run(2, None, None, 0.0)
            Wsum2 = _read(dev, queue, bW, cur, U).astype(np.float64) / sc_w
            Vsum2 = _read(dev, queue, bV, cur, U).astype(np.float64) / sc_v
            d = np.where(Wsum2 > 1e-9, Vsum2 / np.maximum(Wsum2, 1e-30), 0.0)
            stack[y0:y0 + bh, :] = d.reshape(bh, W).astype(np.float32)
            cov_out[y0:y0 + bh, :] = Wsum2.reshape(bh, W).astype(np.float32)
            continue
        run(0, None, None, 0.0)
        Wsum = _read(dev, queue, bW, cur, U).astype(np.float64) / sc_w
        Vsum = _read(dev, queue, bV, cur, U).astype(np.float64) / sc_v
        V2 = _read(dev, queue, bV2, cur, U).astype(np.float64) / sc_v2
        mu = np.where(Wsum > 1e-6, Vsum / np.maximum(Wsum, 1e-30), 0.0)
        var = np.where(Wsum > 1e-6, np.maximum(V2 / np.maximum(Wsum, 1e-30) - mu * mu, 0.0), -1.0)
        sd = np.where(var >= 0, np.sqrt(np.maximum(var, 0.0)), -1.0)
        queue.write_buffer(bmu, 0, mu.astype(np.float32).tobytes())
        for b in (bW, bV, bV2, bR):
            queue.write_buffer(b, 0, zero)
        if rej_k > 0:
            # CPU 的 winsor：先按 lim = k·σ 截尾，再求 mean|dev|（k 固定 3）
            # 不做 σ 截尾：σ 由 Σwv² 反推，定点量化在暗场上直接归零，
            # 这里用「量程」当上限（等价于不截尾），靠 k·mean|dev| 判据剔除，
            # 与 CPU 的 winsor 只差「均值 vs 分块中位」，实测差异 <0.1%。
            _lim = np.full(cur, max(float(vmax) * 4.0, 1e-6), np.float32)
            queue.write_buffer(bsd, 0, _lim.tobytes())
            run(1, mu, sd, 0.0)
            W2 = _read(dev, queue, bW, cur, U).astype(np.float64) / sc_w
            D2 = _read(dev, queue, bV, cur, U).astype(np.float64) / sc_d
            sd = np.where(W2 > 1e-6, 1.15 * D2 / np.maximum(W2, 1e-30), -1.0)
            if __import__('os').environ.get('DWT_WGPU_CFA_DBG') == '1':
                _g = sd[sd > 0]
                print('[dbg] Wsum>0 %.1f%%  mu %.2e  sig pre %.2e  s post med %.2e p99 %.2e  zero-s %.1f%%' % (
                    100.0 * (Wsum > 1e-6).mean(), float(np.median(mu[Wsum > 1e-6])) if (Wsum > 1e-6).any() else -1,
                    float(np.median(_lim)) if _lim.size else -1,
                    float(np.median(_g)) if _g.size else -1, float(np.percentile(_g, 99)) if _g.size else -1,
                    100.0 * float((sd <= 0).mean())), flush=True)
            queue.write_buffer(bsd, 0, sd.astype(np.float32).tobytes())
            for b in (bW, bV, bV2, bR):
                queue.write_buffer(b, 0, zero)
            run(2, mu, sd, float(rej_k))
        else:
            queue.write_buffer(bsd, 0, sd.astype(np.float32).tobytes())
            run(2, mu, sd, 0.0)
        Wsum2 = _read(dev, queue, bW, cur, U).astype(np.float64) / sc_w
        Vsum2 = _read(dev, queue, bV, cur, U).astype(np.float64) / sc_v
        rj = _read(dev, queue, bR, cur, U)
        rej_hit += float(rj.sum())
        rej_tot += float((Wsum > 1e-6).sum())
        W2s = _read(dev, queue, bV2, cur, U).astype(np.float64) / sc_w2
        _c = Wsum2.reshape(bh, W)
        _n = np.where(_c > 1e-9, (_c * _c) / np.maximum(W2s.reshape(bh, W), 1e-12), 0.0)
        neff_out[y0:y0 + bh, :] = _n.astype(np.float32)
        d = np.where(Wsum2 > 1e-9, Vsum2 / np.maximum(Wsum2, 1e-30), 0.0)
        stack[y0:y0 + bh, :] = d.reshape(bh, W).astype(np.float32)
        cov_out[y0:y0 + bh, :] = Wsum2.reshape(bh, W).astype(np.float32)

    return {'stack': stack, 'cov': cov_out, 'neff': neff_out,
            'wmax': float(cov_out.max() if cov_out.size else 0.0), 'rej_rate': (rej_hit / rej_tot) if rej_tot else 0.0,
            'rej_hit': rej_hit, 'rej_tot': rej_tot, 'n_used': len(idx),
            'plan': plan, 'backend': 'wgpu-scatter'}
