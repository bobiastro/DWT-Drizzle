# -*- coding: utf-8 -*-
"""GPU 叠加（CuPy）—— 与 CPU 版 `_drizzle_band` 逐位等价的替代实现。

为什么值得搬：叠加阶段占了总耗时 2/3，热点是"每帧 610 万输入像素 → 9 个
输出落点"的散点累加，305 帧约 33 亿次累加，CPU 上纯内存带宽受限；GPU 上
同样的工作量只要十几毫秒一帧。

设计约束（与 CPU 版一致，便于逐位比对）：
  · 落点是边长 P 的方框（pixfrac），重叠面积按 1/P² 归一；
  · 覆盖 cov 与值×覆盖 num 分开累加，最后 num/cov；
  · 排异：锚点/中位/MAD 只用最多 ANCHOR_N 帧的子集估计，阈值
    thr = rej_k·MAD + m·R·(a-sky)，m 由核心区分位数 p99 估；
  · 合成权重 e = A·R·lnC + B·(1-R)·lnS + sharp_w·lnC + flux_pow·lnT。
"""
import math

import numpy as np

try:
    import cupy as cp
    import cupyx
    _OK = True
except Exception:                                   # pragma: no cover
    cp = None
    cupyx = None
    _OK = False

ANCHOR_N = 48


def available() -> bool:
    """这台机器能不能真的用 GPU：CuPy 能导入、有设备、能跑一次真正的累加。

    只查「有没有设备」不够：上一个版本用 cupyx.scatter_add，它在第一帧才 JIT，
    于是探测通过、跑到一半才炸（缺 NVRTC）。这里把热路径的最小单元先跑一遍。
    """
    if not _OK:
        return False
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            return False
        a = cp.zeros(16, dtype=cp.float32)
        if float(a.sum()) != 0.0:
            return False
        idx = cp.array([0, 1, 1, 3], dtype=cp.int64)
        w = cp.array([1.0, 2.0, 3.0, 4.0], dtype=cp.float64)
        r = cp.bincount(idx, weights=w, minlength=5).astype(cp.float32)
        return [float(x) for x in r] == [1.0, 5.0, 0.0, 4.0, 0.0]
    except Exception:
        return False


def free_mem() -> int:
    """当前可用显存字节数；CuPy 不可用或查询失败返回 0。

    带高预算是按显存算的（num/cov 两个 (n,bh,w) float32），写死一个值会
    在大卡上浪费、在小卡上 OOM，所以运行时问一次设备。
    """
    if not _OK:
        return 0
    try:
        _free, _total = cp.cuda.Device().mem_info
        return int(_free)
    except Exception:
        return 0

def _scatter_gpu(vals, X, Y, P, W, Y0, Y1):
    """把 (X,Y) 处的方框落点铺到输出行 [Y0,Y1)，返回 (num, cov) 两张 GPU 图"""
    bh = Y1 - Y0
    size = bh * W
    m = (Y > Y0 - P) & (Y < Y1 + P) & (X > -P) & (X < W + P)
    if not bool(m.any()):
        z = cp.zeros((bh, W), cp.float32)
        return z, z
    xs = X[m]
    ys = Y[m]
    v = vals[m].astype(cp.float32)
    hx = cp.float32(0.5 * P)
    x0, x1 = xs - hx, xs + hx
    y0, y1 = ys - hx, ys + hx
    k0 = cp.floor(x0).astype(cp.int64)
    l0 = cp.floor(y0).astype(cp.int64)
    noff = int(math.ceil(P)) + 1
    inv = cp.float32(1.0 / (P * P))
    idxs, wgts, wvts = [], [], []
    for a in range(noff):
        kx = k0 + a
        wx = cp.minimum(x1, (kx + 1).astype(cp.float32)) - cp.maximum(x0, kx.astype(cp.float32))
        wx = cp.clip(wx, 0.0, None)
        vx = (kx >= 0) & (kx < W) & (wx > 0)
        if not bool(vx.any()):
            continue
        for b in range(noff):
            ky = l0 + b
            wy = cp.minimum(y1, (ky + 1).astype(cp.float32)) - cp.maximum(y0, ky.astype(cp.float32))
            wy = cp.clip(wy, 0.0, None)
            mm = vx & (ky >= Y0) & (ky < Y1) & (wy > 0)
            if not bool(mm.any()):
                continue
            w = (cp.where(mm, wx, 0) * cp.where(mm, wy, 0)) * inv
            flat = ((ky - Y0) * W + kx).astype(cp.int64)
            if not bool(mm.all()):
                flat = flat[mm]
                w = w[mm]
                vv = v[mm]
            else:
                vv = v
            idxs.append(flat)
            wgts.append(w)
            wvts.append(w * vv)
    if not idxs:
        z = cp.zeros((bh, W), cp.float32)
        return z, z
    idx = cp.concatenate(idxs).astype(cp.int64)
    # 不用 cupyx.scatter_add：它内部走 cupy.RawKernel → 需要 CUDA Toolkit 的
    # NVRTC（nvrtc64_*.dll）做 JIT。只装驱动的机器没有这个 DLL，会直接报
    #   RuntimeError: CuPy failed to load nvrtc64_112_0.dll
    # bincount(weights=) 语义相同（同下标按权重累加），但用预编译内核。
    num = cp.bincount(idx, weights=cp.concatenate(wvts).astype(cp.float32),
                      minlength=size).astype(cp.float32)
    cov = cp.bincount(idx, weights=cp.concatenate(wgts).astype(cp.float32),
                      minlength=size).astype(cp.float32)
    return num.reshape(bh, W), cov.reshape(bh, W)


def _xy_gpu(j0, j1, w, s, ox, oy, theta, ux, uy, k=1.0, d2=0.0, h=0):
    th = math.radians(theta)
    ct, st = cp.float32(math.cos(th)), cp.float32(math.sin(th))
    yy = (cp.arange(j0, j1, dtype=cp.float32) + 0.5)[:, None]
    xx = (cp.arange(w, dtype=cp.float32) + 0.5)[None, :]
    if d2:
        cx0, cy0 = cp.float32(w / 2.0), cp.float32(float(h) / 2.0)
        rc = cp.float32(0.5 * float(min(float(h) or 1.0, float(w))))
        dxp, dyp = xx - cx0, yy - cy0
        f = 1.0 + cp.float32(d2) * (dxp * dxp + dyp * dyp) / (rc * rc)
        xx, yy = cx0 + dxp * f, cy0 + dyp * f
    bx = ct * xx - st * yy
    by = st * xx + ct * yy
    X = cp.float32(s) * (cp.float32(k) * bx + cp.float32(ux)) + cp.float32(ox)
    Y = cp.float32(s) * (cp.float32(k) * by + cp.float32(uy)) + cp.float32(oy)
    return X, Y
