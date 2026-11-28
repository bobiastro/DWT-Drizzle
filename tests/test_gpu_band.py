# -*- coding: utf-8 -*-
"""GPU 带核 vs CPU 带核：同一 ctx 上逐位比对，定位语义差异。"""
import sys
import numpy as np

sys.path.insert(0, 'core')
sys.path.insert(0, '.')
import drizzle
import gpu_band
import gpu_dz

if getattr(gpu_dz, 'cp', None) is None:
    if __name__ == '__main__':
        print('GPU-BAND-SKIP（本机没有 cupy，带核走 wgpu 路径，cupy 版比对无从做）')
        raise SystemExit(0)
    import unittest
    raise unittest.SkipTest('本机没有 cupy，跳过 cupy 带核比对')

rng = np.random.default_rng(7)


class FakeFS:
    """够 _drizzle_band 用的最小帧集：strip(i,j0,j1) 返回一段带噪星场"""

    def __init__(self, n=12, h=400, w=300, sky=0.01):
        self.n = n
        self.h, self.w = h, w
        self.sky = sky
        self.base = rng.normal(sky, 0.001, (h, w)).astype(np.float32)
        yy, xx = np.mgrid[0:h, 0:w]
        for _ in range(60):
            cy, cx = rng.integers(10, h - 10), rng.integers(10, w - 10)
            amp = rng.uniform(0.2, 2.0)
            self.base += (amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 4.0)).astype(np.float32)
        self.arrays = [self.base + rng.normal(0, 0.0005, (h, w)).astype(np.float32) for _ in range(n)]

    def strip(self, i, j0, j1):
        return self.arrays[i][j0:j1]

    def __len__(self):
        return self.n


def make_ctx(fs, n, W, H, s=1.0):
    th = rng.uniform(-3, 3, n)
    kk = np.where(rng.random(n) < 0.2, 1.02, 1.0)
    dx = rng.normal(0, 2, n)
    dy = rng.normal(0, 2, n)
    rot = [np.array([[np.cos(np.radians(t)), -np.sin(np.radians(t))],
                     [np.sin(np.radians(t)), np.cos(np.radians(t))]]) for t in th]
    ctx = {'fs': fs, 'R_out': (rng.random((H, W)) * 0.3).astype(np.float32),
           'C': np.linspace(0.8, 1.2, n), 'S': np.linspace(0.9, 1.1, n),
           'T': np.linspace(0.9, 1.1, n), 'fp': 2.0,
           'dx': dx, 'dy': dy,
           'plan': {'s': s, 'ox': 0.0, 'oy': 0.0, 'W': W, 'H': H},
           'A': 14.0, 'B': 2.0, 'rej_k': 3.5, 'rej_gate': 10.0,
           'sky': 0.01, 'pixfrac': 1.0, 'sharp_w': 0.0,
           'rot': rot, 'theta': th, 'scale': kk,
           'native_shape': (fs.h, fs.w), 'W': W, 's': s,
           'idx': list(range(n)), 'n': n, 'P': 1.0, 'use_rot': True, 'mx': None, 'fs_n': n,
           'bgcoef': None, 'bgdeg': 0, 'dist': None,
           'rows_fn': drizzle._band_rows}
    return ctx


if __name__ == '__main__':
    fs = FakeFS()
    n = len(fs)
    H, W = 120, 220
    ctx = make_ctx(fs, n, W, H)
    a = drizzle._drizzle_band(ctx, 20, 60)
    b = gpu_band.band(ctx, 20, 60)
    for key in ('stack', 'neff', 'cov', 'ncov'):
        d = np.abs(np.asarray(a[key], np.float64) - np.asarray(b[key], np.float64))
        print('%-6s max %.3g  mean %.3g   nan? %s/%s'
              % (key, np.nanmax(d), np.nanmean(d),
                 np.isnan(np.asarray(a[key])).any(), np.isnan(np.asarray(b[key])).any()))
    print('rej cpu %.4f%%  gpu %.4f%%' % (100 * a['rej_num'] / max(a['rej_den'], 1),
                                          100 * b['rej_num'] / max(b['rej_den'], 1)))
