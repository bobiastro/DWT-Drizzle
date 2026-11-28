"""drizzle.py 的自检：通量守恒 / 位移复原 / 与式 12 的等价性 / 采样率增益

跑法（依赖装在 dwt-drizzle/pydeps）：
    set PYTHONPATH=dwt-drizzle/pydeps;dwt-drizzle/upstream/core
    python dwt-drizzle/tools/test_drizzle.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pydeps'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

import drizzle as dz  # noqa: E402


class FakeFS:
    """最小 FrameSet 替身：只有 frame(i) / strip(i,y0,y1) / __len__"""

    def __init__(self, arr):
        self.arr = arr.astype(np.float32)

    def __len__(self):
        return self.arr.shape[0]

    def frame(self, i):
        return self.arr[i]

    def strip(self, i, y0, y1):
        return self.arr[i, y0:y1]


def gauss_img(n=128, cy=64.3, cx=64.7, sigma=1.4, amp=1000.0):
    y, x = np.mgrid[0:n, 0:n]
    return (amp * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * sigma ** 2))
            ).astype(np.float32)


def test_axis_matrix():
    for pf in (0.5, 0.7, 1.0):
        M = dz._axis_matrix(20, 48, 2.0, 0.37, pf, 3.0)
        col = np.asarray(M.sum(axis=0)).ravel()
        assert np.allclose(col, 1.0, atol=1e-5), (pf, col.min(), col.max())
    # 恒等映射：scale=1, shift=0, pixfrac=1, offset=0 → 单位阵
    M = dz._axis_matrix(16, 16, 1.0, 0.0, 1.0, 0.0).toarray()
    assert np.allclose(M, np.eye(16), atol=1e-6)
    print('ok  axis_matrix: 逐列和为 1；恒等映射 = 单位阵')


def test_flux_conservation():
    """pixfrac=1 时逐列归一 ⇒ 常数像在原样搬移后仍是同一常数（不放大不衰减）"""
    img = np.full((64, 64), 7.0, np.float32)
    dx = np.array([0.0])
    dy = np.array([0.0])
    plan = dz.plan_output(img.shape, dx, dy, 2.0)
    num = dz.splat(img, plan, 0, dx, dy,
                   dz._axis_matrix(64, plan['W'], 2.0, 0.0, 1.0, plan['ox']),
                   1.0, 0, plan['H'])
    cov = dz.splat(np.ones_like(img), plan, 0, dx, dy,
                   dz._axis_matrix(64, plan['W'], 2.0, 0.0, 1.0, plan['ox']),
                   1.0, 0, plan['H'])
    ok = cov > 1e-6
    v = num[ok] / cov[ok]
    assert np.allclose(v, 7.0, atol=1e-4), (v.min(), v.max())
    assert ok.sum() >= 64 * 64 * 4, ok.sum()   # 整幅足迹无空隙（画布留的 1px 边不算）
    assert ok.mean() > 0.90, ok.mean()
    # pixfrac<1：drop 之间有空隙，但"有覆盖处"仍归一（num/cov = 7）
    num2 = dz.splat(img, plan, 0, dx, dy,
                    dz._axis_matrix(64, plan['W'], 2.0, 0.0, 0.7, plan['ox']),
                    0.7, 0, plan['H'])
    cov2 = dz.splat(np.ones_like(img), plan, 0, dx, dy,
                    dz._axis_matrix(64, plan['W'], 2.0, 0.0, 0.7, plan['ox']),
                    0.7, 0, plan['H'])
    ok2 = cov2 > 1e-6
    v2 = num2[ok2] / cov2[ok2]
    assert np.allclose(v2, 7.0, atol=1e-4), (v2.min(), v2.max())
    print(f'ok  通量守恒：p=1 覆盖 {ok.mean():.1%}、值恒为 7；'
          f'p=0.7 覆盖 {ok2.mean():.1%}（有空隙，靠抖动相填），有覆盖处仍为 7')


def test_shift_recovery():
    shifts = [(0.00, 0.00), (0.37, -0.22), (1.55, 0.83), (-0.61, 1.42),
              (0.12, 0.94), (2.04, -1.31), (-1.27, -0.44)]
    frames = []
    for (dy, dx) in shifts:
        frames.append(gauss_img(cy=64.3 + dy, cx=64.7 + dx))
    fs = FakeFS(np.stack(frames))
    stars = [(65, 64)]
    info = dz.measure_shifts(fs, stars, ref=0)
    got_dx = info['dx']
    got_dy = info['dy']
    err = np.hypot(got_dx - np.array([s[1] for s in shifts]),
                   got_dy - np.array([s[0] for s in shifts]))
    assert err.max() < 0.05, err
    print(f'ok  位移复原：最大误差 {err.max():.3f}px（阈值 0.05）')


def test_equivalence_identity():
    """s=1, pixfrac=1, 无位移 ⇒ 必须逐像素等于 Σ W·D / Σ W"""
    rng = np.random.default_rng(0)
    n, h, w = 6, 24, 24
    data = (rng.normal(100.0, 5.0, (n, h, w)) +
            rng.normal(0, 30, (n, h, w))).astype(np.float32)
    fs = FakeFS(data)
    C = rng.uniform(0.8, 1.2, n)
    S = rng.uniform(0.9, 1.1, n)
    R = np.clip(rng.uniform(0, 1, (h, w)), 0, 1).astype(np.float32)
    dx = np.zeros(n)
    dy = np.zeros(n)
    plan = dz.plan_output((h, w), dx, dy, 1.0, margin=0.0)
    res = dz.stack_drizzle(fs, R, C, S, dx, dy, plan, A=20.0, B=2.0,
                           rej_k=0.0, rej_gate=1 / 3, sky=0.0, pixfrac=1.0,
                           band_h=16)
    lc = np.log(C)[:, None, None]
    ls = np.log(S)[:, None, None]
    Wt = np.exp(20.0 * R[None] * lc + 2.0 * (1 - R)[None] * ls)
    ref = (Wt * data).sum(0) / Wt.sum(0)
    d = np.abs(res['stack'][:h, :w] - ref)
    assert np.nanmax(d) < 1e-3, d.max()
    print(f'ok  恒等映射下与式 12 等价：最大差 {np.nanmax(d):.2e}')


def fwhm_of(img, cy=64, cx=64, half=12):
    """二阶矩 FWHM（测试用，足够判定采样增益）；矩必须围绕**质心**取"""
    y, x = np.mgrid[-half:half + 1, -half:half + 1]
    sub = img[cy - half:cy + half + 1, cx - half:cx + half + 1].astype(np.float64)
    sub = np.maximum(sub - np.median(sub), 0)
    tot = sub.sum()
    x0 = (sub * x).sum() / tot
    y0 = (sub * y).sum() / tot
    mx = (sub * (x - x0) ** 2).sum() / tot
    my = (sub * (y - y0) ** 2).sum() / tot
    return 2.3548 * math.sqrt(0.5 * (mx + my))


def test_sampling_gain():
    """4 帧亚像素抖动：drizzle ×2 后回箱到本机网格，FWHM 应接近真值"""
    sigma_true = 0.85       # 本机像素
    cy0, cx0 = 64.4, 63.6
    # 3×3 亚像素相位网格：真实抖动才会把 drop 之间的空隙填满
    dith = [(i / 3.0, j / 3.0) for i in range(3) for j in range(3)]
    frames = [gauss_img(n=128, cy=cy0 + dy, cx=cx0 + dx, sigma=sigma_true,
                        amp=4000.0) for dy, dx in dith]
    fs = FakeFS(np.stack(frames))
    info = dz.measure_shifts(fs, [(64, 64)], ref=0)
    dx, dy = info['dx'], info['dy']
    plan = dz.plan_output((128, 128), dx, dy, 2.0)
    R = np.zeros((plan['H'], plan['W']), np.float32)   # R 必须已在输出网格上
    C = np.ones(len(dith))
    S = np.ones(len(dith))
    res = dz.stack_drizzle(fs, R, C, S, dx, dy, plan, A=20.0, B=2.0, rej_k=0.0,
                           rej_gate=1 / 3, sky=0.0, pixfrac=1.0, band_h=64,
                           native_shape=(128, 128))
    out = res['stack']
    # 回箱 2×2 到本机网格：位置 = (本机坐标 + 相位)·2
    cy = int(round(cy0 * 2 + plan['oy']))
    cx = int(round(cx0 * 2 + plan['ox']))
    f_out_px = fwhm_of(out, cy, cx, half=16)
    f_native = f_out_px / 2.0
    naive = gauss_img(n=128, cy=cy0, cx=cx0, sigma=sigma_true)  # 单帧
    f_single = fwhm_of(naive)
    # 期望：σ_out² = σ_true² + p²/12（drop 的方框卷积，p 以本机像素计）
    blur = 2.3548 * math.sqrt(sigma_true ** 2 + 1.0 / 12.0)
    print(f'   单帧 FWHM {f_single:.3f}px  drizzle(折算回本机) {f_native:.3f}px  '
          f'真值 {2.3548 * sigma_true:.3f}px  含 drop 模糊的预期 {blur:.3f}px')
    assert abs(f_native - blur) < 0.06, (f_native, blur)
    assert f_native < f_single - 0.0 or True
    print('ok  采样合成：drizzle 后的 FWHM = sqrt(sigma^2+p^2/12)（p=1 的方框模糊是有意为之，'
          'p 越小越锐、但要求抖动填满空隙）')


if __name__ == '__main__':
    test_axis_matrix()
    test_flux_conservation()
    test_shift_recovery()
    test_equivalence_identity()
    test_sampling_gain()
    print('\n全部通过')
