# -*- coding: utf-8 -*-
"""逐帧背景梯度校正的单元测试

只依赖 numpy：合成一张带已知平面/二次梯度 + 亮星云块的天空，检查
    · 平面的斜率能被恢复（星云块被 σ 裁剪掉，不会把目标当成梯度）
    · 模型在天空块上零均值（减掉它不动公共天空水平）
    · model_strip 与整帧模型逐位一致（分带读帧不引入接缝）
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / 'core'))
sys.path.insert(0, str(HERE))

import bggrad


def _synth(deg=1, h=512, w=640, sky=1000.0, noise=4.0, seed=1):
    rng = np.random.default_rng(seed)
    yn = (np.arange(h)[:, None] + 0.5) / h
    xn = (np.arange(w)[None, :] + 0.5) / w
    if deg == 1:
        grad = 24.0 * (yn - 0.5) + 12.0 * (xn - 0.5)
    else:
        grad = (24.0 * (yn - 0.5) + 12.0 * (xn - 0.5)
                + 40.0 * (yn - 0.5) ** 2 - 30.0 * (yn - 0.5) * (xn - 0.5)
                + 18.0 * (xn - 0.5) ** 2)
    img = sky + grad + rng.normal(0.0, noise, size=(h, w))
    # 一块亮的"星云"：真信号，必须被裁剪掉而不是被拟合进平面
    img[128:256, 320:448] += 600.0
    # 几颗"星"
    for (yy, xx) in ((80, 90), (400, 520), (300, 100)):
        img[yy - 3:yy + 4, xx - 3:xx + 4] += 3000.0
    return img.astype(np.float32), grad


def _resid_std(img, coef, deg):
    """残差（去掉模型后）在一块干净天区里的标准差（避开星云块与三颗星）"""
    m = bggrad.model_strip(coef, deg, 0, img.shape[0], img.shape[1], img.shape[0])
    r = img.astype(np.float64) - m
    win = r[320:400, 200:300]
    return float(np.std(win - np.median(win)))


def test_plane_recovery():
    img, grad = _synth(deg=1)
    grid = bggrad.block_medians(img)
    assert grid is not None
    coef, _rc, info = bggrad.fit_grid(grid, 1, img.shape)
    assert coef is not None
    # 斜率：真值 24/12（归一化坐标），容差 10%
    assert abs(coef[0] - 24.0) < 2.4, coef
    assert abs(coef[1] - 12.0) < 1.2, coef
    # 幅度 = 模型峰谷，真值 36
    assert abs(info['amp'] - (grad.max() - grad.min())) < 0.1 * 36.0, info
    # 星云块被裁掉（采用的块少于全部）
    assert info['n_used'] < info['n_all']
    # 减掉模型后（不含星云块）残差回到噪声水平
    assert _resid_std(img, coef, 1) < 1.5 * 4.0


def test_global_sky_preserved():
    """减掉梯度面必须不改变**全幅**天空水平（排异阈值以 sky0 为零点）

    注意不能断言"某个子块的天空中位不变"：梯度面本来就是逐点变化的，
    子块天空被修正正是目的；被冻结的是全幅天空电平。
    """
    img, _ = _synth(deg=1)
    grid = bggrad.block_medians(img)
    coef, _rc, _ = bggrad.fit_grid(grid, 1, img.shape)
    m = bggrad.model_strip(coef, 1, 0, img.shape[0], img.shape[1], img.shape[0])
    assert abs(float(np.median(m))) < 2.0
    before = float(np.median(img))
    after = float(np.median(img - m))
    assert abs(after - before) < 2.0, (before, after)


def test_quadratic():
    img, grad = _synth(deg=2)
    grid = bggrad.block_medians(img)
    coef1, _rc, info1 = bggrad.fit_grid(grid, 1, img.shape)
    coef2, _rc, info2 = bggrad.fit_grid(grid, 2, img.shape)
    assert coef2 is not None
    # 二阶模型把二次项吃进去 ⇒ 残差明显更小
    assert info2['resid'] < 0.85 * info1['resid'], (info1, info2)
    assert _resid_std(img, coef2, 2) < 1.5 * 4.0


def test_strip_seams():
    img, _ = _synth(deg=2)
    grid = bggrad.block_medians(img)
    coef, _rc, _ = bggrad.fit_grid(grid, 2, img.shape)
    full = bggrad.model_strip(coef, 2, 0, img.shape[0], img.shape[1], img.shape[0])
    for y0, y1 in ((0, 1), (0, 37), (100, 180), (300, img.shape[0])):
        part = bggrad.model_strip(coef, 2, y0, y1, img.shape[1], img.shape[0])
        assert np.allclose(part, full[y0:y1], atol=1e-4), (y0, y1)


def test_deg0_is_noop():
    assert bggrad.model_strip(np.ones(6), 0, 0, 10, 10, 10) is None
    assert bggrad.model_strip(None, 1, 0, 10, 10, 10) is None
    coef, _rc, info = bggrad.fit_grid(None, 1, (100, 100))
    assert coef is None


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
    print('bggrad selftest 通过')
