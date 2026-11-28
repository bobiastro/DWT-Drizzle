# -*- coding: utf-8 -*-
"""回归测试：measure_shifts 走纯平移兜底时不能再崩（best=None 下标访问）。

背景：rotation 候选全灭 → best 被置回 None → 走 `_trans_only` 兜底 →
旧代码继续 `u = np.asarray(best[1])` → TypeError。
这里构造一个「第二帧星点完全不同」的假数据集，逼它走兜底。
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'core'))
sys.path.insert(0, str(ROOT / 'pydeps'))

import drizzle  # noqa: E402


class FakeFS:
    def __init__(self, imgs):
        self.imgs = imgs
        self.n = len(imgs)
        self.shape = imgs[0].shape

    def __len__(self):
        return self.n

    def frame(self, i):
        return self.imgs[i]

    def __getitem__(self, i):
        return self.imgs[i]


def _stars_img(n_star=60, size=400, seed=0, shift=(0.0, 0.0)):
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), np.float32) + 0.01
    xs = rng.uniform(20, size - 20, n_star)
    ys = rng.uniform(20, size - 20, n_star)
    yy, xx = np.mgrid[0:size, 0:size]
    for x, y in zip(xs, ys):
        x += shift[0]
        y += shift[1]
        img += 8.0 * np.exp(-(((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.4 ** 2)))
    return img


def main() -> int:
    a = _stars_img(seed=1)
    b = _stars_img(seed=1, shift=(2.0, -1.0))
    # c 帧的星点位置完全不相干 ⇒ 旋转候选与纯平移都验不过 ⇒ 必须标 bad 而不是崩
    c = _stars_img(seed=99)
    fs = FakeFS([a, b, c])
    stars = [(int(x), int(y)) for x, y in zip(*np.where(a > 4.0))][:40]
    try:
        res = drizzle.measure_shifts(fs, stars, ref=0, log=print)
    except TypeError as e:
        print('FAIL: 仍然崩在 best[1]:', e)
        return 1
    dx = np.asarray(res['dx'], float)
    dy = np.asarray(res['dy'], float)
    print('dx =', np.round(dx, 3), ' dy =', np.round(dy, 3))
    # 第二帧的星点整体挪了 (+2, -1)，对齐它需要相反位移（dx = -u，故为 +2）
    assert np.isfinite(dx[1]) and abs(dx[1] - 2.0) < 0.6, ('第二帧 dx 应约 +2', dx)
    assert abs(dy[1] - (-1.0)) < 0.6, ('第二帧 dy 应约 -1', dy)
    # 第三帧和参考帧不是一个取景：应被标死 + 插值补位，而不是让程序崩掉
    assert np.all(np.isfinite(dx)), ('不该留下 NaN 位移', dx)
    print('MOVEFIX-OK')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
